# A position pool larger than the request window (0054) and the Engram read-ahead reset (0055)

Date: 2026-10-29. Measured on the two-Spark pair this recipe serves from (GB10 ×2, TP2, Mia EAQ 2.9 bpw with the Keys
overlay, `PARALLEL=4`, `PROMPT_CACHE_GIB=1`, `CONTEXT=1048576`). Synthetic prompts only.

## The problem

Two long conversations that alternate (for example two coding-agent sessions, or one session and a second tool-heavy
one) each need an extent of about the conversation's length plus its reply budget in the shared KV pool. Before this
patch the pool was the request window, 1,048,576 positions. Two extents of ~635K positions do not fit together, so the
second session's placement had to retire the first session's kept state. When the user switched back, the first
session was recomputed from position 0: minutes of prefill on every switch.

The cache was doing what it was built to do, keeping the most recent state of a conversation as rows inside the KV
pool. It ran out of room in the pool, not in the cache's budget (`PROMPT_CACHE_GIB`).

## What other recipes do

- **MiaAI-Lab's recipe** (`zig/patches/0008-pool-window.patch`): the pool may be larger than the window
  (`TF_DS_POOL`). The RoPE tables stay at the window (`TF_DS_WINDOW`), so a request still stops at the model's window,
  and the KV caches grow with the pool, about 0.9 GiB a rank per 1,048,576 positions by their measurement. Its 0012
  (by Robert Wojciechowski, @wojo) also clears an Engram read-ahead a cancelled pass leaves behind.
- MiaAI-Lab's later series adds a host-memory spill tier for kept states (states leave the KV pool for host or disk
  memory and come back on a hit). That is a much larger change (thousands of lines across the cache, the lanes and the
  follower) and is **not** part of this release.
- Paged-KV servers (vLLM's automatic prefix caching, SGLang's radix cache) avoid the fixed-extent problem by
  allocating KV in blocks; this engine's dsv41 lanes use contiguous extents with compressed per-layer state, so the
  pool-size approach is the one that fits without a re-architecture.

## What changed

- `zig/patches/0054-dsv41-pool-window.patch` adapts MiaAI-Lab's 0008 to this tree (its imports differ). With
  `POOL` unset nothing changes: the pool is the window, as before.
- The recipe's `POOL` (`scripts/config.sh`, empty by default) passes `TF_DS_POOL=$POOL` and `TF_DS_WINDOW=$CONTEXT`
  to both ranks and `--pool $POOL` to rank 1. `start.sh` refuses a `POOL` that is not a multiple of 2,048 between
  `CONTEXT` and 4,194,304.
- `zig/patches/0055-dsv41-engram-ahead-reset.patch` carries the read-ahead part of MiaAI-Lab's 0012: every prompt pass
  drops a read-ahead an earlier pass left (one that ended between two chunks: a cancel or an error), so no chunk can
  take rows hashed from another prompt's ids. This tree already did this with `TF_DS_PAIR=1` (the recipe's setting);
  now it is unconditional. No switch.

## Evidence

All measured in one guarded window (public proxy stopped, no user traffic, both nodes checked for idle first), against
the deployed predecessor (0053) as the reference.

**Phase 1, `POOL` unset (0054 inert, 0055 on).** Every suite was compared complete-token-array by complete-token-array
with the reference: 9 long-context rows (683,602-token placement with the blocked-placement prefix restore, an
850,123-token cold pass through the SDK, the 45 s cancel at 196,608 tokens resuming at 65,536), 20 ordinary rows
(cache, pressure, four-stream, nested) and 5 feature rows. All identical.

Two-session alternation on the 1M pool (synthetic sessions A and B of 600,000 tokens each, then each extended by
2,000 tokens, then again):

| request | cached | prefill |
| --- | ---: | ---: |
| A1 | 0 | 263.2 s |
| B1 | 0 | 256.1 s |
| A2 (back to A) | **0** | **257.1 s** |

B1's placement retired A1's kept state, so A2 was recomputed from position 0. This reproduces the problem on the model.

**Phase 2, `POOL=1572864`.** The same alternation:

| request | cached | prefill |
| --- | ---: | ---: |
| A1 | 0 | 256.8 s |
| B1 | 0 | 255.6 s |
| A2 (back to A) | **598,016** | **2.49 s** |
| B2 | 598,016 | 2.49 s |
| A3 | 600,064 | 2.44 s |
| B3 | 600,064 | 2.43 s |

A2's complete output equals phase 1's A2 (recomputed cold). The features suite, the stock smoke and validate tools and
an owned decode-disconnect recovery check also passed under `POOL`.

**Memory.** `MemAvailable` right after the load: at most 6,394 / 8,466 MiB (head / worker) with the 1M pool, 6,006 /
8,032 MiB with `POOL=1572864`: ~390-430 MiB less per Spark for 524,288 more positions, in line with MiaAI-Lab's
~0.9 GiB per 1M. Minima under phase 2's load: 4,892 / 6,917 MiB (gate: 4,096 on both). On this site the head reads
~4.8 GiB idle after the tests, against `memguard`'s 2.5 GiB floor.

A long chat request's preparation memory was measured separately, on CPU, because the window's two-session rows are
token prompts (they skip the chat render): the server with this tree's tokenizer and template held ~84 MiB per open
~683K-token chat request (92 MiB idle, 176 MiB with one open, 262 MiB with two, 105 MiB after). MiaAI-Lab's 0023
describes 1.3-1.5 GB per long request in their tree; this tree's 0044 already releases the tokenizer's scratch, so
0023 is not needed here.

**Engram cold reads (a measurement, not a gate).** A fresh, never-seen ~100K-token prose prompt prefilled at
2,627 tok/s, and the same prose behind a different first line at 2,636 tok/s. A fresh pass read ~6.7 GB (head) and
~18.8 GB (worker) from NVMe per 100K tokens without slowing down. Fresh text alone is therefore not the reason two
earlier translated-route test passes ran at ~711 and ~345 tok/s; that remains **unexplained** (one open hypothesis is
NVMe thermal throttling after long sustained reads, not measured).

## Limits

- `POOL` keeps more long conversations resident; it does not make cold prefill faster. The first pass of each long
  session still takes minutes.
- Two sessions stay resident at `POOL=1572864` while each conversation is at most ~737K tokens with a 32,768-token
  reply budget (two extents of the prompt plus the budget, plus the 32,768-position room placement leaves to grow,
  fit 1,572,864); larger sessions, or a third long one, can still retire a kept state. The window's sessions were
  sent with `max_tokens` 32,768, as Pi sends. A larger pool costs about 0.9 GiB a rank per 1M
  positions: measure `MemAvailable` on both Sparks before raising it.
- A kept state also needs room in the prompt cache's own budget (`PROMPT_CACHE_GIB`): one state is ~7.6 MiB, so
  1 GiB holds well over a hundred.
- Rollback: remove `POOL` (or set it empty) and restart; the pool is then the window again.
