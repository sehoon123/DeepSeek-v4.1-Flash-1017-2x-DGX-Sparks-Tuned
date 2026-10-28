# Prefill progress states, translated-stream keep-alives — 2026-10-29

This follow-up carries patches **0051–0053** on top of the qualified 0046–0050 cache and transport fixes. It addresses
how a long prompt pass that is interrupted loses its work, and a remaining body-idle gap on two API routes. It does
**not** make cold prefill itself faster, and it is not a claim that every client-side stall is resolved.

## What changed

- **0051 — rolling prompt progress states** (`PROMPT_PROGRESS_TOKENS`, recipe default 65536; `TF_DS_PROMPT_PROGRESS_TOKENS`
  to the engine; 0 turns it off). Before it, a long cold prompt pass kept states only at its history mark, the shared
  system cuts and its last chunk end. A pass that ended early (a client that left or timed out, a network drop, an
  abort, a proxy restart) kept nothing beyond the system prefix, so a retry recomputed the whole prompt, and a client
  deadline shorter than the cold prefill could never be passed by retrying. Now a long pass also keeps a state every
  65,536 tokens. Each newer state of the same prompt retires the progress state it supersedes, so at most about one
  extra state exists while a long pass runs and none after it completes. A progress state takes room only from
  superseded or progress states and never evicts another conversation's (if nothing fits, nothing is evicted);
  superseded progress states give their room before a history mark evicts anything. Same token positions and chunk
  ends as an uninterrupted pass. Snapshot budget, retention, victim order and serving limits unchanged.
- **0052 — translated-stream keep-alive.** The native OpenAI SSE comments of 0050 did not reach `/v1/messages`
  (Anthropic) or `/v1/responses`, which send nothing during a long prompt pass. These now send Anthropic's documented
  `event: ping` / `{"type":"ping"}` and an SSE comment respectively, every 15 seconds, after the stream opens and between
  whole events. Transport only: never a delta, token, usage item or first token.
- **0053 — overflow wording.** In a narrow band at the window's end the backend's `PromptTooLong` reached clients as a
  generic server error. Pi 1.4.0's own classifier (run directly) read it as neither an overflow nor retryable. It is now
  worded as clients detect an overflow ("exceeds the context window"), so Pi compacts and retries. Text only.

No weights, EXL3 quantization, overlay, arithmetic, sampling, context/output limits, four-stream capacity, 1 GiB
snapshot budget or permanent guards changed.

## Why this mattered for the original incident

Pi 1.4.0's defaults (read in its source): `httpIdleTimeoutMs` = 300,000 ms as undici `bodyTimeout`/`headersTimeout`, and
automatic retry (3 attempts, 2/4/8 s backoff). Before 0050, a cold pass longer than 300 s was cut by the client and
restarted from the system prefix on every retry: up to four ~300 s attempts of lost work. 0050 removed the idle cutoff
on chat completions; 0051 makes any remaining interruption (Esc, a network change, a proxy restart) resume near where it
stopped instead of from zero. The identity of the hop that cut the original sessions on the user's separate PC was not
observed.

## Qualification (actual model, both GB10s, unchanged production settings)

Reserved window 02:19–03:01 KST on 2026-10-29, public proxy closed, temporary Restart=no, both-node absence proofs,
3,072 MiB floors. The baseline is the **complete, promoted 0050 run** of the previous window (the image production
served), verified by hash and reused, **not rerun or counted twice**. The candidate was started through the normal
service `start.sh` with the staged recipe files (the real deployment path), from an empty cache.

| Gate | Result |
| --- | --- |
| Long/placement suite (648K–850K synthetic token sessions) | 9 rows, every complete output token array identical |
| Interrupted pass: owned disconnect 45 s into a 196,608-token cold pass, identical retry | resumed at **65,536** (0050: 0); retry prefill **52.4 s** vs 76.3 s; output identical to the uninterrupted reference |
| Cold prefill with progress states (648,878 / 850,123 tokens: 9 / 12 states) | **−0.09% / +0.21%** vs 0050 |
| Blocked placement (0049) | resumes in place at 679,936, prefill 2.42 s |
| Pi 1.4.0 SDK + production proxy, 850,123-token cold pass | 382.7 s, 25 keep-alives, max body gap 15.08 s, one fetch |
| Ordinary/cache/pressure/four-stream/nested + thinking/tools/image | 20 + 5 responses identical, cache counts identical |
| `/v1/messages`, `/v1/responses` over ~150K-token cold prefills, 25 s client idle deadline | stayed connected (14 / 28 keep-alives, max gap 15.05 s), terminal events, same text as JSON replays |
| Owned decode disconnect + exact recovery; smoke; functional validation | pass (video skipped: no PyAV) |

Server log for the 648,878-token cold pass: `kept 10; 1 states, 7 MiB` — ten states kept as the pass advanced,
one held after it. Minimum sampled MemAvailable in the window: 5,232 / 7,027 MiB (head / worker).

CPU evidence before the window: host tests 273/273, server 69/69, seam 55/55; three removed-fix controls fail at the
intended assertions (no progress marks, no rolling retirement, unrestricted eviction); a fake-backend end-to-end test
shows a retry after a cancel resuming at the last progress state with tokens equal to an uninterrupted pass. Translated
routes over real HTTP (31 s engine gap, 17 s client idle deadline): the 0050 server times out on all three paths, the
candidate completes. Native ReleaseFast build 28/28 (sm121). Upstream `629c524a` plus all 49 patches reproduces all
1,279 files of the compiled tree.

## Deployment identity

- Both ranks: `tensorfold-dsv41:exl3-tp2-b35b0716c6f3` (local COPY-only packaging of the qualified 0050 image with the
  rank-0 native binary `c5f62536…`; rank 1's follower binary is unchanged and identical). No registry image is published.
- Environment identical to 0050 except exactly `TF_DS_PROMPT_PROGRESS_TOKENS=65536` on both ranks.

## Observed but not caused by this change

Both translated-route test prompts were fresh synthetic prose and prefilled at **~711 and ~345 tokens/s** (149,915 and
149,926 tokens in 210.8 s and 434.7 s), against ~2,500 tokens/s for the suites' other long prompts in the same window
(chat route, 0050 and 0053 alike: 131K in 50.2 s, 360K in ~145 s). Those other prompts had been processed in earlier
windows. A cold-cache (Engram row reads from NVMe) effect on genuinely new text is the leading hypothesis, **not yet
verified**; it is being measured separately. Until then, read published cold-prefill speeds as warm-cache figures.

## Limits

- Cold or genuinely changed long prompts still need full computation; keep-alives do not override absolute deadlines.
- A restart still loses all KV. Two long sessions that do not fit the pool together still recompute on every switch;
  a larger shared pool and an NVMe spill tier (as in MiaAI-Lab's recipe) are being evaluated separately.
- No deliberate GPU or transport fault injection; follower save-copy acknowledgement remains open.
