# The local stack of 2026-10-31 .. 2026-11-06 (`zig/patches/0058-local-stack-20261106.patch`)

On 2026-11-06 the pair (2 x DGX Spark, this recipe's model, settings and checkpoint) served a leader built from this
recipe's tree plus the changes below. `0058` is their cumulative diff against the tree `zig/Dockerfile` builds
(TF_REF with 0001-0057). Applying it gives the served source, file for file (1286 files, sha256-compared).

Each change went through these steps before it served:
1. CPU tests in a capped container, with red copies (the old code) failing by assertion where a bug was fixed.
2. A read-only source review.
3. A GPU A/B on the pair:
   - every request of a fixed manifest exact (token ids and text hashes) against the reference arm;
   - decode and TTFT non-regression within 5 %;
   - the prompt-cache counters equal;
   - the engine log clean.

The manifest has 88 requests: prose and code at 1 and 4 streams; sampled, mixed and non-stream requests; long prompts
of 18K-540K tokens and a resumed turn. Later windows added tool-choice cases, two pictures, a pair-under-load case and a
high-offset pool case (92).

## What it changes

**Earlier local changes (2026-10-31 .. 2026-11-03)**
- stability10, copy telemetry v3, the protected-text tokenizer memo, reply preparation, hardening steps 1-2 (prompt-cache
  validity and metrics) and the nested-request queue scan. This tree's research variants of 0041-0044 are also part of
  the diff.
- Each was A/B-checked the same way before it served.

**Bundle 1: kept-state placement**
- The least recently used kept state is set aside first, not the oldest saved.
- A state set aside in a search that found room elsewhere stays valid.
- A shorter state that its longer sibling covers stays valid when that sibling resumes in place.

**Bundle 2 / 2b: memory**
- **C1:** the decode indexer's scratch in fixed row blocks, so it no longer grows with the pool. The same bits: per-row
  programs.
- **B2:** the aside flag of a resumed prompt.
- **Window guard.**

**Bundle 3: memory**
- One host copy of a kept prompt's tokens instead of two (prompt-cache host bytes -50 %).

**Bundle 4: memory**
- The chat template renders into page-allocator scratch.
- A keep-alive connection's arena is capped at 1 MiB.
- The tokenizer's BPE runs in a per-word arena.
- Prepare max RSS fell from 262 to 148 MiB.

**Bundle 4c: two bugs**
- **`Server.callBlank`:** its 256-byte buffer dropped the template's blank lines in the `tool_choice: required` gate. Now
  4096 bytes.
- **`engine.zig` gate first token:** the gate retagged `feed` before reading it, so the first token of every
  `tool_choice: required` reply was the backend's draw counter (a stray "¢"). Now read first.

**Bundle 5: rank 0's stability**
- **Nested ids:** `/v1/completions` with nested token ids past u32 is refused. Before, one request aborted rank 0 with
  an unchecked cast.
- **Lost context:** CUDA 709 joins the lost-context list (exit 70).
- **snap_drop frame:** reserves its frame room before writing.
- **Pair-broken exit (exit 75):**
  - a round or prompt pass that fails on a failed RDMA ring;
  - a follower link that closes (`link.zig` Watch).
- **Recovery drill:** a killed worker now means rank 0 exits 75 in about 7 s, and systemd restarts the pair. /health
  was back in 70 s.
- **Stall exit (opt-in):** `TF_DS_STALL_EXIT_S` exits a stalled lane host; off by default.
- **New metric:** `tensorfold:prompt_tokens_cached_total`.

**Bundle 6: rank 0's vision tower opens on the first picture**
- **When it opens:** only while MemAvailable is at least 6.75 GiB; otherwise that picture's request is refused with a
  message.
- **When it closes:** at the first text-only prompt pass 30 minutes after the last picture.
- **Saving:** about 1.3-1.4 GiB of rank 0 while no picture is open.
- **Cost:** the first picture's TTFT is about +0.9 s (0.6 s with the shard in the page cache).
- **Checks:** the window's two PNG replies were exact; a reopen after the idle close was exact in production.

**Bundle 7: /metrics counters**
- `tensorfold:replies_total{finish=stop|length|other}`;
- `tensorfold:reasoning_tokens_total`;
- `tensorfold:thinking_budget_closed_total`, which counts thinks run to the thinking budget, that is long or looping
  thinks.

**Bundle 8: whole and streamed replies agree; one opt-in policy**
- **Streamed think tail:** a streamed think cut by max_tokens now sends its last characters (scout A-F2). Before, a held
  "<" was lost.
- **Open DSML block at stop:** a reply that ends inside an open DSML call block, on an end token or stop string, now
  shows no markup in the whole reply either, as the stream already hid it (A-F3).
- **`TF_DS_PAIR_UNDER_LOAD=1` (opt-in):** prompt chunks pair while other streams' rounds wait.
  - Trial: a cold 32K prompt under one decoding stream, TTFT -3.7 %.
  - Cost: that stream's largest gap went from 0.92 to 1.68 s.
  - Exact, pairs against singles.

**Bundle 9: the bug hunt's fixes (`zig/patches/0059-dsv41-bug-hunt-20261106.patch`, both ranks)**

Ten read-only scouts went over the served engine; the confirmed findings were fixed, with a red copy of the old code
failing by assertion (or a panic) for each testable one.
- **Server input:**
  - a `<tool_call>` wrapping an unclosed DSML block is kept as text. Before, it sliced past its end (a panic in safe
    builds, undefined in ReleaseFast);
  - a lone surrogate in JSON (`\ud800`) no longer turns off the special-token protection of the whole conversation;
  - top_k 0 or past 56 with a temperature above 0 is refused before the prompt pass, not after it.
- **Pictures:**
  - a PNG's rows are inflated only to the size its header names: a 1x1 header over megabytes of zeros inflated
    everything before;
  - a picture's decode first checks rank 0's room (16 B a pixel from the PNG or JPEG header, 1 GiB above memguard's
    line). A picture that does not fit gets a retry message instead of memguard ending the rank;
  - its decode scratch is freed with the picture, not kept until the reply ends;
  - Pillow's child is started again after a reply it read only in part;
  - 15 PNGs of every color type, depth and Adam7 decode exactly as Pillow 12.3 does.
- **Engine:**
  - all-greedy rounds take the sampling candidates (count 1), so a NaN row faults its stream. Before, it drew token 0
    to max_tokens;
  - a round that fails after a stream finished discards it from the core. Before, the core kept a pointer to the
    freed stream;
  - the Engram row cache's index is rebuilt every cache-size evictions. Its tombstones only grew before, so a missing
    row's lookup probed ever longer;
  - large image-row buffers (host, device) are given back after their prompt and when the tower closes.
- **The pair:** the follower ends (rank 0 then exits 75 and systemd restarts both) when its own RDMA ring failed.
  Before, rank 0 waited on its gathers for good with /health ok.
- **Checks:** CPU 696 tests, five red copies; GPU A/B 92/92 exact against bundle 8 with both ranks built from this
  tree. Decode -0.26 to +0.08 % (noise); TTFT within 0.6 %, prose at 4 streams 5 % lower; top_k 0 refused in 5 ms.
- **The recipe's scripts:** `start.sh` turns an unasked clean exit of rank 0 into exit 75, so `Restart=on-failure`
  restarts it, and `stop.sh` marks a requested stop. `TF_DS_STALL_EXIT_S=300` is served as a drop-in.

**Bundle 10: the improve-scan's exact units (`zig/patches/0060-dsv41-improve-scan-20261106.patch`, both ranks)**

Six read-only scouts (decode path, speculation, prompt cache and TTFT, memory, stability and deploys, the API) looked for
what was left; most levers were under 1 % or changed tokens. Three exact ones went in:
- **Responses streams:** an item's text and arguments grow in place. Before, every delta copied the whole text, so a
  think at the 28,672-token budget held about 0.66 GB a stream until the reply ended (four streams: about 2.6 GB).
- **No drafts for a copy ahead:** when the copy proposer has a copy the next window surely takes, the drafter is asked
  for 0 drafts, so its pass (about 3 ms) is skipped or the stream leaves the batched pass. The old draft was dropped by
  that window. With `TF_DS_COPY_MULTI=1`, or more live streams than the round's rows can give two each, the draft can
  be used, and the old behaviour stays.
- **Follower divergence:** a kept state's restore or a prompt's image rows that fail on rank 1 end the pair (exit 75).
  Before, rank 1 logged and went on with a different state while rank 0 had its own: wrong tokens, no error.
- **Checks:** CPU 679 tests and two red copies; GPU A/B 92/92 exact against bundle 9 (the A arm was bundle 9's saved run of
  the same manifest and settings, which halved the fenced time to 13 minutes). Decode +0.06 to +0.71 % (code at one
  stream the most, as the skipped passes predict), within the spread of a saved arm; TTFT within 4 %.
- **Proxy (same day, `mia-local/auth-proxy.py` in the served setup):** a stream the engine closes before its final event
  ends with an error in the route's format; the warm-ahead and the re-warm count the tokens two next-turn renderings share
  (a thinking chat without tools now keeps a state: 2,048 -> 4,096 on the probe); TCP_NODELAY.
- **The recipe's scripts:** `start.sh` saves an exited container's log before removing it (a crash's restart deleted the
  only record of why).

## Settings the pair serves with (systemd drop-ins over this recipe)
- **`TF_DS_GRAPH_CAP=64`, `TF_DS_ENGRAM_CACHE_MB=128`:** exact. Rank 0's minimum MemAvailable in the bench rose from
  6537 to 7102 MiB; decode and long-prompt TTFT stayed within noise. These are now `scripts/config.sh`'s defaults.
- **`POOL=3145728`** (0054; window 1M):
  - Exact, including a cold prompt placed past the old 2M end behind three 700K holders.
  - Costs +0.8 GiB a rank (NVML), so rank 0's minimum in the bench was 6234 MiB.
  - 2M measured 1.4 s against 139 s TTFT for alternating 340K sessions, against 1M.
- **`PROMPT_CACHE_GIB=0.5`:** the bench used about 40 MB of it, so a 1 GiB budget mostly holds stale entries.
- **`TF_DS_PAIR_UNDER_LOAD=1`.**
- **`TF_DS_STALL_EXIT_S=300`:** a lane host stalled for 5 minutes exits, for systemd to restart the pair.
- **The unit:** `TimeoutStartSec=660` (the worker wait's 600 s), `RestartSec=5`, `SuccessExitStatus=130` (a requested
  stop ends inactive, so the watchdog leaves it stopped).

## Checked and not changed
- **The served round-cost table:** measured marginal costs per row match it within about 20 %. The drafter saturates at
  about 3 drafts a stream.
- **`_index_score_rows` at RB 2/4:** every setting is bit-equal, but RB1/w1 is the fastest, at 4.0 against 5.0-5.5 ms
  at 64K-256K keys (`tools/dsv41/index_knobs_bench.py --tree`).
- **The DSpark pass trim** (under 0.5 % exact) and **TF_DS_PROMPT_ALIAS** (0.1-0.2 GiB, lifetime risk): not done.

## What is not established
- **The follower before bundle 9.** Up to bundle 8, rank 1 ran a follower built after bundle 2b. From bundle 9 (0059)
  both ranks run builds of the same tree, A/B-checked together.
- **Long soak** at these settings, and a VisionShortOfMemory refusal under real memory pressure.
EOF
