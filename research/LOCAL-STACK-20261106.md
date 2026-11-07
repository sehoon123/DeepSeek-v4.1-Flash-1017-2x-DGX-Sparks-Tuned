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

**Bundle 11: stability and speed units (`zig/patches/0061-dsv41-stability-units-20261107.patch`, both ranks)**
- **First drafts:** a stream whose first round copies from its prompt asks for no first drafts when the copy holds,
  through the same rule and copy probe later rounds use since 0060. The first token is read before the first draft is
  asked; it goes out at the same moment, or one drafter pass earlier when no draft is asked.
- **The Pillow child:** each exchange has a 110 s deadline (four times the 27 s estimate for a 134 Mpx picture, under
  the proxy's 120 s wait for a non-streamed reply). A child past it is killed and restarted, and that picture is refused
  in the lane's words; before, a hung child held every later non-PNG picture while /health stayed ok. A retry that fails
  in any way restarts the child too.
- **/metrics:** `requests_total{key,status}` counts every reply, `key="unauthenticated"` when API keys are off (the
  proxy fronts the engine), so 400/500/503 rates show.
- **Checks:** CPU 685 tests and three red copies; GPU A/B 92/92 exact against bundle 10 (saved arm), decode -0.28 to
  +0.13 % (inside the run-to-run spread), TTFT within 4.3 %.
- **The hosts (same night, root):** `vm.swappiness` 10 and `kernel.panic` 10 (`/etc/sysctl.d/90-dsv41-serving.conf`) and
  systemd's `RuntimeWatchdogSec=60` (the board's SBSA watchdog) on both Sparks; old images removed.
- **Left out:** streaming DSML tool calls as they are written. Three implementation attempts were blocked by the model
  provider's request classifier, not by the design.

**Bundle 12: decode kernel path (`zig/patches/0062-dsv41-decode-hostpath-20261107.patch`, both ranks together)**
- **Acceptance beside the absorb:** rank 0 copies a round's candidates and records to the host and records an event
  before the absorb; the host waits on the event, so the drafts' acceptance and the next frame overlap the absorb. The
  follower no longer syncs at a round's end; every host read still waits for its own copy.
- **Per-rank records:** each rank's head reduces its vocabulary half to its top 64 (value, id) records, 528 bytes a
  row, and the ranks gather those instead of the full logits. The host merges the two halves with the candidates
  kernel's integer order key, so a flush-to-zero float mode cannot change a tie; ties go to rank 0's lower ids and a NaN
  in either half is flagged as before.
- **The verify frame (kind 12)** carries the record count; a rank on 0061 and a rank on 0062 refuse each other's frames
  (the pair restarts until both run the same build).
- **Checks:** CPU 689 tests and four red copies; source review PASS. Two GPU A/B windows, 92/92 exact each, against
  bundle 11 (saved arm, and bundle 11 again in the first window). Decode: +0.23 to +0.84 % in three cells across both
  windows; 4-stream prose +0.38 % and -0.87 % (inside the run-to-run spread). The first window failed the 5 % TTFT gate:
  one group of four requests started 48 ms late, and the pooled median of a 4-stream cell sits between two arrival
  clusters. The second window read TTFT by arrival rank (each rank within 5 %, declared before it ran) and passed:
  +1.1 to +1.7 % (4-stream prose). Its pooled median was +9.4 %, so the old gate would have rejected it too: the
  promotion rests on the per-rank gate, which does not see up to two late groups of six.
- **The late groups:** a group of four starts 44-50 ms late outside the prompt pass in about one run in two, in every
  build since bundle 6 at least (and a code group's first prompt pass sometimes takes 31-35 ms longer). The length is
  fixed, which a CPU throttle of the bench would not give; the cause is not known and is likely the server's.

**Bundle 13: pictures' decode memory (`zig/patches/0063-dsv41-picture-memory-20261107.patch`, rank 0)**
- **Measured first** (served image, one CPU): Pillow's child at its peak needs 14.7 (JPEG), 16.3 (AVIF), 19.2 (JPEG 2000)
  and 22.4 (WebP lossless) bytes a pixel, plus the lane's 3 for the RGB it receives; the served PNG decoder 11.5 (8-bit
  RGB) to 18.2 (16-bit RGBA). The slowest decode at the lane's largest picture (134 Mpx) is JPEG 2000, about 15 s, so the
  110 s deadline has about a 7x margin (the JPEG 2000 figure is x4 from 3.7 s at 33.5 Mpx, on blocky noise). The lane's
  check counted formats other than PNG and JPEG as 24 Mpx: a 134 Mpx JPEG 2000 passed it and would take about 2.9 GiB
  for about 15 s (x4 from 644 MiB at 33.5 Mpx, plus the lane's RGB), far longer under memguard's 4.25 GiB line than the
  pair's 2 s hold (MEMGUARD_HOLD_S=2, hard floor 2.5 GiB).
- **The child** checks every picture at the size its header declares before its decode, on the lane's line (4.25 GiB
  plus 1 GiB): 16 bytes a pixel for JPEG, MPO and PNG, 26 for other formats. Pillow's own size check of a picture it is
  about to decode, an embedded one too (ICO, ICNS, BLP, GIF frames, TIFF tiles), passes the same check. A refusal uses
  the lane's words. The child frees each picture after replying (it held up to 1.5 GB until the next one).
- **The PNG decoder** takes IHDR first and once only. Before, the check read the first chunk's size and the decoder the
  last IHDR's, so a 1 MB request could decode 134 Mpx unchecked. Any other PNG is Pillow's, which is the reference's.
- **One decode at a time:** two large pictures cannot pass the check together. A picture can wait behind a slow one.
- **Checks:** CPU 695 tests; red copies without the check, skipping JPEG, IHDR anywhere and without the wrapper each
  fail; the child before and after gives byte-identical answers on 14 inputs; two source reviews PASS. GPU A/B 92/92
  exact against bundle 12. On the candidate: a JPEG, a small GIF and a two-IHDR PNG decode; a 134 Mpx GIF (100 KB) is
  refused in 3 ms and rank 0's MemAvailable does not move. The lock, the wrapper and slow codecs were tested on CPU
  only. Not explained: the first picture's TTFT in the bench was 2.05 s against 1.72 s (both arms open the tower cold;
  the second picture was unchanged).

**Bundle 14: recovery and latency fixes (`zig/patches/0064-dsv41-recovery-latency-20261107.patch`, both ranks)**
- **The first token before the first drafter pass:** the host now sends a stream's first token, and reports its prompt
  pass, as soon as the token is committed; the stream's first drafter pass runs after. Its time used to sit inside every
  drafting reply's time to first token. Measured in the second window: TTFT by arrival rank -0.1 to -3.5 % in every cell
  but one (code c4 rank 1, +2.8 %). `prefill_seconds` now ends at the first token.
- **A drafter pass on a failed ring ends the pair** (exit 75), as a round and a prompt pass did; before, recovery waited
  for the 300 s stall exit.
- **Writes that never block:** each send takes the room the socket has; a client that stops reading is given up after
  write_ms without progress. Before, a write past that room waited with no timeout and held its thread and socket.
- **A peer that sent a byte and closed is gone** (POLLRDHUP), so its reply stops instead of running to max_tokens.
- **The recipe's start.sh** (with this bundle's windows): a failed ssh, or one "not running" reading, no longer restarts a
  healthy pair (two readings 30 s apart); once rank 0 has ended, rank 1 is killed (its engine has no SIGTERM handler, so
  docker stop waited 30 s). `tools/test_watch_rank1.sh` runs the watch loop's own text with stubs.
- **Checks:** CPU 703 test executions and four red copies; two source reviews PASS. The first GPU window was exact but
  failed the TTFT gate (code c4 rank 0 +10 %); the second, on the change above, passed: 92/92 exact. The first window's
  slow replies were request s2 of the code cell (65 prompt tokens against 59) being served first in 3 of 6 groups, not
  the drafter pass as first diagnosed; a 65-token prompt prefills about 35 ms slower than a 59-token one (not explained).
- **The bench** now records each reply's server timings and the bench cgroup's throttling, and gates the count of late
  groups of four. Late groups this time were server-side, inside the prompt pass.

**Bundle 16: fixes from other recipes (`zig/patches/0065-dsv41-gb10-downloads-captures-calibration-20261108.patch`, both ranks)**
- **GB10 downloads through a pinned bounce:** with driver 580.178.04, a synchronous device-to-host copy straight into
  pageable memory can report success while some pages keep their old bytes (TensorFold upstream a3962b45 measured 10 of
  60 runs). GB10 reports itself integrated on both nodes, so every such copy (the drafter's two each pass, the vision
  tower's span, calibration, profiling, tools) now lands in a 4 MiB pinned buffer per thread and is copied on. Decode:
  -0.32 to -0.59 % against the previous day's bundle 14 run, -0.35 to +0.05 % against bundle 14 measured the same night;
  the same binary moved -0.16 to -0.42 % across the day and six timed repeats spread about 0.4-0.5 %, so a cost of up to
  about 0.3 % from the staged copies cannot be excluded.
- **A failed graph capture no longer poisons the next:** a stretch that failed inside its capture left two join flags
  set, so every later capture waited on the dead capture's event and failed (901) until a restart while /health stayed
  ok (MiaAI-Lab pull request #119 by ldenaeyer). It now joins, ends and clears; a failed round also waits out its Engram
  reads, since a late one would be copied into the rows the next round reads.
- **Engram ids checked before a read batch allocates** (bertholomus/TensorFold 12132870): a refused id leaked the
  batch's buffers.
- **MiaAI-Lab 0050, the per-stream draft calibration, compiled in and off** (`TF_DS_T25_DRAFT_CAL`, rank 0 only; their
  greedy prose 79.7 to 85.4 tok/s, code unchanged). Ported with one change from the source review: a head granted no
  drafts moves back toward no correction instead of staying undrafted. On/off A/B on the pair, same image: replies
  identical (92/92, and 34/34 at T 1.0 / top_k 56 with fixed seeds); at this recipe's sampling, paired per request,
  prose -0.8 % (one stream) and +0.3 % (four), code -1.7 % and -2.0 % (standard errors 0.3-0.8 points); greedy prose
  +0.9 %. Left off: the rule set before the run (both prose cells at least +1 %, none below -1 %) was not met.
- **Checks:** CPU 757 test executions and twelve red copies; two source reviews PASS (nonblockers taken: the lock-in,
  the Engram drain, comment order). The first GPU window stopped before any request on a harness defect (stale binary
  hashes; production returned, 92/92 identical, and reopened); the second passed: 92/92 exact, decode -0.3 to -0.6 %,
  TTFT by arrival rank within the gate (prose c4 pooled +5.9 %, from bench-side delays: client minus server 45 ms in its
  late group), long prompts +0.0 to +0.4 %, three late groups against two.

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
