# Kernel rounds (2026-10-20 20:55 - 2026-10-21 02:25)

Where the decode round's time goes below the policy level, what is left of it, and the two changes that came out of it
(`zig/patches/0019`). Every engine run below compared its replies with serial decoding's token hashes: all equal (a
prefetch or a timing probe does no arithmetic). "Head" is rank 0's Spark, "worker" rank 1's.

## How much is left: the round against the read ceiling

- **The read ceiling is 263 GB/s, not 226.** A plain CUDA read of 768 MiB (`tools/bwmax.cu`: 16-byte loads,
  `ld.global.nc.L1::no_allocate`, 256 threads, 4 blocks a SM) reaches 263.4 GB/s (96.5 % of LPDDR5X's 273); an MoE-like
  pattern (40 x 48.7 MiB, one launch a layer) 244-249 GB/s. Torch's 1 GiB copy (226 GB/s) was not the ceiling; this
  corrects `README.md` and `results/bandwidth.txt` as first published.
- **The MoE already reads each expert once a round.** The kit's `grouped_cp` programs are laid out by expert place, with
  every row that picked the expert in its 16-row member tiles, the shared expert first; the grid's rows x 7 places are the
  worst case and unused places exit at once. Rows that share experts share the bytes, so there is no repeated read to remove.
- **Linear layers stream near the ceiling** once overlapping launches are counted once: `wq_b` 12.5 MiB a rank in 57.6 us
  (227 GB/s), `wo_a` + `wo_b` 22.5 MiB in 98.4 us (240 GB/s, PDL overlap 30 us).
- **Bound.** At the served depth (K3, 38.85 ms a round; streaming kernels run 85 % of it) the round reads ~8.8 GB a rank
  (MoE ~5.3 GB for ~18 unique experts a layer from the measured 4.3 ms a row, other weights 2.6 GB, drafter 0.9 GB): at
  263 GB/s that is 33.5 ms, so even perfect kernels and zero latency-bound gaps give at most ~13 %; the latency-bound
  5.7 ms (gathers, router, hc mixes) is the upper bound of everything not streaming (+17 %). What is ours to change there
  is the prefetcher and the gather's host side; the kit's kernels (no bit-exact source) and GPU-initiated RDMA are not.

## Round 1: the L2 prefetcher (2026-10-20)

120 paced launches a round (1 block x 128 threads, 3 us delay, 230 GB/s, 16 MiB budget; ~1.9 GB a round at one stream).
Constants made settings in a dev build, swept 16 runs A..H..A (req-test-x, 11 sparkDash prompts, 4,389 tokens):
prefetch off -4.48 % (it is worth +4.7 %), 260 GB/s -0.13 %, unpaced -0.23 %, 2 blocks -0.76 %, 24 MiB -0.26 %,
200 GB/s -0.13 %, **no delay +0.35 %**. `joinPrefetch` has one call site after the layer loop: prefetch completion never
gates the main stream (a trace heuristic suggesting a 311 ms stall was chance alignment; a shifted-offset test showed it).

## Round 2: the tensor-parallel gather

- The bench (`tf-dsv41-rdma-bench`, both Sparks, 50-gather graphs): 8.4 / 9.4 / 12.2 / 13.4 us at 1 / 2 / 4 / 6 rows.
- In the engine (the kernels' GPU probe switched on): stage 3.4 us + delivery 16.7 us + copy-out 3.3 us = 23.4 us a gather,
  ~94 a round. Prefetch off: 19.0 us but decode -4.6 %: the prefetcher costs each gather ~4.4 us (~0.41 ms a round) and is
  still worth it. Delaying it hurts (10 us -1.2 %, 25 us -3.3 %); starting at once: +0.47 %.

## Round 3: where a gather's delivery goes

A host-side probe in the proxy loop: posting 0.22-0.26 us, flag store 0.03 us, loop 23-29 ns (the proxy software is not the
gap); NIC round trip 3.75 us at one row on the bench, 7.5 us under the served load. The rest (GPU doorbell -> CPU, CPU flag ->
GPU, through pinned host memory) is 8.8-12.5 us of a 16.5 us delivery at the served depth. Idle, the same round trip is
1.15 us on the head and 1.18 us on the worker (`tools/pingpong.cu`, one GPU thread and one host thread through pinned
memory): under the weight streams it is 8-10x slower. Only GPU-initiated RDMA would take it out (days, hang risk; not done).

## Round 4: each prefetch site's budget (2026-10-21, `results/ps-summary.txt`)

Per-site budgets as settings (A: `wo_a`, `wo_b` behind q's linears, 16 MiB; B: the ffn mix, router and shared expert behind
`wo_b`, 8 MiB; C: the next layer's mix and input linears behind the experts, 16 MiB; H: the head after the last layer,
16 MiB). Against the no-delay reference (median of three reps, median over the 11 prompts): A 12 / 24 -0.26 / -0.17 %,
B 4 / 14 -0.81 / -0.50 %, C 8 / 24 -0.75 / +0.18 %, H 0 / 32 -0.18 / -0.50 %. The engine's budgets are the best measured
and stay. The 3 us delay again -0.72 % at one stream; at two and four streams no delay is +0.58 / +0.99 % a stream.

## Round 5: the drafter's pass prefetching (`results/dp-summary.txt`)

The drafter's pass (three DSpark blocks of the same layer type, then the head) had no prefetch at all ("the served L2
prefetch forks move no bytes and are left out"). Giving it the target layers' four sites on a stream of its own, forked and
joined inside the pass: 84.03 / 84.11 / 84.14 against 83.79 / 83.78 / 83.79 tok/s (+0.37 %, 9 of 11 prompts faster, none
slower), four streams +0.47 % a stream; drafts and rounds unchanged.

## In production (`results/decode-0019-ab.txt`)

sparkDash's decode bench on the served pair, the new image against itself with the previous behaviour set through the
environment (`TF_EXL3_L2_DELAY_NS=3000 TF_DS_DRAFT_PF=0`), back to back: prose +2.0 / +0.9 / +0.1 %, code +1.5 / +0.8 /
+0.6 % at 1 / 2 / 4 streams (geomean +1.0 %). `tools/validate.py`: 8 of 8 replies equal to the earlier image's.

## The serving environment

- **Polling `nvidia-smi` costs decode.** A dashboard on the head ran four `nvidia-smi` queries every 2 s (~2 runs a
  second, including `dmon -c 1 -d 1`) on the head and, over ssh, ~1.6 a second on the worker. Pausing it (SIGSTOP)
  during a decode bench: +0.9 % prose and code, both reps; a second identical loop beside it: -7.2 to -7.6 %. Polling
  every 10 s instead (its `POLL_INTERVAL_GPU` / `POLL_INTERVAL_BANDWIDTH`), A/B/A/B on the served pair: +0.5 / +0.8 /
  +0.8 % prose and +1.0 / +1.3 / +1.6 % code at 1 / 2 / 4 streams, geomean +1.0 % (`results/decode-poll-ab.txt`).
  Per-process ioctl counts showed no other GPU poller. Keep GPU monitoring sparse while serving, or off.
- **Plain CPU load costs little.** 18 busy loops on the head's 20 cores: -1.1 % prose, -1.4 % code. Measurement noise of
  0.8-1 % in one sweep came from short commands on the head (`top`, `perf`, `nvidia-smi`) landing in two prompts; with the
  head left alone the same configuration repeated within 0.01 tok/s.
- **A spinning display kthread.** On the head (not the worker), `nvidia-modeset/kthread_q` has used a whole big core since
  boot (100 % in state R, 5 voluntary context switches in three days): `perf` puts 96 % of it in
  `nvEvoWaitForCoreNotifier`, NVKMS waiting for a display-engine notifier that never comes. The head's DRM has simpledrm
  (the EFI framebuffer) beside nvidia-drm with no connectors; the worker has one DRM device and an idle kthread. Its
  measured effect on decode is nil so far (the pinned-memory round trip above is the same on both Sparks); clearing it needs
  a module reload or a reboot.

## Round 6: copy windows of 48 rows (2026-10-21 03:00-04:41, `zig/patches/0021`, `results/wide-copy-windows.txt`)

A lone stream's copy windows were capped at 16 rows by two things, both in the engine's constants rather than its
arithmetic. The window rings keep model.py's RING_EXTRA of 16 rows beyond the 128-key window, so a wider window would
overwrite keys it still reads; and the decode round's kernels are captured (`kit/aot`) for at most 16 rows wherever a
row count becomes a compile-time constant. Earlier "ROUND_ROWS > 16 changes hashes" came from a third place: a prompt
chunk of up to RING_EXTRA rows takes ring mode, and a 17-32-row chunk in ring mode has no captured variant
(`MissingTritonVariant`), so the prompt pass must keep ring mode at 16 rows.

- `_sparse_attn_part` takes the ring's size as a runtime argument and reads keys by position (slot = position % size), so
  48 spare rows change no arithmetic: ring 160 at 16-row windows gave the same hashes as ring 144.
- Of the decode kernels with a row-count constant (`_rowmm_parts` RB 1-16, `_rowmm2r` RB 3-16, `_hc_mix_part` RB up to 8,
  `_hc_mix_pf` RB up to 8), only the router's `_rowmm_parts` (RB = rows) has no variant past 16; its chunk sums run in
  launches of at most 16 rows now (a row's sums do not depend on the rows beside it). The others tile rows by grid.
- Exactness: every run's token hashes equal serial decoding's: natural copies, oracle copies 0 / 25 / 50 / 90 % wrong,
  three streams, temperature 1 (11 prompts against a serial run), 32K / 64K / 126K prompts, four streams at 48 rows.
- Speed (one stream): perfect copies 212 -> 279 tok/s (+31.6 %; 15.4 -> 37.7 tokens a round), a quarter / half wrong
  +13.8 / +17.4 %, nine in ten wrong +4.6 %, the natural mix +6.5 %; text without copies and four streams unchanged.
  Arena +261 MB per 16 rows (48 rows: +522 MB a Spark). Served after the deploy: sparkDash's six cells unchanged (code
  four streams +3.3 %, within that cell's spread); `tools/validate.py` 8 of 8 equal to the earlier replies.

## Round 7: drafts coupled to the sampler's randomness (negative, `results/draft-coupling.txt`)

The target samples with a Gumbel race keyed by (seed, position, token id) through splitmix64 (`sampling.zig`), so a
draft can race the same draws over the drafter's own scores: identical distributions would then always agree. A research
build replaced the Markov loop's `_score` with a kernel doing that for sampled streams (`patches/draft-coupling.diff`, off
by default, not shipped). At temperature 1 (top-k 20, top-p 0.95) it lost 2.1 % tokens a round at the drafter's own
scale; sharpening it (alpha 1.5-10) brought it back to the argmax drafts' rate but never above it (best +0.4 %, noise).
After top-p the target's draw is mostly its argmax, and where it is not, the drafter's runner-ups do not predict which
other token it draws. Every sampled run's hashes equalled serial decoding's.

## Round 8: copy windows on realistic edits, at temperature 1, and their policy (`results/copy-policy.txt`)

- Four of this repository's Python files, each to be written out whole with one scattered edit (a rename, print -> log,
  PASS -> OK, url -> base_url): 48-row windows +18 to +20 % over 16 rows (180.7 -> 213.5-216.9 tok/s), and +18.1 % at
  temperature 1 (175.5 -> 207.3); at temperature 1 the natural copy mix +4.6 %. Every reply equal to a serial run's.
- Long repetitive generations (30 tests, an enum of every HTTP status, a 40-element table, a 12-command CLI): -0.5 % (T=0)
  and -0.7 % (T=1). One T=1 code prompt lost 9 %: short repeated patterns grow windows that sampling then leaves (the
  same 334 tokens accepted, 453 instead of 397 checked).
- Policies tried and rejected: a first window of 12 or 24 rows (dense edits -7 / -9 %; the oracle runs favour it only
  because their misses come per window), restarting a missed window at the tokens it kept (plain text -12 %: spurious
  matches kept wide windows), and an expected-tokens-a-ms cap from each stream's copy record (no change: the costly misses
  follow runs of hits, which the record cannot foresee). Diffs in `patches/`.

## Round 9: prefill, clocks and the existing switches (rejected)

- Prompt chunks of 4,096 rows instead of 2,048: prefill +8 % (32K) to +18 % (64K), but replies to prompts past 2,048
  tokens change: chunk boundaries set how a row's window keys are blocked (ring part, chunk part), so the attention adds
  them in another order. Also +2.5 GB of arena. Exactness rules it out.
- GPU clock range 300-2,418 MHz (the default application clock) instead of 300-2,250: decode +0.15 % (one stream) / +0.45 %
  (four), prefill +1.2-1.7 %, all replies equal; the round's latency-bound part waits on memory and the fabric, not the SM
  clock. Kept at the reference 2,250.
- `TF_DS_IDX_SPLIT=1`, `TF_DS_ENGRAM_LANES=2` (patches 0001, 0002) on 32K / 64K / 126K prompts: within the run-to-run spread
  (two baselines differ by 7 % at 64K); no gain.

## Round 10: the MoE kernel rebuilt from source, deeper staging (negative, `results/kernel-stages.txt`)

The kit's `grouped_cp_kernel` (experts_cb2.cu of the engine's deepseek-v41-tp2 branch) rebuilt as the engine's torch
extension builds it (cpp_extension, `-O3 -lineinfo`, sm_121, nvcc 13.3 in the engine image): all 162 variants' SASS equal
to the kit's, so the kit's MoE kernels can be rebuilt and varied exactly. Instantiating each warp's ring of cp.async
trellis-copy stages at 4 and 5 (the header: the staging changes no bits; every run's hashes equal): S=4 within the noise,
S=5 -1.5 % (one stream) and -2.2 % (four). The kit's S=3 is the best setting. Also swept, defaults kept: the drafter-pass
prefetch budgets (-0.43 to 0.00 %) and the copy windows' entry match (6 / 12 tokens against 8: edits +2.6 / -1.8 %, plain
text -0.6 / +0.4 %).

## Round 11: memory under a mixed served load (`results/soak-memory.txt`)

No leak: device memory grows only with new graph shapes (~7 MB a context bucket) and prompt-cache states (512 MiB budget).
Rounds past 16 rows cost the engine 0.44 GiB at 32 and 0.62 GiB at 48 (its load line: 104.96 / 105.40 / 105.58 GiB), on a head
that showed 4.83-4.87 GiB idle after the load and 4.94 GiB at a 258K prefill's lowest, against a 4 GiB memguard floor. The
recipe's default is therefore 32 rows (edits +14 % instead of +20 %); 48 where the head has the room.

## Round 12: sampled self-repeats copied 15 tokens at most (`zig/patches/0022`, `results/copy-self-cap.txt`)

The server samples at temperature 1 by default and clients that set nothing (Pi) get it. There sparkDash's one-stream
cells lost 4-7 % with 32-row copy windows against 16 (same seeds, same replies): repeats of the reply's own text grow
windows that sampling then leaves, while copies of the prompt (a file being edited) stay on track. Capping a copy by its
match length (`TF_DS_COPY_MATCH_CAP`) changed nothing, since those repeats match long. Capping a sampling stream's copy
of its own reply, the proposer's source position past the prompt, at 15 tokens (`TF_DS_COPY_SELF_CAP=15`) checks 397
instead of 437 tokens on the costly prompt (111.8 -> 119.5 tok/s, its 16-row figure), keeps the edits' 32-row gain
(17.917 tokens a round either way) and leaves greedy decoding alone. Every reply equal to its reference (VF2: copies,
edits, T=1 against known and serial hashes, 32K-126K prompts, four streams).

## Round 13: negative results around the final configuration (2026-10-21 09:30-10:20)

- **The auth proxy** (`8888`, the key check in front of the engine) costs nothing: one 1,200-token reply streamed at
  237.5-238.1 tok/s through it against 238.0-238.5 straight from `8000` (the engine sends a round's tokens as one event).
- **CPU idle states**: with LPI-2 and LPI-3 (231 and 433 us to wake) disabled on both Sparks, one stream 84.73 / 84.37 tok/s
  against 84.63 / 84.68 with them, four streams 38.02 against 37.96: the engine's threads on the round's path spin. The
  governor is `performance` on both already.
- **Calibrating the confidence head for temperature 1**: on 45,094 drafts dumped at temperature 1 the head is already close
  (position 0 lands 0.707 against 0.727 predicted, positions 1-4 a little above their predictions); refitting it per
  position (Platt) gains +0.2 % in the round simulation (bootstrap 5-95 %: +0.08 .. +0.24 %).
- **Prefetched lines kept in L2 longer** (`evict_last` on the bulk prefetch, `research/patches/pf-evict-last.diff`): one stream
  84.39 / 84.47 tok/s against 84.75 / 84.65, four streams 37.85 against 38.01: they then crowd out the next prefetch.
- **Copies growing beside other streams** (`TF_DS_COPY_MULTI=15`, `research/patches/copy-multi.diff`: a copy of the prompt keeps
  its ramp with two to four streams, the round's rows shared out evenly so the windows never pass `ROUND_ROWS`; copies of
  the reply's own text stay at five): every reply equal. Streams that all edit go faster (two: 75.6 -> 97.6 tok/s a
  stream, +29 %; four: 47.1 -> 51.3, +9 %), but a stream that does not copy, sharing rounds with one that does, slows down
  (one pair: prose 34.5 -> 27.3 tok/s while the copy beside it went 84.5 -> 149.5), and the natural mix averages -1.6 %. A
  trade between streams as it stands. Allowing it only when every live stream's last round was a copy (`zig/patches/0023`)
  keeps the gains and slows no one: two streams that both edit +28 %, four +9 %, the natural mix +3.0 % with the prose
  beside a copy unchanged (34.7 against 33.8); served, two edits at a time +29 % (`results/copy-multi.txt`).
- **A Triton launch without its own variant** (logged once in service: `_index_score_rows` at a three-stream 48-row round
  takes the fitting variant that assumes less, same constexprs): with `TF_AOT_WEAKEST=1`, every launch on its most
  general variant, 26 of 26 replies equal their serial and known hashes (one stream, four, three streams editing with
  48-row copies), 1.6 % slower. The fallback changes speed only.
- **The copy's source**: the suffix lookup already takes the latest of the equally long matches (it walks back from the
  newest), the right one when a conversation has printed the same file more than once.

## Round 14: memory outside the engine's own count (`results/memory-savings.txt`)

The head's free memory fell by 2-3 GiB under mixed load and stayed down. Two causes, both cut with the replies and the
speed unchanged. NCCL, which here carries only the prompt passes' all-gathers, held ~1.2 GiB of pinned host buffers for
its default channels on each Spark; `NCCL_MAX_NCHANNELS=8` holds 0.2 GiB and prefills 32K-126K prompts as fast. And the
rounds' CUDA graphs, one per (rows, context bucket, stretch), were never released, ~3.6 MiB each: 453 of them in one
afternoon's checks. `zig/patches/0024` keeps 128 (`TF_DS_GRAPH_CAP`), destroying the least recently replayed first; even
a cap of 32 under a 102-graph load ran as fast. Served, the same checks left 6.7 GiB free at the lowest instead of 4.0.

## Round 15: the drafter's experts, a fused hc post, the prompt pass's timeline (2026-10-21 17:00-18:46, negative)

- **Fewer drafter experts** (`research/patches/draft-share.diff`, `TF_DS_DRAFT_SHARE`): a block's rows sharing row 0's routed
  experts (3 a block instead of up to 15) or its noise rows sharing row 1's: one stream 85.21 -> 80.96 / 83.14 tok/s,
  tokens a round 3.280 -> 2.994 / 3.133. The noise rows' routing matters more than the bytes it saves.
- **A gather's collect fused with the hc post that reads it** (`research/patches/hc-fused-post.diff`, `TF_DS_HC_FUSED`):
  `_hc_post_only`'s fp32 sum, bf16 rounding and fma chain op for op in the RDMA collect kernel, the gathered partials never
  written out. Every reply equal (one stream, four, copies, temperature 1, 32K), and the same speed: the post's 1.5 ms a
  round in the trace was its programmatic-dependent-launch wait for the collect, not work.
- **The prompt pass**: on a 16K prompt the GPU spends 28 % in the MoE's tensor-core GEMM, 14 % in its row kernel, 12 % in
  NCCL's all-gathers, 11 % in the attention's GEMMs. NCCL's default already takes its fastest protocol (`NCCL_PROTO=Simple`
  the same, LL128 +7 %, LL +19 % prefill time). A host log shows a 2,048-row chunk's Engram reads (both layers, ~370 ms,
  ~133K reads/s) running while the GPU works through the chunk before; the gaps a trace shows at the first Engram layer
  come from the profiler on the host. `TF_DS_ENGRAM_LANES=2` (two readers) prefills 1.5 % slower.

## Round 16: prompt chunks two at a time (`zig/patches/0025`, `results/prefill-pairs.txt`)

A long prompt's chunks before the decoder's replay run only the encoder layers, and there the MoE's routed experts read
every expert's weights for ~32 rows each. Running two such chunks side by side, layer by layer, with their routed experts
in one call (the first chunk's steps of a layer before the second's, the gates one per chunk) keeps every reply and cuts
prefill by 4.5 % at 16K tokens, 8.8 % at 32K and 12.2 % at 126K, for ~1.0 GiB more arena. The gain is less than the
weights alone suggest (the gate / up GEMM takes 1.68x as long for twice the rows) and was first lost to Engram reads
started too late; the shipped read-ahead takes the next two chunks' rows onto back buffers from the first Engram layer.
`zig/patches/0026` then puts a pair's NCCL gathers on a side stream beside the other chunk's work and starts the next
chunks' Engram reads earlier on a ring of their own: -9.4 / -12.5 / -13.9 % prefill at 16K / 32K / 126K, served -11 to
-12.5 % at 29K-144K tokens, every reply the same.

## Round 17: what else bounds a prompt and a long context (2026-10-21 22:40 - 10-22 01:30)

- **Decode deep in a context is the drafter, not the kernels.** On a 126K-token context a round takes 37.2 ms (a short
  one's K3 round 38.9 ms): `topk` over the compressed keys, the longer index scores and the merge add ~1-1.5 ms; the MoE
  and linears read the same. What drops is tokens a round, 2.57 against 3.28: the drafter attends to its last 128
  positions only.
- **Eight streams** (`--streams 8`, 16 prompts): 4 x 35.3 -> 8 x 19.4 tokens/s, +10 % aggregate for -45 % a stream, for
  34 MB more arena. Every reply equal; four stays the recipe's setting.
- **A chunk's first-layer Engram rows on two rings** (the back hosts' ring idle until they are read): no change on
  single-chunk prompts (1.6K tokens 1.10 s either way) or two-chunk ones (4K 2.15 s). The reads are not what a short
  prompt waits for.
- **A follow-up turn's TTFT** is the prefill of its tokens since the last kept state: states are kept at 2,048-row chunk
  ends before the replay (other ends would change the chunks a prompt is cut into, and its bits), so a turn redoes up to
  ~2K of the last one's tokens: 1.5-2.3 s for turns of 2-4K tokens on a 30K conversation. Running the last encoder-only
  chunk beside the final one would need the state kept between them, which the pair's shared layers overwrite.
- **Graphs on or off with four mixed streams** (an earlier run had graphs off faster): on 10.03, off 9.94 tokens/s a
  stream (69 captures); one stream 84.70 / 82.80. Not reproduced; captures cost nothing visible.
- **Memory with 0025 / 0026** (served, floor 3 GiB): a full prompt cache (61 states, 465 MiB), a 290,187-token prompt
  (140.0 s, before pairs 168.6-169.3 s) and three conversations at once kept MemAvailable >= 4,998 MiB on the head, 5,768
  on the other Spark (134 turns, no errors).
- Nsight Compute's kernel replay saves the device memory a kernel can write: with the model resident in the Spark's
  unified memory that exhausted the host (the OOM killer took desktop services); profile kernels standalone, not in the
  engine (and not beside a serving engine either: a standalone `--set full` run took the host under the memory guard's
  floor).
- **The prompt MoE kernels against the tensor cores.** GB10 sustains 110.3 TFLOPS of `mma.sync` m16n8k16 fp16 x fp16 ->
  fp32 (1,021 FLOP a clock an SM at 2.22 GHz). The gate / up kernel (`grouped_mma_kernel<2, 2, 10, FOLD, KB 4>`, run
  standalone on a pair's routing: 4,096 rows, 6 of 384 experts + the shared one) takes 18.9 ms, 35.9 TFLOPS of useful
  work (33 %); 2,048 rows 13.0 ms, 25.9 TFLOPS. Its 64-row member blocks are 72.8 % full on a pair, 53.8 % on a single
  chunk. Nsight Compute (standalone): 106 registers and 44.3 KB of shared memory a block hold it to 2 blocks (16 warps)
  an SM (100 KB of shared memory an SM); no warp eligible 59 % of the time; stalls a issued instruction: barrier 1.91,
  wait 1.71, math pipe 1.23, short scoreboard 0.96, long scoreboard 0.70; 64M shared-memory bank conflicts.
- **Every bit-exact setting the kit has, and two it lacks, are slower** (16K / 32K prefill, every hash equal): the
  served gate / up `KB 4` against `KB 2` (+6 / +10 %); the down kernel's (column tiles, prefetch, row tiles a group)
  served `8,1,2` against `8,1,1` (+1 %), `8,2,1` (+2 %), `4,2,2` (+2.5 %) and, built for this (SASS of every kit kernel
  unchanged), `4,1,4` (+2 %) and `4,2,4` (+3.5 %). A step's tiles and a group's rows do not change an output's mma chains
  or their order; four warps keep the down kernel's K ranges. `__launch_bounds__(256, 3)` gives 80 registers (48 B
  spilled) but still 2 blocks (shared memory), 4 % slower, the same bits. The kit's warp-decoding `mma2` (other bits; N
  a multiple of 256, so not this model's 1,152) is only ~10 % faster at N 1,024.
- **Where its time goes**: with every routed expert at exactly 64 rows a pair takes 17.4 ms (448 programs), at random
  routing 18.9 ms (615); a program of 32 rows takes 85 % of one of 64, so a program's cost is its weight decode and
  pipeline, not its mma work. A variant decoding each tile once for up to 128 rows (16 warps, each warp's rows, columns,
  k tiles and chains unchanged) gives the same Z bit for bit and decodes 32 % fewer tiles on a pair, but is 18 % slower
  (2,048 rows +59 %): one 16-warp barrier domain an SM instead of two of 8 hides less (the kit's own note: 128 rows a
  program were slower on GB10). The decode saved would need a pipeline without the step barrier (asynchronous barriers,
  warps that decode for others) to show.
- **Three programs an SM instead** (shipped: `zig/kit-patches`, `zig/patches/0027`, `results/experts-three-programs.txt`):
  two row stages instead of three, the staged rows unpadded with their 16-byte chunks XOR-swizzled by row (ldmatrix
  still conflict-free), `__launch_bounds__(256, 3)`: 33,024 B and 80 registers (48 B spilled), three programs an SM. The
  same Z bit for bit; 4,096 rows 19.0 -> 17.7 ms (-7 %), 2,048 rows 13.0 -> 12.5 ms (-4 %). In the engine, the kit's
  experts cubin rebuilt from its source with this one change (the other 1,744 kernels with the kit's SASS): prefill
  -1.8 / -2.8 / -2.4 % at 16K / 32K / 126K tokens (four interleaved runs, every hash equal), served TTFT -4.1 / -1.1 /
  -1.5 % at 29K / 70K / 144K tokens (replies the same). The kit is still fetched and checked by sha256; the image builds
  the cubin and the engine reads it in the kit's one's place.
- What is left there is a kernel without the per-step barrier (asynchronous barriers, warps that decode for others),
  keeping each output's 16 chains and their sum order, and the down kernel's 255 registers (two 4-warp programs an SM).
- **The decode MoE kernel is already where it should be**: `grouped_cp_kernel` runs four 4-warp programs an SM at 128
  registers (`__launch_bounds__(128, 4)`); five (96 registers, 112-120 B spilled) decode 82.4 / 82.6 tokens/s against
  84.9 / 84.7, six (80 registers, 208-224 B spilled) 75.2 / 74.8 (one stream, every hash equal). It streams the weights;
  spills cost more than the extra programs hide. Four streams' round rows (16 / 24 / 32 / 48) decode the same (~38.1 a
  stream); with copies 48 rows +7.0 % against 32.

## Round 18: Engram's table reads, and the prompt MoE kernels once more (2026-10-22, `zig/patches/0028`, `results/engram-reads.txt`)

A request's first chunk waits for its first Engram layer's rows (~200-290 ms) with only the embedding and layer 0 on
the GPU, and the NVMe serves them at its random-read rate (two 4 KiB reads a row: its 256 FP8 values and its 8 scales;
reading on two rings was no faster). Reading fewer rows is what helps. The engine's n-gram hash, ported to Python and
checked against the kit's six `engram.json` cases, over a code-heavy stream (300K tokens of the engine's sources and
this recipe's files, 146 chunks): 46 % of a chunk's ids on rank 0 (its columns: the 2- and 3-grams) and 30 % on rank 1
(3- and 4-grams) repeat a row named before in the same chunk, and of the distinct rows a 264 MiB cache of the rows
read before holds 50 % (rank 0) and 26 % (rank 1): 27 % and 52 % of today's reads left. `zig/patches/0028` reads a
prompt chunk's distinct rows only (then copies them to every place that names them) and keeps the rows read last, by
prompt chunks and decode rounds, in host memory (256 MiB a Spark, CLOCK replacement), for every ring of the process.
The tables are read-only files, so every row has the bytes its read would have. Engine, one stream, prompts from one
document: 1.6K tokens 1.243 -> 0.998 s (-20 %), 4K -3.5 %, 16K -3.3 %, 32K -2.5 %; every prefill and decode hash equal
(11 prompts and their 4-token replies, `req-test-x`, `req-long-x`, four mixed streams).

Negative, the same bits each time: the experts' input rotation (`rot_in_kernel`, one warp a 128-value block, ~171 GB/s
of writes on a pair) with eight warps a program (7.4 ms against 3.4 on a pair: rows scattered) or one warp walking a
row's 40 blocks (3.5 ms); the down kernel (`grouped_rows_kernel`, 255 registers, two 4-warp programs an SM) forced to
three or four programs (168 or 128 registers: +43 % / +310 % on a pair) and its smaller tilings forced likewise (4,2,2 /
8,1,1 at three or four programs: +6 to +45 %).

## Round 19: the next turn's prefill, ahead of it (2026-10-22, `tools/warm_ahead_proxy.py`, `results/warm-ahead.txt`)

A turn's prompt state is kept at the last 2,048-token chunk end before its last 128 tokens, so the next turn of the
conversation re-prefills from there: the previous prompt's tail (up to ~2,200 tokens) as well as what is new. The
reply's tokens were decoded, not prompt-processed (other kernels, other bits), so their state cannot be kept. What can
be done exactly is to run the prompt pass the next turn would run, sooner: when the client is idle after a reply, send
the conversation and the reply (as the client will send it back) to the server, so the cache keeps the state at the
deepest chunk end every next request can resume (shared tokens + 3 - 128 at most). The first design kept a deeper one
(the shared tokens' last chunk end, reached with a filler message): a short next message could not resume it, and the
store, which frees first a state a newer one extends, had freed the conversation's earlier state: the next turn
re-prefilled 39K tokens (first token 16.5 s). With the rule, 39K-token conversations, three turns, a client sending
replies back as received: the third turn resumed at 38,912 instead of 36,864, its first token 0.54-0.59 s instead of
1.40-1.57 s; a third turn arriving 0.4 s into the warm-ahead (closed) still resumed at 38,912 (1.19 s). The resumed
reply has the bits of the same request from an empty cache. In use (a coding agent, 37K -> 55K tokens over 14 turns):
the client's replies came back as predicted 19 of 19 times and 8 warm-aheads moved its resume point a chunk on.
Shipped as a token prompt (the shared ids up to the chunk end and 128 past it, `/v1/completions`): the server runs 128
rows past the chunk end instead of up to ~2,200, so a next request arriving meanwhile waits less (an agent loop with a
1 s tool: 1.48 -> 1.18 s), and it starts at once after tool calls. A restart empties the cache: the proxy, having seen
the server unreachable, re-runs the recent conversations when clients are quiet, in 16K-token pieces (each kept, so a
request arriving meanwhile loses at most the piece in flight). A 38K-token conversation: re-warmed in 20 s (three
pieces, each resuming the last), its next turn's first token 1.17 s (resumed 36,864) instead of 16.19 s (from 0).

## Round 20: the draft policy where agents spend their time (2026-10-22, `results/policy-long-sampled.txt`)

An agent workload is mostly decode (prompts ~70K tokens, the cache resuming all but ~2K of them, ~860 tokens decoded
a turn with long reasoning, sampled at the server's defaults): ~85 % of the time goes to decode, at fewer tokens a
round than the short benchmarks (2.6-2.9 against 3.3 for prose). Replayed with the same shape (52K and 78K-token tool
conversations, reasoning at max effort, T 1.0 / top-p 0.95 / top-k 20, seeded): the shipped policy (the drafter's
calibrated confidences) 66.06 and 65.85 tok/s; confidence bias -0.5 / +0.5 / +1.0: -0.6 / -3.8 / -6.8 % (more drafts
verified: more tokens a round, longer rounds); two drafts always trusted -1.7 %; copies from 3-token matches: the same.
All the same bits. The calibration holds where context is long and decoding sampled; no switch to change.

## Round 21: the host between sampled rounds (2026-10-22, `zig/patches/0029`, `results/sampler-pinned.txt`)

At temperature > 0 the round's logits come to the host and each row is drawn there (top-k 20 + 8 candidates, top-p,
Gumbel-max with position-keyed uniforms): nothing of the next round can start before. `--profile` on rank 0 (eight
seeded T=1 requests, 6.1 rows a round): the logits copy 0.37 ms a round (3 MB into a pageable buffer: the driver stages
it, ~9 GB/s) and the draws 0.37 ms, of a 45.5 ms round. The candidates' search went through the 129,280 logits one at
a time with a branch each; past the first k, a later logit enters only above the last kept value (its id is higher
than every kept one), so a 16-wide compare skips a block with none above it, and the rest take the same insertion: the
same (value, id) pairs (a test against the one-at-a-time loop over ties, NaN and infinities). Standalone: 71.7 -> 17.1 us
a row on a big core, 125.9 -> 27.7 on a little one. In the engine: draws 0.37 -> 0.12 ms, and with the logits buffer
pinned (`cuMemHostAlloc`, the same 24.8 MB) the copy 0.37 -> 0.07 ms. Decode 110.00 / 110.00 -> 111.41 / 111.31 tok/s
(+1.2 %, interleaved), every run's tokens equal. The rest of the host side of a round is small: the frame to rank 1
0.003 ms, the drafter's results two copies of a few hundred bytes; the round itself is the GPU's (forward 41.5 ms with
its issue overlapped, the drafter's pass 3.4 ms).

## Round 22: the request's tokens (2026-10-22, `zig/patches/0030`, `results/tokenize-memo.txt`)

Every request is rendered and tokenized whole: at 133-140K tokens `/tokenize` (the chat route: JSON, template, encode)
took 73-92 ms a turn, the encode alone 57 ms (ReleaseSafe, a big core). The tokenizer splits a text at its added tokens
that are not normalized (for this model the end-of-sentence a reply ends with) and encodes each segment by itself, from
its bytes and whether it starts the text; a conversation's next request repeats the previous one's segments up to its
last reply. So the server keeps the last four long texts with their segments and ids, and a new text takes the ids of
the leading segments it has alike (the same bounds and token, inside the common prefix) and encodes the rest. The same
ids: random texts against encode (the added token plain and with lstrip / rstrip), and 37 requests of served
conversations (tool calls, dropped reasoning, warm-ahead texts) against the server's ids. Encode 57-59 -> 2.0-2.8 ms at
135-140K tokens, 4.8-5.5 -> 0.1-1.9 ms at 13-15K; a conversation's second request still encodes whole (its first
had no end-of-sentence yet).

## Round 23: a request arriving during another's prompt (2026-10-22, `zig/patches/0031`, `results/prefill-nest.txt`)

A short request resumed from the cache waited 29.6 s behind another client's 60K-token prompt: between a pass's chunks
the host ran only the other streams' decode rounds (`TF_PREFILL_INTERLEAVE_MS`), and the next prompt pass started when
this one ended. Now, in that `between`, the queue's first request runs its whole pass when it would prefill at most
`PREFILL_NEST_TOKENS` (8192; its prompt past the kept state it resumes, as the store's lookup finds it, no pictures).
Three pieces of per-pass state had to become the stream's own again after such a pass, on both ranks: the slot and
extent the pass is set up with (a chunk frame names neither; a `fill_resume` frame sets them up again, rank 1 keeping
each slot's ids), the Engram read-ahead (keyed by start and length only: the nested prompt's chunks may share both, so
the next chunk reads its rows anew), and the pairs' back-host ring (keyed by start). The host keeps the outer request's
cancel for its own pass (a nested pass's `between` takes the others'), and nests one level. Served, a 68.6K-token prompt
with a 1.9K-token request sent 3 s into it: that request's first token after 2.02-2.43 s instead of 27.64 s, the long
prompt's after 34.6 s instead of 29.3; two short requests (one at T=1) 2.34 and 2.98 s (2.40 and 9.03 s in a first build
that nested one a `between` and let a nested pass hold the host's near-end gate); a 12.7K-token one (over the limit)
waits as before; the long request closed by its client while the short pass ran ended cleanly. Every reply
equal to the same prompt alone (10 of 10). Streaming itself is even: a reply's events (a round's tokens each) arrive
32.2 ms apart at the median, p99 ~41 ms, the same through the proxy as from the server.

## Round 24: upstream since the pin (2026-10-22, `zig/patches/0032`, `results/rdma-idle-nap.txt`, `results/indexer-split.txt`)

MiaAI-Lab's recipe moved on 10-21. Their `0010-rdma-idle-nap.patch` (William J Riley's TensorFold PR #27, ported to the
Zig ring by Mykhailo Tamarin) applies unchanged on this stack: carried as 0032 with their settings (`RDMA_IDLE_S` 2, `RDMA_NAP_US` 500). An
idle server's busiest thread went from 99.9 % of a core to 0.3 % on each Spark; sampled decode the same (+0.06 %).
Their revision of 0001 (aaa4xu, 2550cda4) turns the prefill indexer split on by default from token 131,072; on this
stack the split pays earlier (43K -0.95 %, 161K -2.74 % against that threshold, 16K the same, the same bits), so the
recipe sets `TF_DS_IDX_SPLIT=1`, which the 0001 it carries already reads. Their 0008 (a POOL setting) is left out.
BertholomusAI's engine branch moved on too (17 commits past the pin). Its `cf6aae2b` (evilpsycho42, PR #35) applies
unchanged and fixes what reproduced here: a request sharing a conversation's prefix and ending under 128 tokens past its
kept chunk end (a new session's first message after a shared system prompt can) got that state from the store, which the
lanes refused, and the store dropped it as a failed restore: the conversation's next turn then resumed 6,144 instead of
8,192 (2.33 s instead of 1.31). Carried as 0033. Left out: the tool-only reply's held whitespace (`6568611e`: it changes
what a client receives and so what it sends back, which the warm-ahead proxy learns), the AIO refusal leak (`12132870`:
it does not apply over 0028, and the row ids are always in range), the opt-in switches `TF_DS_SOLO_OVERHEAD` and
`TF_DS_NO_COPIES`, and the pass-in-pieces scheduler (this recipe's 0031 nests short passes instead).

## Round 25: the decode round at 78K tokens, sampled (2026-10-23, `zig/patches/0034`, `results/topk.txt`)

An Nsight trace of rank 1 decoding a 78K-token tool conversation at T=1 (94 rounds, 42.96 ms each): the GPU idle 2.4 % of
the window (a new process's graph instantiations 24 ms of the 95, launches and the peer's frames the rest); the MoE
26.8 ms, the linears 9.9 ms, the gathers 3.6 ms (their waits in), the hc mixes ~4.5 ms (PDL waits in), the sparse
attention 2.4 ms, the heads 2.1 ms, and this engine's own top-k of the indexer's keys 0.59 ms (eight a round). That one
ran a row on one block: the pruned selection's 32,768 candidates took 89.7 us and the 610 tile maxima 23.9 us, standalone.
Keys in registers with a warp's scan of each digit's bins, and the long rows in slices over many blocks then the top-k of
the slices' top-k sets (the same set: the keys are unique): 16.3 and 4.9 us. Interleaved twice: 42.13 / 42.13 -> 41.81 /
41.91 ms a round (+0.6 %), every run's tokens equal; short contexts (the fused Triton selection) and prefill unchanged.
At the same 78K / T=1 shape, the switches rechecked (interleaved twice each, every run's tokens equal): the paced L2
prefetch still pays (off: 72.7 / 72.6 against 76.1 / 75.9 tok/s, -4.4 %; short prompts -5.1 %), its 16 MiB budget is the
best of 12 (-1.5 %), 16 and 20 (the same); both ranks' containers on the ten big cores alone change nothing (75.7 / 75.9
against 76.5 / 73.5). The context buckets are powers of two, so a growing conversation recaptures its round graphs once
a doubling; the indexer scores the whole bucket, ~0.1 ms a round more than its rows need at 78K. The MoE reads ~6.4 GB
a rank a round in ~26.8 ms, ~90 % of the read ceiling; what is left is the gathers' latency and the small kernels.

## What is left

The remaining recoverable time is latency-bound and outside the code we can change bit-exactly: the kit's kernels
(rewriting `grouped_cp` / `glinear` would need their exact reduction order), the gather's pinned-memory path (GPU-initiated
RDMA), and the host's round trip between rounds (~0.35-0.45 ms: acceptance read on the host before the next graph).
Copy windows past 48 rows would need the round buffers' max_rows (48) and more captured kernels. The larger lever for
text without copies is tokens a round: a better drafter (training), not kernels.
