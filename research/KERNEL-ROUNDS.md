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
- **The copy's source**: the suffix lookup already takes the latest of the equally long matches (it walks back from the
  newest), the right one when a conversation has printed the same file more than once.

## What is left

The remaining recoverable time is latency-bound and outside the code we can change bit-exactly: the kit's kernels
(rewriting `grouped_cp` / `glinear` would need their exact reduction order), the gather's pinned-memory path (GPU-initiated
RDMA), and the host's round trip between rounds (~0.35-0.45 ms: acceptance read on the host before the next graph).
Copy windows past 48 rows would need the round buffers' max_rows (48) and more captured kernels. The larger lever for
text without copies is tokens a round: a better drafter (training), not kernels.
