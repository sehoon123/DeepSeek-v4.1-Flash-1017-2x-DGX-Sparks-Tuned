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

## What is left

The remaining recoverable time is latency-bound and outside the code we can change bit-exactly: the kit's kernels
(rewriting `grouped_cp` / `glinear` would need their exact reduction order), the gather's pinned-memory path (GPU-initiated
RDMA), and the host's round trip between rounds (~0.35-0.45 ms: acceptance read on the host before the next graph). The
larger lever is tokens a round: a better drafter (training), not kernels.
