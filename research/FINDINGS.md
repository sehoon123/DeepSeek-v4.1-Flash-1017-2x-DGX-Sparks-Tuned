> **Lab notebook (2026-10-20, Korea time), kept as written; later entries correct earlier ones.** Corrections to read first:
> (1) an early per-kernel estimate of the MoE at "63 % of bandwidth" was wrong: with the down projection's PDL weight preload
> the MoE phase streams at ~228 GB/s; (2) the read ceiling is 226 GB/s by a 1 GiB torch read and up to ~249 GB/s apparent for
> the LM-head GEMV (which may include L2-prefetched bytes), so "MoE at the ceiling" means within about 0-8 %; (3) the Markov
> loop costs 0.16 ms a round, not 0.65 ms; (4) "every run equal to serial" holds for the runs with a serial or expect
> reference (558 records, 0 mismatches); two data-collection runs had none. Host names and paths are generalized.

# DSV4.1-1017 Ablit — round-2 research (2026-10-20 10:17–)
Invariant: target bits unchanged (proposals/scheduling only); rows/round <= 16 (row-invariant decode kernels).

## Measured cost model (rank-0 lane profile, profile/run1, production image da8e1eb22cc6)
step(1 row)=25.4 ms (fwd 21.6 + DSpark pass 3.5 + host 0.3); step(4)=38.5; step(6)=46.1; C4-sampled host draws 0.65 ms.
Server round time == offline model (c1prose 38.87 vs 39.06 ms, c1code 40.65 vs 40.36): no serving-loop overhead.
Phase (eager, 1.68 rows): attn matmuls 6.1 ms (~roofline), attn small ops 2.9, MoE 12.5 (~91% BW), gathers 1.9, hc mixes 1.3, engram/head/post 1.6.

## Rejected
- sampled host overhead (~1%), serving-loop overhead (0%), tree drafting (extra MoE row ~4.3 ms > expected gain),
  IDX_SPLIT / ENGRAM_LANES (upstream: no gain / -1.5..-2.6%), tap correction (DSpark taps = layers 37-39, after ablit 10-35),
  VRAM -10% (weights ~99.5 of 104.96 GiB on rank 0; RoPE tables 0.53 GiB).

## Validated bottleneck: copy window (H-COPY)
verbatim copy: 5.93 tok/round, 99.1% accepted, rows 5.97/6 (window saturated), 123.6 tok/s; copy+edit 5.15 tok/round, 111 tok/s.
Core already doubles copy_width after a fully accepted lone round (engine.zig:371) but DSV caps exact_width=window_rows=6.
Limits allow 16: RING_EXTRA=16, decode_rows=16, max_absorb=48. Plan: exact_width/wc/verify cap 16, first_copy_rows=6
(first copy window unchanged), DSpark drafts still <=5, lone stream only. Estimate: verbatim +35-45%, edits ~0%, SparkDash 0%.

## H-COPY deployed (2026-10-20 12:13 KST)
Patch `zig/patches/0017-dsv41-copy-window.patch` (default off; on via `TF_DS_COPY_ROWS=16 TF_DS_COPY_SKIP=1` in scripts/local.sh).
Image `tensorfold-dsv41:exl3-tp2-c28f5fb6b9c2` (sha256:70d6532d…), identical on both ranks; rollback image da8e1eb22cc6 kept.
- exact_width 16 for a lone stream's copy windows (first copy 5 → 11 → 15 on full accepts); drafter windows unchanged (<=5 drafts).
- a lone stream's fully accepted copy window skips the next DSpark pass (its drafts would be dropped by the plan).
- TF_DS_* policy parsing moved into Lanes.applyEnv (serve and tf-dsv41-lanes identical): offline == server (sha + rounds 3/3).
- tf-dsv41-lanes test proposer (`"oracle"` request field, --oracle-miss/--oracle-seed) for rollback stress.
Exactness: serial refs == server drafted; 62/62 oracle/natural runs equal to serial (miss 0/25/50/90 %, parallel 3,
skip on/off, T=1 sampled, EOS and max_tokens ends); server 5/5 known shas; SparkDash wave outputs 12/12; validate.py all pass.
Speed (server): verbatim copy 123.6 → 204.1 tok/s (+65%), copy+edit 111.0 → 141.1 (+27%); SparkDash prose 64.21/91.82/126.87
(−0.2/0.0/+2.1 %), code 99.17/142.51/154.69 (+3.9/+1.0/+0.6 %). Downtime 433 s for the build/restart.
Rollback: rm zig/patches/0017-*.patch; cp research-20261020b/local.sh.pre-hcopy scripts/local.sh; restart while idle.

## Remaining tracks (assessed, not started)
- DSpark re-distillation on the Ablit target: the drafter is 4,839 tensors / 6.75 GiB served (3 blocks x 128 EXL3-4bit
  experts ~13.6B params + markov heads + attention); originals in the official repo's shards 44-46. Needs tap/label capture
  from this engine, a PyTorch DSpark trainer, EXL3 re-quantization and rank-cache rebuild: multi-day. Upper bound from the
  stock README gap: prose C1 +17 %, prose C2 +15 %, code C1 +11 %.
- Kernel fusion: ~6 ms/round latency-bound (gathers 1.9, hc mixes 1.3, attention small ops 2.9, engram 0.45) mostly in the
  kit's recorded Triton kernels; bit-exact fusion needs their arithmetic reproduced; estimate +5-8 %.

## DSpark confidence head: Ablit-only load bug found and fixed (2026-10-20 12:40-13:20)
The Ablit graft casts `mtp.2.confidence_head.proj.weight` to F32 (for another engine); this engine reads it as f16
(`f16ToF32`), so it ran with the F32 bytes' first half reinterpreted as f16. Proof (calib/runA, smoke dump): the engine's
logged logit == features . (f32 bytes read as f16) with corr +1.000000 / max err 6.9e-6, while the true weight gives corr
-0.04; Markov half of the dumped features == embed[previous draft] 735/735. Broken head: mean p 0.28 vs landing 0.75,
AUC 0.45; true head: mean p 0.72, AUC 0.84. Drafter gate biases are all zero (unaffected), so draft tokens were fine:
only the per-round depth decision was blind (this is why the stock rule drafted 0.68/round and A2/P8 helped).
Fix: patch 0018 `TF_DS_CONF_FILE` loads the true fp32 weight (exactly f16/bf16-representable) into confw.
Offline A/B (calib/runB, every run equal to serial):
| condition | A2/P8 | fixed head + stock rule | change |
| C1 eval 34 prompts (incl. Korean, thinking), 2 reps | 68.68 / 68.31 | 74.33 / 74.50 | +8.6 % |
| T=1 sampled 15 prompts | 59.08 | 64.36 | +8.9 % |
| C2 SparkDash 11 prompts (per-stream) | 51.81 | 55.42 | +7.0 % |
| C4 SparkDash 11 prompts (per-stream) | 33.82 | 37.58 | +11.1 % |
Measured cost table: no further gain (74.25). Simulated headroom left: oracle +34 % vs K3, fixed head +14.8 %.

## Confidence-head fix deployed (2026-10-20 13:27 KST)
Image `tensorfold-dsv41:exl3-tp2-39b972fbfeab` (sha256:47ea893c…, both ranks), patches 0017 + 0018; local.sh: A2/P8 removed,
`TF_DS_CONF_FILE=/hf/tensorfold-dsv41/conf-fix/conf_head_f32.bin` (the true fp32 head; the server refuses to start without it).
Server checks: 5/5 known shas; rounds equal the offline fixed-head runs (164/74/141); validate.py all pass; SparkDash waves 12/12
and sampled waves 12/12 equal; Pi canary; public 401/200.
SparkDash greedy (2 reps) vs the H-COPY image with A2/P8: prose 64.21/91.82/126.87 -> 66.81/100.75/134.15 (+4.0/+9.7/+5.7 %),
code 99.17/142.51/154.69 -> 114.32/148.21/155.38 (+15.3/+4.0/+0.4 %). Sampled T=1 vs the morning image: +7.1/+8.1/+11.1 %
prose, +13.3/+3.0/+5.1 % code. Versus the stock README: prose -10.8/-4.5/+3.2 %, code +4.0/+6.0/-0.1 %.
Rollback: rm zig/patches/0018-*.patch; cp research-20261020b/local.sh.pre-conffix scripts/local.sh; restart while idle.
Next headroom (simulation): oracle +34 % vs K3, fixed head +14.8 %: a Platt/refit of the head on dumped data (TF_DS_CALIB_FILE,
TF_DS_CALIB_PLATT are in 0018) is the cheap follow-up; body re-distillation and kernel fusion remain multi-day.

## Rejected: Dinkelbach (average-reward) depth criterion (2026-10-20 13:40, calib/sim2.py on the 187-round smoke dump)
The allocator maximizes expected tokens / round time per round; the long-run-optimal rule maximizes tokens - lambda*time
(lambda = achieved throughput). Simulation: fixed lambda 0.04-0.12 -> -0.1 % .. -2.3 %, per-request fixed point -0.5 %,
causal EMA +0.2 % versus today's ratio rule (79.0 tok/s simulated). No gain beyond noise: not implemented.

## Verified: the Ablit graft changes nothing else that this engine loads (2026-10-20 14:05, header-diff.py)
All 39 shards' headers, stock (HTTP range, no weights) vs served: 192,495 tensors each, none missing either side, 0 shape
differences, exactly 4 dtype differences: mtp.{0,1,2}.ffn.gate.bias and mtp.2.confidence_head.proj.weight F16 -> F32.
The gate biases are all zero (unaffected); the confidence head was the only live casualty. The 104 wo_b tensors keep their
stock shape and dtype (values differ). With the publisher's equal Ablit/stock speeds on an engine that reads the F32 head
correctly (48.1 vs 48.2 tok/s), there is no evidence that the Ablit edit lowers drafter acceptance: body re-distillation
is not justified.
## Rejected: per-position Platt calibration (in-sample -0.8 %, bootstrap -0.91 % [-1.85, -0.31]) despite better log loss.
## nsys on the head hangs inside the engine image (even a 5-call ctypes probe); the 13:44 window was stopped after 555 s.

## Round 3 (2026-10-20 14:14 ->): host, kernel timeline and long context
### W1 (14:14:22-14:24:05, 583 s down; every run equal to serial)
- Rejected H-cpu: 11 SparkDash prompts, default 83.38 / 82.78 tok/s, big cores (5-9,15-19) 82.94, little cores (0-4,10-14)
  82.55. The host's core speed does not bound the round. Rounds 1338/1339 vary run to run: the allocator's overhead EMA is
  timing-fed (the tokens do not change).
- nsys works on rank 1 only (rank 0 hangs in this image); its distortion is 0.3 % (K3 traced 76.78 vs 77.03 tok/s).
- Rank-1 timeline, K3 (247 rounds of 38.85 ms in the steady window; the production policy's run gives the same picture):
  any kernel 99.2 % of the time, compute kernels 98.8 %, the DRAM-streaming kernels (MoE grouped_cp, glinear) 85.4 %.
  A round = 33.1 ms with a streaming kernel running + 5.7 ms without (of which the GPU is idle 0.45 ms). In the 5.7 ms:
  rdma_collect 1.9, hc_post_only 1.5, hc_finish_s 0.8, drafter _rowmm_parts 0.76, sparse_attn_part 0.73, merge_rot2 0.57,
  hc_mix_part 0.56, rdma_stage 0.41 (prefetch 3.8 runs beside). MoE: two fixed grids, 2016 CTAs 35.2/round x 393 us and
  1120 CTAs 35.2/round x 195 us (time ratio 2:1 = the gate+up : down byte ratio, i.e. bandwidth-proportional); grouped_cp
  24.9 ms of the round. Not a host or idle-GPU problem: the remainder is small latency-bound kernels.
- Where the code lives: grouped_cp/glinear are kit cubins, hc/sparse attention kit Triton AOT (no source here); only the L2
  prefetch and the RDMA gather are this repo's CUDA. The gather is host-proxied (pinned host ring, a busy-polling proxy,
  RDMA write with immediate, flags polled by the collecting kernel) with PDL, own-early, unroll and fence mode on as
  served; ~26 us of un-overlapped time a gather, 89 a round. Only GPU-initiated RDMA would cut it (multi-day): not tried.
### W2 (14:39:57 ->): long context (first exactness check of 0017/0018 there)
- 32K / 64K / 125K-token prompts (this engine's sources, a summary task), 400 tokens greedy: serial 44.92 tok/s; the
  production policy 67.40 tok/s, 3/3 equal to serial; --profile 1: step 37.14 ms, forward 33.23 ms at 3.24 rows a round
  (short context: 38.76 / 34.88 at 4.07). Against forward(rows) = 21.6 + 4.32 (rows - 1) the long context costs ~+2 ms a
  round; its lower speed is acceptance (2.50 vs 3.18 tokens a round), not the round's cost.
### W2 results (14:39:57-15:54:19, 4,462 s down incl. the appended H-pool runs; every run equal to serial)
- Copy threshold (TF_DS_COPY_ENTER; 36 prompts = 34 held-out + copyv/copye; ce-compare.py): baseline 8 twice 79.03 / 79.26
  tok/s (per-request geomean -0.13 / +0.14 %); 4 -1.31 %, 6 -0.29 %, 12 +0.29 % (under the pre-set +1 % bar; copye 71 -> 73
  rounds). Kept 8.
- Policy ceiling (policy-ceiling.py on K5 dumps: 125,030 drafts from 120 greedy + 70 sampled fit prompts and the 11
  SparkDash prompts; logged logits = the fixed head's within 1.9e-5). Held-out prompts only, simulated with the lone-stream
  cost table: VAL greedy (23 prompts) oracle +17.5 %, per-position Platt +0.3 % [+0.1, +0.5], joint model (all five logits,
  stream EMA) +0.2 %, head refit (L2 to w0, f16-rounded) +0.2 %, MLP-64 -2.2 %; VAL sampled (13) oracle +17.2 %, best +0.3 %
  [-0.1, +0.7]; TEST SparkDash (11) oracle +15.6 %, best +0.2 %. AUC 0.842 -> 0.847 at best. The oracle gap is not
  predictable from what the allocator sees: the depth policy is at its ceiling. Nothing met the bar (+3 %, 95 % lower bound
  > 0): nothing deployed.
- MoE (rank-1 nsys, K0 vs K3): two fixed grids a layer, 504 / 280 CTAs at 1 row and 2016 / 1120 at 4 rows (= rows x 7 experts
  x 72 / 40 tiles, 7 = 6 routed + shared, inferred); 293.7 vs 587.8 us a layer, time following the distinct experts' bytes.
  Coverage (array-backed nsys-cover2.py; reproduces K3's 85.4 % / 5.69 ms): MoE and linear kernels never run at once
  (0.00 ms). K0 round 25.98 ms = MoE 10.34 + linear 10.59 + 5.05 ms with no streaming kernel (idle 0.35). The 1-row forward
  MoE reads ~1.76 GB (assumed 6.3 MiB an expert a rank) in <= 10.3 ms: >= 170 GB/s (~195 for the forward's share), while the
  L2 prefetch reads beside it, so DRAM use is higher; the kernel's headroom is unknown (a few to ~20 % of MoE time) and it
  is a kit cubin without source.
- H-pool rejected: the configured window does not change the round. 1M 82.61 / 83.06, 512K 83.11 / 83.05, 256K 83.11 tok/s
  (1M-vs-1M noise 0.54 %); outputs identical at every window; rank-1 nsys at 512K 81.60 vs 81.76 at 1M (377 rounds each).
  The README's sparkDash table was taken at CONTEXT=524288 with clocks locked at 2,200 MHz (here 2,250): neither explains the
  prose C1 gap. What remains: checkpoint (stock vs Ablit), prompt or engine version; a stock A/B needs ~141 GB of downloads.
- After the window: same image (39b972fbfeab) and env on both nodes, smoke all PASS (qualification/smoke-round3.*), proxy 401
  without a key, memguard idle.
### Round 3 conclusion
No exact change worth deploying. A C1 round is ~85 % DRAM-streaming kernels (MoE or linear, never both) and ~15 % latency-
bound kernels running alone: the RDMA gather (host-proxied) and the kit's hc / sparse-attention kernels (Triton AOT /
cubins, no source). Upper bound if all of that vanished: +17 % (K3) / +24 % (1 row). Paths left, each multi-day: bit-exact
fusion of the hc / sparse-attention kernels, GPU-initiated RDMA, a faster EXL3 MoE kernel (needs the kit's source), a
better drafter (the only way past the depth policy's ceiling).

### W3 (17:29:43-17:40:10, 627 s down; every run equal to serial via expect_sha): skew, rails, wire floor, layers
- Clock marks (TF_DS_CLOCK_MARKS=1 --profile 1, both ranks): the production policy's forward phases sum to 32,123 / 32,111
  us a round on rank 0 / 1, gathers 23.6-25.2 us each on both (gather_a 898 / 932, gather_m 942 / 959 us a round): no skew
  (rank 1 waits 51 us a round more, 0.16 %). At K0 only, rank 0 runs ~3.5 % slower in compute phases and rank 1 waits.
- RDMA wire floor (perftest ib_write_lat, both rails alike): 1.44 us (64 B), 2.90 (4 KB), 4.44 (16 KB), 5.87 (32 KB), 8.66
  (64 KB). A production gather (80 KB fp32, 40 KB a rail) is ~6.6 us on the wire against ~24 us measured: ~17 us of stage /
  fence / proxy / flag / collect overhead, 76 gathers a round = ~1.3 ms (3.3 % of a K3 round).
- One rail (--rdma rocep1s0f1) vs two: 83.11 / 83.33 vs 83.38 / 83.43 tok/s (-0.36 / -0.10 %), 11/11 equal: rejected.
- Layer-end clocks (=2, production policy): 40 layers, median 820 us, fastest L21 649; slowest L1 1,209 and L14 1,123 (the
  two Engram layers, +98 / +79 MiB of extra weights), L0 1,021, L2 948, L39 939.
### Bandwidth and byte accounting (17:00-18:15)
- GB10 (worker, production idle): 48 SMs, L2 24 MiB, 100 KiB smem a SM. torch read 1 GiB 225.9 GB/s; 2 / 8 / 32 / 128 MiB
  reads 96 / 143 / 165 / 175 GB/s (transfer-size limited).
- Checkpoint (headers): routed experts 3 bpw (12.66 MiB an expert), shared experts 5 bpw (21.1 MiB a layer), attention
  5 bpw (~76-82 MiB a layer), router fp16 3.75 MiB, head 6 bpw [320, 8080, 96] (237 MiB a rank), drafter experts 3 x 2,160
  MiB, drafter main_proj [15360 -> 5120] 4 bpw.
- K0 layer on rank 1 (nsys, PDL-aware): MoE 0 -> 222.9 us for 48.5 MiB a rank (6 routed + shared, halves) = 228 GB/s, at
  the measured read ceiling; attention linears ~41 MB in 190.7 us (~217 GB/s). The earlier per-kernel "63 %" was wrong: it
  ignored the down projection's PDL weight preload during gate/up. A K0 forward reads ~3,764 MiB a rank: 17.5 ms at
  226 GB/s against ~21 ms measured (~86 %); the rest is latency chains (gathers, router, exposed hc).
- Draft pass (3.64 ms at K0): main_proj + KV prep 0.33, three DSpark blocks 2.02 (MoE 5 rows x 4 slots), draft head (5-row
  vocab projection) 1.06, Markov loop 0.16 (not 0.65: the first _bias launch's mean held its PDL wait), confidence 0.06.
- Rejected: draft-head vocabulary pruning. 247 distinct replies (90,444 tokens, 11,989 ids) need 436-450 of 505 128-row
  blocks a rank for 99 % coverage, 463-464 for 99.5 %.
### Correction (exactness audit, calib/verify-exact.py)
803 records in 41 offline runs: 529 matched their serial-derived expect_sha, 29 matched by serial / prefix / peer
agreement, 0 mismatches; 193 records (the K5 dump runs d-fit-g / d-fit-s and runA/smoke) had no independent reference, so
their "all_equal" is vacuous. "Every run equal to serial" above holds for the runs with a reference; the deployed
configuration's exactness rests on those and on validate.py (8/8 greedy replies equal to the 08:45 deployment's).
### Where a K3 C1 round (38.85 ms) goes, and what is left
DRAM-streaming at the measured ceiling ~85 %; latency chains ~15 % (gathers 1.84 ms, router 0.71, hc mixes 0.76, Engram
0.51, sparse-attention / indexer partly hidden). Exact engineering left, each multi-day: zero-copy or GPU-initiated RDMA
(realistically 0.4-0.8 ms), bit-exact hc / router fusion (~0.4-0.6 ms; kit Triton without source), prefetch pace (<= 1 %):
~2-4 % together. The one lever with >= 10 % headroom is drafter landing: +2.4 % a point; +4.1 pt (SparkDash) or +6.3 pt
(diverse prompts) for +10 % -- DSpark self-distillation (multi-day; the serving GPUs cannot train while they serve).
