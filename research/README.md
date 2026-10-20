# Decode research

What was measured on one pair of DGX Sparks while making this recipe's changes, 2026-10-19 and 20 (Korea time). None of it
is needed to run the recipe.

**While measuring**: two DGX Sparks (GB10, 128 GB each, GPU clocks capped at 2,250 MHz), linked by both CX7 PCIe links as
two RoCE rails; `ABLIT=1`, `CONTEXT=1048576`, `PARALLEL=4`; Mia's AI Lab's recipe at `e65785cd` with
`zig/patches/0010`-`0018` and the drafter's confidence head from a file, image `tensorfold-dsv41:exl3-tp2-39b972fbfeab`.
That is the image, and those are the confidence-head values, this recipe gives such an install (its stock shards were
pruned). Read every number as this pair's. The tools keep the original machines' layout: `/home/USER`, the ssh host
`worker`, the link addresses `10.10.10.x` and `10.10.11.x`.

| | |
| --- | --- |
| [FINDINGS.md](FINDINGS.md) | the notebook of 10-20: profiling, the confidence-head diagnosis, copy windows, depth-policy ceilings, the round's kernels, bandwidth and RDMA; corrections are marked where they were made |
| [OVERNIGHT-OPTIMIZATION.md](OVERNIGHT-OPTIMIZATION.md) | the overnight search over draft-depth policies, run while the confidence head was still misread (its patches 0005-0011 are 0010-0016 here) |
| [KERNEL-ROUNDS.md](KERNEL-ROUNDS.md) | 10-20 / 10-21: the read ceiling (263 GB/s), the L2 prefetcher, the tensor-parallel gather, each prefetch site, the drafter's prefetch (`zig/patches/0019`), the serving environment (GPU polling, CPU load), copy windows of 48 rows (`zig/patches/0021`), and drafts coupled to the sampler's randomness (negative) |
| [patches/](patches/) | research-only engine diffs, not in the image: `draft-coupling.diff` (round 7) |
| [GITHUB-COMPARISON.md](GITHUB-COMPARISON.md) | this pair against the figures other DeepSeek-V4.1-Flash recipes publish (10-19) |
| [results/](results/README.md) | data, and the outputs the notes quote |
| [tools/](tools/README.md) | the offline harness and the analysis scripts |
| [REFERENCES.md](REFERENCES.md) | what was read or used, and its license |

## In short

- **The confidence head was misread.** DSpark scores each draft with `mtp.2.confidence_head.proj.weight`; the overlay
  stores it as F32 and the engine reads F16. As read, the scores ranked drafts worse than chance (AUC 0.45 against which
  drafts landed; the true head: 0.84). Offline, on held-out prompts, the true head gave +8.6 % (one stream), +8.9 %
  (temperature 1), +7.0 % and +11.1 % (two and four streams); served, with the stock depth rule back, sparkDash's six cells
  +6.4 % (`results/decode-table.txt`). Mia's AI Lab's recipe fixed the same problem the same day, in the checkpoint
  (`32907a1`, after issues #11 and #13 there).
- **Copy windows (0017)**: 62 of 62 runs equal to serial decoding; a file copied from the prompt 123.6 → 204.1 tok/s
  verbatim and 111.0 → 141.1 with edits; sparkDash's six cells +1.2 % (`results/decode-table.txt`).
- **Where a round goes** (one stream at a fixed depth of 3, 38.85 ms; `results/coverage.txt`): a DRAM-streaming kernel
  (MoE, linear layers, heads, drafter) runs during 33.2 ms of it (85 %), near the read ceiling (MoE ~228 GB/s; a plain
  CUDA read reaches 263 GB/s and an MoE-like pattern 244-249, `KERNEL-ROUNDS.md`; the 226 GB/s first given here was
  torch's 1 GiB copy, not the ceiling). The other 5.7 ms are latency-bound: the 76 tensor-parallel all-gathers of
  ~24 µs each (their wire floor is ~6.6 µs, `results/rdma-latency.txt`) and the small kernels `coverage.txt` lists (hc
  mixes, attention's parts); the GPU idles 0.45 ms.
- **Kernel rounds** (`KERNEL-ROUNDS.md`): the round reads ~8.8 GB a rank at the served depth, 33.5 ms at the read
  ceiling against 38.85 ms measured, so kernels and gaps hold at most ~13 %. The prefetcher's first wave starting at once
  and the drafter's pass prefetching its blocks (`zig/patches/0019`) gave +1.0 % served (six sparkDash cells, geomean);
  every site's budget, pacing and block count were swept and the engine's are the best measured. A dashboard polling
  `nvidia-smi` twice a second cost the served decode 0.9 %.
- **Copy windows of 48 rows** (`KERNEL-ROUNDS.md` round 6, `zig/patches/0021`): perfect copies +31.6 %, a natural copy mix
  +6.5 %, replies equal serial decoding in every check; the 16-row cap was the window rings' spare rows and the router's
  captured variants, not arithmetic.
- **The depth rule is at its ceiling**: richer predictors of how many drafts land gain at most +0.3 % on held-out prompts,
  where an oracle would gain +16-17 % (`results/policy-ceiling.txt`).
- **The drafter decides the rest**: a point of landing rate is worth ~2.4 %; +10 % needs +4.1 points on sparkDash's prompts
  and +6.3 on varied ones (`results/gate-drafter.txt`), i.e. a drafter re-trained on the served model's own outputs.
- **Did not help** (FINDINGS.md): copy-window entry at 4, 6 or 12 tokens instead of 8 (`results/copy-threshold.txt`); a
  256K or 512K window instead of 1M; pinning the host to the big or the little cores; one RDMA rail instead of two
  (`results/rails.txt`); pruning the draft head's vocabulary (99 % of tokens need 436-450 of a rank's 505 blocks,
  `results/vocab-coverage.txt`); tree drafting; RDMA, prefetch, graphs, AIO or the side stream off; more than 16 rows a
  round (it changes output bits).
- **Exactness**: 856 offline results in 48 runs; 611 compared with a serial reference or its hash, none different
  (`results/exactness-audit.txt`).
- **This recipe as our production** (2026-10-20 20:38): the same image and confidence-head values as before the switch,
  replies equal to the saved ones (`tools/validate.py`, 8 of 8), sparkDash's six cells +0.6 % (noise;
  `results/decode-recipe-tuned-vs-conffix.txt`).
- **Against Mia's AI Lab's README** (prose, one stream: 66.8 here, 74.9 there with the stock checkpoint): not explained
  here. Their own measurement after the dtype fix has the overlay at 70.7 against stock's 75.1 on their prose bench, so part
  of the gap is likely the overlay's weights.
- **Overnight** (`results/overnight-final.json`): the depth policies of 0010-0016 gained +37.0 % on the six cells over that
  night's start, with the head still misread; with the true head and the stock rule the cells stand +46.9 % over it.

These notes, the measurements and the tools were produced with an AI coding agent working on the owner's machines, under
the owner's direction.
