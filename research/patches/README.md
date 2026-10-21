# research/patches

Research-only diffs, not in the image (`zig/patches/` holds what the image applies). Each was measured and rejected
(`../KERNEL-ROUNDS.md`); they apply after this recipe's `zig/patches/0019`-`0021` unless noted.

| Diff | Round | What it does |
| --- | --- | --- |
| `draft-coupling.diff` | 7 | the drafter's Markov `_score` replaced by a kernel racing the target sampler's Gumbel draws for sampled streams (`TF_DS_DRAFT_COUPLE`, `TF_DS_DRAFT_COUPLE_SCALE`) |
| `copy-restart.diff` | 8 | a missed copy window restarts at the tokens it kept (`TF_DS_COPY_RESTART`; the core lanes engine and config) |
| `copy-ev.diff` | 8 | copy windows capped at the most expected tokens a millisecond from each stream's copy record (`TF_DS_COPY_EV`) |
| `experts-stages.diff` | 10 | against the engine's `deepseek-v41-tp2` source (`src/tensorfold/cuda/exl3`), not this recipe: `grouped_cp_launch` also instantiating 4 and 5 cp.async stages, built as the engine's torch extension builds it (cpp_extension, `-O3 -lineinfo`, sm_121) |
| `copy-multi.diff` | 13 | copies of the prompt growing beside other streams, rows shared evenly (`TF_DS_COPY_MULTI=N`): +29 % / +9 % when two / four streams all edit, a non-copying stream beside a copying one -21 %, the natural mix -1.6 % |
| `pf-evict-last.diff` | 13 | the bulk L2 prefetch marking its lines `evict_last` (`TF_DS_PF_HINT=1`): -0.3 % one stream, -0.4 % four |
