# Results

Measured 2026-10-19 and 20 (Korea time) on the setup in [../README.md](../README.md). Each file, and how it was made
(tools are under [../tools/](../tools/README.md)).

| File | |
| --- | --- |
| `decode-hotfix-final.json`, `decode-hcopy.json`, `decode-conffix.json` | sparkDash decode waves (temperature 0, 400 tokens, two of each): before, after 0017, after 0017 and the true confidence head (`bench/conc-probe-dsv.mjs`) |
| `sampled-default-1.json`, `sampled-conffix.json` | the same at the server's sampling (temperature 1), 10:25 and 13:32 (`bench/conc-probe-sampled.mjs`) |
| `decode-upstream-prompts.json`, `decode-steady-c1.json`, `idle-probe-1.json` | with upstream sparkDash's stream labels; six waves of prose at one stream; requests after two- and four-minute idle gaps |
| `decode-recipe-tuned.json`, `decode-recipe-tuned-vs-conffix.txt` | the same bench on this recipe, run as our production after the switch (20:40), against `decode-conffix.json` |
| `decode-0019-old.json`, `decode-0019-new1.json`, `decode-0019-new2.json`, `decode-0019-ab.txt` | the same bench on the image with patches 0019 and 0020, against itself with the previous prefetch behaviour set through the environment (2026-10-21 02:09-02:24) |
| `wide-copy-windows.txt`, `decode-0021.json` | copy windows of 16 to 48 rows (the harness) and sparkDash's bench after deploying them (2026-10-21 03:00-04:55) |
| `copy-policy.txt`, `copy-self-cap.txt`, `copy-multi.txt`, `rows-32-48.txt`, `multiturn-t1.txt`, `soak-concurrent.txt`, `kernel-stages.txt`, `soak-memory.txt`, `memory-savings.txt`, `prefill-pairs.txt` | copy windows on realistic edits and at temperature 1 with the policies tried; the MoE kernel rebuilt with deeper staging; memory under a mixed served load |
| `draft-coupling.txt` | drafts racing the target sampler's Gumbel draws at temperature 1 (negative) |
| `decode-poll-ab.txt` | the same bench with a dashboard's `nvidia-smi` polling every 2 s against every 10 s, A/B/A/B (2026-10-21 02:35-02:45) |
| `pf-summary.txt`, `pr-summary.txt`, `hp-summary.txt`, `ps-summary.txt`, `dp-summary.txt` | the kernel rounds' offline sweeps: the prefetcher's constants, the gather probes (GPU and host side), each prefetch site's budget, the drafter's prefetch (`KERNEL-ROUNDS.md`) |
| `overnight-final.json`, `overnight-soak.txt` | the overnight run's start, its soak medians and the seven soak cycles |
| `decode-table.txt` | the decode tables of the README and the notes (`calib/decode-table.py`) |
| `exactness-audit.txt` | every offline result against its serial reference or hash (`calib/verify-exact.py`) |
| `coverage.txt` | the time of a round during which a DRAM-streaming kernel runs, depth 3 and 0, from rank-1 traces (`calib/nsys-cover2.py`) |
| `layer-timeline-k0.txt`, `draft-pass-k0.txt` | one layer and one draft pass, kernel by kernel, at one row (`calib/nsys-layer.py`, `calib/nsys-draftpass.py`) |
| `clock-marks.txt` | both ranks' phase and layer GPU clocks (`TF_DS_CLOCK_MARKS`; `calib/clock-compare.py`, `calib/clock-layers.py`) |
| `checkpoint-bytes.txt` | bytes by tensor class, and what a rank reads in a one-row forward (`calib/ckpt-accounting.py`) |
| `bandwidth.txt`, `memory.txt` | GB10 read bandwidth (`bench/bwprobe.py`, transcribed); the engine's and both nodes' memory while serving |
| `rdma-latency.txt`, `rails.txt` | `ib_write_lat` on both rails (`calib/latprobe.sh`); one rail against two (`calib/rail-compare.py`) |
| `policy-ceiling.txt`, `gate-drafter.txt` | depth-policy predictors on held-out prompts (`calib/policy-ceiling.py`); decode against the drafter's landing rate (`calib/gate-drafter.py`) |
| `copy-threshold.txt`, `vocab-coverage.txt` | copy-window entry at 4 / 6 / 8 / 12 tokens (`calib/ce-compare.py`); how much of the vocabulary the replies use (`calib/vocab-coverage.py`) |
| `validate-round3.log` | Mia's AI Lab's `tools/validate.py` against the served model, replies compared with the morning's |
| `decode-final-greedy.json`, `decode-final-sampled.json` | the served decode bench after rounds 6-13 (greedy and T=1, prose and code at 1 / 2 / 4 streams) |
| `experts-three-programs.txt` | the prompt MoE's gate/up kernel with three programs per SM (`zig/kit-patches/0001`): standalone and served |
| `engram-reads.txt` | Engram reads deduplicated per chunk and the host row cache (`zig/patches/0028`): simulation and prompt times |
| `warm-ahead.txt` | `tools/warm_ahead_proxy.py`: next-turn first tokens with and without the warm-ahead; re-warm after a restart |
| `prompt-tail.txt` | a prompt ending within the window past a kept state, before and after `zig/patches/0033` |
| `indexer-split.txt` | the prefill indexer split (`zig/patches/0001`) at 16K / 43K / 161K: none, from 131K, every chunk |
| `rdma-idle-nap.txt` | the RDMA proxy's idle nap (`zig/patches/0032`): idle CPU and decode, before and after |
| `prefill-nest.txt` | requests arriving during another's long prompt pass, with and without `zig/patches/0031` |
| `tokenize-memo.txt` | the server's tokenize reusing the previous request's ids (`zig/patches/0030`): checks and times |
| `sampler-pinned.txt` | sampled decoding's host side: the logits copy and the draws a round, before and after `zig/patches/0029` |
| `policy-long-sampled.txt` | the draft policy's switches on long-context sampled decode (Round 20) |

The traces (Nsight Systems exports) and the drafter dumps (`TF_DS_CALIB_FILE`: the confidence head's inputs, i.e. the
model's hidden states) are not included: they are large, and the dumps are derived from the model.
