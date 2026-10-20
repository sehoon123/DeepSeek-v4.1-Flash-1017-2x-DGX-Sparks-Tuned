# Results

Measured 2026-10-19 and 20 (Korea time) on the setup in [../README.md](../README.md). Each file, and how it was made
(tools are under [../tools/](../tools/README.md)).

| File | |
| --- | --- |
| `decode-hotfix-final.json`, `decode-hcopy.json`, `decode-conffix.json` | sparkDash decode waves (temperature 0, 400 tokens, two of each): before, after 0017, after 0017 and the true confidence head (`bench/conc-probe-dsv.mjs`) |
| `sampled-default-1.json`, `sampled-conffix.json` | the same at the server's sampling (temperature 1), 10:25 and 13:32 (`bench/conc-probe-sampled.mjs`) |
| `decode-upstream-prompts.json`, `decode-steady-c1.json`, `idle-probe-1.json` | with upstream sparkDash's stream labels; six waves of prose at one stream; requests after two- and four-minute idle gaps |
| `decode-recipe-tuned.json`, `decode-recipe-tuned-vs-conffix.txt` | the same bench on this recipe, run as our production after the switch (20:40), against `decode-conffix.json` |
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

The traces (Nsight Systems exports) and the drafter dumps (`TF_DS_CALIB_FILE`: the confidence head's inputs, i.e. the
model's hidden states) are not included: they are large, and the dumps are derived from the model.
