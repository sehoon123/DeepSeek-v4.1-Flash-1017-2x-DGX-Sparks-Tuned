# Tools

The scripts as they were used, with the original machines' layout (`/home/USER`, the ssh host `worker`, the link addresses
`10.10.10.x` / `10.10.11.x`): read them before running any. Scripts that need NumPy or PyTorch were run inside the engine
image (`docker run --rm --network none --entrypoint python3 tensorfold-dsv41:exl3-tp2-39b972fbfeab ...`); the probes need a
local sparkDash checkout (not included). Nothing here is needed to run the recipe.

- `header-diff.py`: every tensor's dtype and shape, the stock checkpoint (HTTP range requests of its headers) against the
  overlay checkpoint. It found the confidence head's F32 cast.
- `calib/calib-run3.sh OUT SPEC DEADLINE`: the offline harness. It stops the serving service, runs both ranks
  (`tf-dsv41-lanes`) with the flags and environment of each spec line (`name|requests|parallel|rank-0 flags|rank-1
  flags|env|max seconds`; `!` lines run a command) on a request file, records token hashes, rounds and profiles, and
  restores the service, also on failure. `NSYS1=1` traces rank 1 with Nsight Systems (rank 0 hangs under it).
  `spec-W2.txt`, `spec-W3.txt` are examples; `summarize.py`, `add-expect.py` (expected hashes from a serial run) and
  `ms-per-round.py` read its output.
- Requests: `prompts_fit.py`, `tok2.py`, `mkreq2.py`, `mklong.py` (long prompts from the engine's own sources), `sd-dump.mjs`
  (sparkDash's decode prompts). Request files are not included: they hold sparkDash's prompt texts.
- Exactness and decode: `verify-exact.py` (every result against its reference), `decode-table.py`.
- The drafter: `diag-conf.py` (the confidence-head diagnosis), `check-dump.py`, `sim2.py`, `platt-test.py`,
  `policy-ceiling.py`, `gate-drafter.py` (landing rate against decode), `ce-compare.py`, `vocab-coverage.py`.
- Kernels and memory: `nsys-analyze.py`, `nsys-cover2.py` (time covered by DRAM-streaming kernels), `nsys-moe-grid.py`,
  `nsys-layer.py`, `nsys-draftpass.py`, `ckpt-bytes.py`, `ckpt-accounting.py`, `clock-compare.py`, `clock-layers.py`.
- RDMA: `latprobe.sh` (`ib_write_lat` on both rails), `rail-compare.py`.
- `hcopy/`: the copy-window exactness runs (oracle copies with a share of wrong tokens, against serial decoding).
- `profile/`: the lane profiler run (`--profile`) on sparkDash's prompts.
- `bench/`: sparkDash-based decode and prefill probes (`conc-probe-dsv.mjs`, `conc-probe-sampled.mjs`, `idle-probe.mjs`,
  the prompt-interleave canaries, `prefill-sparkdash.mjs`) and `bwprobe.py` (GPU read bandwidth).
