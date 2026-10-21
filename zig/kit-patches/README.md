# Kit patches

The kit (BertholomusAI's, fetched at `KIT_REF` and checked against its `MANIFEST` by `scripts/prepare.sh`) holds cubins
built from the EXL3 kernels of BertholomusAI's TensorFold, branch `deepseek-v41-tp2` (its Python engine's torch
extensions). `zig/Dockerfile` fetches that branch at `TF_PY_REF` (`scripts/config.sh`), applies the patches here
(`git apply`, filename order), and builds `experts_cb2.cu` as that engine does (`build_experts.py`: PyTorch's
cpp_extension, `-O3 -lineinfo`, sm_121) into the image's `/opt/tensorfold/kit-overlay/experts_cb.cubin`. The engine
reads it in place of the kit's `cubins/experts_cb.cubin` when `TF_DS_EXPERTS_CUBIN` names it (`zig/patches/0027`; this
recipe's default). Set `TF_DS_EXPERTS_CUBIN=` (empty) in `scripts/local.sh` for the kit's own cubin.

| Patch | What |
|---|---|
| `0001-experts-mma-three-programs.patch` | `grouped_mma_kernel` (the prompt chunks' gate / up): two row stages instead of three, a staged row unpadded (64 halfs) with its 16-byte chunks XOR-swizzled by row so that the eight rows an ldmatrix reads still fall in distinct banks (4 k tiles a step; other steps keep the padding), and `__launch_bounds__(256, 3)`: 33,024 B of shared memory and 80 registers, three programs an SM instead of two. Each warp's rows, columns, k tiles and mma chains, and their sums, are unchanged. |

Checked: the kernel's output bit for bit against the kit's (standalone, a pair's routing and a chunk's); every other
kernel of the rebuilt cubin has the kit's SASS (1,744 of 1,762, the 18 `grouped_mma_kernel` variants differ); prefill
hashes at 16K / 32K / 126K tokens, `req-long-x`, the served validation replies and long-prompt replies equal. Measured:
the kernel -7 % on a 4,096-row pair, -4 % on a 2,048-row chunk; prefill -1.8 / -2.8 / -2.4 % at 16K / 32K / 126K
tokens. Details: `research/KERNEL-ROUNDS.md`, round 17.
