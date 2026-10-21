# Build the EXL3 experts extension's codebook-2 translation unit (experts_cb2.cu) as the deepseek-v41-tp2 engine's
# tensorfold/cuda/build.py compiles it (PyTorch's cpp_extension, -O3 -lineinfo, the GPU's -gencode; no GPU is needed to
# compile) and keep its sm_121 cubin: the kit's cubins/experts_cb.cubin made from source with zig/kit-patches applied.
# usage: build_experts.py SRC_DIR BUILD_DIR OUT_CUBIN (run by zig/Dockerfile)
import glob
import os
import shutil
import subprocess
import sys

os.environ["TORCH_CUDA_ARCH_LIST"] = "12.1"
os.environ.setdefault("MAX_JOBS", "2")
import torch.utils.cpp_extension as ce  # noqa: E402

src, build, out = sys.argv[1:4]
os.makedirs(build, exist_ok=True)
try:
    ce.load(name="tensorfold_exl3_experts_v19", sources=[os.path.join(src, "experts_cb2.cu")],
            extra_cuda_cflags=["-O3", "-lineinfo", "-gencode=arch=compute_121,code=sm_121"],
            extra_include_paths=[src], build_directory=build, verbose=False)
except Exception as e:  # importing the module needs the whole extension (and a GPU); its object file is what is kept
    print("experts: load:", type(e).__name__)
objs = glob.glob(os.path.join(build, "*.o"))
if len(objs) != 1:
    sys.exit(f"experts: expected one object file, found {objs}")
x = os.path.join(build, "x")
os.makedirs(x, exist_ok=True)
subprocess.run(["cuobjdump", "-xelf", "all", objs[0]], cwd=x, check=True, capture_output=True)
cubins = glob.glob(os.path.join(x, "*.sm_121.cubin"))
if len(cubins) != 1:
    sys.exit(f"experts: expected one sm_121 cubin, found {cubins}")
os.makedirs(os.path.dirname(out), exist_ok=True)
shutil.copy(cubins[0], out)
print("experts: cubin", out, os.path.getsize(out), "bytes")
