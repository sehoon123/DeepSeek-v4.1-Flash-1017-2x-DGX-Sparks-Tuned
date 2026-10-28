# Research only: original BM128/BK32 accumulation, partition only independent
# output rows of the original two-term BF16 Hadamard epilogue. No sum reassociation.
import triton
import triton.language as tl

@triton.jit
def _store_ep(acc, rm, rn, h, SVH, BIAS, OUT, M, o_stride,
              HAS_BIAS: tl.constexpr, SCALE: tl.constexpr):
    top = acc.to(tl.bfloat16)
    rest = (acc - top.to(tl.float32)).to(tl.bfloat16)
    y = tl.dot(rest, h, tl.dot(top, h))
    y = y * SCALE * tl.load(SVH + rn).to(tl.float32)[None, :]
    if HAS_BIAS:
        y += tl.load(BIAS + rn).to(tl.float32)[None, :]
    tl.store(OUT + rm[:, None] * o_stride + rn[None, :],
             y.to(OUT.dtype.element_ty), mask=(rm < M)[:, None])

@triton.jit(do_not_specialize=["M"])
def _gemm_ep(X, W, H, SVH, BIAS, OUT, M, o_stride, K: tl.constexpr, N: tl.constexpr,
             BM: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
             HAS_BIAS: tl.constexpr, SCALE: tl.constexpr, EP: tl.constexpr):
    tl.static_assert(BM == 128 and BK == 32)
    tl.static_assert(EP == 32 or EP == 64)
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    per = GROUP * (N // 128)
    first = (pid // per) * GROUP
    rows = tl.minimum(nm - first, GROUP)
    pm = first + (pid % per) % rows
    pn = (pid % per) // rows
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * 128 + tl.arange(0, 128)
    rk = tl.arange(0, BK)
    ok = rm < M
    acc = tl.zeros((BM, 128), dtype=tl.float32)
    for k0 in range(0, K, BK):
        x = tl.load(X + rm[:, None] * K + (k0 + rk)[None, :], mask=ok[:, None], other=0.0)
        w = tl.load(W + (k0 + rk)[:, None] * N + rn[None, :])
        acc = tl.dot(x, w, acc)
    hi = tl.arange(0, 128)
    h = tl.load(H + hi[:, None] * 128 + hi[None, :])
    a0, a1 = tl.split(acc.reshape((2, 64, 128)).trans(1, 2, 0))
    er = pm * BM + tl.arange(0, EP)
    if EP == 64:
        _store_ep(a0, er, rn, h, SVH, BIAS, OUT, M, o_stride, HAS_BIAS, SCALE)
        _store_ep(a1, er + 64, rn, h, SVH, BIAS, OUT, M, o_stride, HAS_BIAS, SCALE)
    else:
        a00, a01 = tl.split(a0.reshape((2, 32, 128)).trans(1, 2, 0))
        a10, a11 = tl.split(a1.reshape((2, 32, 128)).trans(1, 2, 0))
        _store_ep(a00, er, rn, h, SVH, BIAS, OUT, M, o_stride, HAS_BIAS, SCALE)
        _store_ep(a01, er + 32, rn, h, SVH, BIAS, OUT, M, o_stride, HAS_BIAS, SCALE)
        _store_ep(a10, er + 64, rn, h, SVH, BIAS, OUT, M, o_stride, HAS_BIAS, SCALE)
        _store_ep(a11, er + 96, rn, h, SVH, BIAS, OUT, M, o_stride, HAS_BIAS, SCALE)
