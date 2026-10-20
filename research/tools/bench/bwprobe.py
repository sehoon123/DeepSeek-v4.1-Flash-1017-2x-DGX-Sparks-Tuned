# Achievable DRAM read bandwidth on this GPU (L2-cold, rotating offsets) and its device limits.
import torch, statistics as st
p = torch.cuda.get_device_properties(0)
for a in ('name', 'major', 'minor', 'multi_processor_count', 'max_threads_per_multi_processor', 'L2_cache_size', 'regs_per_multiprocessor',
          'shared_memory_per_block', 'shared_memory_per_block_optin', 'shared_memory_per_multiprocessor', 'total_memory'):
    print(f'{a}: {getattr(p, a, "?")}')
N = 1 << 30
x = torch.randint(-2**31, 2**31 - 1, (N // 4,), dtype=torch.int32, device='cuda'); y = torch.empty_like(x); torch.cuda.synchronize()
def bench(fns, reps=15):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); ts = []
    for _ in range(reps):
        s.record()
        for f in fns: f()
        e.record(); torch.cuda.synchronize(); ts.append(s.elapsed_time(e) / len(fns))
    return st.median(ts)
t = bench([lambda: torch.amax(x)]); print(f'read 1 GiB (amax int32): {N / t / 1e6:6.1f} GB/s  {t:.3f} ms')
t = bench([lambda: torch.sum(x.view(torch.float16).float() if False else x, dtype=torch.int64)]); print(f'read 1 GiB (sum int64): {N / t / 1e6:6.1f} GB/s')
t = bench([lambda: y.copy_(x)]); print(f'copy 1 GiB (read+write): {2 * N / t / 1e6:6.1f} GB/s combined')
for S in (2 << 20, 8 << 20, 32 << 20, 128 << 20):
    n = S // 4; k = N // S
    fns = [(lambda j=j: torch.amax(x[j * n:(j + 1) * n])) for j in range(min(k, 64))]
    t = bench(fns, reps=8); print(f'read {S >> 20:4d} MiB chunks back to back (L2-cold): {S / t / 1e6:6.1f} GB/s  {t * 1e3:7.1f} us each')
