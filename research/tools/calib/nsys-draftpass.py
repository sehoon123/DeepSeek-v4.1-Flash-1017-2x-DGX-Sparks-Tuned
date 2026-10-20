# The draft pass of one steady round on a rank-1 nsys export: from the main LM head (505-CTA glinear) to the next round's first MoE.
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); names = dict(db.execute('select id, value from StringIds'))
k = [(s, e, st, names.get(n, '?'), gx * gy * gz, ds) for s, e, st, n, gx, gy, gz, ds in db.execute(
    'select start, end, streamId, shortName, gridX, gridY, gridZ, dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
t0 = k[0][0]; mid = t0 + (k[-1][1] - t0) * 0.55
heads = [i for i, x in enumerate(k) if x[0] > mid and x[3] == 'glinear_kernel' and x[4] == 505]
i0 = heads[0]; j = next(i for i in range(heads[1], len(k)) if k[i][3] == 'grouped_cp_kernel' and k[i][4] == 504)
seg = k[i0:j]; base = seg[0][0]
print('head launches (us from the first, dyn smem, us long):', [(round((k[i][0] - base) / 1e3), k[i][5], round((k[i][1] - k[i][0]) / 1e3)) for i in heads[:4]])
print(f'main head start -> next round first MoE: {(k[j][0] - base) / 1e3:.0f} us, {len(seg)} kernels')
by = collections.Counter(); cnt = collections.Counter()
for s, e, st, n, g, ds in seg: by[(n, g)] += e - s; cnt[(n, g)] += 1
print('kernel (grid)                        count   sum_us')
for (n, g), d in by.most_common(16): print(f'  {n[:30]:30s} ({g:5d}) {cnt[(n, g)]:4d} {d / 1e3:8.1f}')
print('order (kernel x consecutive count, us since the main head started):')
prev, run, first = None, 0, 0
for s, e, st, n, g, ds in seg:
    if (n, g) != prev:
        if prev: print(f'   {(first - base) / 1e3:7.0f}  {prev[0][:28]}({prev[1]}) x{run}')
        prev, run, first = (n, g), 0, s
    run += 1
print(f'   {(first - base) / 1e3:7.0f}  {prev[0][:28]}({prev[1]}) x{run}')
