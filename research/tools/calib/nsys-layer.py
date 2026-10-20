# One steady-state layer on a rank-1 nsys export: every kernel between two MoE gate/up launches, with start, length, gap, stream.
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); grid0 = int(sys.argv[2]) if len(sys.argv) > 2 else 504; names = dict(db.execute('select id, value from StringIds'))
k = [(s, e, st, names.get(n, '?'), gx * gy * gz) for s, e, st, n, gx, gy, gz in db.execute(
    'select start, end, streamId, shortName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL order by start')]
t0 = k[0][0]; mid = t0 + (k[-1][1] - t0) * 0.55
i0 = next(i for i, x in enumerate(k) if x[0] > mid and x[3] == 'grouped_cp_kernel' and x[4] == grid0)
i1 = next(i for i in range(i0 + 1, len(k)) if k[i][3] == 'grouped_cp_kernel' and k[i][4] == grid0)
seg = k[i0:i1 + 1]; base = seg[0][0]; run_end = base
print(f'one layer (MoE gate/up -> next gate/up): {(seg[-1][0] - base) / 1e3:.1f} us, {len(seg) - 1} kernels')
print('  start_us   dur_us  gap_us  stream  kernel (grid)')
for s, e, st, n, g in seg:
    print(f'  {(s - base) / 1e3:8.1f} {(e - s) / 1e3:8.1f} {(s - run_end) / 1e3:7.1f}  {st:6d}  {n[:30]} ({g})'); run_end = max(run_end, e)
