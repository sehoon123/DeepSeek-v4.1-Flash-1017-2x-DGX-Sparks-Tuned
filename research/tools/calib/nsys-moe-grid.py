# grouped_cp_kernel launches by grid size in the steady-state window: separates the forward's MoE (one grid a layer half) from the drafter's.
import sqlite3, sys, collections
db = sqlite3.connect(sys.argv[1]); mpr = float(sys.argv[2]); names = dict(db.execute("select id, value from StringIds"))
k = [(s, e, names.get(n, '?'), gx * gy * gz) for s, e, n, gx, gy, gz in db.execute("select start, end, shortName, gridX, gridY, gridZ from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
t0 = k[0][0]; dens = collections.Counter(int((x[0] - t0) / 1e9) for x in k); peak = max(dens.values()); hot = sorted(s for s in dens if dens[s] >= 0.5 * peak)
w0 = t0 + (hot[0] + 0.2 * (hot[-1] - hot[0] + 1)) * 1e9; w1 = t0 + (hot[0] + 0.8 * (hot[-1] - hot[0] + 1)) * 1e9; nr = (w1 - w0) / (mpr * 1e6)
g = collections.defaultdict(list)
for s, e, n, grid in k:
    if w0 <= s < w1 and n == 'grouped_cp_kernel': g[grid].append(e - s)
tot = 0
for i, (grid, d) in enumerate(sorted(g.items(), key=lambda x: -sum(x[1]))):
    tot += sum(d)
    if i < 10: print(f'  grid {grid:6d} CTAs: {len(d)/nr:6.1f}/round  mean {sum(d)/len(d)/1e3:7.1f} us  {sum(d)/nr/1e6:6.3f} ms/round')
print(f'  grouped_cp total {tot/nr/1e6:.3f} ms/round over {nr:.0f} rounds')
