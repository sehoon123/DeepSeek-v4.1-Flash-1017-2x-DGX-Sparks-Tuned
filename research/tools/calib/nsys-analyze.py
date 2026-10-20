# Kernel-level attribution of a rank-1 nsys sqlite export: busy/idle in the steady-state decode, idle gaps, per-round time by kernel.
import sqlite3, sys, collections, statistics
db = sqlite3.connect(sys.argv[1]); q = lambda s, *a: db.execute(s, a).fetchall()
tabs = {r[0] for r in q("select name from sqlite_master where type='table'")}
names = dict(q("select id, value from StringIds"))
ev = []
for t, kind in (('CUPTI_ACTIVITY_KIND_KERNEL', 'k'), ('CUPTI_ACTIVITY_KIND_MEMCPY', 'c'), ('CUPTI_ACTIVITY_KIND_MEMSET', 's')):
    if t not in tabs: continue
    cols = [r[1] for r in q(f"pragma table_info({t})")]
    nm = 'shortName' if 'shortName' in cols else None
    for r in q(f"select start, end, streamId{', ' + nm if nm else ''} from {t}"):
        ev.append((r[0], r[1], r[2], kind, names.get(r[3], '?') if nm else kind))
ev.sort(); print('events', len(ev), 'kernels', sum(e[3] == 'k' for e in ev))
t0, t1 = ev[0][0], max(e[1] for e in ev)
# steady-state window: the middle 60 % of the span with the densest kernel launches (decode), excluding load/prefill/tail
dens = collections.Counter(int((e[0] - t0) / 1e9) for e in ev if e[3] == 'k')
secs = sorted(dens); peak = max(dens.values()); hot = [s for s in secs if dens[s] >= 0.5 * peak]
w0 = t0 + (hot[0] + 0.2 * (hot[-1] - hot[0] + 1)) * 1e9; w1 = t0 + (hot[0] + 0.8 * (hot[-1] - hot[0] + 1)) * 1e9
win = [e for e in ev if w0 <= e[0] < w1]
print(f'decode seconds {hot[0]}..{hot[-1]} (kernels/s peak {peak}); analysing {(w1 - w0) / 1e9:.1f} s, {len(win)} events')
busy = 0; cur_s = cur_e = None; gaps = []
for s, e, *_ in win:
    if cur_e is None: cur_s, cur_e = s, e; continue
    if s > cur_e:
        busy += cur_e - cur_s; gaps.append(s - cur_e); cur_s, cur_e = s, e
    else: cur_e = max(cur_e, e)
busy += cur_e - cur_s; span = w1 - w0
print(f'GPU busy (union of all streams) {busy / span * 100:.1f} %, idle {100 - busy / span * 100:.1f} %')
for lo, hi in ((0, 5e3), (5e3, 20e3), (20e3, 100e3), (100e3, 1e6), (1e6, 1e12)):
    g = [x for x in gaps if lo <= x < hi]; print(f'  idle gaps {lo/1e3:6.0f}-{hi/1e3:<8.0f}us: n={len(g):6d} total {sum(g) / span * 100:5.1f} % of time')
# rounds: the most frequent kernel per second ~ once a layer; use the gap > 300 us as round boundaries
bounds = [win[0][0]] + [win[i + 1][0] for i in range(len(win) - 1) if win[i + 1][0] - max(x[1] for x in win[max(0, i - 50):i + 1]) > 300e3]
ng = max(1, len(bounds) - 1); mpr = float(sys.argv[2]) if len(sys.argv) > 2 else span / ng / 1e6
nr = span / (mpr * 1e6); print(f'rounds by idle>300us: {ng} ({span / ng / 1e6:.2f} ms each) | by the leader\'s ms/round {mpr}: {nr:.1f}')
by = collections.defaultdict(lambda: [0, 0])
for s, e, st, k, n in win: by[n][0] += e - s; by[n][1] += 1
tot = sum(v[0] for v in by.values())
print('top kernels by GPU time in the window (per round ms, count per round, mean us):')
for n, (d, c) in sorted(by.items(), key=lambda x: -x[1][0])[:22]:
    print(f'  {d / nr / 1e6:6.3f} ms  {c / nr:7.1f}/round  {d / c / 1e3:8.1f} us  {n[:70]}')
small = sum(c for n, (d, c) in by.items() if d / c < 10e3); print(f'kernels per round {sum(c for d, c in by.values()) / nr:.0f}, of which mean < 10 us: {small / nr:.0f}; summed kernel time {tot / nr / 1e6:.2f} ms/round')
