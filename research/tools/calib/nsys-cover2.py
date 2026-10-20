# Memory-light coverage (array-backed, streaming): time with a DRAM-streaming kernel (MoE grouped_cp / glinear) running,
# the rest, and which kernels run in the uncovered time. Same window rule and numbers as nsys-cover.py.
import sqlite3, sys, collections
from array import array
db = sqlite3.connect(sys.argv[1]); mpr = float(sys.argv[2]); names = dict(db.execute("select id, value from StringIds"))
ids = {v: k for k, v in names.items()}
STREAM = {ids.get('grouped_cp_kernel'), ids.get('glinear_kernel')}; PREF = ids.get('tf_ds_l2_prefetch_kernel')
t0 = db.execute("select min(start) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
dens = collections.Counter(r[0] for r in db.execute("select (start - ?) / 1000000000 from CUPTI_ACTIVITY_KIND_KERNEL", (t0,)))
peak = max(dens.values()); hot = sorted(s for s in dens if dens[s] >= 0.5 * peak)
w0 = int(t0 + (hot[0] + 0.2 * (hot[-1] - hot[0] + 1)) * 1e9); w1 = int(t0 + (hot[0] + 0.8 * (hot[-1] - hot[0] + 1)) * 1e9)
S, E, N = array('q'), array('q'), array('i')
for s, e, n in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start >= ? and start < ? order by start", (w0, w1)):
    S.append(s); E.append(e); N.append(n)
span = w1 - w0; nr = span / (mpr * 1e6)
def union_len(sel):
    tot = 0; cs = ce = None
    for i in range(len(S)):
        if not sel(N[i]): continue
        if ce is None or S[i] > ce:
            if ce is not None: tot += ce - cs
            cs, ce = S[i], E[i]
        elif E[i] > ce: ce = E[i]
    return tot + (ce - cs if ce is not None else 0)
U_all = union_len(lambda n: True); U_comp = union_len(lambda n: n != PREF); U_str = union_len(lambda n: n in STREAM)
print(f'{sys.argv[1].split("/")[-1]}: window {span/1e9:.1f} s ~ {nr:.0f} rounds of {mpr} ms, {len(S)} kernels')
print(f'  any kernel {U_all/span*100:5.1f} % | any compute kernel (no prefetch) {U_comp/span*100:5.1f} % | MoE/linear streaming {U_str/span*100:5.1f} %')
print(f'  per round: streaming covered {U_str/nr/1e6:.2f} ms, NOT covered {(span - U_str)/nr/1e6:.2f} ms (of which compute idle {(span - U_comp)/nr/1e6:.2f} ms)')
# MoE and linear overlap: union(MoE) + union(linear) - union(both)
U_moe = union_len(lambda n: n == ids.get('grouped_cp_kernel')); U_lin = union_len(lambda n: n == ids.get('glinear_kernel'))
print(f'  MoE alone-union {U_moe/nr/1e6:.2f} ms, linear {U_lin/nr/1e6:.2f} ms, both running at once {(U_moe + U_lin - U_str)/nr/1e6:.2f} ms a round')
# uncovered intervals (complement of the streaming union), then each non-streaming kernel's time inside them
gaps = array('q'); cs = ce = None
for i in range(len(S)):
    if N[i] not in STREAM: continue
    if ce is None: 
        if S[i] > w0: gaps.extend((w0, S[i]))
        cs, ce = S[i], E[i]
    elif S[i] > ce: gaps.extend((ce, S[i])); cs, ce = S[i], E[i]
    elif E[i] > ce: ce = E[i]
if ce is not None and ce < w1: gaps.extend((ce, w1))
by = collections.Counter(); g = 0; ng = len(gaps) // 2
for i in range(len(S)):
    if N[i] in STREAM: continue
    s, e = S[i], E[i]
    while g < ng and gaps[2 * g + 1] <= s: g += 1
    j = g
    while j < ng and gaps[2 * j] < e:
        by[N[i]] += min(gaps[2 * j + 1], e) - max(gaps[2 * j], s); j += 1
print('  time in the uncovered intervals by kernel (ms/round; overlaps double count):')
for n, d in by.most_common(10): print(f'    {d/nr/1e6:6.3f}  {names.get(n, "?")[:60]}')
