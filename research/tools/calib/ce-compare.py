# Per-request decode speed of the copy-threshold runs against the mean of the two baselines (ce8, ce8b).
import json, sys, math, os
O = sys.argv[1]; runs = ['ce8', 'ce8b', 'ce4', 'ce6', 'ce12']
def load(n):
    f = f'{O}/{n}.r0.jsonl'
    if not os.path.exists(f): return None
    pre, out = {}, {}
    for line in open(f):
        if not line.startswith('{') or '"request"' not in line: continue
        d = json.loads(line)
        if 'prefill_s' in d: pre[d['request']] = d['prefill_s']
        elif 'rounds' in d: out[d['name']] = (d['tokens'], d['seconds'] - pre.get(d['request'], 0), d['rounds'], d.get('equal'))
    return out
R = {n: load(n) for n in runs}; R = {n: v for n, v in R.items() if v}
if 'ce8' not in R or 'ce8b' not in R: sys.exit('baselines missing')
names = sorted(R['ce8']); base = {k: (R['ce8'][k][1] + R['ce8b'][k][1]) / 2 for k in names}
for n, v in R.items():
    tok = sum(v[k][0] for k in names); sec = sum(v[k][1] for k in names); eq = sum(bool(v[k][3]) for k in names)
    g = math.exp(sum(math.log(base[k] / v[k][1]) for k in names) / len(names))
    copy = ' '.join(f'{k} {v[k][0] / v[k][1]:.1f} tok/s ({v[k][2]} rounds)' for k in ('copyv', 'copye') if k in v)
    rest = math.exp(sum(math.log(base[k] / v[k][1]) for k in names if not k.startswith('copy')) / max(1, len([k for k in names if not k.startswith('copy')])))
    print(f'{n:5s} total {tok / sec:6.2f} tok/s  per-request geomean vs base {(g - 1) * 100:+5.2f}%  (non-copy {(rest - 1) * 100:+5.2f}%)  equal {eq}/{len(names)}  | {copy}')
