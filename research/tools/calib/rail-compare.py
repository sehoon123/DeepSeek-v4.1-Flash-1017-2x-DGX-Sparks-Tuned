import json, sys, math
O = sys.argv[1]; runs = ['rail2-a', 'rail2-b', 'rail1-a', 'rail1-b']
def load(n):
    pre, out = {}, {}
    for line in open(f'{O}/{n}.r0.jsonl', errors='replace'):
        if not line.startswith('{') or '"request"' not in line: continue
        d = json.loads(line)
        if 'prefill_s' in d: pre[d['request']] = d['prefill_s']
        elif 'rounds' in d: out[d['name']] = (d['tokens'], d['seconds'] - pre.get(d['request'], 0), d['rounds'], d.get('equal'), d.get('expect_sha', ''))
    return out
R = {}
for n in runs:
    try: R[n] = load(n)
    except FileNotFoundError: pass
base = {k: (R['rail2-a'][k][1] + R['rail2-b'][k][1]) / 2 for k in R['rail2-a']}
for n, v in R.items():
    tok = sum(x[0] for x in v.values()); sec = sum(x[1] for x in v.values()); eq = sum(bool(x[3]) and bool(x[4]) for x in v.values())
    g = math.exp(sum(math.log(base[k] / v[k][1]) for k in v) / len(v))
    print(f'{n:8s} decode {tok / sec:6.2f} tok/s  per-request geomean vs 2-rail mean {(g - 1) * 100:+5.2f}%  equal-to-serial {eq}/{len(v)}  rounds {sum(x[2] for x in v.values())}')
