# SparkDash decode cells: medians of the reps, change per cell and geomean (plus the range when reps are paired worst/best).
# Inputs: two conc-probe JSONs, or an overnight FINAL.json (its "baseline" or "soak_median") as the "before".
import json, math, sys, collections
def load(f, field='baseline'):
    d = json.load(open(f)); c = collections.defaultdict(list)
    if 'waves' in d:
        for w in d['waves']:
            if w['n'] in (1, 2, 4): c[(w['type'], w['n'])].append(w['aggregateTps'])
        return f"{d['label']} {d['started'][:16]}Z", c
    for k, v in d[field].items():
        t, n = k.split('.c'); c[(t, int(n))].append(v)
    return f'{f.split("/")[-1]}:{field}', c
def med(v): v = sorted(v); m = len(v) // 2; return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2
fa, fb = sys.argv[1], sys.argv[2]; field = sys.argv[3] if len(sys.argv) > 3 else 'baseline'
la, A = load(fa, field); lb, B = load(fb, field)
print(f'before: {la}   after: {lb}')
lr = []; lo = []; hi = []
for key in sorted(A, key=lambda k: (k[0] != 'prose', k[1])):
    a, b = med(A[key]), med(B[key]); lr.append(math.log(b / a)); lo.append(math.log(min(B[key]) / max(A[key]))); hi.append(math.log(max(B[key]) / min(A[key])))
    print(f'  {key[0]:5s} C{key[1]}: {a:7.2f} -> {b:7.2f}  {100 * (b / a - 1):+6.2f} %   reps {A[key]} -> {B[key]}')
g = lambda v: 100 * (math.exp(sum(v) / len(v)) - 1)
print(f'  geomean {g(lr):+.2f} %  (reps paired worst / best: {g(lo):+.2f} % / {g(hi):+.2f} %)')
