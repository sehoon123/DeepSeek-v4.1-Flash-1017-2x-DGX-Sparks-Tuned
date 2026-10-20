# Drafter-accuracy gate: how much tokens/s a gain in landing rate would buy at the best fixed depth, from the K5 dumps.
# Model: tokens per round = 1 + sum_j prod_{i<=j} p_i (p_j = P(draft j lands | reached)); round time = measured forward + DSpark.
import numpy as np, json
W = 5376
rec = np.dtype([('req', '<u4'), ('round', '<u4'), ('tok', '<u4'), ('conf', '<f4'), ('j', 'u1'), ('label', 'u1'), ('temp', 'u1'), ('held', 'u1'), ('feat', '<u2', (W,))])
assert rec.itemsize == 10772
def fnv(s):
    h = 2166136261
    for c in s.encode(): h = ((h ^ c) * 16777619) & 0xffffffff
    return h
SPLIT = json.load(open('/work/split.json')); role = {}
for n in SPLIT['train']: role[fnv(n)] = 'train'
for n in SPLIT['val']: role[fnv(n)] = 'val'
for n in SPLIT['test'] + ['smoke_ko']: role[fnv(n)] = 'test'
FWD = {1: 21.6, 2: 26.0, 3: 30.3, 4: 34.56, 5: 38.35, 6: 42.11}     # ms per forward at 1..6 rows (profile fit)
def T(K): return (FWD[1] + 0.3) if K == 0 else FWD[1 + K] + 3.8      # + DSpark pass 3.5 + host 0.3 when drafting
def tpr(p, K):
    acc, pr = 0.0, 1.0
    for j in range(K): pr *= p[j]; acc += pr
    return 1 + acc
def tps(p, K): return tpr(p, K) / T(K) * 1000
def best(p): return max((tps(p, K), K) for K in range(6))
def shift(p, d, only=None): return [min(0.999, x + (d if (only is None or j in only) else 0.0)) for j, x in enumerate(p)]
def gain(p, q): return best(q)[0] / best(p)[0] - 1
def need(p, target):
    lo, hi = 0.0, 0.3
    for _ in range(40):
        mid = (lo + hi) / 2
        if gain(p, shift(p, mid)) >= target: hi = mid
        else: lo = mid
    return hi
def landing(a, sel):
    m = sel & (a['label'] != 2); J = a['j'][m].astype(int); L = a['label'][m] == 1
    return [float(L[J == j].mean()) for j in range(5)], int(m.sum())
sets = {}
a = np.memmap('/out/d-fit-g.bin', dtype=rec, mode='r'); g = np.array([role.get(int(r), 'other') for r in a['req']])
sets['all fit-greedy (in-sample)'] = landing(a, np.ones(len(a), bool)); sets['VAL greedy (23 prompts)'] = landing(a, g == 'val')
a2 = np.memmap('/out/d-fit-s.bin', dtype=rec, mode='r'); s2 = np.array([role.get(int(r), 'other') for r in a2['req']])
sets['VAL sampled T=1 (13 prompts)'] = landing(a2, s2 == 'val')
a3 = np.memmap('/out/d-test.bin', dtype=rec, mode='r'); t3 = np.array([role.get(int(r), 'other') for r in a3['req']])
sets['TEST SparkDash (11 prompts)'] = landing(a3, t3 == 'test')
print('set                               n_drafts  p0    p1    p2    p3    p4   | best tok/s (K) | tok/round@K5 model')
for name, (p, n) in sets.items():
    b, K = best(p)
    print(f"{name:33s} {n:7d}  " + ' '.join(f'{x:.3f}' for x in p) + f" | {b:6.1f} (K{K}) | {tpr(p, 5):.3f}")
print()
p = sets['TEST SparkDash (11 prompts)'][0]
print('uniform landing-rate gain on TEST -> throughput gain at the best fixed depth:')
for d in (0.01, 0.02, 0.03, 0.05):
    print(f'  +{d*100:.0f} pt on every position: {gain(p, shift(p, d)) * 100:+5.1f} %')
print('same +3 pt on one group of positions only:')
for lab, only in (('p0 only', {0}), ('p1 only', {1}), ('p2-p4 only', {2, 3, 4})):
    print(f'  +3 pt {lab:10s}: {gain(p, shift(p, 0.03, only)) * 100:+5.1f} %')
print(f'uniform gain needed for +10 % on TEST: +{need(p, 0.10) * 100:.1f} pt; for +5 %: +{need(p, 0.05) * 100:.1f} pt')
pv = sets['VAL greedy (23 prompts)'][0]
print(f'uniform gain needed for +10 % on VAL greedy: +{need(pv, 0.10) * 100:.1f} pt')
