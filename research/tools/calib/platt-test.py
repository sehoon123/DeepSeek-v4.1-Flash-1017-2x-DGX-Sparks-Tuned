import numpy as np, collections, sys
W = 5376
rec = np.dtype([('req','<u4'),('round','<u4'),('tok','<u4'),('conf','<f4'),('j','u1'),('label','u1'),('temp','u1'),('held','u1'),('feat','<u2',(W,))])
a = np.fromfile(sys.argv[1], dtype=rec); w0 = np.fromfile('/work/conf_head_f32.bin', dtype=np.float32).astype(np.float64)
x = (a['feat'].astype(np.uint32) << 16).view(np.float32).astype(np.float64); lg = x @ w0
y = (a['label'] == 1).astype(float); reached = a['label'] != 2
sig = lambda z: 1 / (1 + np.exp(-z))
def logloss(p, t): p = np.clip(p, 1e-6, 1 - 1e-6); return -(t * np.log(p) + (1 - t) * np.log(1 - p)).mean()
def platt(z, t, iters=50):                               # Newton fit of t ~ sigmoid(A z + B)
    A, B = 1.0, 0.0
    for _ in range(iters):
        p = sig(A * z + B); g = np.array([((p - t) * z).sum(), (p - t).sum()]) + 1e-6 * np.array([A - 1, B])
        h = np.array([[(p * (1 - p) * z * z).sum() + 1e-6, (p * (1 - p) * z).sum()], [(p * (1 - p) * z).sum(), (p * (1 - p)).sum() + 1e-6]])
        d = np.linalg.solve(h, g); A -= d[0]; B -= d[1]
    return A, B
par = {}; pc = lg.copy()
print('position  n   landed   mean p(raw)  Platt(A,B)   logloss raw -> platt')
for j in range(5):
    m = reached & (a['j'] == j)
    if m.sum() < 20: par[j] = (1.0, 0.0); continue
    A, B = platt(lg[m], y[m]); par[j] = (A, B); pc[a['j'] == j] = A * lg[a['j'] == j] + B
    print(f'   {j}    {m.sum():3d}   {y[m].mean():.3f}     {sig(lg[m]).mean():.3f}     ({A:5.2f},{B:+5.2f})     {logloss(sig(lg[m]), y[m]):.4f} -> {logloss(sig(pc[m]), y[m]):.4f}')
fwd = {1: 21.6, 2: 26.0, 3: 30.3, 4: 34.56, 5: 38.35, 6: 42.11}; OH = 3.8; T = lambda k: fwd[1 + k] + OH
rounds = collections.defaultdict(list)
for i in range(len(a)): rounds[(int(a['req'][i]), int(a['round'][i]))].append(i)
def sim(prob, idxs):
    tok = ms = 0.0
    for idx in idxs:
        S = np.cumprod([prob[i] for i in idx]); acc = int(sum(a['label'][i] == 1 for i in idx)); best, bk = 1 / T(0), 0
        for k in range(1, len(idx) + 1):
            v = (1 + S[:k].sum()) / T(k)
            if v > best: best, bk = v, k
        tok += 1 + min(acc, bk); ms += T(bk)
    return tok / ms * 1000
keys = sorted(rounds); allidx = [rounds[k] for k in keys]
raw, cal = sim(sig(lg), allidx), sim(sig(pc), allidx)
print(f'in-sample simulated throughput: raw head {raw:.1f} tok/s | Platt per position {cal:.1f} tok/s ({(cal / raw - 1) * 100:+.1f}%)')
rng = np.random.default_rng(0); reqs = sorted(set(k[0] for k in keys)); gains = []
for _ in range(200):                                     # leave-one-request-out style stability: refit on a bootstrap of rounds
    pick = rng.choice(len(allidx), len(allidx)); tr = [allidx[i] for i in pick]
    pcb = lg.copy()
    for j in range(5):
        ids = np.array([i for idx in tr for i in idx if a['j'][i] == j and reached[i]])
        if len(ids) >= 20:
            A, B = platt(lg[ids], y[ids]); mk = a['j'] == j; pcb[mk] = A * lg[mk] + B
    gains.append(sim(sig(pcb), allidx) / raw - 1)
print(f'bootstrap of the refit: mean gain {np.mean(gains) * 100:+.2f}%  5-95%: {np.percentile(gains, 5) * 100:+.2f}% .. {np.percentile(gains, 95) * 100:+.2f}%')
