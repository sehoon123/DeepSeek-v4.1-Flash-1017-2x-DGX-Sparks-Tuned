import numpy as np, collections, sys
W = 5376
rec = np.dtype([('req','<u4'),('round','<u4'),('tok','<u4'),('conf','<f4'),('j','u1'),('label','u1'),('temp','u1'),('held','u1'),('feat','<u2',(W,))])
a = np.fromfile(sys.argv[1], dtype=rec); w0 = np.fromfile('/work/conf_head_f32.bin', dtype=np.float32).astype(np.float64)
x = (a['feat'].astype(np.uint32) << 16).view(np.float32).astype(np.float64); p = 1 / (1 + np.exp(-(x @ w0)))
fwd = {1: 21.6, 2: 26.0, 3: 30.3, 4: 34.56, 5: 38.35, 6: 42.11}; OH = 3.8; T = lambda k: fwd[1 + k] + OH
rounds = collections.defaultdict(list)
for i in range(len(a)): rounds[(int(a['req'][i]), int(a['round'][i]))].append(i)
byreq = collections.defaultdict(list)
for (r, n), idx in sorted(rounds.items()):
    acc = int(sum(a['label'][i] == 1 for i in idx)); byreq[r].append((np.cumprod([p[i] for i in idx]), acc))
def ratio_k(S):
    best, bk = 1 / T(0), 0
    for k in range(1, len(S) + 1):
        v = (1 + S[:k].sum()) / T(k)
        if v > best: best, bk = v, k
    return bk
def lam_k(S, lam):
    best, bk = 1 - lam * T(0), 0
    for k in range(1, len(S) + 1):
        v = 1 + S[:k].sum() - lam * T(k)
        if v > best: best, bk = v, k
    return bk
def rate(seq, chooser):
    tok = ms = 0.0
    for S, acc in seq:
        k = chooser(S); tok += 1 + min(acc, k); ms += T(k)
    return tok, ms
allr = [v for seq in byreq.values() for v in seq]
base_t, base_m = rate(allr, ratio_k); print(f'rounds {len(allr)} requests {len(byreq)}; ratio rule (today): {base_t/base_m*1000:.1f} tok/s')
for lam in (0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12):
    t, m = rate(allr, lambda S: lam_k(S, lam)); print(f'  fixed lambda={lam:.2f}: {t/m*1000:6.1f} tok/s ({(t/m/(base_t/base_m)-1)*100:+5.1f}%)')
# per-request self-consistent lambda (best case), and a causal EMA estimate
tt = mm = 0.0
for seq in byreq.values():
    lam = 0.07
    for _ in range(30):
        t, m = rate(seq, lambda S: lam_k(S, lam)); lam = t / m
    t, m = rate(seq, lambda S: lam_k(S, lam)); tt += t; mm += m
print(f'  per-request fixed point lambda: {tt/mm*1000:6.1f} tok/s ({(tt/mm/(base_t/base_m)-1)*100:+5.1f}%)')
tt = mm = 0.0
for seq in byreq.values():
    ema = 0.07
    for S, acc in seq:
        k = lam_k(S, ema); tok = 1 + min(acc, k); ms = T(k); tt += tok; mm += ms; ema += 0.1 * (tok / ms - ema)
print(f'  causal EMA(0.1) lambda:         {tt/mm*1000:6.1f} tok/s ({(tt/mm/(base_t/base_m)-1)*100:+5.1f}%)')
print('chosen k histogram  ratio:', dict(collections.Counter(ratio_k(S) for S, _ in allr)), '| lambda 0.07:', dict(collections.Counter(lam_k(S, 0.07) for S, _ in allr)))
