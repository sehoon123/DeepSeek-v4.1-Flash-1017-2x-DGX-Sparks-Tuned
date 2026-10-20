# Held-out policy ceiling from K5 dumps: does any predictor beat the fixed head's survivals under the allocator's ratio rule?
import numpy as np, json, sys, collections, torch
torch.set_num_threads(16); W = 5376
rec = np.dtype([('req','<u4'),('round','<u4'),('tok','<u4'),('conf','<f4'),('j','u1'),('label','u1'),('temp','u1'),('held','u1'),('feat','<u2',(W,))])
def fnv(s):
    h = 2166136261
    for c in s.encode(): h = ((h ^ c) * 16777619) & 0xffffffff
    return h
files = sys.argv[1].split(','); SPLIT = json.load(open('/work/split.json'))
role = {}
for n in SPLIT['train']: role[fnv(n)] = 'train'
for n in SPLIT['val']: role[fnv(n)] = 'val'
for n in SPLIT['test'] + ['smoke_ko']: role[fnv(n)] = 'test'
A = []; src = []
for fi, f in enumerate(files):
    a = np.fromfile(f, dtype=rec); A.append(a); src.append(np.full(len(a), fi))
a = np.concatenate(A); src = np.concatenate(src); n = len(a)
X = torch.from_numpy((a['feat'].astype(np.uint32) << 16).view(np.float32))
w0 = torch.from_numpy(np.fromfile('/work/conf_head_f32.bin', dtype=np.float32))
f16 = lambda t: t.half().float()  # the engine runs the head in f16: judge refits with f16-rounded weights
lg0 = (X @ w0).numpy().astype(np.float64)
err = np.abs(lg0 - a['conf']).max(); print(f'records {n} from {len(files)} dumps; integrity: |logged - fixed-head logit| max {err:.2e} ({"FIXED HEAD" if err < 1e-2 else "NOT the fixed head!"})')
grp = np.array([role.get(int(r), 'other') for r in a['req']]); y = (a['label'] == 1).astype(np.float64); reached = a['label'] != 2
key = np.array([f'{s}:{r}:{t}' for s, r, t in zip(src, a['req'], a['temp'])])
# causal stream features: EMA of the landed fraction over earlier rounds, and the previous round's fraction
order = np.lexsort((a['j'], a['round'], key)); ema = np.zeros(n); prev = np.zeros(n)
rounds = collections.OrderedDict()
for i in order: rounds.setdefault((key[i], int(a['round'][i])), []).append(i)
state = {}
for (k, r), idx in rounds.items():
    e, p = state.get(k, (0.6, 0.6)); ema[idx] = e; prev[idx] = p
    frac = sum(y[i] for i in idx) / len(idx); state[k] = (0.8 * e + 0.2 * frac, frac)
J = a['j'].astype(int); sig = lambda z: 1 / (1 + np.exp(-z))
def auc(s, t):
    o = np.argsort(s); t = t[o]; npos = t.sum(); nneg = len(t) - npos
    return float((np.arange(1, len(t) + 1)[t > 0].sum() - npos * (npos + 1) / 2) / max(1, npos * nneg))
def ll(p, t): p = np.clip(p, 1e-6, 1 - 1e-6); return float(-(t * np.log(p) + (1 - t) * np.log(1 - p)).mean())
def fit_lr(F, t, l2=1e-3, iters=200):
    F = torch.tensor(F); t = torch.tensor(t); w = torch.zeros(F.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=iters, line_search_fn='strong_wolfe')
    def cl():
        opt.zero_grad(); z = F @ w; loss = torch.nn.functional.binary_cross_entropy_with_logits(z, t) + l2 * (w[:-1] ** 2).sum(); loss.backward(); return loss
    opt.step(cl); return w.detach().numpy()
# design for the joint model: per position, [own logit, mean of the round's 5 logits, next logit, ema, prev, sampled, 1]
rl = collections.defaultdict(lambda: np.zeros(5)); 
for i in range(n): rl[(key[i], int(a['round'][i]))][J[i]] = lg0[i]
meanl = np.array([rl[(key[i], int(a['round'][i]))][:a['held'][i]].mean() for i in range(n)])
nxt = np.array([rl[(key[i], int(a['round'][i]))][J[i] + 1] if J[i] + 1 < a['held'][i] else 0.0 for i in range(n)])
base = np.stack([lg0, meanl, nxt, ema, prev, a['temp'].astype(float), np.ones(n)], 1)
F = np.zeros((n, 5 * base.shape[1]))
for p in range(5): F[J == p, p * base.shape[1]:(p + 1) * base.shape[1]] = base[J == p]
tr = (grp == 'train') & reached
probs = {'fixed head (today)': sig(lg0)}
wP = fit_lr(np.stack([lg0 * (J == p) for p in range(5)] + [(J == p).astype(float) for p in range(5)] + [np.zeros(n)], 1)[tr], y[tr], 0.0)
print('   Platt (scale, shift) per position: ' + ', '.join(f'({wP[q]:.3f}, {wP[5 + q]:.3f})' for q in range(5)))
probs['Platt per position'] = sig(np.stack([lg0 * (J == p) for p in range(5)] + [(J == p).astype(float) for p in range(5)] + [np.zeros(n)], 1) @ wP)
wJ = fit_lr(F[tr], y[tr], 1e-4); probs['joint logits+EMA'] = sig(F @ wJ)
# refit of the head itself: init w0, per-position bias, L2 toward w0; then a small MLP correction on top of the head logit
Xt = X[torch.from_numpy(tr)]; yt = torch.tensor(y[tr]); Jt = torch.from_numpy(J[tr])
best = None
for lam in (1e-1, 1e-2):
    w = w0.clone().double().requires_grad_(True); b = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=60, line_search_fn='strong_wolfe'); Xd = Xt.double()
    def cl():
        opt.zero_grad(); z = Xd @ w + b[Jt]; loss = torch.nn.functional.binary_cross_entropy_with_logits(z, yt) + lam * ((w - w0.double()) ** 2).sum(); loss.backward(); return loss
    opt.step(cl)
    wq = f16(w.detach().float()).double(); pr = sig((X.double() @ wq + b.detach()[torch.from_numpy(J)]).numpy())
    np.asarray(wq.float()).tofile(f'/refit/refit-{lam}.bin'); np.asarray(b.detach().numpy(), dtype=np.float64).tofile(f'/refit/refit-{lam}.bias'); print(f'   refit lambda {lam}: bias {[round(float(x), 3) for x in b.detach()]}  |w-w0|/|w0| {float((w.detach() - w0.double()).norm() / w0.double().norm()):.3f}')
    m = (grp == 'val') & reached; v = ll(pr[m], y[m]) if m.any() else ll(pr[tr], y[tr])
    if best is None or v < best[0]: best = (v, lam, pr)
probs[f'head refit (lambda {best[1]})'] = best[2]
mu = Xt.mean(0); sd = Xt.std(0) + 1e-3
net = torch.nn.Sequential(torch.nn.Linear(W + 7, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
Zall = torch.cat([(X - mu) / sd, torch.from_numpy(np.stack([lg0 / 10, ema, prev, a['temp'].astype(float), J / 4.0, meanl / 10, nxt / 10], 1)).float()], 1)
opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); idx_tr = np.where(tr)[0]; yT = torch.from_numpy(y).float()
for ep in range(8):
    perm = np.random.default_rng(ep).permutation(idx_tr)
    for s in range(0, len(perm), 2048):
        bi = torch.from_numpy(perm[s:s + 2048]); opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Zall[bi]).squeeze(-1), yT[bi]); loss.backward(); opt.step()
with torch.no_grad(): probs['MLP 64 on head inputs+EMA'] = sig(net(Zall).squeeze(-1).double().numpy())
fwd = {1: 21.6, 2: 26.0, 3: 30.3, 4: 34.56, 5: 38.35, 6: 42.11}; OH = 3.8; T = lambda k: fwd[1 + k] + OH
R = [(k, idx) for (k, r), idx in rounds.items()]
def sim(p, sel, choose=None):
    tok = collections.Counter(); ms = collections.Counter()
    for k, idx in sel:
        acc = int(sum(y[i] for i in idx)); h = len(idx)
        if choose: kk = choose(h, acc)
        else:
            S = np.cumprod([p[i] for i in idx]); best_v, kk = 1 / T(0), 0
            for c in range(1, h + 1):
                v = (1 + S[:c].sum()) / T(c)
                if v > best_v: best_v, kk = v, c
        tok[k] += 1 + min(acc, kk); ms[k] += T(kk)
    return tok, ms
for set_name, cond in (('VAL greedy', lambda k: k.split(':')[2] == '0'), ('VAL sampled', lambda k: k.split(':')[2] == '1'), ('TEST SparkDash', lambda k: True)):
    want = 'val' if set_name.startswith('VAL') else 'test'
    sel = [(k, idx) for k, idx in R if grp[idx[0]] == want and cond(k)]
    if not sel: continue
    res = {}
    for name, p in [('fixed K3', None), ('fixed K5', None), ('oracle', None)] + list(probs.items()):
        ch = {'fixed K3': lambda h, acc: min(3, h), 'fixed K5': lambda h, acc: h, 'oracle': lambda h, acc: acc}.get(name)
        res[name] = sim(p, sel, ch)
    keys = sorted(set(k for k, _ in sel)); base_t, base_m = res['fixed head (today)']
    rate = lambda tm, ks: sum(tm[0][k] for k in ks) / sum(tm[1][k] for k in ks) * 1000
    b0 = rate(res['fixed head (today)'], keys); rng = np.random.default_rng(1)
    m = np.isin(grp, [want]) & reached
    print(f'== {set_name}: {len(sel)} rounds, {len(keys)} prompts; fixed head {b0:.1f} tok/s (simulated)')
    for name, tm in res.items():
        r = rate(tm, keys); boots = []
        for _ in range(300):
            ks = list(rng.choice(keys, len(keys))); boots.append(rate(tm, ks) / rate(res['fixed head (today)'], ks) - 1)
        extra = f' AUC {auc(probs[name][m], y[m]):.3f} logloss {ll(probs[name][m], y[m]):.4f}' if name in probs else ''
        print(f'   {name:30s} {r:6.1f}  {(r / b0 - 1) * 100:+5.1f}%  95% CI [{np.percentile(boots, 2.5) * 100:+.1f}, {np.percentile(boots, 97.5) * 100:+.1f}]{extra}')
