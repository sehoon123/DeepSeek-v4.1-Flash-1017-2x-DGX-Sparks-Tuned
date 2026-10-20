# Integrity checks of a confidence-head dump: (1) features x w0 == the logit the engine logged, (2) label shape per round,
# (3) the engine's own drafted/accepted counts per request. Runs in the engine image (numpy), CPU only.
import numpy as np, json, struct, sys, os, glob, collections
W = 5376
rec = np.dtype([('req', '<u4'), ('round', '<u4'), ('tok', '<u4'), ('conf', '<f4'), ('j', 'u1'), ('label', 'u1'), ('temp', 'u1'), ('held', 'u1'), ('feat', '<u2', (W,))])
name = sys.argv[1]; out = '/out'
a = np.fromfile(f'{out}/{name}.bin', dtype=rec); print('records', len(a), 'bytes/rec', rec.itemsize)
ck = '/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036/checkpoint'; w0 = None
for f in sorted(glob.glob(ck + '/model-0003[89]-of-00039.safetensors')):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n)); base = 8 + n
        for k, v in h.items():
            if k.endswith('confidence_head.proj.weight'):
                fh.seek(base + v['data_offsets'][0]); raw = fh.read(v['data_offsets'][1] - v['data_offsets'][0]); print('head tensor', k, v['dtype'], v['shape'], f.split('/')[-1])
                w0 = np.frombuffer(raw, dtype={'F32': np.float32, 'BF16': np.uint16, 'F16': np.float16}[v['dtype']]).reshape(-1)
                if v['dtype'] == 'BF16': w0 = (w0.astype(np.uint32) << 16).view(np.float32)
assert w0 is not None and w0.shape[0] == W, 'no confidence head weight found'
x = (a['feat'].astype(np.uint32) << 16).view(np.float32)
l32 = x.astype(np.float64) @ w0.astype(np.float64); l16 = x.astype(np.float64) @ w0.astype(np.float16).astype(np.float64)
for tag, l in (('f32 weights', l32), ('f16-rounded weights', l16)):
    e = np.abs(l - a['conf']); print(f'(1) logit recompute [{tag}]: max|err|={e.max():.3e} mean={e.mean():.3e} corr={np.corrcoef(l, a["conf"])[0,1]:.8f}')
bad = 0; rounds = collections.defaultdict(list)
for i in range(len(a)): rounds[(int(a['req'][i]), int(a['round'][i]))].append(i)
for key, idx in rounds.items():
    lab = [int(a['label'][i]) for i in idx]; js = [int(a['j'][i]) for i in idx]
    ok = js == list(range(len(idx))) and all(v == 1 for v in lab[:lab.count(1)]) and (lab.count(0) <= 1) and (lab == [1] * lab.count(1) + [0] * lab.count(0) + [2] * lab.count(2))
    bad += not ok
print(f'(2) label shape: {len(rounds)} rounds, {bad} malformed; held values {dict(collections.Counter(int(h) for h in a["held"]))}')
def fnv1a(s):
    h = 2166136261
    for c in s.encode(): h = ((h ^ c) * 16777619) & 0xffffffff
    return h
byreq = collections.defaultdict(lambda: [0, 0])
for r, lab in zip(a['req'], a['label']):
    byreq[int(r)][0] += 1; byreq[int(r)][1] += int(lab == 1)
tot_bad = 0
for l in open(f'{out}/{name}.r0.jsonl'):
    try: j = json.loads(l)
    except Exception: continue
    if 'token_sha' not in j: continue
    got = byreq.get(fnv1a(j['name']), [0, 0]); ok = (got[0] == j['drafted'] and got[1] == j['accepted']); tot_bad += (not ok)
    print(f"(3) {j['name']:9s} engine drafted={j['drafted']:4d} accepted={j['accepted']:4d} | dump records={got[0]:4d} landed={got[1]:4d} {'OK' if ok else 'MISMATCH'}")
print('RESULT', 'PASS' if (bad == 0 and tot_bad == 0 and np.abs(l16 - a['conf']).max() < 1e-2 or np.abs(l32 - a['conf']).max() < 1e-2) and bad == 0 and tot_bad == 0 else 'FAIL')
