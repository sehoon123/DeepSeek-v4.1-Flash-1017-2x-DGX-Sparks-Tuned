import numpy as np, json, struct, glob
W = 5376; D = 5120
rec = np.dtype([('req','<u4'),('round','<u4'),('tok','<u4'),('conf','<f4'),('j','u1'),('label','u1'),('temp','u1'),('held','u1'),('feat','<u2',(W,))])
a = np.fromfile('/out/smoke.bin', dtype=rec)
ck = '/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036/checkpoint'; T = {}
for f in sorted(glob.glob(ck + '/model-0003[89]-of-00039.safetensors')):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n)); base = 8 + n
        for k, v in h.items():
            if k.startswith('mtp.2.') and (k.endswith('confidence_head.proj.weight') or k.endswith('markov_head.embed.weight')) or k.endswith('ffn.gate.bias'):
                if k.startswith('mtp.'):
                    fh.seek(base + v['data_offsets'][0]); T[k] = (v['dtype'], v['shape'], fh.read(v['data_offsets'][1] - v['data_offsets'][0]))
for k, (dt, sh, raw) in sorted(T.items()): print('tensor', k, dt, sh, len(raw), 'bytes')
x = (a['feat'].astype(np.uint32) << 16).view(np.float32).astype(np.float64)
# (A) the Markov half of the features: row j (j>=1) must be embed[token of draft j-1] (the token the step read)
dt, sh, raw = T['mtp.2.markov_head.embed.weight']; E = (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(sh)
ok = tot = 0
for i in range(1, len(a)):
    if a['j'][i] >= 1 and a['round'][i] == a['round'][i-1] and a['req'][i] == a['req'][i-1]:
        tot += 1; ok += np.array_equal(x[i, D:].astype(np.float32), E[a['tok'][i-1]])
print(f'(A) Markov embedding half equals embed[previous draft token]: {ok}/{tot}')
# (B) which weight does the engine's logit come from?
raw = T['mtp.2.confidence_head.proj.weight'][2]
cands = {'f32 as stored': np.frombuffer(raw, np.float32),
         'first half of the f32 bytes read as f16': np.frombuffer(raw[:W*2], np.float16).astype(np.float32),
         'first half read as bf16': (np.frombuffer(raw[:W*2], np.uint16).astype(np.uint32) << 16).view(np.float32)}
for name, w in cands.items():
    w = w.astype(np.float64); l = x @ w
    good = np.isfinite(l)
    c = np.corrcoef(l[good], a['conf'][good])[0, 1] if good.sum() > 2 else float('nan')
    print(f'(B) {name:42s} finite={good.mean():.3f} corr={c:+.6f} max|err|={np.nanmax(np.abs(l - a["conf"])):.3e} |w| range=({np.nanmin(np.abs(w)):.2e},{np.nanmax(np.abs(w)):.2e})')
q = np.percentile(a['conf'], [1, 10, 50, 90, 99]); s = 1 / (1 + np.exp(-a['conf'].astype(np.float64)))
print('(C) logged conf logit percentiles 1/10/50/90/99:', np.round(q, 2), '| sigmoid mean', round(s.mean(), 3), '| reached-draft landing rate', round((a['label'] == 1).sum() / (a['label'] != 2).sum(), 3))
for k in sorted(T):
    if k.endswith('ffn.gate.bias'):
        dt, sh, raw = T[k]; b32 = np.frombuffer(raw, np.float32); b16 = np.frombuffer(raw[:len(raw)//2], np.float16) if dt == 'F32' else None
        print(f'(D) {k} {dt}: true range ({b32.min():+.3f},{b32.max():+.3f}) | as-f16-misread range ({np.nanmin(b16):+.3e},{np.nanmax(b16):+.3e})' if dt == 'F32' else f'(D) {k} {dt}')
