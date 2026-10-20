# Bytes of the served checkpoint by tensor class (headers only) and the bytes a rank reads in a 1-row (K0) forward.
import json, struct, glob, re, collections, sys
CK = sys.argv[1]; ITEM = {'F16': 2, 'BF16': 2, 'F32': 4, 'I16': 2, 'I32': 4, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'I64': 8}
T = {}
for f in sorted(glob.glob(CK + '/model-*.safetensors')):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == '__metadata__': continue
        el = 1
        for d in v['shape']: el *= d
        T[k] = (v['dtype'], tuple(v['shape']), el * ITEM.get(v['dtype'], 1))
M = 2 ** 20
def cls(k):
    if '.ffn.experts.' in k: return 'routed'
    if '.ffn.shared_experts.' in k: return 'shared'
    if '.attn.' in k: return 'attn'
    if '.ffn.gate' in k: return 'router'
    return 'other'
layers = collections.defaultdict(collections.Counter); nexp = collections.Counter()
for k, (dt, sh, b) in T.items():
    m = re.match(r'layers\.(\d+)\.', k)
    if m:
        L = int(m.group(1)); layers[L][cls(k)] += b
        if '.ffn.experts.' in k and k.endswith('w1.trellis'): nexp[L] += 1
print(f'tensors {len(T)}, total {sum(b for _, _, b in T.values()) / 2 ** 30:.2f} GiB, layers {len(layers)}')
for L in (0, 1, 2, 14, 20, 39):
    c = layers[L]; n = nexp[L]
    print(f'  layer {L:2d}: {n} routed experts x {c["routed"] / max(n, 1) / M:.2f} MiB, shared {c["shared"] / M:.1f} MiB, attention {c["attn"] / M:.1f} MiB, router {c["router"] / M:.2f} MiB, other {c["other"] / M:.1f} MiB')
for k in ('head.trellis', 'mtp.0.main_proj.trellis'):
    if k in T: print(f'  {k}: {T[k][0]} {list(T[k][1])} = {T[k][2] / M:.1f} MiB, {T[k][1][-1] // 16} bits a weight')
mtp = sum(b for k, (dt, sh, b) in T.items() if k.startswith('mtp.') and '.ffn.experts.' in k)
print(f'  drafter (mtp) routed experts: {mtp / M:.0f} MiB')
# a rank reads half of every split tensor; a K0 round reads 6 routed experts and the shared expert of every MoE layer
routed = sum(6 * layers[L]['routed'] / nexp[L] / 2 for L in layers if nexp[L]); rest = sum((layers[L]['shared'] + layers[L]['attn'] + layers[L]['other']) / 2 + layers[L]['router'] for L in layers)
head = sum(b for k, (dt, sh, b) in T.items() if k.startswith('head.')) / 2
tot = routed + rest + head
print(f'K0 forward, per rank: routed {routed / M:.0f} MiB + shared/attention/other/router {rest / M:.0f} MiB + head {head / M:.0f} MiB = {tot / M:.0f} MiB')
for bw in (226, 249): print(f'  at {bw} GB/s: {tot / (bw * 1e9) * 1e3:.2f} ms')
