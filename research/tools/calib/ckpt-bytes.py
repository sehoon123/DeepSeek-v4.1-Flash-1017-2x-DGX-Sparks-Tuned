# Bytes of every tensor class in the served checkpoint (headers only), to turn kernel durations into achieved DRAM bandwidth.
import json, struct, glob, re, collections, sys
CK = '/home/USER/.cache/huggingface/tensorfold-dsv41/6d54be4f-ablit-0bdc2036/checkpoint'
ITEM = {'F16': 2, 'BF16': 2, 'F32': 4, 'I16': 2, 'U16': 2, 'I32': 4, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E8M0': 1, 'I64': 8, 'F64': 8, 'U32': 4}
T = {}
for f in sorted(glob.glob(CK + '/model-*.safetensors')):
    with open(f, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]; h = json.loads(fh.read(n))
    for k, v in h.items():
        if k == '__metadata__': continue
        el = 1
        for d in v['shape']: el *= d
        T[k] = (v['dtype'], tuple(v['shape']), el * ITEM.get(v['dtype'], 1))
print('tensors', len(T), 'total %.2f GiB' % (sum(b for _, _, b in T.values()) / 2**30))
def pat(k): return re.sub(r'experts\.\d+\.', 'experts.E.', re.sub(r'^layers\.\d+\.', 'layers.L.', k))
for layer in (2, 5):
    print(f'--- layer {layer}: classes (count, example shape/dtype, bytes each, class total MiB)')
    cls = collections.OrderedDict()
    for k, (dt, sh, b) in T.items():
        if k.startswith(f'layers.{layer}.'):
            p = pat(k); c = cls.setdefault(p, [0, dt, sh, b, 0]); c[0] += 1; c[4] += b
    for p, (n, dt, sh, b, tot) in cls.items(): print(f'  {n:5d} {p[9:]:46s} {dt:5s} {str(sh):24s} {b/2**20:9.3f} MiB each  total {tot/2**20:9.1f} MiB')
print('--- non-layer tensors')
cls = collections.OrderedDict()
for k, (dt, sh, b) in T.items():
    if not k.startswith('layers.'):
        p = re.sub(r'\.\d+\.', '.N.', k); c = cls.setdefault(p, [0, dt, sh, b, 0]); c[0] += 1; c[4] += b
for p, (n, dt, sh, b, tot) in cls.items():
    if tot > 5 * 2**20: print(f'  {n:5d} {p:50s} {dt:5s} {str(sh):24s} total {tot/2**20:9.1f} MiB')
print('layer count', len({k.split(".")[1] for k in T if k.startswith("layers.")}))
