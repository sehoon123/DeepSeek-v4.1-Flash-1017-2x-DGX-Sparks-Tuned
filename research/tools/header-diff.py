# Compare every tensor's (dtype, shape) between the stock Mia checkpoint (headers fetched by HTTP range, no weights) and the served Ablit one.
import json, struct, os, sys, urllib.request, collections
tok = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
REPO, REV = 'Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw', '6d54be4fca338d79b17c3b03cc9a423bb3ac6718'
ABL = '/home/USER/.cache/huggingface/tensorfold-dsv41/6d54be4f-ablit-0bdc2036/checkpoint'
def rng(url, a, b):
    r = urllib.request.Request(url, headers={'Authorization': 'Bearer ' + tok, 'Range': f'bytes={a}-{b}'})
    with urllib.request.urlopen(r, timeout=60) as x: return x.read()
def stock_header(fn):
    url = f'https://huggingface.co/{REPO}/resolve/{REV}/{fn}'
    n = struct.unpack('<Q', rng(url, 0, 7))[0]; return json.loads(rng(url, 8, 8 + n - 1))
def local_header(path):
    with open(path, 'rb') as fh: n = struct.unpack('<Q', fh.read(8))[0]; return json.loads(fh.read(n))
S = {}; A = {}; fetched = 0
for i in range(1, 40):
    fn = f'model-{i:05d}-of-00039.safetensors'
    for k, v in stock_header(fn).items():
        if k != '__metadata__': S[k] = (v['dtype'], tuple(v['shape']), fn)
    for k, v in local_header(f'{ABL}/{fn}').items():
        if k != '__metadata__': A[k] = (v['dtype'], tuple(v['shape']), fn)
    fetched += 1
print(f'shards compared {fetched}; stock tensors {len(S)}, served tensors {len(A)}')
only_s = sorted(set(S) - set(A)); only_a = sorted(set(A) - set(S)); print('only in stock:', len(only_s), only_s[:4], '| only in served:', len(only_a), only_a[:4])
dt = [(k, S[k][0], A[k][0]) for k in S if k in A and S[k][0] != A[k][0]]; sh = [(k, S[k][1], A[k][1]) for k in S if k in A and S[k][1] != A[k][1]]
print('dtype differences:', len(dt))
for k, a, b in dt[:12]: print('   ', k, a, '->', b)
print('shape differences:', len(sh), '| by suffix:', dict(collections.Counter(k.rsplit('.', 1)[-1] + ' in ' + ('mtp' if k.startswith('mtp') else 'layers.%s.%s' % (k.split('.')[1], k.split('.')[2]) if k.startswith('layers') else 'other') for k, _, _ in sh).most_common(6)))
wo = [k for k, _, _ in sh if '.attn.wo_b.' in k]; print('   shape-changed tensors that are attn.wo_b of layers 10-35:', sum(1 for k in wo if 10 <= int(k.split('.')[1]) <= 35), 'of', len(sh))
for k, a, b in [x for x in sh if '.attn.wo_b.' not in x[0]][:6]: print('   NON-wo_b shape change:', k, a, '->', b)
print('stock dtype of the confidence head:', S.get('mtp.2.confidence_head.proj.weight', ('?',))[0], '| served:', A.get('mtp.2.confidence_head.proj.weight', ('?',))[0])
print('stock dtype of ffn.gate.bias mtp.0:', S.get('mtp.0.ffn.gate.bias', ('?',))[0], '| layers.5 gate bias stock/served:', S.get('layers.5.ffn.gate.bias', ('-',))[0], A.get('layers.5.ffn.gate.bias', ('-',))[0])
