import json, hashlib
I = json.load(open('ids_all.json')); F = json.load(open('prompts_fit.json')); SD = json.load(open('prompts_sd.json'))
def val(n): return int(hashlib.md5(n.encode()).hexdigest(), 16) % 5 == 0          # ~20 % held out of every fit
VAL = [x['name'] for x in F if val(x['name'])]; TRAIN = [x['name'] for x in F if not val(x['name'])]
def req(name, n, t=0.0, seed=None):
    d = {'name': name, 'prompt': I[name], 'max_tokens': n, 'temperature': t, 'top_k': 20, 'top_p': 0.95}
    if seed is not None: d['seed'] = seed
    return d
def put(fn, reqs): json.dump({'requests': reqs}, open(fn, 'w'))
put('req-smoke.json', [req('sd_prose0', 200), req('sd_code0', 200), req('smoke_ko', 200)])
put('req-fit-g.json', [req(x['name'], 384) for x in F])
put('req-fit-s.json', [req(x['name'], 384, 1.0, 1000 + i) for i, x in enumerate(F[:70])])
put('req-test.json', [req(x['name'], 400) for x in SD])
put('req-eval.json', [req(n, 320) for n in VAL] + [req(x['name'], 400) for x in SD])
put('req-cost.json', [req('sd_prose0', 300), req('sd_code0', 300), req('f001', 300), req('f040', 300)])
json.dump({'val': VAL, 'train': TRAIN, 'test': [x['name'] for x in SD]}, open('split.json', 'w'))
print('train', len(TRAIN), 'val', len(VAL), 'test', len(SD), '; val sample', VAL[:6])
