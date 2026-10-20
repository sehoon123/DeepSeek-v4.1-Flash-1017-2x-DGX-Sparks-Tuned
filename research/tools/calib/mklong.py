import json, jinja2, tokenizers, glob, os
M = '/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036/model'
tc = json.load(open(f'{M}/tokenizer_config.json')); tok = tokenizers.Tokenizer.from_file(f'{M}/tokenizer.json')
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True); env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(Exception(m))
tpl = env.from_string(open(f'{M}/chat_template.jinja').read()); sp = lambda x: x if isinstance(x, str) else (x or {}).get('content', '')
files = sorted(glob.glob('/corpus/**/*.zig', recursive=True)); parts = []
for f in files:
    t = open(f, errors='replace').read(); parts.append((f.replace('/corpus/', ''), t, len(tok.encode(t, add_special_tokens=False).ids)))
task = '\n\nBased only on the code above, write a technical summary (about 300 words) of how one speculative decoding round works in this engine: drafting, verification, acceptance and the caches.'
out = {}
for name, target in (('long32', 32000), ('long64', 64000), ('long128', 128000)):
    body, n = [], 0
    for path, t, k in parts:
        if n + k > target: continue
        body.append(f'// file: {path}\n{t}'); n += k
        if n > target * 0.97: break
    text = '\n'.join(body) + task
    ids = tok.encode(tpl.render(messages=[{'role': 'user', 'content': text}], add_generation_prompt=True, enable_thinking=False, thinking=False, bos_token=sp(tc.get('bos_token')), eos_token=sp(tc.get('eos_token'))), add_special_tokens=False).ids
    out[name] = ids; print(name, 'files', len(body), 'prompt tokens', len(ids))
json.dump({'requests': [{'name': k, 'prompt': v, 'max_tokens': 400, 'temperature': 0.0, 'top_k': 20, 'top_p': 0.95} for k, v in out.items()]}, open('/work/req-long.json', 'w'))
