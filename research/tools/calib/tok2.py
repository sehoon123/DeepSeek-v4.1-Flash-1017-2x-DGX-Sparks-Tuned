import json, jinja2, tokenizers
M = '/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036/model'
tc = json.load(open(f'{M}/tokenizer_config.json')); tok = tokenizers.Tokenizer.from_file(f'{M}/tokenizer.json')
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True); env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(Exception(m))
tpl = env.from_string(open(f'{M}/chat_template.jinja').read())
sp = lambda x: x if isinstance(x, str) else (x or {}).get('content', '')
P = json.load(open('/work/prompts_all.json')); out = {}; tails = {}
for p in P:
    t = tpl.render(messages=[{'role': 'user', 'content': p['text']}], add_generation_prompt=True, enable_thinking=p['think'], thinking=p['think'], bos_token=sp(tc.get('bos_token')), eos_token=sp(tc.get('eos_token')))
    out[p['name']] = tok.encode(t, add_special_tokens=False).ids; tails[p['think']] = t[-30:]
json.dump(out, open('/work/ids_all.json', 'w'))
L = sorted(len(v) for v in out.values()); print('rendered', len(out), 'prompts; tokens min/median/max', L[0], L[len(L) // 2], L[-1], '; tails', {str(k): v for k, v in tails.items()})
