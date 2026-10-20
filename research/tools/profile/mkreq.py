import json, sys, jinja2, tokenizers
M = '/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036/model'
tc = json.load(open(f'{M}/tokenizer_config.json'))
tok = tokenizers.Tokenizer.from_file(f'{M}/tokenizer.json')
env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
env.globals['raise_exception'] = lambda m: (_ for _ in ()).throw(Exception(m))
tpl = env.from_string(open(f'{M}/chat_template.jinja').read())
def special(x): return x if isinstance(x, str) else (x or {}).get('content', '')
def ids(prompt):
    text = tpl.render(messages=[{'role': 'user', 'content': prompt}], add_generation_prompt=True, enable_thinking=False, thinking=False,
                      bos_token=special(tc.get('bos_token')), eos_token=special(tc.get('eos_token')))
    return tok.encode(text, add_special_tokens=False).ids, text
P = json.load(open('/work/prompts.json'))['prompts']
out = {}
for t in ('prose', 'code'):
    out[t] = []
    for p in P[t]:
        i, text = ids(p); out[t].append(i)
print(json.dumps({'counts': {t: [len(x) for x in out[t]] for t in out}, 'tail': repr(ids(P['prose'][0])[1][-120:])}))
json.dump(out, open('/work/prompt_ids.json', 'w'))
