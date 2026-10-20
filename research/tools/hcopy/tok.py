import json, jinja2, tokenizers
M='/hf/tensorfold-dsv41/6d54be4f-ablit-0bdc2036/model'
tc=json.load(open(f'{M}/tokenizer_config.json')); tok=tokenizers.Tokenizer.from_file(f'{M}/tokenizer.json')
env=jinja2.Environment(trim_blocks=True,lstrip_blocks=True); env.globals['raise_exception']=lambda m:(_ for _ in ()).throw(Exception(m))
tpl=env.from_string(open(f'{M}/chat_template.jinja').read())
sp=lambda x: x if isinstance(x,str) else (x or {}).get('content','')
P=json.load(open('/work/prompts.json')); out={}
for k,p in P.items():
    t=tpl.render(messages=[{'role':'user','content':p}],add_generation_prompt=True,enable_thinking=False,thinking=False,bos_token=sp(tc.get('bos_token')),eos_token=sp(tc.get('eos_token')))
    out[k]=tok.encode(t,add_special_tokens=False).ids
json.dump(out,open('/work/ids.json','w')); print({k:len(v) for k,v in out.items()})
