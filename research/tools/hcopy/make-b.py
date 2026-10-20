import json,sys
out=sys.argv[1]
ref={}
for l in open(f'{out}/a-ref.r0.jsonl'):
    try: j=json.loads(l)
    except Exception: continue
    if 'token_sha' in j: ref[j['name']]=j
R=json.load(open('req-ref.json'))['requests']
missing=[r['name'] for r in R if r['name'] not in ref]
assert not missing, f'missing serial replies {missing}'
for k,v in {'c1prose':'a7513d5f16e9','c1code':'1b56d685e63a'}.items(): print('drafted(server)==serial', k, ref[k]['token_sha'], v, 'OK' if ref[k]['token_sha']==v else 'DIFF')
orc=[dict(r,oracle=ref[r['name']]['reply'],expect_sha=ref[r['name']]['token_sha']) for r in R]
json.dump({'requests':orc},open('req-oracle.json','w'))
nat=[dict(r,expect_sha=ref[r['name']]['token_sha']) for r in R if r['name'] in ('copyv','copye','c1prose','c1code')]
json.dump({'requests':nat},open('req-nat.json','w'))
srv={'c1prose':'a7513d5f16e9','c1code':'1b56d685e63a','prose0':'3f948825eb71'}; rounds={'c1prose':159,'c1code':102,'prose0':142}
for l in open('/tmp/a-v0-pass.jsonl'):
    try: j=json.loads(l)
    except Exception: continue
    if 'token_sha' in j: print('V0',j['name'],'sha',j['token_sha'],'server',srv[j['name']],'SHA_OK' if j['token_sha']==srv[j['name']] else 'SHA_DIFF','rounds',j['rounds'],'server',rounds[j['name']],'ROUNDS_OK' if j['rounds']==rounds[j['name']] else 'ROUNDS_DIFF')
print('serial refs',{k:(v['tokens'],v['reason'],v['token_sha']) for k,v in ref.items()})
