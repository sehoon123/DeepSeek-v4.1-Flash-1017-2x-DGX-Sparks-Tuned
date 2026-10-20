import json,sys
reqs=[];adm={};summ=None
for l in open(sys.argv[1]):
    try: j=json.loads(l)
    except Exception: continue
    if 'prefill_s' in j: adm[j['name']]=j['prefill_s']
    elif 'token_sha' in j: reqs.append(j)
    elif 'error' in j or 'step' in j: print('  ERROR',json.dumps(j)[:300])
for r in reqs:
    dec=max(1e-9,r['seconds']-adm.get(r['name'],0)); t=r['tokens']
    print(f"  {r['name']:8s} tok={t:4d} {r['reason']:6s} sha={r['token_sha']} equal={r['equal']} rounds={r['rounds']:4d} tok/round={(t-1)/max(1,r['rounds']):.2f} drafted={r['drafted']} accepted={r['accepted']} min_rows={r['min_rows']} tps={(t-1)/dec:.1f}")
if not reqs: print('  NO REPLIES')
