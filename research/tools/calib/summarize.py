import json, sys
reqs = []; adm = {}; prof = None
for l in open(sys.argv[1]):
    try: j = json.loads(l)
    except Exception: continue
    if 'prefill_s' in j: adm[j['name']] = j['prefill_s']
    elif 'token_sha' in j: reqs.append(j)
    elif 'profile' in j: prof = j['profile']
    elif 'error' in j or 'step' in j: print('  ERROR', json.dumps(j)[:300])
T = R = 0; secs = 0.0
for r in reqs:
    dec = max(1e-9, r['seconds'] - adm.get(r['name'], 0)); t = r['tokens']; T += t - 1; R += r['rounds']; secs += dec
    if len(sys.argv) > 2 or len(reqs) <= 6:
        print(f"  {r['name']:9s} tok={t:4d} {r['reason']:6s} sha={r['token_sha']} equal={r['equal']} rounds={r['rounds']:4d} tok/round={(t-1)/max(1,r['rounds']):.2f} drafted={r['drafted']} accepted={r['accepted']} tps={(t-1)/dec:.1f}")
if reqs: print(f"  TOTAL requests={len(reqs)} tokens={T} rounds={R} tok/round={T/max(1,R):.3f} decode_tps={T/secs:.2f} all_equal={all(r['equal'] for r in reqs)}")
else: print('  NO REPLIES')
if prof: print('  PROFILE', json.dumps({k: prof[k] for k in ('rounds', 'rows_a_round', 'step_ms', 'forward_ms', 'pass_ms', 'enqueue_ms', 'absorb_ms')}))
