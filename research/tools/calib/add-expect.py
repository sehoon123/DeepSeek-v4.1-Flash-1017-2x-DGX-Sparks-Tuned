import json, sys
ref, src, dst = sys.argv[1:4]
sha = {}
for l in open(ref):
    try: j = json.loads(l)
    except Exception: continue
    if 'token_sha' in j: sha[j['name']] = j['token_sha']
R = json.load(open(src))['requests']; missing = [r['name'] for r in R if r['name'] not in sha]
assert not missing, f'no serial reply for {missing}'
json.dump({'requests': [dict(r, expect_sha=sha[r['name']]) for r in R]}, open(dst, 'w'))
if len(sys.argv) > 4: json.dump({'requests': [dict(r, expect_sha=sha[r['name']]) for r in R if r['name'].startswith('sd_')]}, open(sys.argv[4], 'w'))
print('expect_sha added', len(R), dst)
