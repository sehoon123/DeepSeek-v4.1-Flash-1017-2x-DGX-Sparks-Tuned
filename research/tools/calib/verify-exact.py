# Independent exactness audit of every offline run: group requests by identity (prompt ids + sampling params), collect the
# trusted reference shas (serial runs and expect_sha), and check every run's output against them and against each other.
import json, glob, os, re, hashlib, collections
CAL = os.getcwd(); runs = []
for hl in sorted(glob.glob(CAL + '/run*/harness.log')):
    d = os.path.dirname(hl)
    for line in open(hl, errors='replace'):
        m = re.search(r'\] run (\S+) req=(\S+) par=\S+ r0=\'([^\']*)\' r1=\'([^\']*)\'', line)
        if m: runs.append((d, m.group(1), m.group(2), m.group(3), m.group(4)))
reqcache = {}
def reqs(f):
    if f not in reqcache: reqcache[f] = json.load(open(f'{CAL}/{f}'))['requests']
    return reqcache[f]
def kid(r):
    x = {k: v for k, v in r.items() if k not in ('name', 'expect_sha')}
    return hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:12]
recs = []; missing_files = set()
for d, name, rf, x0, x1 in runs:
    jf = f'{d}/{name}.r0.jsonl'
    if not os.path.exists(jf) or not os.path.exists(f'{CAL}/{rf}'): missing_files.add((d.split('/')[-1], name, rf)); continue
    R = reqs(rf)
    for line in open(jf, errors='replace'):
        if not line.startswith('{') or '"token_sha"' not in line: continue
        o = json.loads(line); i = o['request']
        if i >= len(R): continue
        recs.append(dict(run=f'{d.split("/")[-1]}/{name}', k=kid(R[i]), name=o['name'], tokens=o['tokens'], sha=o['token_sha'], exp=o.get('expect_sha', ''),
                         reply=o.get('reply', []), serial='--serial' in x0, flags=(x0 + ' | ' + x1).strip(' |')))
print(f'runs parsed {len(runs)} (missing files for {len(missing_files)}), result records {len(recs)}, distinct requests {len({r["k"] for r in recs})}')
trusted = collections.defaultdict(set)
for r in recs:
    if r['exp']: trusted[r['k']].add(r['exp'])
    if r['serial']: trusted[r['k']].add(r['sha'])
# a trusted sha is only usable for a record with the same token count: serial runs and expects were taken at the request's max_tokens
byk = collections.defaultdict(list)
for r in recs: byk[r['k']].append(r)
bad = []; ok_exp = ok_cross = unverified = 0; unver_runs = collections.Counter(); multi = 0
for k, L in byk.items():
    ref = {}
    for r in L:
        if r['serial']: ref[r['tokens']] = (r['sha'], r['reply'])
    for r in L:
        if r['exp']: ref.setdefault(r['tokens'], (r['exp'], None))
    for r in L:
        if r['serial']: continue
        t = ref.get(r['tokens'])
        if t is not None:
            if t[0] == r['sha']: (ok_exp if r['exp'] else ok_cross).__class__; ok_exp += 1 if r['exp'] else 0; ok_cross += 0 if r['exp'] else 1
            else: bad.append((r['run'], r['name'], r['tokens'], r['sha'], t[0]))
            continue
        # no serial/expect reference for this token count: use a longer serial reply as a prefix reference, else cross-run agreement
        pref = [v[1] for tk, v in ref.items() if v[1] is not None and tk > r['tokens']]
        if pref and r['reply'] and pref[0][:len(r['reply'])] == r['reply']: ok_cross += 1; continue
        peers = [q for q in L if q is not r and q['tokens'] == r['tokens']]
        if peers:
            multi += 1
            if all(q['sha'] == r['sha'] for q in peers): ok_cross += 1
            else: bad.append((r['run'], r['name'], r['tokens'], r['sha'], 'peers ' + ','.join(sorted({q['sha'] for q in peers}))))
        else: unverified += 1; unver_runs[r['run']] += 1
n_serial = sum(r['serial'] for r in recs)
print(f'serial reference records {n_serial}; non-serial records checked against an expect/serial sha: {ok_exp} with expect_sha, {ok_cross} via serial/prefix/peer agreement ({multi} by peer agreement only)')
print(f'MISMATCHES: {len(bad)}'); [print('  ', b) for b in bad[:12]]
print(f'no independent reference (only a vacuous self-check): {unverified} records in {len(unver_runs)} runs'); [print('  ', k, v) for k, v in unver_runs.most_common(12)]
