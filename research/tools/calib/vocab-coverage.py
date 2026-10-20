# How many of a rank's 505 128-row head blocks the tokens of every distinct reply need (vocabulary pruning of a draft head).
import json, glob, collections, sys
seen = set(); cnt = collections.Counter()
for f in glob.glob(sys.argv[1] + '/run*/*.r0.jsonl'):
    for line in open(f, errors='replace'):
        if not line.startswith('{') or '"reply"' not in line: continue
        d = json.loads(line)
        if d.get('token_sha') in seen: continue
        seen.add(d.get('token_sha')); cnt.update(d['reply'])
V = 129280; half = V // 2; tot = sum(cnt.values())
print(f'distinct replies {len(seen)}, tokens {tot}, distinct token ids {len(cnt)}')
for r in (0, 1):
    blk = collections.Counter()
    for t, c in cnt.items():
        if r * half <= t < (r + 1) * half: blk[(t - r * half) // 128] += c
    mass = sum(blk.values()); acc = 0; need = {}
    for i, c in enumerate([c for _, c in blk.most_common()], 1):
        acc += c
        for q in (0.99, 0.995, 0.999):
            if q not in need and acc >= q * mass: need[q] = i
    print(f'  rank {r}: {mass} tokens over {len(blk)} of 505 blocks; for 99 % coverage {need.get(0.99)} blocks, 99.5 % {need.get(0.995)}, 99.9 % {need.get(0.999)}')
