# Per-phase GPU clock gaps (us a round) of both ranks side by side, from a clock-marks run.
import json, sys, glob
O, name = sys.argv[1], sys.argv[2]
def grab(path):
    out = {}
    for line in open(path, errors='replace'):
        i = line.find('{"rank"')
        if i < 0: continue
        try: d = json.loads(line[i:])
        except Exception: continue
        for key in ('round_clock_us', 'round_launches', 'profile', 'follow_profile'):
            if key in d: out[key] = d[key]
    return out
r0 = grab(f'{O}/{name}.r0.jsonl'); r1 = grab(f'{O}/{name}.r1.log')
c0, c1 = r0.get('round_clock_us', {}), r1.get('round_clock_us', {})
print(f'{name}: rounds r0 {c0.get("rounds")} r1 {c1.get("rounds")}')
keys = [k for k in c0 if k != 'rounds'] + [k for k in c1 if k != 'rounds' and k not in c0]
tot0 = tot1 = 0.0
print('  phase        r0 us/round  r1 us/round   r1-r0   count/round')
for k in keys:
    a = c0.get(k, [0, 0]); b = c1.get(k, [0, 0]); tot0 += a[0]; tot1 += b[0]
    if max(a[0], b[0]) < 5 and not k.startswith('L'): continue
    print(f'  {k:10s} {a[0]:12.1f} {b[0]:12.1f} {b[0] - a[0]:+8.1f}   {max(a[1], b[1]):6.1f}')
print(f'  {"TOTAL":10s} {tot0:12.1f} {tot1:12.1f} {tot1 - tot0:+8.1f}')
