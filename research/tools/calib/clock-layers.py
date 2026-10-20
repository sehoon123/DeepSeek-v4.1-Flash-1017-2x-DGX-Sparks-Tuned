# Layer-end GPU clocks (TF_DS_CLOCK_MARKS=2) of both ranks: per-layer time a round, its spread, the slowest and fastest layers.
import json, sys, statistics as st
def grab(p):
    for line in open(p, errors='replace'):
        i = line.find('{"rank"')
        if i >= 0 and 'round_clock_us' in line: return json.loads(line[i:])['round_clock_us']
O, n = sys.argv[1], sys.argv[2]; L0, L1 = grab(f'{O}/{n}.r0.jsonl'), grab(f'{O}/{n}.r1.log')
lay = sorted(int(k[1:]) for k in L0 if k.startswith('L')); v0 = [L0[f'L{i}'][0] for i in lay]; v1 = [L1.get(f'L{i}', [0])[0] for i in lay]
s = sorted(zip(v0, lay))
print(f'{n}: {len(lay)} layers, sum r0 {sum(v0):.0f} us r1 {sum(v1):.0f} us a round; median {st.median(v0):.0f}, max |r0 - r1| {max(abs(a - b) for a, b in zip(v0, v1)):.0f}')
print('  fastest:', [(l, round(v)) for v, l in s[:5]], ' slowest:', [(l, round(v)) for v, l in s[-6:]])
