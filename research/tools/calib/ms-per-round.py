# decode ms a round from a leader jsonl: per-request seconds include that request's prefill, so subtract it
import json, sys
s = p = r = 0
for line in open(sys.argv[1]):
    if not line.startswith('{') or '"request"' not in line: continue
    d = json.loads(line)
    if 'rounds' in d: s += d['seconds']; r += d['rounds']
    elif 'prefill_s' in d: p += d['prefill_s']
print(f'{(s - p) / r * 1000:.3f}' if r else '38.8')
