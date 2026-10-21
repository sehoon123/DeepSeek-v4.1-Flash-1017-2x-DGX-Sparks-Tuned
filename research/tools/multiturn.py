import json, sys, time, urllib.request
URL = sys.argv[1]
TOPICS = [
  ["Write a Python module `ringbuf.py` with a fixed-size ring buffer class (push, pop, peek, __len__, __iter__) and docstrings.",
   "Add type hints everywhere and a `resize` method that keeps the newest items. Show the whole file again.",
   "Write pytest tests for every method, including resize edge cases.", "Rename the class to `RingBuffer` and the module's functions accordingly; print the whole file.",
   "Explain in a short paragraph the complexity of each method.", "Add thread safety with a lock; print the whole file again.",
   "Now port the class to Rust with the same API.", "Summarize the differences between the Python and Rust versions in a table."],
  ["Explain how speculative decoding works in large language models, step by step.", "How does a copy (prompt lookup) drafter differ from a learned draft model?",
   "Write a short story (about 300 words) about a GPU that learns to dream.", "Continue the story for another 300 words, keeping the same characters.",
   "Rewrite the first paragraph of the story in a more formal style.", "List ten ideas for improving decode speed on memory-bound hardware.",
   "Pick the three best ideas and explain each in detail.", "Write a haiku about each of those three ideas."],
  ["Write a bash script that backs up a directory to a timestamped tar.zst, keeping the last 7 archives.",
   "Add command-line options for the source, the destination and the number of archives kept, with usage text. Print the whole script.",
   "Add a dry-run mode and logging to a file. Print the whole script again.", "Write a systemd service and timer that run it daily.",
   "Explain each line of the timer unit.", "Convert the script to Python with argparse; print the whole program.",
   "Add unit tests for the rotation logic.", "Review the Python program for bugs and print a corrected version."],
]
def chat(msgs):
    b = {'model': 'DeepSeek-V4.1-Flash-1017', 'messages': msgs, 'max_tokens': 1500, 'stream': True, 'stream_options': {'include_usage': True},
         'chat_template_kwargs': {'enable_thinking': False}}
    first = last = None; text = []; usage = None
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL + '/v1/chat/completions', data=json.dumps(b).encode(), headers={'Content-Type': 'application/json'}), timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data: ') or line == 'data: [DONE]': continue
            j = json.loads(line[6:])
            if j.get('usage'): usage = j['usage']
            for c in j.get('choices', []):
                d = c.get('delta', {}).get('content')
                if d:
                    t = time.time(); first = first or t; last = t; text.append(d)
    n = usage['completion_tokens']
    return ''.join(text), usage['prompt_tokens'], n, (first - t0), ((n - 1) / (last - first) if last and last > first else 0)
convs = [[] for _ in TOPICS]
for turn in range(8):
    for ci, topic in enumerate(TOPICS):
        convs[ci].append({'role': 'user', 'content': topic[turn]})
        try:
            txt, pt, n, ttft, tps = chat(convs[ci])
            convs[ci].append({'role': 'assistant', 'content': txt})
            print(json.dumps({'conv': ci, 'turn': turn, 'prompt': pt, 'tokens': n, 'ttft': round(ttft, 2), 'tps': round(tps, 1)}), flush=True)
        except Exception as e:
            print(json.dumps({'conv': ci, 'turn': turn, 'error': str(e)[:200]}), flush=True)
