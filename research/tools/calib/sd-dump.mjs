import { pickDecodeBenchPrompts } from '/home/USER/sparkDash/src/shared/llmPrompts.js';
const out = [];
for (const t of ['prose', 'code']) { const seen = new Set(); for (const n of [1, 2, 4]) for (const p of pickDecodeBenchPrompts(n, t)) seen.add(p); [...seen].forEach((p, i) => out.push({ name: `sd_${t}${i}`, text: p, think: false })); }
console.log(JSON.stringify(out));
