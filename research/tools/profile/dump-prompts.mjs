import * as P from '/home/USER/sparkDash/src/shared/llmPrompts.js';
const out = {};
for (const t of ['prose', 'code']) { const seen = new Set(); for (let i = 0; i < 200; i++) for (const p of P.pickDecodeBenchPrompts(4, t)) seen.add(p); out[t] = [...seen]; }
console.log(JSON.stringify({ exports: Object.keys(P), counts: { prose: out.prose.length, code: out.code.length }, prompts: out }));
