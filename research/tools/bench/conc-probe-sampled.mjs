#!/usr/bin/env node
// Concurrency waves with SparkDash's own prompt picker and request function (unmodified imports), 400-token SAMPLED streams (T/TOP_P/TOP_K/SEED env),
// the aggregate computed as SparkDash's runConcurrencyWave does: total decode tokens / (last token - first token window).
// node conc-probe.mjs LABEL OUTPUT [REPS] [TYPES] [CONCURRENCIES]
import fs from 'node:fs';
import crypto from 'node:crypto';
import { runStreamingRequest, applyThinkingFlags, stripFillForceFields, closeLlmStreamAgent } from '/home/USER/sparkDash/server/collectors/LlmStreaming.js';
import { pickDecodeBenchPrompts } from '/home/USER/sparkDash/src/shared/llmPrompts.js';

const [label, output, repsArg = '2', typesArg = 'prose,code', concArg = '1,2,4,8'] = process.argv.slice(2);
if (!label || !output) throw Error('usage: conc-probe.mjs LABEL OUTPUT [REPS] [TYPES] [CONCURRENCIES]');
const reps = Number(repsArg), types = typesArg.split(','), concs = concArg.split(',').map(Number), maxTokens = 400;
const base = 'http://127.0.0.1:8000';
const median = a => { const s = [...a].sort((x, y) => x - y); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function idle() {
  for (let i = 0; i < 120; i++) {
    const h = await (await fetch(base + '/health', { signal: AbortSignal.timeout(10000) })).json();
    if (h.ok !== false && !h.warming && !h.busy && !h.requests_running) return h;
    await sleep(1000);
  }
  throw Error('not idle');
}
function body(prompt, tokens, forced) {
  const b = { model: 'DeepSeek-V4.1-Flash-1017', messages: [{ role: 'user', content: prompt }], max_tokens: tokens, temperature: Number(process.env.T ?? 1.0), top_p: Number(process.env.TOP_P ?? 0.95), top_k: Number(process.env.TOP_K ?? 20), seed: (seedBase += 1),
    stream: true, stream_options: { include_usage: true } };
  applyThinkingFlags(b, b.model, false);
  if (forced) Object.assign(b, { min_tokens: tokens, ignore_eos: true, stop: [] });
  return b;
}
async function wave(type, n, tokens) {
  const prompts = pickDecodeBenchPrompts(n, type);
  const h0 = await idle();
  const results = await Promise.all(prompts.map(async p => {
    let r = await runStreamingRequest(base + '/v1/chat/completions', body(p, tokens, true), AbortSignal.timeout(300000),
      { debug: false, collectContent: true, retryOnThinking400: true, thinking: false });
    if (r.error && /^HTTP 400\b/.test(r.error)) r = await runStreamingRequest(base + '/v1/chat/completions', stripFillForceFields(body(p, tokens, true)), AbortSignal.timeout(300000), { debug: false, collectContent: true, retryOnThinking400: true, thinking: false });
    return r;
  }));
  const h1 = await idle();
  const ok = results.filter(r => !r.error && r.decodeTokens > 0);
  const firsts = ok.map(r => r.tFirst), lasts = ok.map(r => r.tLast);
  const window = Math.max(...lasts) - Math.min(...firsts);
  const agg = ok.length && window > 0 ? ok.reduce((s, r) => s + r.decodeTokens, 0) / window * 1000 : 0;
  const delta = k => (typeof h1[k] === 'number' && typeof h0[k] === 'number') ? h1[k] - h0[k] : null;
  return { type, n, ok: ok.length, failed: results.length - ok.length, aggregateTps: Math.round(agg * 100) / 100,
    perRequestMedianTps: Math.round(median(ok.map(r => r.decodeTps)) * 100) / 100, ttftMedianMs: Math.round(median(ok.map(r => r.ttftMs))),
    rounds: delta('rounds_total'), drafted: delta('drafted_total'), accepted: delta('accepted_total'),
    outputs: ok.map(r => crypto.createHash('sha256').update(r.content || '').digest('hex').slice(0, 10)), errors: results.filter(r => r.error).map(r => r.error) };
}
let seedBase = Number(process.env.SEED ?? 1000);
const report = { label, started: new Date().toISOString(), reps, types, concs, maxTokens, waves: [] };
const save = () => fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
try {
  await idle();
  for (const t of types) { await wave(t, 1, 32); await wave(t, Math.max(...concs), 32); }   // warm-ups, not recorded
  for (const t of types) for (const n of concs) for (let i = 0; i < reps; i++) { const w = await wave(t, n, maxTokens); report.waves.push(w); save();
    console.log(JSON.stringify({ type: t, n, rep: i, agg: w.aggregateTps, per: w.perRequestMedianTps, ttft: w.ttftMedianMs, failed: w.failed })); }
  report.passed = report.waves.every(w => w.failed === 0); save();
} catch (e) { report.error = e.stack; report.passed = false; save(); throw e; } finally { await closeLlmStreamAgent(); }
