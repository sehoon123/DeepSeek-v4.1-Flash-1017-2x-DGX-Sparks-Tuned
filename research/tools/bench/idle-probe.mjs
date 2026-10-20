#!/usr/bin/env node
// Lone-stream decode after idle gaps: node idle-probe.mjs LABEL OUTPUT TYPE GAPS_SECONDS_CSV (no warm-ups; the base url, prompt picker and request function are conc-probe-dsv.mjs's)
import fs from 'node:fs';
import crypto from 'node:crypto';
import { runStreamingRequest, applyThinkingFlags, stripFillForceFields, closeLlmStreamAgent } from '/home/USER/sparkDash/server/collectors/LlmStreaming.js';
import { pickDecodeBenchPrompts } from '/home/USER/sparkDash/src/shared/llmPrompts.js';
const [label, output, type = 'prose', gapsArg = '0,0,0'] = process.argv.slice(2);
const base = 'http://127.0.0.1:8000';
const maxTokens = 400;
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
  const b = { model: 'DeepSeek-V4.1-Flash-1017', messages: [{ role: 'user', content: prompt }], max_tokens: tokens, temperature: 0, top_p: 1,
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
const rows = [];
try {
  await idle();
  for (const g of gapsArg.split(',').map(Number)) {
    if (g > 0) await sleep(g * 1000);
    const t0 = new Date().toISOString(); const w = await wave(type, 1, maxTokens);
    rows.push({ gap: g, start: t0, end: new Date().toISOString(), agg: w.aggregateTps, ttft: w.ttftMedianMs, failed: w.failed, out: w.outputs[0] });
    console.log(JSON.stringify(rows[rows.length - 1])); fs.writeFileSync(output, JSON.stringify({ label, type, rows }, null, 2) + '\n');
  }
} finally { await closeLlmStreamAgent(); }
