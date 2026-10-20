#!/usr/bin/env node
import fs from 'node:fs';
import { randomUUID } from 'node:crypto';
import { runStreamingRequest, applyThinkingFlags, closeLlmStreamAgent } from '/home/USER/sparkDash/server/collectors/LlmStreaming.js';
import { buildPrefillPrompt, repeatsForSize, medianOf, rateFromSample, timeoutMsForSize } from '/home/USER/sparkDash/server/collectors/PrefillBench.js';
const [output] = process.argv.slice(2); if (!output) throw Error('output required');
const base='http://127.0.0.1:8000', model='DeepSeek-V4.1-Flash-1017';
const sizes=[8192,16384,32768,65536,131072,262144];
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
async function idle(){for(let i=0;i<1800;i++){const h=await(await fetch(base+'/health',{signal:AbortSignal.timeout(10000)})).json();if(h.status==='ok'&&!(h.live?.connections))return h;await sleep(1000);}throw Error('not idle');}
function body(prompt){const b={model,messages:[{role:'user',content:prompt}],max_tokens:8,temperature:0,top_p:1,stream:true,stream_options:{include_usage:true}};applyThinkingFlags(b,model,false);return b;}
async function one(target,label){const prompt=buildPrefillPrompt(target,`${label}-${randomUUID()}`);const r=await runStreamingRequest(base+'/v1/chat/completions',body(prompt),AbortSignal.timeout(timeoutMsForSize(target)),{collectContent:false,retryOnThinking400:true,thinking:false});return {target,prompt_tokens:r.prefillTokens,cached_tokens:r.cachedPromptTokens,ttft_ms:r.ttftMs,server_prompt_ms:r.serverPromptMs,server_prompt_n:r.serverPromptN,completion_tokens:r.completionTokens,error:r.error||null};}
const report={label:'dsv1017-ablit-context524288-sparkdash-prefill',started:new Date().toISOString(),sizes,samples:[],summary:{}};
try{
 await idle(); await one(512,'warmup');
 const cal=[];for(let i=0;i<3;i++){const r=await one(16,'cal');if(!r.error&&r.ttft_ms>0)cal.push(r.ttft_ms);}const overhead=medianOf(cal);report.calibration={samples_ms:cal,median_ms:overhead};
 for(const size of sizes){const reps=repeatsForSize(size);const a=[];for(let i=0;i<reps;i++){const r=await one(size,`s${size}-r${i}`);r.rate=rateFromSample({promptTokens:r.prompt_tokens,ttftMs:r.ttft_ms,cachedTokens:r.cached_tokens,serverPromptMs:r.server_prompt_ms,serverPromptN:r.server_prompt_n},overhead);a.push(r);report.samples.push(r);fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({size,rep:i,tokens:r.prompt_tokens,ttft:r.ttft_ms,tps:r.rate.tps,error:r.error}));}
 report.summary[size]={reps,median_prompt_tokens:medianOf(a.map(x=>x.prompt_tokens)),median_ttft_ms:medianOf(a.map(x=>x.ttft_ms)),median_tps:medianOf(a.map(x=>x.rate.tps)),methods:[...new Set(a.map(x=>x.rate.method))],errors:a.filter(x=>x.error).map(x=>x.error)};}
 report.passed=report.samples.every(x=>!x.error);report.finished=new Date().toISOString();fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
}catch(e){report.passed=false;report.error=e.stack;fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');throw e;}finally{await closeLlmStreamAgent();}
