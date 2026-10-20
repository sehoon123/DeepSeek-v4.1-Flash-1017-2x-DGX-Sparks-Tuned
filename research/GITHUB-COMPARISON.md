> Run 2026-10-19 on the deployment before the work in this repository (recipe 1afc8910, confidence head misread); kept for
> the record. Current numbers are in the README.

# GitHub performance claim reproduction

Run 2026-10-19 against production DeepSeek-V4.1-Flash-1017 Ablit, native Zig TP2, P4, DSpark on, thinking off. The main recipe README numbers are for **stock** Mia EAQ at `CONTEXT=524288`; the optional Ablit section explicitly says stock performance/correctness does not validate Ablit mode.

## Matched sparkDash decode

SparkDash's unmodified real prose/code prompt picker and streaming client, temperature 0, 400 forced tokens, two repetitions per cell. Aggregate is SparkDash's total post-first tokens divided by the earliest-first to latest-last token window.

| Workload | README stock | Ablit 524K | Difference | Ablit 1M |
|---|---:|---:|---:|---:|
| Prose C1 | 74.9 | 49.28 | -34.2% | 50.36 |
| Prose C2 aggregate | 105.5 | 77.81 | -26.2% | 79.36 |
| Prose C4 aggregate | 130.0 | 102.52 | -21.1% | 104.50 |
| Code C1 | 109.9 | 63.16 | -42.5% | 64.20 |
| Code C2 aggregate | 139.8 | 87.72 | -37.3% | 89.30 |
| Code C4 aggregate | 155.5 | 111.90 | -28.0% | 113.93 |

TTFT does match the README closely: prose C1/C2/C4 132/243/402 ms against 131/243/411; code 200/297/522 against 200/291/511. Every output hash is identical between this deployment's 1M and 524K runs. The context arena is therefore not the decode gap.

The overlay publisher's separate Ablit receipt reports three 512-token essay decode cells 45.1/42.4/43.6 (mean 43.7 tok/s). This deployment's SparkDash prose C1 49–50 tok/s exceeds that separate Ablit receipt, while the recipe's random-word `tools/bench.py` C1 42.6 is close to it. Do not compare the overlay's 192-token e2e refusal-gate number directly to SparkDash decode.

## Matched sparkDash cold prefill at 524K

Used the local SparkDash `PrefillBench.js` prompt builder, warmup, three 16-token overhead calibrations, unique salt per sample and its repeat counts (3 through 32K, 2 through 128K, 1 at 256K).

| Label | README tok/s | Measured tok/s | Difference | Measured TTFT |
|---:|---:|---:|---:|---:|
| 8K | 2,166 | 2,165.66 | -0.02% | 3.971 s |
| 16K | 2,183 | 2,153.02 | -1.37% | 7.797 s |
| 32K | 2,175 | 2,157.65 | -0.80% | 15.375 s |
| 64K | 2,136 | 2,122.47 | -0.63% | 31.065 s |
| 128K | 2,079 | 2,068.26 | -0.52% | 63.562 s |
| 256K | 2,000 | 1,999.38 | -0.03% | 131.300 s |

Prefill is reproduced within 0.02–1.37% at every size.

## Clock diagnosis

Under the matched decode load:

- Head: 268 samples at >=50% utilization; clock median/min/max 2236/2229/2236 MHz, utilization median 96%, power median 48.55 W, max 76 C, every throttle mask 0.
- Worker: 270 samples; 2229/2223/2236 MHz, utilization median 96%, power median 47.79 W, max 77 C, every throttle mask 0.

The historical ~700 MHz head fault is absent. During the longer prefill suite, clocks remained normal at head/worker medians 2229/2223 MHz. Head briefly reached 85 C and reported NVML mask `0x20` (software thermal slowdown) for 13 samples, with a 2112 MHz minimum; worker stayed throttle-free. Despite that brief thermal event, prefill matched the README, so it does not explain the decode gap.

## Conclusion

The GitHub claim is **partially reproduced**:

- Cold prefill and TTFT: yes, essentially exact.
- Decode throughput: no, 21–43% below the stock README depending on workload/concurrency.
- Hardware clocks: normal; not the cause.
- Context 1M vs 524K: not the cause; outputs are identical and 1M was ~2% faster in this interleaved time window.

The decisive configuration difference is the requested Ablit overlay versus the README's stock checkpoint. It changes 104 attention-output tensors, and speculative decoding speed depends on target/drafter agreement. This engine's health response does not expose draft/accept counters, so the contribution cannot be numerically separated without restoring the deleted stock shards. Per user instruction, stock mode was pruned and is not redownloaded for an A/B.

Raw files: `decode-1m.json`, `decode-524k.json`, `prefill-524k.json`, and their `*-gpu-{head,worker}.jsonl` telemetry.
