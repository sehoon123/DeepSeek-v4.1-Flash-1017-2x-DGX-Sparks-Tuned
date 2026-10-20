> Measured before the DSpark confidence-head fix (patch 0018). With the head read correctly the deployment went back to the
> stock depth rule; the adaptive-K patches described here stay in the series, off unless their variables are set. Patch
> numbers here are the pre-hotfix ones: 0005-0011 below are 0010-0016 in ../zig/patches/. Paths are generalized.

# DeepSeek-V4.1-Flash-1017 Ablit overnight decode optimization

Window: 2026-10-20 00:13–10:00 KST. Hardware: two GB10 Sparks, TP2 over two RoCE rails. Target checkpoint and all target output bits stayed fixed; only speculative proposal policy and engine scheduling code changed. Every adopted candidate had identical output hashes on SparkDash prose/code C1/C2/C4.

## Starting point

Matched production baseline (medians from the final A/B controls):

| | C1 | C2 aggregate | C4 aggregate |
|---|---:|---:|---:|
| Prose | 49.11 | 77.48 | 102.06 |
| Code | 62.74 | 87.29 | 111.43 |

The publisher's stock checkpoint README is faster on decode; cold prefill was already reproduced within 0.02–1.37%. Loaded clocks were normal, so the decode gap was not the historical 700 MHz fault.

## Main finding

The stock DSpark confidence policy was too conservative against the target-only Ablit overlay. Forcing useful drafts recovered performance without changing target output.

Phase-1 fixed K sweep, one matched run:

| Policy | Geometric change | Important behavior |
|---|---:|---|
| drafts off | -12.5% | rejected |
| K0 | -22.7% | rejected |
| K1 | +9.1% | insufficient |
| K2 | +25.1% | good |
| K3 | **+32.2%** | robust; every cell +18% or more |
| K4 | +29.6% | prose C4 regression |
| K5 | +31.1% | fastest code C1/C2, prose regression |

Acceptance instrumentation then showed K5 accepted about 1.7–2.1 drafts/round on prose and 3.2–3.7 on code. Several request-local routers were implemented and tested (periodic probes, exponential backoff, K3-running-mean routing, bounded K5 samples). The most stable global policy was periodic adaptive A2/P8:

```text
TF_DS_ADAPT_K=1
TF_DS_ADAPT_PERIOD=8
TF_DS_ADAPT_ACCEPT=2
TF_DS_ADAPT_MAX_LIVE=2
```

It tests K5 periodically only with at most two live streams, promotes on observed acceptance, and uses K3 at higher concurrency.

## Final matched result

Final hotfix image, A2/P8, two independent pre-hotfix verification windows plus post-hotfix soak:

| | Baseline | Final hotfix median | Change | GitHub stock README |
|---|---:|---:|---:|---:|
| Prose C1 | 49.11 | ~64.0 | **+30%** | 74.9 |
| Prose C2 | 77.48 | ~91.7 | **+18%** | 105.5 |
| Prose C4 | 102.06 | ~126.5 | **+24%** | 130.0 |
| Code C1 | 62.74 | ~95.6 | **+52%** | 109.9 |
| Code C2 | 87.29 | ~142.7 | **+63%** | 139.8 |
| Code C4 | 111.43 | ~154.4 | **+39%** | 155.5 |

Geometric improvement is approximately **+35%**. The requested overall +20% target is exceeded. Code C2 exceeds the stock README; code C4 and prose C4 are within about 1% and 3%; C1 and prose C2 remain below stock. The prose C2 cell is the limiting improvement (~18%): K3 is already its measured optimum, so a further target requires a better/retrained drafter rather than a global depth policy.

## Source changes and image

Upstream hotfix was rebased and applied during the run:

- Recipe `e65785cd3cde8e23c38d60a732a014abb6ccb533`.
- Engine pin `629c524a0ea57512a5342abb498d1b7c25353343`.
- Upstream prefill interleave (250 ms), image-keyed cache, shared-prefix fixes, and held/hard memguard behavior.

Local decode research patches were rebased as `zig/patches/0010`–`0016` after upstream 0005/0006. They add confidence floor/bias, acceptance instrumentation and optional routing policies. All experimental policies are default-off; production enables only A2/P8 through environment variables.

Final image:

```text
tensorfold-dsv41:exl3-tp2-da8e1eb22cc6
tf.patches=da8e1eb22cc6
```

Both nodes use identical image content. The target checkpoint, Ablit overlay, Engram, rank caches and kernel kit were unchanged.

## Rejected hypotheses

- CPU/GPU clock fault: absent; decode clocks 2236/2229 MHz, throttle masks zero.
- Context 1M versus 524K: outputs identical; 1M was ~2% faster in the short interleaved comparison.
- Drafts off / K0: materially slower.
- Round rows 24–48: adaptive offered no gain; fixed-K combinations changed output hashes or failed concurrency, rejected.
- Round-cost slope changes: flat table gave +24%, below fixed/adaptive policy.
- Side, prefetch, Engram AIO, graphs, RDMA off: all slower (RDMA off about -13%).
- Confidence bias alone: at most ~3%.
- K3-minimum plus bias: ~33%, no improvement over K3.
- Running-mean and bounded-probe classifiers: did not reliably identify code C1.
- Probe backoff: early noisy failures prevented useful K5 promotion.
- Newer unrelated TensorFold forks: reviewed; their published numbers use different packs/engine architecture and do not supersede the measured A2/P8 result here.

## Hotfix behavior

Hotfix decode did not regress. A 100K prefill submitted after a decoding stream no longer blocked it for the entire prompt, but QoS is not complete isolation: the 400-token stream took 29.3 s at 13.7 tok/s while the long prefill ran (normally ~6 s / ~64 tok/s). The long prefill took ~54 s. Default `PREFILL_INTERLEAVE_MS=250` is retained because it is the upstream measured value.

## Upstream hotfix and final soak

At 08:30 the recipe hotfix was rebased rather than deferred: recipe `e65785cd`, engine pin `629c524`, upstream prompt-prefill interleave and image-cache patches. Local decode patches were renumbered to 0010–0016 and replayed cleanly on the new pin. Final image: `tensorfold-dsv41:exl3-tp2-da8e1eb22cc6` (`tf.patches=da8e1eb22cc6`).

Seven post-hotfix soak cycles (two repetitions per C1/C2/C4 cell) all passed. Medians and full ranges:

| | Median | Min–max |
|---|---:|---:|
| Prose C1 | 64.19 | 63.60–64.31 |
| Prose C2 | 91.69 | 91.09–92.12 |
| Prose C4 | 126.26 | 125.53–126.94 |
| Code C1 | 95.65 | 94.41–96.09 |
| Code C2 | 142.69 | 141.38–143.22 |
| Code C4 | 154.41 | 152.13–154.76 |

Maximum spread was 1.78%; NRestarts and memguard actions stayed zero.

## Safety and validation

- Every adopted benchmark output hash equals baseline.
- Final smoke and full validate pass: text, thinking, tools, image, 21.9K long prompt, 8/8 concurrent parity.
- Head minimum MemAvailable during candidate decode remained about 6.4 GiB; memguard action count 0.
- Final loaded clocks remained normal; no OOM, panic, rank failure, or service restart.
- The configured model window remains 1M, but memory load qualification remains 256K as documented in the deployment report.

Raw data are split by phase under `phase1-runs` through `phase9-runs`, with final cross-checks under the live `runs/` archive at the time of each phase. Queue files and exact local configurations are retained beside them.
