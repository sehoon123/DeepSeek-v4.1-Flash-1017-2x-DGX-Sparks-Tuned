# Cache placement and long-stream stability — 2026-10-28

Follow-up: [PROGRESS-STATES.md](PROGRESS-STATES.md) (0051–0053, 2026-10-29) adds rolling prompt progress states, keep-alives on the
Anthropic/Responses stream routes and overflow wording, qualified and deployed.

This follow-up carries patches **0048–0050**, on top of the previously qualified 0046/0047 lifecycle fixes.
It addresses a reproduced server-side cache-loss path and an independently tested HTTP body-idle risk.
It does **not** establish every cause of the original client incident or claim that all stalls are eliminated.

## What changed

- **0048 — faulted-job reference:** retire the job's borrowed Entry/saved pointer before prefix invalidation can free it. Later reporting must not dereference the freed entry. CPU fault-path regression, not a claim of actual-GPU NaN recovery testing.
- **0049 — placement:** first ask whether an extent fits around live streams alone. If it cannot, return ContextFull without destroying saved states. After other victims have been removed, recheck the source's original extent before discarding the source; classify copying from the resulting base, not the initial placement attempt. Invalid/stale sources remain unusable.
- **0050 — native OpenAI SSE:** the request thread sends a standard `: keep-alive` comment every 15 seconds while waiting for engine events. No extra writer thread, model text, generated token, usage item or TTFT event. A closed writer uses existing cancellation/drain behavior. Nonstream JSON and translated outputs without the optional callback are unchanged.

No weights, EXL3 quantization, overlay, arithmetic, sampling, cache budget/retention, context/output limits or four-stream capacity changed. The oldest-other/source-last victim order and backend final validity checks remain in force. Comments do not override absolute client deadlines and do not accelerate cold computation.

## Actual-model blocked-placement reproduction

Two GB10s, TP2, EAQ EXL3 2.9 bpw plus the same Keys overlay; context pool 1,048,576, four streams, max output 32,768 and prompt snapshots 1 GiB. The same site settings remained active, including round rows 48, prefill interleave 250 ms, nest allowance 8,192, paired prefill/scratch reuse, phase arena, graphs, mHC RB2 and EP32 AOT.

The fixtures are **synthetic token arrays, not captured user conversations**:

1. Seed a 648,878-token session, then an 8,940-token second session.
2. Grow the first to 681,402 tokens. Its kept prefix reaches 679,936 at base 0.
3. Grow the second to 45,241 tokens and let it really decode 2,500 tokens.
4. While it is still decoding, submit the first session grown to 683,602 tokens, still reserving all 32,768 output tokens.

The small session occupies an extent starting at 714,752; the grown large session needs 716,800 contiguous positions. Before the small session finishes, no eviction can make the large placement feasible. The old path nevertheless invalidated the 647,168/679,936-token states at base 0, with only **30 MiB of the 1,024 MiB snapshot budget used**. The restore-failed counter did not reveal this loss.

| 683,602-token request | 0047 baseline | 0048–0050 candidate |
| --- | ---: | ---: |
| Restored prefix | 0 | 679,936 |
| Actual prefill | 296.689 s | 2.427 s |
| Client first generated token | 315.968 s | 21.740 s |

The candidate still waits roughly 19 seconds for the other live extent; it does not shrink a request's reservation. Its log then records `placement resumes in place: source 679936 at 0, extent 716800`. Both sessions' complete output token arrays match baseline. Most long geometry replies are only ACK: this is a cache/placement regression, **not a long-recall quality benchmark or general kernel-speed claim**.

## Long streaming transport

- CPU-only real-HTTP fixture: a 31-second engine-output gap with a 17-second body-idle deadline. Old direct plain, direct tool-bearing and proxied tool-bearing streams all time out. Candidate streams pass with comments near 15/30 seconds, identical output and true TTFT near 31 seconds. An owned early close drains and recovery matches.
- The exact SDK/private-proxy runner also passes in a no-GPU private network namespace under a minimal service PATH.
- Actual model: **850,123 fresh tokens**, through the locally installed Pi 1.4.0 OpenAI SDK and a private loopback instance of the unchanged production proxy. Dummy credentials only; no user's traffic/configuration was read.
- Node v22.23.3 / embedded Undici 6.28.1 uses a default 300,000 ms body-idle timeout. It was not overridden. A separate 900-second absolute test bound prevents heartbeats from masking a hang.
- Candidate prefill **381.915 s**, **25 comments**, maximum received-body gap **15.062 s**, **one fetch / zero retries**, explicit `[DONE]`. Full output, usage and finish match the complete baseline response. Keepalives are not counted as first generated tokens.

This verifies the mitigation on the tested stack, not the identity of the original user's disconnecting hop or configuration on their separate PC.

## Qualification and operational evidence

- Host tests **267/267**, server tests **69/69**, seam tests **50/50** (overlap; do not add as independent totals).
- Four removed-fix controls fail at the intended assertions: live-feasibility, in-place source preservation, actual move classification, and faulted-job reference retirement.
- Native build **28/28** steps, Zig 0.17.0 and CUDA 13.3.73, sm121, requested ReleaseFast (build summary labels the native target `safe`).
- Full public patch-chain reconstruction matches **1,279 tracked engine-tree files**.
- **34 full-token-array comparisons / 68 complete responses**: 9 long/placement cases, 20 ordinary/cache/pressure/four-stream/nested cases, 5 complete-token thinking/tool/image features. Request hashes/identities, full array hashes, non-cache usage, finish and cache/timing checks are independent gates. Ordinary cache counts are exact; only intended placement/cancel-retry differences are allowed.
- Legitimately changed prefixes miss. Owned zero-token prefill cancellation drains, then the identical retry matches baseline. Separate owned decode cancellation and complete recovery match baseline.
- Smoke and functional validation pass, including eight-way versus serial replies. Video: skipped: PyAV unavailable.
- Candidate cache-failed counter maximum: **0**. No matching CUDA/NaN/OOM/unknown-snapshot errors in the captured qualification logs. This is finite test evidence, not proof of absence of all faults.

The first window completed the full baseline and six matching candidate responses, then stopped because the test service PATH did not resolve `node`. No SDK model request was sent at that failure. The guard restored 0047, smoke/config/source checks and original restart/public access policies. After CPU-testing the exact SDK runner and pinning its executable path, a new 10-minute idle check/reservation was taken. The unchanged complete baseline was verified and reused, **not rerun or counted twice**; the unchanged candidate was rerun from empty state and all gates passed.

Public proxy closed, temporary Restart=no, reservation/maintenance markers and both-node production-absence proof protected each replacement. Only owned synthetic requests were disconnected. The 3,072 MiB per-node floor, contact/time guards and permanent serving guards stayed in place. Sampled MemAvailable minima:

| Window | Head MiB | Worker MiB |
| --- | ---: | ---: |
| First window (baseline / partial candidate / verified rollback) | 5058 | 6889 |
| Second window (complete candidate) | 5599 | 7399 |

## Deployment identity

- Local image: `tensorfold-dsv41:exl3-tp2-67309ead97da`
- Both-node image ID: `sha256:27b75f33f380af5ba6fffeaed24e26c938f75d140505210e1419fb93ad00094e`
- Native SHA256: `97be35393f4f54fcd70f6719aed181a3445f1bfc4dba8dfc12e70e346b66bb01`
- Complete environment, command, mounts and host limits equal baseline. Source/prepared marker verified; services active with original on-failure policies, no automatic restarts, batch 4, not warming, public unauthenticated request returns 401.
- Packaging was COPY-only on the qualified 0047 image. Full source construction remains the ordered patch chain in `zig/Dockerfile`. A fresh end-to-end build of the entire recipe Dockerfile was not rerun; source reapplication and native compilation were verified separately. The previous 0047 image is retained for guarded rollback.

This is a **GitHub source/evidence release**, not a registry image or model-weight upload. Synthetic request fixtures and complete response arrays accompany the release; no live client bodies, conversations, credentials or Burp traffic are included.

## Limits and remaining work

- Cold or genuinely changed very long prompts still take minutes; the shared 1M pool can legitimately queue streams. Heartbeats must not be mistaken for model progress.
- A restart cannot resurrect previously invalidated KV. First real requests after maintenance can be cold; history and limits were not truncated to hide that cost.
- The private-use-placeholder tokenization candidate is separate and not included.
- Follower save-copy acknowledgement/protocol fault injection is not implemented here.
- Actual GPU NaN injection was not performed; the fault-reference regression uses the CPU engine seam.
- This release does not claim every Pi/MCP stall is fixed or that no critical issues remain.
