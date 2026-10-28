# Prompt-cache lifecycle hardening — 2026-10-28

This records the earlier 0046/0047 deployment. The subsequently qualified and deployed
[0048–0050 follow-up](CACHE-PLACEMENT-SSE.md) adds fault-reference retirement, placement preservation
and native OpenAI SSE keep-alives; its evidence and limitations are recorded separately.

## Released and deployed: 0046 + 0047

These are host-side correctness fixes, not new CUDA kernels or a change to model
arithmetic. They were qualified against the actual 0045 runtime on two GB10s and
installed on the serving site at 14:40–14:41 KST on 2026-10-28.

### 0046: validity, refresh and nested ownership

- An optional pure family predicate lets exact-prefix lookup skip invalid saved
  rows and choose the longest remaining valid matching prefix. Other families
  without the predicate retain their original behavior.
- An invalid duplicate key no longer suppresses a fresh save. Replacement uses
  the original budget path and preserves shared-cut status. Valid duplicates
  still need no copy.
- A previously short nested-prefill estimate is revalidated before admission.
- Before nested cache mutation, an outer pass retires its already-consumed
  snapshot reference. This prevents subsequent reporting through an entry that
  nested eviction/replacement may have freed.

The backend still performs its final usability and placement checks. Invalidation
*after* lookup still fails closed. Changed prompt tokens never qualify for reuse
of nonmatching KV. The victim-selection rule and budgets are unchanged; consumed
restore accounting now precedes nested mutation, so its timing/LRU ordering is
not claimed to be identical.

### 0047: rollback a failed snapshot save

Previously, `Lanes.snapSaveFn` registered a newly allocated snapshot, then made
fallible frame/send/copy calls. Its failure cleanup freed the buffer/object but
left the pointer in the family list. Later placement or validity traversal could
therefore access a freed snapshot.

Cleanup ownership now transfers to the registered-list drop operation after
allocation/append succeeds. A later failure unregisters before freeing and uses
the normal peer-drop path. This also attempts to discard a save the peer may have
already mirrored. It does not add a reliable peer-save acknowledgement protocol.

## Qualification evidence

### CPU and source

- 262/262 host tests; 68/68 server tests and 49/49 engine-seam tests. Counts overlap.
- Five negative controls fail at the intended assertions without the respective
  fixes: invalid-longer lookup, invalid-duplicate refresh, stale short nesting
  estimate, outer-reference retirement and failed-save rollback.
- The save-failure regression invokes the real family/Model/DeviceBuffer path,
  replacing only CUDA driver function pointers. Thirty-two injected failures,
  followed by successful save/drop, leave no registered snapshot or counted
  device-buffer allocation behind. No CUDA library/context/device is opened.
- CPU fixtures cover 1,000 invalidation/refresh cycles, budget pressure, delayed
  peer drops, and exact lookup on a 516,919-token history. These are not full-model
  GPU tests at that length.
- Applying the complete recipe patch chain to the pinned source reproduces all
  425 files under `zig/src` exactly; all Zig build/source files were also compared
  without differences.
- Native compilation: 28/28 steps, Zig 0.17.0, CUDA compiler 13.3.73, sm_121.

### Actual model / native HTTP

A fresh 0045 resident and then the candidate ran the same 20-request synthetic
suite: **40 complete responses / 20 complete token-array comparisons**. The gate
checks request hashes, every output token, independently recomputed full SHA256,
usage, finish reason and cache/timing fields—not just a short token-hash prefix.

Cases include:

- About 131K tokens: cold, warm, appended and changed-early prompts. Appending
  preserves prefix reuse; changing the early input correctly misses.
- Three unrelated prompts of 360,229 / 360,216 / 360,174 tokens: together they
  exceed the unchanged 1,048,576-position pool. Replay and refreshed warm reuse
  follow the resulting placement invalidation.
- Four approximately 16K streams: solo outputs equal concurrent cached outputs,
  including fixed-seed sampling.
- A short nested request completes before the outer approximately 131K request's
  first output token. Both complete outputs equal the reference.
- During the candidate run, an additional owned synthetic stream is disconnected;
  its request ID is correlated to the server's `client-left` record. A subsequent
  complete recovery reply matches the baseline. The cancelled stream is not
  counted as a complete-response comparison.

Both main runs logged five stale-state transitions. The cache's final `failed`
count was **1 on 0045, 0 on the candidate**. Output arrays matched. This demonstrates
that the exercised stale-state path no longer attempts the known-invalid restore;
it is not proof that every historical long wait had this cause, or that all
recomputation is avoidable.

The ordinary smoke suite passed. Functional checks passed for text, thinking,
tool calls/results, an image, a 21,875-token prompt, and eight concurrent-vs-solo
comparisons. **Video was skipped because PyAV was unavailable.**

## Preserved contract and operational safety

Model, EXL3 2.9-bpw quantization, matching Keys overlay, TP2, weights, arithmetic,
sampling, four-stream capacity, 1M context pool, output limits, 1-GiB snapshot
budget and permanent memory guards were preserved. The qualified site's existing
0045 opt-ins were unchanged, including EP32, paired prefill/reuse and phase arena.

The window required ten minutes of quiet evidence, a reservation lock and an
exclusive maintenance marker. The public proxy was stopped and both restart
policies temporarily set to `no`; only one two-node model resident ran at a time.
Both-node model absence was checked before each replacement. Sampled minimum
MemAvailable was **4,971 MiB / 6,581 MiB**, above the 3,072-MiB floor. Original
`on-failure` policies and public authentication enforcement were restored, with
both services healthy and zero automatic restarts at the final check. Planned
model replacements did occur; zero automatic restarts does not mean uninterrupted
service during maintenance.

Qualified local runtime:

- Image tag: `tensorfold-dsv41:exl3-tp2-0078cefc720d`
- Identical image ID on both nodes:
  `sha256:238e2aa0355f38e804bd19702c40ad6f9a2889d46eb79311489c1f9ec9e84c31`
- Native SHA256:
  `6b8474c5f15ddb0994ebbbf8372c6d35f1ae2d1269ccbbcffe6cc5139d143eea`

This local runtime was COPY-packaged on the qualified 0045 image; existing kernels
and AOT assets were retained. **A complete recipe Dockerfile build was not rerun.**
The source recipe remains the reproducible build definition. No model weights or
private request/client data are included in this publication. No kernel-throughput
or general latency percentage is claimed; the cancellation check also introduced
intentional contention during candidate qualification.

## Limits and further findings

- Longest-valid-shorter selection and snapshot-buffer budget saturation have
  focused CPU tests. The actual-model suite exercised warm reuse and token-pool
  placement pressure, not every possible fallback shape or a saturated 1-GiB
  snapshot-buffer budget.
- Recoverable CUDA/transport failures were not injected into the actual two-node
  model. A follower can still log `follow_refused` after its own save-copy failure
  without a positive acknowledgement to the head. That protocol path needs
  separate two-node fault/recovery qualification.
- Further review found a separate NaN error-path reference issue: prefix
  invalidation can free a job's selected cache entry before final reporting.
  A focused CPU negative control reproduces the outstanding-reference defect;
  a separate follow-up passes 50 engine-seam tests. It was not part of 0046/0047.
  **It has since been incorporated as 0048 in the qualified/deployed
  [0048–0050 follow-up](CACHE-PLACEMENT-SSE.md).** Deliberate actual-GPU NaN injection
  remains untested; CPU fault regression and normal-path native qualification are distinct evidence.
- The previously isolated private-use placeholder/tokenization correction is
  also separate. It intentionally changes tokens for the triggering inputs and
  requires its own native/actual-model qualification.

The release fixes demonstrated code defects. It does not assert that the original
client-specific latency incident is conclusively explained or that no critical
issues remain.
