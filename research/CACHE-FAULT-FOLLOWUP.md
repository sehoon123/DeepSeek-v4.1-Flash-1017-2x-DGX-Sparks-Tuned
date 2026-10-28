# Draft: fault invalidation must retire the job's cache reference

**CPU-tested only. Not native/actual-model qualified, not part of the 0046/0047
release, and not deployed.** The serving image remains the one documented in
`CACHE-LIFECYCLE.md`.

## Finding

`LaneHost.forgetFaulted` removes every saved prefix that might hold a numerical
fault. One can be the request's selected `job.entry`. Its subsequent
`finish -> Job.reported` path compares the restored length with `e.at` before
`Store.resumed` checks whether that entry is still registered. The comparison can
therefore dereference an entry that the invalidation just freed.

A relevant case is a resumed prompt failing before any keep or interleave callback
has retired the consumed reference. The existing NaN-isolation regression did
not attach a prompt cache, and so did not exercise this lifetime interaction.
This is not evidence of a particular historical crash or of a GPU NaN's cause.

## Candidate 0048

Clear the job's selected Entry and saved-state pointer before `forgetPrefixes`
can destroy entries. Preserve cached-token statistics and the existing prefix
invalidation itself. No kernel, arithmetic, weight, sampler, budget or capacity
change is proposed.

A focused CPU fixture uses the real host/cache operations: it creates a saved
prefix, associates it with a successfully resumed job, invalidates the faulted
prefixes, and then invokes normal final reporting. It checks retired references,
no cache hit reported for the failed job, one failed state, and an empty store.

- Without the fix, the intended `job.entry == null` assertion fails after
  invalidation. The negative control stops there rather than deliberately
  dereferencing freed memory.
- With the fix, all **50 engine-seam tests pass** under the testing allocator.
- No model, CUDA context, network or GPU was used; the isolated test container had
  a 1-GiB hard memory cap and both-node memory-floor monitoring.
- Patch application against the released source chain was checked.

Native compilation and real-model cache/error/recovery qualification remain
required. Do not merge/deploy this draft solely because its CPU regression passes.
The separate follower acknowledgement and tokenizer-placeholder findings also
remain outside this candidate.
