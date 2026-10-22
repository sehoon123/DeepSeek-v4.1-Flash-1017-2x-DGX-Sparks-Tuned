# Changelog

## sehoon123/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned

This recipe's changes to Mia's AI Lab's at `31eea46`. Everything below this section is Mia's AI Lab's changelog; its
issue and pull-request numbers are their repository's.

- **A prompt ending just past a kept state no longer drops it** (`zig/patches/0033`: TensorFold's `cf6aae2b` by
  evilpsycho42, PR #35, unchanged): such a prompt (one sharing a longer conversation's prefix and ending under 128 tokens
  past its chunk end) resumes an earlier state and the conversation keeps its own; before, the state was handed to the
  lanes, refused, dropped as a failed restore, and the prompt ran from 0.
- **The prefill indexer split on every eligible chunk** (`TF_DS_IDX_SPLIT=1`, `scripts/config.sh`; `zig/patches/0001`,
  MiaAI-Lab's): 161K-token prefill -2.74 % against splitting only from token 131,072 (MiaAI-Lab's later default, aaa4xu's
  2550cda4) and -4.2 % against none, 43K -0.95 %, 16K +0.00 %; the same bits.
- **The RDMA proxy naps when idle** (`zig/patches/0032`: MiaAI-Lab's `0010-rdma-idle-nap.patch`, unchanged, with their
  `RDMA_IDLE_S` 2 and `RDMA_NAP_US` 500; William J Riley's TensorFold PR #27, ported by Mykhailo Tamarin): a quiet server no longer spins a core on
  each Spark; decode's gathers never nap.
- **Requests needing little fill inside a long prompt pass** (`zig/patches/0031`, `PREFILL_NEST_TOKENS`, 8192): a
  request that arrives during another's prompt pass and would prefill at most that many tokens runs its whole pass
  between two of the long one's chunks (the long pass then sets its slot up again on both ranks and reads its next
  chunk's Engram rows anew); one level; the outer request's cancel waits for its own pass. A 1.9K-token request 3 s into
  a 68.6K-token prompt: first token 27.6 -> 2.0-2.5 s, a second one sent 0.5 s later 2.98 s; every reply equal to the
  same prompt's alone.
- **The server's tokenizer reuses the previous request's ids** (`zig/patches/0030`): the leading segments (split at the
  end-of-sentence tokens) a request shares with one of the last four long texts keep their ids; a 135K-token
  conversation's next request 57 -> 2-3 ms to tokenize; the same ids.
- **`scripts/prepare.sh` refuses to build a new image beside a running server** (`PREPARE_BESIDE_SERVER=1` overrides):
  the build's compiles took the head under the memory guard's floor, which stopped the server.
- **Sampled decoding's host side** (`zig/patches/0029`): a round's logits are copied to pinned memory (0.37 -> 0.07 ms a
  round) and each row's top candidates are found 16 logits at a time (0.37 -> 0.12 ms): T=1 decode +1.2 % (110.00 ->
  111.41 / 111.31 tok/s, two pairs, one stream), the same bits; greedy decoding (the device's argmax) is unchanged.
- **Warm-ahead proxy** (`tools/warm_ahead_proxy.py`, optional): an OpenAI / Anthropic API proxy that, while a client is
  idle after a reply, has the prompt cache keep the state at the next chunk end the conversation and its reply reached
  (those tokens up to it and 128 past it as a token prompt, `max_tokens 1`; at once after tool calls), so the next turn
  resumes ~2,048 tokens later. Only chunk ends every next request can use; a client is warmed ahead once it has shown how it sends replies
  back (learned per system prompt and tools); a client request closes a warm-ahead in flight. 39K-token conversations:
  a turn's first token 1.40-1.57 -> 0.54-0.59 s when the reply crossed a chunk end; the same reply bits as from an
  empty cache. After a server restart it re-runs the recent conversations in 16K-token pieces while clients are quiet
  (`REWARM`): a 38K-token conversation's next turn 16.2 -> 1.17 s after a restart. `PROMPT_CACHE_GIB`
  (`scripts/config.sh`, unset: the server's 0.5) sets the prompt cache's budget.
- **Engram reads: each row once, the recent ones from memory** (`zig/patches/0028`; `TF_DS_ENGRAM_DEDUP=0`,
  `TF_DS_ENGRAM_CACHE_MB=0` turn them off): a prompt chunk's Engram rows are read from the NVMe once each, however many
  of its n-grams name them (30-46 % of a chunk's 24,576 ids repeat one), and the rows read last, by prompt chunks and
  decode rounds, stay in host memory (256 MiB a Spark by default) for later chunks and requests to copy instead of
  reading. A request's first chunk waits for its first Engram layer's rows, so short prompts gain most: 1.6K-token
  prompts -20 % prefill time, 4K -3.5 %, 16K -3.3 %, 32K -2.5 % (prompts from one document, as a conversation's turns
  are). The tables are read-only, so every byte is the same: prefill and decode hashes equal; ~0.3 GiB more a Spark.
- **The prompt MoE's gate / up kernel three programs an SM** (`zig/kit-patches`, `zig/patches/0027`; `TF_DS_EXPERTS_CUBIN`
  set empty turns it off): the image now rebuilds the kit's `cubins/experts_cb.cubin` from its source (BertholomusAI's
  TensorFold, branch `deepseek-v41-tp2`, at `TF_PY_REF`) with one change to `grouped_mma_kernel`: two row stages instead
  of three, the staged rows unpadded with their 16-byte chunks swizzled, and `__launch_bounds__(256, 3)`, so three programs
  fit an SM (they were two: 44.3 KB of shared memory and 106 registers each). Every output's mma chains and their order
  are unchanged: the kernel's output is the same bit for bit (standalone, on a pair's and a chunk's routing) and so are
  the prefill hashes; the cubin's other 1,744 kernels have the kit's SASS. The kernel -7 % on a pair, -4 % on a single
  chunk; prefill -1.8 / -2.8 / -2.4 % at 16K / 32K / 126K tokens (four interleaved runs), served TTFT -1 to -4 %. The
  kit itself is still fetched and checked as before; the engine reads the rebuilt cubin from
  `/opt/tensorfold/kit-overlay/` in its place.
- **Prefill pairs, faster still** (`zig/patches/0026`, with 0025; `TF_DS_PAIR_OVERLAP=0` turns it off): a pair's NCCL
  gathers on a side stream beside the other chunk's work (the main stream waits it before an NCCL call of its own), and
  the next two chunks' Engram rows read from the pass's first Engram layer on, on a ring of their own. 16K / 32K /
  126K-token prompts -9.4 / -12.5 / -13.9 % prefill time against no pairs (0025 alone: -4.5 / -8.8 / -12.2 %); ~0.13 GiB
  more. Replies unchanged.
- **Prefill 5-14 % faster** (`zig/patches/0025`, `TF_DS_PAIR=1` by default): a long prompt's chunks before the decoder's
  replay (encoder-only, the bulk of it) go through the encoder layers two at a time, the first chunk's steps before the
  second's at every layer and their routed experts in one call (twice the rows per expert for the same weight reads);
  the gates stay one per chunk. The next two chunks' Engram rows are read on back host buffers from the first Engram
  layer on. 16K / 32K / 126K-token prompts -4.5 / -8.8 / -12.2 % prefill time, ~1.0 GiB more arena a Spark. Replies
  unchanged.
- **Less memory, the same speed**: `NCCL_MAX_NCHANNELS=8` (`scripts/config.sh`): NCCL carries only the prompt passes'
  all-gathers, and its default channels held ~1.2 GiB of pinned host buffers on each Spark; 8 hold 0.2 GiB, prefill within
  noise (32K-126K). The CUDA graphs of the rounds kept at most 128 (`zig/patches/0024`, `TF_DS_GRAPH_CAP=128`, the least
  recently replayed destroyed first): they piled up with every new (rows, context bucket) shape, ~3.6 MiB each, 1.6 GiB
  after a day's mixed load. Replies unchanged (an all-gather copies; a recaptured graph runs the same launches).
- **Copies beside other streams** (`zig/patches/0023`; `TF_DS_COPY_MULTI=15` by default): when every live stream's last
  round was a copy, their copies of the prompt keep growing beside each other, up to 15 tokens and an even share of a
  round's rows (never past `ROUND_ROWS`). Two streams that both edit +28 %, four +9 %, sparkDash-like mixes +3 %, no stream
  slower (an unguarded version slowed a drafting stream beside a copy by 21 %). Replies unchanged.
- **Sampled self-repeats copied 15 tokens at most** (`zig/patches/0022`; `TF_DS_COPY_SELF_CAP=15` by default): at
  temperature > 0 a copy window over the reply's own earlier text (not the prompt) takes at most 15 tokens; the copy
  proposer now says where its copy starts. Recovers the 4-7 % that 32-row windows cost one-stream sampled decode on
  repetitive text, keeps the edits' gain (copies of the prompt). Replies unchanged.
- **Copy windows of up to 48 rows** (`zig/patches/0021`; `ROUND_ROWS=32` by default, `TF_DS_COPY_ROWS` following it; 48 for
  ~0.26 GiB more a Spark: realistic edits +14 % at 32 rows, +20 % at 48): the window
  rings get 48 spare rows (model.py's RING_EXTRA is 16; keys are read by position, so the ring's size changes no
  arithmetic, and a prompt chunk still takes ring mode at 16 rows or fewer), and the router's chunk sums run in launches
  of at most 16 rows, the captured variants' widest. Replies equal serial decoding (oracle copies 0-90 % wrong, three
  streams, temperature 1, 32K-126K prompts); perfect copies +31.6 %, a natural copy mix +6.5 %, other text unchanged.
  `start.sh` accepts `TF_DS_COPY_ROWS` up to `ROUND_ROWS`.
- **The L2 prefetcher retuned, and the drafter's pass prefetching too** (`zig/patches/0019`): the prefetcher's first wave
  starts at once (the engine waited 3 us; +0.35 to +0.72 % decode at one stream in five windows, +0.6 / +1.0 % a stream
  at two / four streams), the drafter's pass prefetches its blocks' weights like the target's layers (+0.37 %, 9 of 11
  prompts faster, none slower), and the prefetcher's constants are settings (`TF_EXL3_L2_DELAY_NS`,
  `TF_EXL3_L2_RATE_GBPS`, `TF_DS_PF_BLOCKS`, `TF_DS_PF_BUDGET_MIB`, each site's `TF_DS_PF_{A,B,C,E,H}_MIB`,
  `TF_DS_DRAFT_PF=0`, `TF_DS_DPF_{A,B,C,H}_MIB`; the engine's budgets measured best and are the defaults). Token hashes
  equal serial decoding's in every run. `start.sh` forwards `TF_EXL3_*` to both ranks.
- **Upstream's DSML tool-call fix** (`zig/patches/0020`): TensorFold commit `4803479e` by Manrique (@hashd1ve), merged
  upstream as #25 after this recipe's pin, carried as published: DSML calls are read from their own tags, so a
  `<tool_call>` in the prose or a `</tool_call>` in an argument no longer drops or misroutes a call.
- **Wide copy windows, on by default** (`zig/patches/0017`; `TF_DS_COPY_ROWS=16`, `TF_DS_COPY_SKIP=1` in
  `scripts/config.sh`, checked by `start.sh`): a stream running alone checks up to 16 copied tokens a round instead of 6,
  and skips the drafter's pass after a round that accepted its whole copy. Replies equal serial decoding (62 of 62 runs:
  copies 0-90 % wrong, three streams, temperature 1). Measured on the pair when it was added: a JavaScript file copied from
  the prompt 123.6 -> 204.1 tok/s verbatim and 111.0 -> 141.1 with every `const` turned into `let`; sparkDash's six decode
  cells +1.2 % (code, one stream +3.9 %).
- **The drafter's confidence head without the stock shards** (`scripts/ablit_conf_head.py` and its test, `zig/patches/0018`):
  `scripts/ablit_stock_dtypes.py` needs the stock shards; on an overlay checkpoint whose stock shards are gone, `prepare.sh`
  checks the head instead (a pure cast of F16 values) and writes it to `conf_head_f32.bin`, and `start.sh` gives it to both
  ranks as `TF_DS_CONF_FILE`; neither the checkpoint nor the weight files are rewritten. Found here from the drafter's own
  scores (as read, they ranked drafts worse than chance, AUC 0.45; the true head 0.84); with the true head and the stock
  depth rule back, sparkDash's six cells +6.4 %.
- **`ABLIT_PRUNE_STOCK=1`**: with `ABLIT=1`, removes the 24 stock shards the overlay replaced once its checkpoint is verified
  (127 GB; `scripts/prune_ablit_stock.py` writes its plan first and removes a blob only when no other snapshot uses it), and
  preparation stops downloading stock. The dtype step runs before it.
- **`start.sh`**: a SIGTERM (systemd stopping the service) always reaches `./stop.sh` (a foreground `docker logs -f`
  deferred the trap, so a stop could leave the ranks running); range checks for the `TF_DS_*` variables of
  `zig/patches/0010`-`0018`; `TF_DS_FIX_K` reaches rank 1 as `--fix-k`.
- **Research patches, off unless set**: draft-depth policies (`0010`-`0016`), 0018's calibration dumps and Platt scaling.
  `research/`: the measurements and tools.
- `scripts/config.sh` also searches `~/.local/bin` (last) for the `hf` CLI: a systemd user service's PATH lacks it, and
  `prepare.sh` stopped there with "install the Hugging Face CLI" when run from one.
- `GHCR_IMAGE` names this repository's namespace; nothing is published there, so `prepare.sh` builds the image.

## Unreleased

- `prepare.sh`'s "cannot mount ... over NFS" error shows the last four lines of docker's output, not only the last (which was always
  docker's "Run 'docker run --help'" line, never the reason); the README's NFS section says what to check (issue
  [#15](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/15), reported by **aoeu10**). The README's memory note now
  records the 464K-token head kill at `CONTEXT=655360` and keeps the 512K guidance (issue
  [#17](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/17), reported by **tanchunsiong**).
- `ABLIT=1`: the drafter drafts again (**sunlinhua-dotcom**, issue [#11](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/11) and
  [pull request #12](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/12); the same finding, with a second measurement
  on stock-vs-grafted decode, was reported by **Ye1806431561** in issue [#13](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/13)
  and pull request [#14](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/14), which relinks the unedited shards to stock instead;
  #12 is the one taken). The overlay's `apply_mia29_wo_b_graft.py` casts four MTP tensors from F16 to F32 for another
  engine; this one uploads `mtp.2.confidence_head.proj.weight` as stored and reads it as F16 (`draft.zig`), so from the F32 copy the
  confidence head computed garbage and the engine drafted 0.2-0.8 tokens a round while acceptance stayed high. `scripts/ablit_stock_dtypes.py`
  (prepare.sh, after the overlay, and once on a checkpoint prepared before) gives the four tensors stock's F16 bytes back after checking that
  each is a pure cast (`scripts/test_ablit_stock_dtypes.py`, run in the image, covers the give-back, the untouched hardlinked stock shard, the second run and the refusal); every other tensor, the 104 overlay tensors included, is left as it is, and the checkpoint's headers then match stock's.
  A checkpoint prepared before this rebuilds both ~106 GB weight files on the next start. **Measured on the pair** (the same F16 bytes written
  into both ranks' weight files in place; after and stock back to back, same script): ABLIT decode is back at stock's speed, default sampling
  50.8 -> 68.6 tok/s (stock 69.1, 99%), code 60.8 -> 114.8 (stock 110.8), prose 52.5 -> 70.7 (stock 75.1), structured 48.7 -> 142.1 (stock 137.8),
  single stream; 4 streams prose 104 -> 134.9 (131.0), code 112 -> 157.4 (155.6); acceptance equal to stock's.
  The script was run on copies of the real shards 38 and 39: four tensors given back, equal to stock's bytes, the other 9,788 equal to the
  overlay's, the originals untouched, a second run writes nothing; a tensor that is not a pure cast is refused (exit 1).
- `start.sh` re-reads the memory available on both Sparks for up to `MIN_AVAIL_WAIT_S` seconds (default 30; 0 reads once) before it
  refuses a start under `MIN_AVAIL_GIB`. Right after an image build or a large copy the page cache is counted as available only as it
  is written back: 111.1 to 111.6 GiB was read on a worker a few seconds after a build, 119 GiB once it settled, and the 112 GiB check
  refused a start the Spark could take. The README's `MIN_AVAIL_GIB` default reads 112, as `scripts/config.sh` has it (it said 109).
  By **ezoushen**, [pull request #7](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/7).
- `THINKING_BUDGET` (empty by default): passes `--thinking-budget` to rank 0's `tensorfold-native serve`, so a turn that would spend its whole `MAX_TOKENS` thinking
  has its think block closed after that many reasoning tokens and the answer continues (a request's own `thinking_budget` wins). Checked on the pair: a budget of 1,500
  gave about 1.5K reasoning tokens and a finished answer. README: the engine refuses a `top_k` above 56 (`TooManyCandidates`) and `top_k` 0 or -1.
  By **ezoushen**, [pull request #8](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/8).

- Issues #3 and #2 in the engine (patches, both **off or on as noted, not yet merged**): `zig/patches/0005-prefill-interleave.patch` (**Coinupbtc**,
  [pull request #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5)) gives the streams already live decode rounds and their cancels between the
  chunks of a long prompt pass, `PREFILL_INTERLEAVE_MS` (default 0: off); `0006-image-prompt-cache.patch` lets a prompt with pictures resume and keep prompt states,
  keyed by the pictures' content (on; `TF_DS_IMAGE_CACHE=0` turns it off). The patches apply in order on the pin, and the engine's host tests pass with all five (21/21 steps).
  **Measured on the pair** (2026-10-19, image `exl3-tp2-8eb5919ff0a4`, interleave 250 ms): a 120K-token cold prompt no longer freezes a decoding stream (longest gap 1.3 s, was
  90 s; the prompt takes 99 s, was 91 s) and a cancel is answered 0.74 s after the client leaves, inside the other prompt's pass; a conversation with a picture resumes
  26,624 of 27,142 tokens on turn 2 (0.8 s, cold 14 s) and another picture with the same text stays cold; `tools/exact.py`: 31 of 31 cases IDENTICAL to the previous
  image (4 concurrent equal to the 4 solo, resumes, 100K prompt). `PREFILL_INTERLEAVE_MS` defaults to 250 (0: off).
  An earlier version of 0005 written here was dropped for the pull request's, which joins the Engram read-ahead only when another stream wants a round.
- memguard is looser (issue #4): the default floor goes from 4 to 2.5 GiB and a kill needs the memory to stay under it for `MEMGUARD_HOLD_S` (4)
  seconds without once coming back over it; under `MEMGUARD_HARD_GIB` (1) in two samples 0.25 s apart it still kills at once. A dip that
  recovers is logged with its length and lowest value, not killed. Measured here: a 120K-token cold prompt beside a decoding stream took
  the head to 3.3 GiB, over the old 4 GiB floor. The decision logic was tested with a fake `/proc/meminfo` and a stub `docker` (a dip that
  recovers, a sustained dip, a crash under the hard floor, flapping); no rank was killed or spared by it on the pair yet.
- Issues [#2](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/2),
  [#3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/3),
  [#4](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/4) (reported by lukemdanastasi): `TF_REF` moves from
  `3ff3b2ac` to `629c524` (the branch tip of 2026-10-19). It brings `9f82099` (a long prompt pass lets streams near their
  `max_tokens` finish before the next pass; kept prompts hold their extent), `629c524` (a shared prefix no longer stales the first
  conversation's kept states) and the template and JSON-writer fixes. `zig/patches/0003` (cache placement) is removed: the engine does this
  itself now and the patch no longer applies. Image `tensorfold-dsv41:exl3-tp2-bd95cdfc712d`; the engine's host tests pass with patches 0001, 0002 and 0004
  applied (`zig build test`: 21/21 steps, 221/221 tests). **Served on the pair** (2026-10-19, `tools/smoke.py` passes): cold prompts of 131K and 262K
  tokens prefill at 1,608 and 1,868 tok/s; `tools/prompt_reuse.py` resumes 99-100% of a 159K and a 261K-token history after another
  conversation's request (1.5-1.9 s a turn). `MemAvailable` on the head (6.5 GiB after the load) fell to 4.9 GiB at 159K, 4.2 GiB at 261K and
  **3.3 GiB** while a 120K-token cold prompt prefilled beside a decoding stream: under the default 4 GiB `MEMGUARD_GIB` (this pair runs 2.5).
  Issue #3 is **not** helped by the pin: that decoding stream got no token for 90.1 s while the other prompt prefilled (16 tokens in 90 s). Not fixed, because they are engine behaviour: prompts with images run whole and keep nothing (#2); a prompt pass runs
  to completion before other streams decode and a cancel waits for it (#3, only the tip's narrow mitigation applies).
  For #4 `start.sh` now reports `MemAvailable` on both Sparks after the load and warns when it is within 3 GiB of `MEMGUARD_GIB`, and the
  README says what holds the memory above the floor (nothing at run time) and what to do on a shared head.
- First version: DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw on two DGX
  Sparks, TensorFold's native Zig engine with BertholomusAI's `dsv41` family (`deepseek-v41-zig` @ `3ff3b2ac`), their kit
  (`abe07c7`) and their weight-file tool (`deepseek-v41-zig-tp4` @ `5940208`). Scripts adapted from the GLM-5.3-Flash
  recipe. Not served yet: see the README's Status.
- From the GLM-5.3-Flash recipe: the node and NFS helpers, `stop.sh`, the start banner, the layered settings,
  `scripts/zig_flags.py` (run by `start.sh` against the image's `capabilities --json` before anything is stopped),
  `scripts/publish-image.sh` and the registry pull, the API checks in `tools/` (`toolcheck.py`, `needle.py`,
  `end_of_turn.py`, `anthropiccheck.py`), the Hermes Agent page and the `.github` templates. Not ported, because they read
  the GLM Python engine's `/health` counters or its prompt cache, patches or settings: `tools/hermes_benchmark.py`,
  the pool, kept-state, display-KV and CPU/GPU tests, the spill tier, `SHARED_PREFIX`, the three and four Spark scripts,
  and the Ablit weights. (`tools/prompt_reuse.py` is ported since, with its cached-token lookup adapted.)
- `model/chat_template.jinja` writes each piece of the prompt out as it is made instead of adding it to one growing
  string (fix by hashd1ve, issue #1): rank 0's host memory no longer grows with messages x prompt length (40K tokens in
  2,000 messages: +1,555 MiB before, +244 MiB after, as the issue measured). The rendered prompt is identical (3,000 random
  conversations compared on jinja2).
- `stop.sh` counts running requests from the native server's `/health` (`live.connections`).
- Served and measured on one pair of Sparks (README, Status). With sparkDash's real prompts:
  prefill ~2,140-2,180 tok/s from 8K to 64K prompts (2,000 at 256K), decode 74.9 tok/s (prose) and 109.9 tok/s
  (code) for one stream, 130.0 and 155.5 tok/s for four. `tools/bench.py`'s random-word prompts give less decode (55-61 tok/s, 113-117 for four): the drafter
  cannot predict them.
- Both GPUs must share one clock ceiling: a Spark locked at 1,800 MHz against 2,200 MHz on the other cost ~3-6% of prefill
  and ~3% of decode (TP=2 meets twice a layer, so the slower GPU sets the pace).
- Rank 0 reads Engram's tables from local disk (`HEAD_ENGRAM_DIR`): over NFS a random read took ~940 us against ~90 us
  locally, and prefill was ~690-890 tok/s instead of ~1,500-2,000.
- Memory safety: `scripts/memguard.sh` (kills a rank's container under `MEMGUARD_GIB`), the `MIN_AVAIL_GIB` preflight, setup
  stops a running server before it builds or downloads, and `HEAD_ENGRAM_DIR` is checked (both shards, exact sizes) before
  anything starts.
- `HEAD_NFS_SERVER` / `HEAD_NFS_VOLUME`: rank 0 reads the checkpoint over NFS through a docker volume (sshfs does not work:
  it lists files as type "unknown" and the engine skips them).
- `API_KEY=1`: an opt-in API key (off by default; the API is open on the network otherwise).
- `tools/bench.py`, `tools/smoke.py` and `tools/prompt_reuse.py`.
- Engine patches in `zig/patches/` (off by default; bit-identical to the unpatched image on 31 of 31 prompts): prompt-cache placement
  (`TF_DS_CACHE_PLACEMENT=1`, **on**: a resumed turn no longer misses the cache because another conversation's kept rows were in
  the way), a second Engram AIO ring (`TF_DS_ENGRAM_LANES=2`, off: 1.5-2.6% slower) and a prefill indexer row split
  (`TF_DS_IDX_SPLIT=1`, off: no measurable gain). `tools/exact.py` records and compares replies.
- `zig/patches/0004-call-gate.patch`: the engine enforces `tool_choice` `"required"` and a named function (taken from the spark1/spark2
  deployment's copy; built into the running image `exl3-tp2-7c9bf5d36350`: required, named, streaming and thinking-on requests
  return valid tool calls, auto/none/absent are unchanged).
- `RANK1_PROFILE` (rank 1 `--profile`); both GPUs must share one clock ceiling; each Spark can run from a local copy of the
  cache (no NFS), with `PREPARE=0`.
