<h1 align="center">DeepSeek-V4.1-Flash-1017 EXL3 on 2x DGX Sparks with TensorFold</h1>

<p align="center">
  <sub>sehoon123's recipe, built on <a href="https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks">Mia's AI Lab's</a>: the same setup, with an exact wide copy-window decode path</sub>
</p>

Serve **DeepSeek-V4.1-Flash-1017** from two NVIDIA DGX Sparks (GB10, 128 GB each, linked by their ConnectX-7 ports)
through an OpenAI-compatible API, with the full **1,048,576-token window**, **image input**, tool calling, thinking on
or off, and **DSpark** speculative decoding from the checkpoint's own draft blocks, using
[TensorFold](https://github.com/ashhart/TensorFold)'s native Zig engine and the 2.9 bit EXL3 quantization
[`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
(196 GiB; experimental: its model card has no perplexity or KL measurement against the unquantized model).

- Checkpoint: `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` (39 shards, routed experts at 3 bits a
  weight, 2 bits in layers 18-22, attention at 5, shared experts at 4-5, head at 6, DSpark blocks at 4; the vision tower
  and the DSpark blocks are in it; the revision is pinned in `scripts/config.sh`), plus Engram's two tables, unquantized,
  from `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10` (shards 47 and 48, ~203 GB; the quantization leaves them out)
- API model id: `DeepSeek-V4.1-Flash-1017`, port 8888
- Engine: TensorFold native (Zig 0.17.0), BertholomusAI's `dsv41` family at TP=2, one rank on each Spark
- One command on the first Spark: `./start.sh` sets up both Sparks and starts both ranks; `./stop.sh` stops them
- Optional `ABLIT=1`: keeps Mia EAQ 2.9 bpw and applies the matching [Keys overlay by drowzeys](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated).
  Requires your Hugging Face token and access to that gated repository; see [Abliterated mode](#abliterated-mode).

## Latest stability release — 2026-10-28

Patches **0046–0050** harden saved-cache ownership, preserve valid prefixes when placement is temporarily blocked by
live streams, and send 15-second native OpenAI SSE keep-alives during long waits. The 0048–0050 follow-up passed
**34 complete output-token-array comparisons** against 0047, including a real 850K-token SDK/proxy request with
381.91 seconds of cold prefill and no retry. Model, quantization, arithmetic, limits and cache budget are unchanged.
See [the qualification report and its limits](research/CACHE-PLACEMENT-SSE.md),
[0046/0047 lifecycle details](research/CACHE-LIFECYCLE.md), and the
[source/evidence release](https://github.com/sehoon123/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned/releases/tag/cache-placement-sse-20261028).
Cold prefill can still take minutes; SSE comments do not override absolute client deadlines. This is not a claim
that every client-side stall is resolved. The decode measurements below retain their original dates and baselines.

## What this recipe adds

This is [Mia's AI Lab's recipe](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks) at [`31eea46`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/commit/31eea4608f8bb292a92e2f7a1db400e928620c78) with the changes below (follow Mia's AI Lab on
[X](https://x.com/MiaAI_lab), sponsor them on [GitHub](https://github.com/sponsors/MiaAI-Lab)). The setup, the commands,
the settings and the engine pin are theirs, and so is the rest of this README, whose issue and pull-request numbers are
their repository's.

- **Wide copy windows, on by default** (`zig/patches/0017` and `0021`; `ROUND_ROWS=32` and `TF_DS_COPY_ROWS` equal to it,
  `TF_DS_COPY_SKIP=1` in `scripts/config.sh`). When a reply copies text from its prompt or from itself (an edited file, a
  quotation, a list written again), a stream that runs alone checks up to 32 copied tokens a round instead of 6, and skips
  the drafter's pass after a round that accepted its whole copy. With 16-row windows (0017 alone): a JavaScript file from
  the prompt copied verbatim 123.6 → 204.1 tok/s, with every `const` turned into `let` 111.0 → 141.1. From 16 to 48 rows
  (0021, one stream): writing out four Python files whole with a scattered edit each (a rename, print -> log, ...) +14 %
  at 32 rows and +20 % at 48 (180.7 -> 206.3 / 216.9 tok/s; at temperature 1 +13 / +18 %), copies of the reply +21 / +32 %;
  text without copies unchanged. Each 16 rows past 16 cost ~0.26 GiB of each Spark's memory at load (the wider rounds'
  buffers), and more once several conversations run at once: three or four of them left a 48-row head 0.5-0.8 GiB less
  free than a 32-row one. So 32 is the default; set `ROUND_ROWS=48` where the head has the room, with `MEMGUARD_GIB` at 3
  or below ("Memory and safety"; with 0022 below, 48 rows copy 5.5-6.8 % faster again and lose nothing elsewhere,
  `research/results/rows-32-48.txt`). At temperature > 0 a copy of
  the reply's own earlier text takes at most 15 tokens a round (`zig/patches/0022`, `TF_DS_COPY_SELF_CAP=15`): sampled
  repeats leave their copy sooner, and full-width windows over them cost one-stream sampled decode 4-7 %; copies of the
  prompt keep the full width. When every live stream copies (agents writing out the files they edit), their copies keep
  growing beside each other, up to 15 tokens and an even share of a round's rows (`zig/patches/0023`,
  `TF_DS_COPY_MULTI=15`): two streams that both edit +28 %, four +9 %, and no stream that drafts is slowed.
- **1.0 GiB less pinned host memory on each Spark, and no slow creep**: `NCCL_MAX_NCHANNELS=8` (NCCL carries only the prompt
  passes' all-gathers; its default channels held ~1.2 GiB, prefill as fast with 8), and at most 128 of the rounds' CUDA
  graphs kept (`zig/patches/0024`, `TF_DS_GRAPH_CAP=128`; each new (rows, context bucket) shape added ~3.6 MiB for good,
  1.6 GiB after a day's mixed load).
- **Prefill 9-14 % faster on long prompts**: `zig/patches/0025`-`0026`, `TF_DS_PAIR=1`. Chunks before the decoder's replay
  (only the encoder layers) go two at a time, their routed experts in one call, their gathers beside the other chunk's
  work; ~1.1 GiB more arena a Spark. And the prompt MoE's gate / up kernel runs three programs an SM instead of two
  (`zig/kit-patches`, the kit's experts cubin rebuilt from its source in the image, `zig/patches/0027`): prefill another
  2-3 %, the same bits.
- **Short prompts 20 % sooner**: `zig/patches/0028`. A chunk's Engram table rows are read from the NVMe once each (a
  third or more of its n-grams name a row another one names), and the rows read last stay in host memory (256 MiB a
  Spark), so a conversation's next turn reads only what is new; the first chunk of every request waits for those reads.
- **Sampled decoding ~1.2 % faster**: `zig/patches/0029`. Between a round and the next, the host drew each row's token
  from its logits: their copy (~3 MB a round, to pageable memory) took 0.37 ms and the top-k search over the 129K
  vocabulary 0.37 ms; with pinned memory and the search 16 logits at a time, 0.07 + 0.12 ms. The same bits.
- **A long conversation's next request read ~55 ms sooner**: `zig/patches/0030`. The server tokenized every request
  whole (57 ms at 133K tokens); it keeps the last long texts' ids and encodes only what follows the previous reply
  (2-3 ms). The same ids.
- **Long prompts' prefill another 3-4 % faster**: `TF_DS_IDX_SPLIT=1` (MiaAI-Lab's `zig/patches/0001`, now on for every
  eligible chunk): the prefill indexer's rows are split between the two Sparks (161K tokens 71.3 -> 68.3 s, 43K -0.95 %,
  16K the same; the same bits).
- **A quiet server no longer spins a core on each Spark**: `zig/patches/0032`, MiaAI-Lab's RDMA idle nap (their 0010,
  unchanged; TensorFold PR #27 by William J Riley, ported by Mykhailo Tamarin): the gather proxy sleeps between looks after 2 s with nothing to do
  (99.9 -> 0.3 % of a core idle); decode is the same (+0.06 %).
- **A request arriving during another's long prompt starts in ~2 s, not after it**: `zig/patches/0031`,
  `PREFILL_NEST_TOKENS=8192`. Such a request whose own pass needs at most that many tokens (past its kept state) runs
  between two of the long pass's chunks: a 1.9K-token request sent 3 s into a 68.6K-token prompt had its first token
  after 2.0-2.4 s instead of 27.6 s; the long prompt took ~5 s longer. The same bits for both.
- **A conversation's next turn up to ~60 % sooner (optional)**: `tools/warm_ahead_proxy.py`, an OpenAI / Anthropic API
  proxy in front of the server. While a client is idle after a reply, it has the server's prompt cache keep the state at
  the next 2,048-token chunk end the conversation and its reply reached, so the next turn resumes there instead of
  re-prefilling those tokens (39K-token conversations: a turn's first token 1.40-1.57 -> 0.54-0.59 s when the reply
  crossed a chunk end). Replies are the same bits; any harness (a client is warmed ahead once it has shown how it sends
  replies back). See *Warm-ahead proxy* below.
- **The drafter's confidence head when the stock shards are gone.** Mia's AI Lab's recipe gives the overlay's F32 copy of
  `mtp.2.confidence_head.proj.weight` stock's F16 bytes back (`scripts/ablit_stock_dtypes.py`), which needs the stock
  shards. When they are gone, preparation checks the head and writes it to `conf_head_f32.bin`, and both ranks load it
  from there (`TF_DS_CONF_FILE`, `zig/patches/0018`); the checkpoint and the ~106 GB weight files stay as they are.
- **`ABLIT_PRUNE_STOCK=1`**, for `ABLIT=1` on a smaller disk: once the overlay checkpoint is verified, the 24 stock shards
  it replaced are removed (127 GB) and are not downloaded again ([Abliterated mode](#abliterated-mode)).
- **`start.sh`**: a SIGTERM (systemd stopping the service) always reaches `./stop.sh` and stops both ranks (a foreground
  `docker logs -f` used to defer the trap); the `TF_DS_*` variables of `zig/patches/0010`-`0018` are range-checked;
  `TF_DS_FIX_K` reaches rank 1 as `--fix-k`; `TF_EXL3_*` settings reach both ranks too.
- **The L2 prefetcher, retuned, and the drafter's own** (`zig/patches/0019`). The engine touches the next kernels'
  weights into L2 while latency-bound work runs; its first wave now starts at once instead of 3 µs late (+0.35 to
  +0.72 % decode at one stream in five measurement windows, +0.6 / +1.0 % a stream at two / four), and the drafter's
  pass prefetches its blocks' weights the way the target's layers do (+0.37 %, 9 of 11 prompts faster, none slower).
  Each prefetch site's budget became a setting; the engine's budgets measured best and stay. A prefetch does no
  arithmetic: every run's token hashes equal serial decoding's.
- **Upstream's DSML tool-call fix** (`zig/patches/0020`: TensorFold commit `4803479e` by Manrique, @hashd1ve, merged
  upstream after this recipe's engine pin): a `<tool_call>` in the prose, or a `</tool_call>` inside an argument (a file
  being written), no longer drops a DeepSeek-V4.1 tool call or streams a wrong one.
- **Research patches, off unless their variables are set**: draft-depth policies (`0010`-`0016`) and 0018's calibration
  switches ([zig/patches/README.md](zig/patches/README.md)).
- **[research/](research/README.md)**: the measurements behind these changes: where a 38.9 ms decode round goes on this
  pair, the exactness checks, what did not help, and the tools.

The historical decode checks below compared replies with serial decoding (one token a round, no drafts) or with its token
hashes: 62 of 62 copy-window runs (copies 0-90 % wrong, three streams at once, temperature 1, EOS and length endings) and
611 decodes of the later offline runs, up to 125K-token prompts, and every run of the kernel rounds (patch 0019), none
different ([research/](research/README.md)).

The runtime uses the full ordered [patch chain](zig/patches/README.md) and kit overlays in `zig/Dockerfile`.
The currently qualified local image is `tensorfold-dsv41:exl3-tp2-67309ead97da`; its exact build/deployment scope is
recorded in the stability report above. No container-registry image is published: `./start.sh` builds a missing
local image on the head and copies it to the worker. Source/evidence releases do not include model weights.

<!-- MEASURED -->
Measured on our pair with this commit, run as our production (`ABLIT=1`, `CONTEXT=1048576`, the defaults above; sparkDash's
decode bench, the streams together, median of two waves, 2026-10-21 12:18-12:22, a dashboard polling `nvidia-smi` every 10 s):

| Decode, tok/s | 1 stream | 2 streams | 4 streams |
| --- | --- | --- | --- |
| Prose, greedy | 68.2 | 102.4 | 137.4 |
| Code, greedy | 116.3 | 152.3 | 166.1 |
| Prose, temperature 1 (the served default) | 70.8 | 94.8 | 123.2 |
| Code, temperature 1 | 108.4 | 137.1 | 163.5 |

Against the same pair on 2026-10-20 before this recipe's later patches (greedy: Mia's AI Lab's hotfix image with 0010-0016;
temperature 1: that morning's run): +10.9 % and +10.5 % geomean (`research/results/decode-final-*.json`, `decode-table.txt`).
sparkDash's prompts copy little; replies that copy go further: four files written out with a scattered edit +14 %
(temperature 0) and +13 % (temperature 1) over 16-row copy windows, two such edits at a time +29 % with 0023
(`research/results/wide-copy-windows.txt`, `copy-policy.txt`, `copy-multi.txt`). `tools/validate.py` found the replies
equal to the earlier images' (8 of 8). Polling `nvidia-smi` every 2 s instead of 10 s cost 1 %: keep GPU monitoring sparse
while serving. The stock checkpoint (`ABLIT=0`) was not measured with these changes; for it, see Mia's AI Lab's numbers
under Status.
<!-- /MEASURED -->

## Status

This section is Mia's AI Lab's, for their recipe at `31eea46` (the stock checkpoint unless it says otherwise); this
recipe's own measurements are under [What this recipe adds](#what-this-recipe-adds).

What is checked, and how:

- `DRY_RUN=1 ./start.sh` prints the link it found (on a pair of Sparks) and both ranks' docker commands; `./stop.sh`
  and `scripts/*.sh` pass `bash -n`.
- The kit's RoPE tables, made from this checkpoint's `config.json`, match the kit's recorded sha256 (4 of 4), and the
  token map made from its `tokenizer.json` is byte for byte the one the kit's Engram fixtures were recorded with.
  `scripts/prepare.sh` checks both on every run. The checkpoint holds all 43 `gate.bias_vl` tensors (F32) the image
  routing bias is made from.
- `zig/Dockerfile` builds (`tensorfold-dsv41:exl3-tp2-136c93b359b4`, 24.9 GB, about 6 minutes of Zig compiling on a
  Spark): `tensorfold-native 1.0.2`, `tf-dsv41-lanes`, `tf-dsv41-rank-cache` and the three inputs' scripts are in it.
- `tf-dsv41-rank-cache` plans the TP=2 split: run on a copy of the real headers (tensor bytes left out) of an earlier 43-shard
  2.9 bpw EXL3 checkpoint with the same tensor widths, it gives each rank 143,770 entries and 98.9 GiB, the ~98 GiB a
  Spark the weights need. (Not re-run on the 39-shard checkpoint.)

Served and measured on one pair of Sparks (the head also runs a desktop and ~18 small containers; the worker is otherwise
idle). Two sets of numbers, from two tools:

**[sparkDash](https://github.com/MiaAI-Lab/sparkDash)** (real prose and code prompts, as sparkDash reported them, with the
server at `CONTEXT=524288`, up to 4 streams, drafts on):

| Decode, tok/s | 1 stream | 2 streams | 4 streams |
| --- | --- | --- | --- |
| Prose | 74.9 (TTFT 131 ms) | 105.5 total, 55.0 each (TTFT 243 ms) | 130.0 total, 34.9 each (TTFT 411 ms) |
| Code | 109.9 (TTFT 200 ms) | 139.8 total, 74.8 each (TTFT 291 ms) | 155.5 total, 44.4 each (TTFT 511 ms) |

| Prefill, prompt tokens | 8K | 16K | 32K | 64K | 128K | 256K |
| --- | --- | --- | --- | --- | --- | --- |
| tok/s | 2,166 | 2,183 | 2,175 | 2,136 | 2,079 | 2,000 |
| TTFT | 4.0 s | 7.7 s | 15.3 s | 30.9 s | 63.2 s | 131.3 s |

An earlier sparkDash run, with one Spark's GPU locked at 1,800 MHz (the other at 2,200), read prose 74.6 and code 108.1
tok/s for one stream, 140.6 and 152.7 for four, and prefill 2,008-2,062 tok/s from 4K to 64K prompts (1,877 at 256K).

**`tools/bench.py`** (thinking off, temperature 0, random-word prompts so no prefix cache is hit and the drafter has little
to predict; medians of one or two repetitions, so differences of a few percent are noise):

| | Result |
| --- | --- |
| Prefill, 2K / 8K / 30K / 100K-token prompts | ~1,500-1,600 / ~1,960 / ~2,000 / ~1,950 tok/s (TTFT 1.2 s / 3.9 s / 14.4 s / 49 s) |
| Decode, 1 stream | 55-61 tok/s |
| Decode, 4 streams together | 113-117 tok/s |
| Drafting (DSpark) | ~2.2-2.4 tokens a round on random-word prompts (about 50 of 88 and 58 of 93 drafted tokens accepted) |

The lower `bench.py` decode numbers are the prompt, not the engine: real text drafts much better than random words.

These need **Engram's tables on local disk on both Sparks**: with the head reading them over NFS (`HEAD_ENGRAM_DIR` unset
and a remote `HF_CACHE`) a random 4 KiB read took ~940 us against ~90 us from local disk, and prefill was ~690-890 tok/s.
BertholomusAI's figures for the earlier checkpoint on two Sparks (decode 89.8 / 59.0 / 106.6 tok/s for code / prose /
structured, 4 streams 118.9 tok/s together, prompts at ~2,400 tok/s up to 128K) are the reference these are compared with:
with real prompts decode is above them (one stream: prose 74.9 against 59.0, code 109.9 against 89.8; four streams:
130.0 and 155.5 against 118.9), and prefill is ~9-11% under them from 8K to 64K prompts (~2,140-2,180 against ~2,400 tok/s).

A needle-in-a-haystack test at `CONTEXT=524288` (`tools/bench.py --suite needle`, a passphrase hidden at 10%, 50% and 90%
depth of a 479,012-token prompt) found it 3 of 3 times; each prefill took ~280 s (~1,710 tok/s). The head's lowest free
memory during it was 2.67 GiB, 0.17 GiB above the 2.5 GiB floor `memguard` was set to, so a 512K window is the practical
limit for a head that carries ~6 GiB of other software. A reported run at `CONTEXT=655360` (issue #17; 4.9 GiB free after the load on
a head with its other services stopped) fell under the 2 GiB floor for the 4 s `MEMGUARD_HOLD_S` and had rank 0 killed on a turn
that took a cache-hit history from 408K to 464K tokens: the head's free memory falls as a history grows (see the CHANGELOG: 6.5 GiB after
the load, 4.9 at 159K tokens, 4.2 at 261K), and a larger `CONTEXT` only allows longer histories than the head can carry. Keep `CONTEXT` at 524288 or less on
such a head, or free more of its memory first. A 100K-token prompt also ran without errors at the full
1,048,576-token window (3.14 GiB free at the lowest) and at 524,288 tokens.

**Engine patches** (`zig/patches/`, three, written for this recipe). Each is **off by default**, so the image behaves like
the pinned engine: its replies' token hashes equal the unpatched image's on 31 of 31 prompts (1 to 100,000 tokens, cached-prefix
resumes, four at once, a three-conversation scenario; `tools/exact.py`). Each is switched on with a `TF_DS_*` variable, which
`start.sh` gives to both ranks, and each was tested on its own:

| Switch | Result |
| --- | --- |
| `TF_DS_CACHE_PLACEMENT=1` (gone) | Measured on the earlier pin: in the three-conversation scenario conversation B's second turn used to miss the prompt cache and re-read its whole 6K-token prompt; with the patch it reused 4,096, with an identical reply. The pin is now `629c524`, whose engine places kept states itself (a conversation sharing a prefix copies the shared rows to a free extent); the patch was removed and the scenario was not re-measured on it. |
| `TF_DS_ENGRAM_LANES=2` (off) | A second I/O ring for the prompt's Engram reads. Bit-identical, but prefill 1.5-2.6% *slower* from 8K to 131K: the drive is the limit. |
| `TF_DS_IDX_SPLIT=1` (off) | Splits the prefill indexer's rows between the two ranks. Bit-identical, with no measurable gain on this pair (modeled at +1% at 64K and +2% at 128K; the undisturbed 32K repetitions tie the baseline). If the two ranks ever disagree on the setting the pair hangs, so it stays off. |

Two limits of BertholomusAI's engine, reported in issues [#2](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/2) and
[#3](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/3), are patched here (`zig/patches/0005`, `0006`), and measured on the pair
(engine `629c524` + patches; the replies of 31 `tools/exact.py` prompts, 4 concurrent among them, are identical to the previous image's):
- **A prompt pass no longer freezes the other streams** (`0005`, by Coinupbtc, pull request #5; `PREFILL_INTERLEAVE_MS`, 250 by default, 0 turns it off): between two
  chunks of a long prompt pass, the streams already live get decode rounds and their cancels are taken. A 120K-token cold prompt used to leave a decoding
  stream without a token for 90 s; now its longest gap is 1.3 s, and a cancel is answered 0.74 s after the client leaves, in the middle of the other
  prompt's pass. The cold prompt itself takes about 10 % longer (99 s against 91 s).
- **A prompt with pictures resumes and keeps prompt states** (`0006`; `TF_DS_IMAGE_CACHE=0` turns it off): the cache keys the pictures' positions by the
  picture's content. With a picture first and 27K tokens after it, turn 2 resumed 26,624 tokens in 0.8 s (cold: 14 s); another picture with identical text was
  cold, and the first picture again hit with a byte-identical reply.
`ENGINE=jl` interleaves prompt pieces with other slots' decode (`TF_DSV41_PREFILL_PIECES`); its image prompt reuse was not tested here.

What is **not** checked: image input and `tools/validate.py` on this pair; other cabling, hardware or a head with less
memory to spare; days of uptime; and the engine's behaviour at the memory floor under heavy concurrent long prompts (the
watchdog exists because that can freeze a Spark; see [Memory and safety](#memory-and-safety)).

## Requirements

- **Two DGX Sparks** (or two GB10 systems with 128 GB unified memory), with nothing else large on them: the weights take
  ~98 GiB a Spark and rank 0 holds ~104 GiB on the GPU, so a Spark's memory is the limit, not its disk or its link.
  `start.sh` refuses to start unless `MIN_AVAIL_GIB` (112) is available on each Spark. A head that also runs a desktop
  and development tools (~6 GiB here) was left with ~3.1 GiB at the worst point of a 100K-token prompt at the full window
  and ~3.7 GiB at 524,288 tokens: give the head less to do, or lower `CONTEXT`.
- **The same GPU clock ceiling on both Sparks** (`nvidia-smi -lgc 0,2200`; check with `nvidia-smi -q -d CLOCK`). With TP=2 the
  ranks meet twice a layer, so the slower GPU sets the pace: one Spark found locked at 1,800 MHz cost about 3-6% of prefill
  and 3% of decode against 2,200 MHz on both (rigmark, same prompts; the GPUs peaked at 85 / 81 C and nothing throttled).
- **A direct ConnectX-7 link:** a QSFP cable between the CX7 ports and an IPv4 address on each end in one private subnet
  (e.g. `192.0.2.1/24` and `192.0.2.2/24`; `ping` must work), with a RoCE v2 GID (`start.sh` checks). One QSFP port of a
  Spark reaches the GB10 over two PCIe Gen5 x4 links, so it appears as two netdevs and two RoCE devices (`enp1s0f0np0` /
  `enP2p1s0f0np0`, `rocep1s0f0` / `roceP2p1s0f0`). Address both twins on each Spark (one subnet each is fine, the same
  MTU on both Sparks): both are then used as two rails; one rail is one x4 (~112 Gb/s of the port's 200).
- **Key-based ssh** from the first Spark (the head, which runs `./start.sh` and the API) to the second (the worker):
  `ssh-copy-id user@<worker>`; check with `ssh -o BatchMode=yes user@<worker> true`.
- Docker with the NVIDIA container runtime and your user in the `docker` group, on both Sparks; `git`, `python3` and the
  `hf` CLI (`pip install -U huggingface_hub`) on the head.
- **Disk.** Head: ~505 GB (the checkpoint 196 GiB, Engram's tables ~203 GB, rank 0's weight file ~106 GB) plus ~25 GB for
  the image under Docker's root. Worker: ~310 GB (Engram's tables and rank 1's weight file) plus the image; the worker
  reads the checkpoint from the head over NFS.
- **NFS.** The worker mounts the head's Hugging Face cache read-only. On the head, once (needs root; the worker needs
  nothing installed):

  ```bash
  sudo apt install nfs-kernel-server
  echo "$HOME/.cache/huggingface <worker CX7 address>(ro,no_subtree_check)" | sudo tee -a /etc/exports
  sudo exportfs -ra
  ```

  If `prepare.sh` then stops with "cannot mount this Spark's ... over NFS", read the lines under it: they are the end of docker's own
  error (the reason is above docker's closing "Run 'docker run --help'" line). Check `sudo exportfs -v` on the head: the export must
  list the *worker's* address on the link (the user@address in the message), the exported path must be `NFS_PATH`, and the worker must
  reach the head's port 2049; the worker's docker mounts the volume itself, so it needs the host's NFS mount helper (`nfs-common` on
  Ubuntu) if it reports a missing `mount.nfs` helper.
  `prepare.sh` then creates a read-only docker NFS volume on the worker (`NFS_VOLUME`, no sudo) and checks that the worker
  sees every file the ranks read. `NFS_PATH` is the exported path as the worker mounts it (default: the head's `HF_CACHE`;
  `/` for an NFSv4 export with `fsid=0`), `NFS_SERVER` the head's address (default: its address on the link).

  **Or no NFS at all.** Each Spark can run from a local copy: set `HF_CACHE` to a local tree holding the checkpoint's small files
  and its vision shard (`model-00039-of-00039`, the only shard read at run time), the served `model/` folder, the kit, the token
  map and, on the head, rank 0's weight file; set `NFS_VOLUME` to the same path as a *host path* (the worker bind-mounts it) and
  leave `HEAD_NFS_SERVER` empty. Set `PREPARE=0` too: `scripts/prepare.sh` assumes a complete cache and would start downloading
  the 196 GiB checkpoint into a partial one.

## Quick start

On the head:

```bash
git clone https://github.com/sehoon123/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned.git
cd DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks-Tuned
cp scripts/local.sh.example scripts/local.sh     # set WORKER=user@192.0.2.2 in it (and FABRIC_PEER, see below)
DRY_RUN=1 ./start.sh                              # prints what it would do; changes and starts nothing
./start.sh
```

`WORKER` is the worker's ssh target. The ranks talk over the route to it: if it is on another network than the link, set
`FABRIC_PEER` to the worker's CX7 address. The worker needs no copy of this repository.

The first run sets up both Sparks (see below): the image on each (two Zig builds, ~6 minutes on a Spark), the checkpoint (196 GiB)
and Engram's tables (~203 GB) downloaded on the head, the kit checked, both ranks' weight files written, Engram's tables
and rank 1's weight file on the worker's disk. Later starts load ~98 GiB of weights on each Spark. `start.sh` shows each
step and the server's log, runs a smoke test through both ranks, and prints `DeepSeek-V4.1-Flash-1017 is now LIVE!
on port 8888` with the endpoint.

Any OpenAI client works with `base_url = "http://<head-address>:8888/v1"` and the model `DeepSeek-V4.1-Flash-1017`.
The model thinks before it answers unless a request turns thinking off, so give replies enough `max_tokens`.

```bash
./start.sh restart                                       # restart both ranks, e.g. after changing a setting
./stop.sh                                                # stop both ranks and free their GPU memory
docker logs -f dsv41-1017-exl3-tf                        # rank 0's log (here)
ssh <worker> docker logs -f dsv41-1017-exl3-tf           # rank 1's log (why it exited, if it did)
python3 tools/validate.py http://<head-address>:8888 --model DeepSeek-V4.1-Flash-1017   # text, thinking, tools, image, long prompt
API_URL=http://<head-address>:8888 python3 tools/toolcheck.py    # also needle.py (a passphrase in ~195k tokens), end_of_turn.py
```

`stop.sh` saves each rank's log (gzipped, newest 10 kept, `LOG_KEEP`) to `~/.cache/dsv41-tensorfold/logs` before it removes
the container.

## Abliterated mode

`ABLIT` accepts **`0` or `1`** (default `0`). Set it in the environment, `scripts/local.sh` or `.env`.
`ABLIT=0` serves the stock Mia EAQ v1 2.9 bpw checkpoint. `ABLIT=1` keeps that checkpoint and applies
`exl3_wo_b_l10_35_mia29.safetensors` from
[`drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated`](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated).
This repository supplies overlays, not a full checkpoint. The recipe uses its **matching Mia 2.9 bpw** overlay and
`apply_mia29_wo_b_graft.py`; the FP8, LibertAI 3.5 bpw and Jayleaton 2.8 bpw sidecars are for other packs.

Before preparing or starting with `ABLIT=1`:

1. Log in to Hugging Face and obtain access by accepting the terms on the
   [gated repository page](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated).
2. Use a Hugging Face read token from **the same account**, with permission to read that gated repository.
   Authenticate on the head with `hf auth login`, or export `HF_TOKEN` as below. A token alone does not grant repository access.

```bash
hf auth login                 # enter your HF read token, after obtaining repository access
ABLIT=1 ./scripts/prepare.sh   # optional: prepare separately; start.sh also prepares automatically
ABLIT=1 ./start.sh restart    # restart both ranks with the overlay
ABLIT=0 ./start.sh restart    # switch back to stock
```

Alternatively, supply your token through the environment without writing it into a command's history:

```bash
read -rsp 'Hugging Face read token: ' HF_TOKEN; echo
export HF_TOKEN
ABLIT=1 ./start.sh restart
unset HF_TOKEN
```

Both `scripts/prepare.sh` and `start.sh` print the token and gated-access requirement when `ABLIT=1`, including a link
to the repository. Preparation downloads the matching overlay before the image build and large checkpoint downloads,
and reports an access error if the download fails. `hf` uses the token on the head; the recipe does not pass `HF_TOKEN`
in command arguments or forward it as an environment variable to the worker or serving containers. A saved login token
inside `HF_CACHE` is part of the shared cache; use `HF_TOKEN` or a token path outside that cache to keep it out of the mounts.
`DRY_RUN=1 ABLIT=1 ./start.sh` prints the notice without downloading or checking access.

The overlay is pinned at `0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0` (`ABLIT_REVISION`, a full commit hash).
It replaces 104 attention tensors in layers 10–35. Its application script also converts four MTP tensors from F16 to F32
for another Zig engine, as described in the upstream `TENSORFOLD-ZIG-MIA.md`; preparation then gives them their stock F16
back (`scripts/ablit_stock_dtypes.py`). This recipe's engine reads the drafter's `mtp.2.confidence_head.proj.weight` as F16
whatever the file holds: from the F32 copy it drafts 0.2–0.8 tokens a round and decode falls 25–45% below stock. A
checkpoint prepared before that step gets it on the next start, which also rebuilds both ~106 GB weight files. The stock Hugging Face snapshot stays intact:
preparation hardlinks unchanged shards and writes changed shards under a separate
`$HF_CACHE/tensorfold-dsv41/<model-revision>-ablit-<overlay-revision>/checkpoint` directory. **Allow extra disk space for
rewritten shards (up to another checkpoint's size), the ~651 MiB sidecar, and a separate ~106 GB rank cache on each Spark.**
Applying the overlay runs on CPU using the image's Python, PyTorch and safetensors.

Both ranks build their caches from the modified checkpoint. Abliterated data and worker cache paths include the overlay
revision, including when `WORKER_CACHE_DIR` is customized, so stock caches cannot be reused accidentally. Keep the same
`ABLIT` setting for preparation and startup; put `ABLIT=1` in your local configuration to make it persistent. Use
`restart` when switching modes on a running server. Startup checks both running containers' mode and overlay revision;
a separately prepared checkpoint cannot make a stock server count as abliterated. The API model name remains `SERVED_NAME`.
The stock performance and correctness results above do not validate the abliterated mode on this recipe's engine.

**This recipe: `ABLIT_PRUNE_STOCK=1`.** With `ABLIT=1`, it removes the 24 stock shards whose 104 `wo_b` tensors the overlay
replaced once the overlay checkpoint is verified (`scripts/prune_ablit_stock.py`: it writes its plan first, removes a blob
only when no other snapshot links to it, and can be re-run), and preparation stops downloading stock: 127 GB less on the
head. `ABLIT=0` afterwards downloads those shards again. The dtype step above runs before the pruning. On a checkpoint
pruned before that step existed (or whose stock shards were deleted by hand), preparation checks the confidence head
instead and, when it is F32, writes it to `conf_head_f32.bin` for both ranks (`scripts/ablit_conf_head.py`,
`TF_DS_CONF_FILE`, `zig/patches/0018`), rewriting neither the checkpoint nor the weight files. This recipe's own
measurements ([What this recipe adds](#what-this-recipe-adds)) are of the abliterated mode.

## What `start.sh` and `scripts/prepare.sh` do

To update: `./stop.sh`, `git pull`, `./scripts/prepare.sh` (it refuses to build a new image beside a running server:
the build's compiles need a few GB of the memory the server leaves), `./start.sh`.

`scripts/prepare.sh` (run by `start.sh` when something is missing; safe to re-run) on the head:

1. checks docker, the GPU, ssh and the link;
2. builds the image from `zig/Dockerfile` (the pinned engine plus the inputs' tools) and copies it to the worker;
3. downloads the checkpoint and Engram's two shards (with their index and config) at their pinned revisions, and checks the
   shards' sizes (`VERIFY_SHA=1` also their sha256, as the model card lists them: the 1017 tables are new, the older
   release's must not be used);
4. fetches the kit at `KIT_REF`, makes its four RoPE tables and the image routing bias from the checkpoint, and checks every
   file against the kit's MANIFEST (the image bias may differ: it is 1017's own);
5. writes Engram's token map and rank 0's weight file from the checkpoint (no GPU), and the folder rank 0 serves (the
   checkpoint's files linked, with `model/chat_template.jinja`: the release ships its chat format as `encoding/encoding.py`,
   which the server cannot run);
6. creates the worker's NFS volume and checks it;
7. copies Engram's tables to the worker's disk (a decode round reads ~70 random 4 KiB rows from them; over NFS that took most
   of a round) and writes rank 1's weight file there from the checkpoint over NFS.

`start.sh` then starts rank 1 on the worker (`tf-dsv41-lanes`, following) and rank 0 here (`tensorfold-native serve`), waits
for the API, and runs a smoke test.

## Configuration

Settings come from the environment, `scripts/local.sh` or `.env`; the defaults and their notes are in
[`scripts/config.sh`](scripts/config.sh). The ones you are likely to touch:

| Setting | Default | |
| --- | --- | --- |
| `MODEL_ID`, `MODEL_REVISION` | `Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw` @ `6d54be4f` | the checkpoint. The Hub repo was renamed from `...-1017-EAQ-v1-2.9-bpw` (the old name redirects), and the Hugging Face cache is keyed by the repo's name and revision: a copy under the old name or an older revision (e.g. `e8ac7684`, which the Hub no longer has) is not found under the default, and `scripts/prepare.sh` would download 196 GiB. It now stops when the disk cannot hold that. To keep serving a copy you have, set both to what `~/.cache/huggingface/hub` holds |
| `ABLIT` | 0 | `0`: stock Mia EAQ 2.9 bpw; `1`: apply the matching gated Keys overlay; requires your HF token and repository access (see Abliterated mode) |
| `ABLIT_REVISION` | `0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0` | pinned Keys overlay and application script; full commit hash |
| `ABLIT_PRUNE_STOCK` | 0 | this recipe: `1` (with `ABLIT=1`) removes the 24 stock shards the overlay replaced once its checkpoint is verified (127 GB); `ABLIT=0` then downloads them again (see Abliterated mode) |
| `HF_TOKEN` | unset | Hugging Face read token for downloads; alternatively authenticate with `hf auth login` on the head |
| `CONTEXT` | 1048576 | a request's positions at most; rank 1's `--pool` follows it |
| `PARALLEL` | 4 | requests at once (the engine's streams, 1 to 16; the family was measured at 4) |
| `THINKING`, `REASONING_EFFORT` | 1, empty | think first unless a request turns it off; `low` / `medium` / `high` / `xhigh` as the default effort |
| `THINKING_BUDGET` | empty | reasoning tokens a reply may spend before the engine closes the think block itself and lets the answer continue (`--thinking-budget`; a request's `thinking_budget` wins); empty: no cap. A hard task can otherwise spend its whole `MAX_TOKENS` thinking and end with no text and no tool call. A model that keeps rambling after the cut is not stopped by it. For long agentic or tool-calling jobs, leave it empty or set it large: a small budget cuts the reasoning mid-plan, which can make the model repeat a tool call it already made |
| `TEMPERATURE`, `TOP_P`, `TOP_K` | 1.0, 0.95, 20 | the server's sampling defaults; a request's own values win. This engine refuses a `top_k` above 56 with `TooManyCandidates` (the finite candidate list its sampler keeps), so `TOP_K` is at most 56; `top_k` 0 or -1 is refused too, with a different, misleading message |
| `DRAFTS` | 1 | the checkpoint's DSpark draft blocks (drafted replies equal plain ones) |
| `TF_DS_COPY_ROWS`, `TF_DS_COPY_SKIP` | 16, 1 | this recipe: wide copy windows (`zig/patches/0017`): a stream running alone checks up to 16 copied tokens a round, and `1` skips the drafter's pass after a round that accepted its whole copy; replies do not change. `6`, `0`: the engine's own |
| `RDMA`, `GRAPHS`, `SIDE`, `PREFETCH`, `ENGRAM_AIO` | 1 | the engine's served settings |
| `MAX_TOKENS` | 32768 | a reply's budget when a request sets none |
| `PROMPT_CACHE_GIB` | empty (the server's 0.5) | this recipe: the prompt cache's budget a Spark; a kept state is ~7 MiB at 30-40K tokens (its rows stay in the KV pool), so `1` keeps ~140 |
| `MIN_AVAIL_GIB`, `MIN_AVAIL_WAIT_S` | 112, 30 | `start.sh` refuses to start with less memory available than this on either Spark (`0` skips the check); it re-reads the memory for up to `MIN_AVAIL_WAIT_S` seconds first, because the page cache a build or a large copy just left counts as available only once it is written back (`0` reads once) |
| `MEMGUARD_GIB`, `MEMGUARD_HOLD_S`, `MEMGUARD_HARD_GIB` | 2.5, 4, 1 | `scripts/memguard.sh` kills a rank's container when the memory available stays under the floor for `MEMGUARD_HOLD_S` seconds without coming back over it, or at once when it falls under the hard floor. A dip that recovers is logged and not killed (decimals fine; `MEMGUARD_GIB=0` turns the guard off) |
| `MEMGUARD_SPARE` | 0 | 1: before a rank's container, `scripts/memguard.sh` kills the largest other process of this user (resident plus GPU memory, 512 MiB or more; no shell, terminal, session, agent or editor server, nothing of this recipe), up to 3 in one dip, and the container only if `MemAvailable` is not back over the floor 2 s later (0.5 s under the hard floor): on a head that also runs an agent's tools or builds, one of those dies instead of the server |
| `HEAD_ENGRAM_DIR` | empty | a local folder with Engram's two shards for rank 0 (as `WORKER_ENGRAM_DIR` is on the worker); empty reads them from `HF_CACHE` |
| `HEAD_NFS_SERVER`, `HEAD_NFS_VOLUME` | empty | rank 0 reads the checkpoint over NFS through a docker volume the docker daemon mounts (no root, survives a reboot); empty uses `HF_CACHE` as a host path. sshfs does not work: it lists files as type "unknown", which the engine skips |
| `TF_DS_ENGRAM_LANES`, `TF_DS_IDX_SPLIT`, ... | unset | the engine patches' switches ([`zig/patches/README.md`](zig/patches/README.md), with this recipe's 0010-0018); a `TF_DS_*` variable you export goes to both ranks |
| `RANK1_PROFILE` | empty | `1` or `2` starts rank 1 with `--profile`, which prints its prefill phase timings when rank 0 stops first (adds a sync per chunk: for attribution, not for timing) |
| `API_KEY` | 0 | `1` makes rank 0 require a key (`Authorization: Bearer`; generated in `$STATE_DIR/api-key`, mode 600). The API is open on the network by default |
| `HOST`, `PORT`, `SERVED_NAME` | 0.0.0.0, 8888, `DeepSeek-V4.1-Flash-1017` | |
| `WORKER`, `FABRIC_PEER`, `NFS_*` | | see Requirements |
| `TF_REF`, `INPUTS_REF`, `KIT_REF` | pinned | the engine, the inputs' tools and the kit (changing one rebuilds or re-checks) |
| `PULL` | 1 | `0` never pulls a published image (`GHCR_IMAGE`, `IMAGE_TAG`, `IMAGE_DIGEST`). None is published for this recipe's patches (`exl3-tp2-39b972fbfeab`), so `scripts/prepare.sh` builds it; Mia's AI Lab's image of their pins (`ghcr.io/miaai-lab/deepseek-v4.1-flash-1017-2x-dgx-sparks-tensorfold:exl3-tp2-7c9bf5d36350`) lacks them |

Extra arguments to `./start.sh` go to rank 0's `tensorfold-native serve` after the defaults; every `TENSORFOLD_*` and
`TF_DS_*` variable you set goes to both ranks. Images are always on (rank 0 runs the vision tower).

### Warm-ahead proxy

`tools/warm_ahead_proxy.py` (optional, standard library only) sits between clients and the server:

```
LISTEN=0.0.0.0:8888 BACKEND=127.0.0.1:8000 KEY_FILE=~/.config/my-api-key python3 tools/warm_ahead_proxy.py
```

with the server on `HOST=127.0.0.1 PORT=8000` (`scripts/local.sh`). `KEY_FILE` (optional) makes it require
`Authorization: Bearer <the file's contents>`; `WARM_AHEAD=0` makes it a plain proxy. Replies pass through byte for
byte. The server keeps a prompt's state only at 2,048-token chunk ends before its last 128 tokens, so a next turn
re-prefills from the last such end. After a reply, if no client request comes for `WARM_DELAY` seconds (3), the proxy
asks the server's `/tokenize` how many tokens the conversation and the reply share with any next request and, when
that reaches a chunk end no kept state covers yet, sends those tokens up to the chunk end and 128 past it as a token
prompt (`/v1/completions`, `max_tokens 1`): the cache keeps that chunk end. It is one every next request can use, and
the state is the one the client's own request would compute, so replies are the same bits (checked against the same
request from an empty cache). After a reply ending in tool calls it starts at once (`WARM_DELAY_TOOL`, 0): the client
runs its tools meanwhile. What it learned of each client is kept in `WARM_STATE` (hashes only). A client request closes a warm-ahead in flight (the server stops at its current chunk). The proxy appends the
reply as the server sent it and learns, per client (its system prompt and tools), how the client sends replies back;
a client is warmed ahead only once its next request has matched. After the server restarts (the proxy saw it
unreachable; the prompt cache is empty), the conversations of the last `REWARM_HOURS` (2) run again, in pieces of
`REWARM_PIECE` tokens (16,384) each keeping its chunk end, while clients are quiet (30 s), so their next turns resume
where they were (`REWARM=0`: off; conversations are kept in memory only). Neither a warm-ahead nor a re-warm piece is
sent while the server has a request in flight, also one that does not come through the proxy (a warm-ahead waits up
to 2 minutes for it). Measurements: `research/results/warm-ahead.txt`.

## Memory and safety

A Spark's GPU shares its memory with the host. A rank that takes all of it does not get a clean error: the machine stops
answering and has to be power-cycled (this happened once, with the engine's reserve lowered to 10 GiB). So:

- the engine's own reserve (`TENSORFOLD_MEMORY_RESERVE_GIB`, 12.2 GiB) is left alone: do not lower it to make the weights fit;
- `start.sh` checks `MIN_AVAIL_GIB` on both Sparks before it stops or starts anything, and stops a running server before
  it builds or downloads;
- `scripts/memguard.sh` runs on both Sparks while the server is up and kills the rank's container (kill first, log after)
  on either of two levels, sampling every 0.25 s: `MemAvailable` under `MEMGUARD_HARD_GIB` (1 GiB) in two samples in a row, or under
  `MEMGUARD_GIB` (2.5 GiB) for `MEMGUARD_HOLD_S` (4) seconds without once coming back over it. A long prompt's workspace takes a few GiB
  for a moment and gives them back; such a dip is logged with its length and lowest value (`... and recovered: no kill`) and the rank
  lives. It is a brake, not a guarantee: a very fast allocation can outrun it. Its log is
  `~/.cache/dsv41-tensorfold/logs/memguard.log`;
- **what holds `MemAvailable` above the floor while a rank runs: nothing**, with `ENGINE=zig`. The 12.2 GiB reserve only
  shapes what the engine allocates when it loads (its boot line reads `4 streams at once, 0.00 GiB each ... of 0.0 GiB left after a
  12.2 GiB reserve` on a head with a 524,288-token window: nothing is left over for the streams); a prompt's prefill workspace, the
  prompt cache (512 MiB), the graphs and everything else on the Spark then draw from what `MemAvailable` still shows, and
  `memguard` kills the rank when that reaches `MEMGUARD_GIB` ([issue #4](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/4):
  a head with a desktop, sparkDash and hermes read 7 GiB right after the load and was killed at 4.085 GiB during a cache-hit turn at 265K tokens).
  `start.sh` now prints `MemAvailable` on both Sparks after the load and warns when either is within 3 GiB of the floor. On a head
  that is that close: stop what does not need to run there (the desktop session: `sudo systemctl set-default multi-user.target`
  and reboot; other containers; sparkDash and similar on the worker or another machine), keep `MIN_AVAIL_GIB` as it is, and do not
  raise `MEMGUARD_GIB` (a higher floor kills sooner). The defaults are the loose end of what has been run here (2.5 GiB floor, 4 s hold, 1 GiB
  hard floor; the long-prompt runs in Status used 2.5): a lower floor or a longer hold trades the kill for the risk of freezing the Spark. `ENGINE=jl` does check at run time: it refuses admissions under a 4 GiB hard floor, falls
  back to 2K prompt segments and runs a missing graph eagerly instead of capturing it ([docs/jl-engine.md](docs/jl-engine.md)); its boot check also sizes the pool to what the
  head can spare.
- the kernel's OOM killer does not see a rank's GPU memory (it is not the process's resident memory), and a tmpfs file is
  no process's at all: once a shared-memory writer takes what is left (here once: Nsight Compute's buffers in `/dev/shm`,
  run inside the engine), the killer ends the desktop and session processes, the user's systemd, then the rank, and frees
  nothing. `/dev/shm` is half the RAM by default; capped, such a writer gets `ENOSPC` instead: add
  `tmpfs /dev/shm tmpfs rw,nosuid,nodev,inode64,size=4G 0 0` to `/etc/fstab` (systemd-remount-fs applies it at boot) and
  run `sudo mount -o remount,size=4G /dev/shm` for now. The engine's container shares the host's (`--ipc=host`) and uses
  some 33 MiB of it. Do not profile with Nsight Compute, or run compute-sanitizer on the engine, beside a loaded rank;
- the KV cache is small (compressed FP4 with block scales; about 0.5 GiB at 262,144 tokens and ~1.8 GiB at the full window,
  by two memory readings); it is sized from `CONTEXT` when the model loads and shared by the streams. Most of the
  memory is the weights (~99 GiB, of which ~3.5 GiB are the drafter's), so the head's other software decides what is left.

## Repository layout

```
start.sh, stop.sh            start and stop both ranks
scripts/config.sh            every setting and its default (layered: environment, scripts/local.sh, .env)
scripts/nodes.sh             the worker over ssh, the CX7 link and its RoCE rails, NFS volume, NCCL settings
scripts/prepare.sh           image, downloads, kit, token map, weight files, the worker's NFS volume and disk
scripts/ablit_stock_dtypes.py gives the overlay's cast MTP tensors stock's F16 bytes back (Mia's AI Lab's)
scripts/ablit_conf_head.py   the confidence head as an fp32 file when the stock shards are gone (this recipe; and its test)
scripts/prune_ablit_stock.py ABLIT_PRUNE_STOCK=1: removes the stock shards the overlay replaced (this recipe)
scripts/memguard.sh          the memory watchdog start.sh runs on both Sparks (kills a rank's container below MEMGUARD_GIB)
scripts/zig_flags.py         start.sh's check of rank 0's serve flags against the image's `capabilities --json`
scripts/publish-image.sh     pushes the built image to ghcr.io (only on purpose)
zig/Dockerfile, zig/patches  the engine image; patches to the pinned engine (Mia's AI Lab's five, this recipe's nine)
model/chat_template.jinja    the chat template rank 0 serves
tools/validate.py            an end-to-end check of a running server
tools/bench.py               prefill, decode, multi-stream and needle measurements as JSON, with `--compare A.json B.json`
tools/exact.py               records a server's replies (token hashes) on a fixed prompt set and compares two records: proves a new image equals the old one
tools/smoke.py               a quick correctness check: health, stop strings, streaming, a tool call and its result
tools/prompt_reuse.py        a conversation resumes from its kept prompt state (from the GLM-5.3-Flash recipe)
tools/warm_ahead_proxy.py    an optional API proxy that warms the prompt cache ahead of a conversation's next turn (this recipe)
tools/toolcheck.py, needle.py, end_of_turn.py, anthropiccheck.py   single checks, from the GLM-5.3-Flash recipe
docs/zig-engine.md           how the pieces fit, and what is not verified
docs/hermes-agent.md         Hermes Agent as a client (by Steve Darlow)
research/                    this recipe's decode measurements, notes and tools (not needed to run it)
```

## License

This recipe's own code, scripts and documentation: Apache-2.0 ([`LICENSE`](LICENSE), [`NOTICE`](NOTICE)). The weights keep
their own MIT license (DeepSeek's; Mia's AI Lab's quantization). TensorFold, BertholomusAI's fork and kit, and the other
third-party parts keep theirs ([`NOTICE`](NOTICE)). This recipe's changes to Mia's AI Lab's are sehoon123's, under the same
license; `NOTICE` lists them.

## Credits

- **[Mia's AI Lab](https://github.com/MiaAI-Lab)**: the recipe this one is built on, at [`31eea46`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/commit/31eea4608f8bb292a92e2f7a1db400e928620c78): its
  scripts, documentation, measurements and engine patches 0001-0006 (Apache-2.0).
- **[DeepSeek](https://huggingface.co/deepseek-ai)**: DeepSeek-V4.1-Flash-1017, its weights, its DSpark draft blocks, its
  Engram tables and its reference inference code (MIT).
- **Mia's AI Lab** ([Mia-AiLab](https://huggingface.co/Mia-AiLab)): the EAQ v1 EXL3 2.9 bpw quantization served here,
  [`Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-1017-EXL3-EAQ-v1-2.9-bpw)
  (MIT), the earlier release
  [`Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw`](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw) whose
  per-tensor widths it equals, and the chat template (`model/chat_template.jinja`, written for that earlier release).
- **[turboderp](https://github.com/turboderp-org)**: [ExLlamaV3](https://github.com/turboderp-org/exllamav3), the EXL3 format
  and its quantizer (MIT).
- **[drowzeys](https://huggingface.co/drowzeys)**: authored the optional [Keys abliteration overlay and Mia EAQ 2.9 bpw
  application script](https://huggingface.co/drowzeys/keys-DeepSeek-V4.1-Flash-1017-Abliterated/tree/0bdc2036113c4fdda38ef6005b3cbc0bc0dfa0f0)
  used by `ABLIT=1` (MIT).
- **[TensorFold](https://github.com/ashhart/TensorFold)** by Ash Hart ([ashhart](https://github.com/ashhart)) and the
  TensorFold contributors (Apache-2.0): the native Zig engine and its OpenAI-compatible server.
- **[BertholomusAI (Albert Lee, @bertholomus)](https://github.com/bertholomus)** (Apache-2.0): the DeepSeek-V4.1-Flash
  family on TensorFold's native engine (the two-node tensor-parallel split, the EXL3 kernels' use, Engram, DSpark drafting,
  the RDMA ring, the vision tower), in their fork
  [bertholomus/TensorFold, branch `deepseek-v41-zig`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig) at
  [`3ff3b2a`](https://github.com/bertholomus/TensorFold/commit/3ff3b2ac1d71a789cb930068c46d9abed025eefa), and its
  weight-file and token-map tools, in their branch
  [`deepseek-v41-zig-tp4`](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-zig-tp4) at
  [`5940208`](https://github.com/bertholomus/TensorFold/commit/59402086381a42c74d53053443224eec2790b4c5); the kernel kit,
  the served settings and the two-node launch commands, in their recipe
  [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) at
  [`abe07c7`](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/commit/abe07c7d509a0199cc7fec15263be2aa52ac652f).
  Commits in that branch by [sayyidfareed](https://github.com/sayyidfareed) (RDMA GID and MTU),
  [Lucas Peters](https://github.com/lucas-peters) (indexer alignment; the kit's `_topk_sel` variant), lepetitprince716-prog
  (expert-run loading) and LeetJoe (image placeholder escaping).
- **[Coinupbtc](https://github.com/Coinupbtc)** (@Coinupbtc) authored `zig/patches/0005-prefill-interleave.patch`, in
  [pull request #5](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5) (commit [`27e02c6`](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/pull/5/commits/27e02c605eae0997ed9161baaf583dd3ad431bb3)): decode rounds and cancels for the other live streams between the chunks of a long prompt pass.
- **Mia's AI Lab**: this recipe's scripts, adapted from
  [GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
  (branch `zig-engine`), with code by webzone (both PCIe links of a port as twin rails), [Alexbob0](https://github.com/Alexbob0)
  (workers by host name; `tools/end_of_turn.py`) and john (each rank's rendezvous address); `tools/anthropiccheck.py` by
  Eduardo Florencio ([eduffd](https://github.com/eduffd)), and the Hermes Agent page by Steve Darlow
  ([@kerpopule](https://github.com/kerpopule)).
- **hashd1ve** ([hashd1ve](https://github.com/hashd1ve)): the change to `model/chat_template.jinja` that writes each piece of the
  prompt out as it is made instead of adding it to one growing string, which cut rank 0's memory growth with long
  conversations ([issue #1](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-1017-2x-DGX-Sparks/issues/1)).
- **Mia's AI Lab**: this recipe's own engine patches, `zig/patches/0001` and `0002` (changes to BertholomusAI's engine, written
  for this recipe, Apache-2.0): the prefill indexer split between the two ranks and a second AIO ring for Engram reads (a third, prompt-cache placement,
  was dropped when the pin moved); also `tools/exact.py` (token-hash replay check) and `scripts/memguard.sh`.
  The idea of splitting the indexer is drowzeys's, from the GLM-5.3-Flash recipe's design (no code of it was used).
  `0004` (`tool_choice` enforcement) is Mia's AI Lab's Zig call gate from its Flash Next CUDA engine, carried into the
  DeepSeek-V4.1 lanes; that gate is a port of the call gate in TensorFold's Python engine (`call_gate.py`), by the
  TensorFold contributors (Apache-2.0).
- **NVIDIA**: the DGX Spark and GB10 hardware, and the PyTorch container (`nvcr.io/nvidia/pytorch:26.07-py3`) the image is
  built on (CUDA, NCCL, cuBLAS; under NVIDIA's container license).

- **sehoon123**: this recipe's changes ([What this recipe adds](#what-this-recipe-adds)), Apache-2.0.

Full list, with commits: [`CREDITS.md`](CREDITS.md).
