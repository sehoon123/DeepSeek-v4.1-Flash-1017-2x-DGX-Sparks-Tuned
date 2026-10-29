# The embedding table in the GB10 display reservation (0056) and a 2,097,152-position pool

Date: 2026-10-29. Measured on this recipe's pair: two MSI EdgeXpert (MS-C931) GB10 Sparks, TP2, Mia EAQ 2.9 bpw with the
Keys overlay, `PARALLEL=4`, `PROMPT_CACHE_GIB=1`, `CONTEXT=1048576`. Synthetic prompts only, public proxy stopped.

## Why

A Spark's firmware keeps part of its memory for a screen: the scanout carve-out, ~2 GiB, which Linux and ordinary CUDA
allocations never use and `MemAvailable` never counts. A headless Spark leaves it idle. Meanwhile each rank of this
engine holds the model's whole BF16 embedding table (129,280 x 5,120, 1,262.5 MiB) in ordinary memory, and the head is
the Spark with the least room (it also runs the server and the vision tower). Moving that table into the carve-out gives
the head (and the worker) back ~1.23 GiB, which this release spends on a larger shared KV pool.

The newer firmware option for the same reservation (DGX Spark OTA2610, UEFI "Display Reserved Memory": Auto / 0.5 GB,
NVIDIA's DGX Spark User Guide) shrinks it to 0.5 GB on a headless Spark. It was not available here: these MSI nodes run
OTA2607 (UEFI 5.36_1.8.0), and neither the stable nor the testing LVFS remote had a newer MSI MS-C931 firmware. The two
approaches add up: once the firmware option exists, the reservation can be shrunk to what the span needs.

## Prior art

- coolbho3k (emihuang) found that the headless GB10 display reservation is reachable from CUDA
  (NVIDIA developer forum, 2026-09).
- christopherowen/spark-ds41f, `experiments/2026-09-29-display-carveout-kv` (Apache-2.0), measured the access paths:
  a DRM dumb buffer exported as a dma-buf and imported with `cuImportExternalMemory` streams at `cudaMalloc`'s speed
  (235 GB/s sequential reads), but small random reads are slow (7 GB/s for 576-byte records). KV caches placed there
  lost 4-11 % on gathers; the embedding and output head placed there cost nothing measurable. 0056 follows that
  placement: only a table read once a step goes there.
- glm53-zig-engine's 0072 (by ezoushen) uses the same reservation for KV in GLM-5.3's pool.

## What changed

- `zig/patches/0056-dsv41-display-embed.patch`: with `TF_DS_DISPLAY_EMBED=1`, after the weights load and before any
  kernel or CUDA graph takes the table's address, each rank checks `/sys/class/drm/card0-*/status` (no display
  connected), creates a DRM dumb buffer of the table's size on `/dev/dri/card0` (4096 x N x 32 bpp, whole 16 MiB),
  exports it as a dma-buf, imports it with `cuImportExternalMemory`, copies the table there, compares the two copies
  byte for byte, frees the ordinary buffer and points the embedding lookups (prompt chunks, decode rounds, drafter) at
  the span. Off by default. Fails closed: with the switch on, modeset off, a connected display, too little room or no
  card device stops the load.
- `scripts/config.sh` / `start.sh`: `DISPLAY_EMBED` (0), `DISPLAY_EMBED_FALLBACK_POOL` (empty), `DISPLAY_CARD`
  (`/dev/dri/card0`). With `DISPLAY_EMBED=1`, start.sh probes both Sparks after the previous server stops
  (`scripts/display_probe.py`, in the engine's image as root with the card: modeset on, no display connected, a dumb
  buffer of the table's size and its export). Both pass: `--device /dev/dri/card0` and the switch on both ranks. Either
  fails: the known-good layout instead (no span; `POOL=DISPLAY_EMBED_FALLBACK_POOL` when set), with a warning saying why.
- The host needs nvidia-drm with modeset on. DGX OS ships `nvidia-drm-options-modeset0`
  (`/etc/modprobe.d/zz-nvidia-drm-override.conf`, `modeset=0`, "for system compatibility"); this site added a later
  file, `/etc/modprobe.d/zzz-tensorfold-nvidia-drm-modeset.conf` with `options nvidia-drm modeset=1`
  (`modprobe --show-depends nvidia-drm` reads `modeset=1 modeset=0 modeset=1`: the last wins), ran
  `update-initramfs -u` and rebooted both nodes. Rollback: delete that file, `update-initramfs -u`, reboot.

## Evidence

**The reservation on these nodes (MSI EdgeXpert, modeset on, headless).** Both nodes: five connectors (HDMI and four
USB-C), all disconnected; the largest dumb buffer at least 2,040 MiB; a 1,264 MiB buffer and its dma-buf export both
succeeded; `MemTotal` unchanged (124,544 MiB): the buffer is not Linux memory. The worker ran the deployed engine at
modeset on before the head rebooted (smoke test passed).

**Phase 1: `DISPLAY_EMBED=1`, `POOL` unset, against the deployed 0055's complete run at the same pool.**

- All 34 complete output token arrays identical: 9 long-context (683,602-token placement with the blocked-placement
  restore at 679,936; an 850,123-token cold pass through the SDK with 25 keep-alives and at most 15.1 s between body
  bytes; the 45 s cancel at 196,608 tokens resuming at 65,536), 20 ordinary (cache, pressure, four-stream, nested), 5
  feature rows.
- Both ranks logged `display reservation (zig/patches/0056): embed.weight's 1262.5 MiB there (1264.0 MiB span on
  /dev/dri/card0, 5 connectors, none connected)`.
- Rank 0's device memory at load: 106.12 GiB (0055) -> 104.89 GiB (0056), -1.230 GiB, the table's 1.233 GiB.
- Cold prefill: 648,878 tokens 280.57 -> 280.26 s (x0.9989), 850,123 tokens 382.00 -> 380.51 s (x0.9961). Decode on
  the eight ordinary rows of 32-36 tokens: x1.028-x1.149, median x1.058 against 0055. These are short rows run at a
  different time; the span cannot make decode faster, so this is read as no slowdown, not as a speed-up.

**Phase 2: `DISPLAY_EMBED=1`, `POOL=2097152`.** Three synthetic sessions of 600,000 / 602,000 / 604,000 tokens,
alternating:

| turn | cached | prefill |
| --- | ---: | ---: |
| A1 / B1 / C1 (cold) | 0 | 256.0 / 255.2 / 255.4 s |
| A2 / B2 / C2 | 598,016 | 2.48 / 2.48 / 2.49 s |
| A3 / B3 / C3 | 600,064 | 2.43 / 2.44 / 2.44 s |

A and B are the pool window's frozen prompts; their complete outputs equal the deployed 0055's at `POOL=1572864`.
Features, the stock smoke and validate tools and an owned decode-disconnect recovery check passed.

**Memory.**

| | 0055, `POOL=1572864` (before) | 0056, `POOL=2097152` |
| --- | ---: | ---: |
| `MemAvailable` right after the load, head / worker | 6,291 / 8,820 MiB | 6,750 / 9,068 MiB |
| engine GPU memory (nvidia-smi), head / worker | 109,949 / 108,532 MiB | 109,710 / 108,293 MiB |
| `MemAvailable` minima under the window's load, head / worker | 4,892 / 6,917 MiB (two sessions) | 5,713 / 7,857 MiB (three sessions) |

The pool's step from 1,572,864 to 2,097,152 positions cost ~1,023 MiB a rank (the KV planes and the buffers sized by
the pool); the table's move freed 1,262.5 MiB. Net: a third more pool with ~0.25-0.46 GiB more `MemAvailable` right
after the load than before, and minima ~0.8-0.9 GiB higher under a heavier test (three long sessions instead of two).

## Limits

- Headless Sparks only. A display connected at start (or modeset off) makes start.sh fall back to the known-good
  layout (no span, `POOL=1572864` here), and says so.
- The span is about 1.26 of the reservation's ~2 GiB; the console framebuffer of a display plugged in later takes
  8-33 MiB of the rest. Not tested with a display plugged in while the engine runs.
- Three sessions fit `POOL=2097152` while each is at most ~643K tokens with a 32,768-token reply budget (three
  extents of the prompt plus the budget, rounded to 2,048, plus placement's 32,768 positions of room after the first
  two); two sessions fit up to ~999K tokens each. A fourth long session can still retire a kept state.
- Long-term stability of DRM dumb buffers held for days by a CUDA process is not yet observed beyond this window.
