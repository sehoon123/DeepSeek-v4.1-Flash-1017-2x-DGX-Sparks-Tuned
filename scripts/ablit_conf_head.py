#!/usr/bin/env python3
"""ablit_conf_head.py CHECKPOINT OUT: the drafter's confidence head as an fp32 file, for an ABLIT=1 checkpoint whose stock
shards are gone (this recipe; scripts/prepare.sh runs it instead of scripts/ablit_stock_dtypes.py then).

The Keys overlay's apply_mia29_wo_b_graft.py casts mtp.2.confidence_head.proj.weight from F16 to F32, and this engine reads
that tensor as F16 whatever the file holds: from the F32 copy the drafter's confidence scores are noise and it drafts far
too little. scripts/ablit_stock_dtypes.py gives the tensor stock's F16 bytes back, but it compares with the stock shards,
and ABLIT_PRUNE_STOCK=1 removes them. This script only reads CHECKPOINT (a header and the tensor's 21 KB; standard library):

  - F16 (the dtype step ran before the pruning): nothing to do, and a stale OUT is removed;
  - F32: checks that it holds 5,376 finite values (5,120 hidden + 256 Markov features), each exactly an F16 value (a pure
    cast: no new values), and writes them unchanged, as raw little-endian fp32, to OUT (atomically; left alone when it is
    equal already). start.sh then sets TF_DS_CONF_FILE to OUT, and zig/patches/0018 loads it on both ranks in place of
    the checkpoint's head.

Exits 1 on anything else (no such tensor, another dtype or shape, a value that is not an F16 value), 2 on bad usage.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import sys
from pathlib import Path

HEAD = "mtp.2.confidence_head.proj.weight"
N = 5376


def fail(msg: str) -> int:
    print(f"ablit_conf_head.py: {msg}", file=sys.stderr)
    return 1


def is_f16(v: float) -> bool:
    try:
        return struct.unpack("<e", struct.pack("<e", v))[0] == v
    except (OverflowError, struct.error):
        return False


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: ablit_conf_head.py CHECKPOINT OUT", file=sys.stderr)
        return 2
    ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
    shard = json.loads((ckpt / "model.safetensors.index.json").read_text())["weight_map"].get(HEAD)
    if shard is None:
        return fail(f"{HEAD} is not in {ckpt}/model.safetensors.index.json")
    with open(ckpt / shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n)).get(HEAD)
        if meta is None:
            return fail(f"{HEAD} is not in {shard}")
        dtype, shape, (a, b) = meta["dtype"], meta["shape"], meta["data_offsets"]
        if dtype == "F16":
            note = ""
            if out.exists():
                out.unlink()
                note = f" (removed {out})"
            print(f"{HEAD} is F16 in {shard}: the engine reads it from the checkpoint{note}")
            return 0
        if dtype != "F32" or math.prod(shape) != N or b - a != 4 * N:
            return fail(f"{HEAD} in {shard} is {dtype} {shape}; expected F16, or F32 with {N} values")
        f.seek(8 + n + a)
        raw = f.read(4 * N)
    if len(raw) != 4 * N:
        return fail(f"{shard} ends inside {HEAD}")
    for i, v in enumerate(struct.unpack(f"<{N}f", raw)):
        if not math.isfinite(v):
            return fail(f"{HEAD}[{i}] is {v}")
        if not is_f16(v):
            return fail(f"{HEAD}[{i}] = {v!r} is not an F16 value: the F32 tensor is not a pure cast of F16 weights")
    sha = hashlib.sha256(raw).hexdigest()[:12]
    if out.exists() and out.read_bytes() == raw:
        print(f"{HEAD} is F32 in {shard}: {out} is up to date (sha256 {sha})")
        return 0
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_bytes(raw)
    os.chmod(tmp, 0o644)
    os.replace(tmp, out)
    print(f"{HEAD} is F32 in {shard}: wrote its {N} values to {out} (sha256 {sha}) for TF_DS_CONF_FILE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
