#!/usr/bin/env python3
"""Regression test for ablit_conf_head.py (standard library only): python3 scripts/test_ablit_conf_head.py

Builds one-shard checkpoints whose drafter confidence head is F32 (a pure cast of F16 values), F16, F32 with a value that
is not an F16 value, the wrong size, or missing; checks the file it writes, that an equal file is left alone, that a stale
file goes when the head is F16, and the refusals.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).with_name("ablit_conf_head.py")
HEAD = "mtp.2.confidence_head.proj.weight"
N = 5376
SHARD = "model-00001-of-00001.safetensors"


def checkpoint(d: Path, tensors: dict) -> Path:
    """tensors: name -> (dtype, shape, raw bytes); written as one safetensors shard and its index."""
    d.mkdir()
    header, blob = {}, b""
    for name, (dtype, shape, raw) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(blob), len(blob) + len(raw)]}
        blob += raw
    h = json.dumps(header).encode()
    h += b" " * (-len(h) % 8)
    (d / SHARD).write_bytes(struct.pack("<Q", len(h)) + h + blob)
    (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: SHARD for k in tensors}}))
    return d


def run(ckpt: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-I", str(SCRIPT), str(ckpt), str(out)], capture_output=True, text=True)


def main() -> int:
    vals = [((i * 37) % 2001 - 1000) / 1024 for i in range(N)]  # all exact F16 values
    f32, f16 = struct.pack(f"<{N}f", *vals), struct.pack(f"<{N}e", *vals)
    other = ("blk.0.weight", ("F16", [4], struct.pack("<4e", 1, 2, 3, 4)))
    bad = list(vals)
    bad[7] = 0.1  # not an F16 value
    fails = 0

    def check(ok: bool, what: str, r: subprocess.CompletedProcess) -> None:
        nonlocal fails
        print(("ok    " if ok else "FAIL  ") + what + ("" if ok else f"  [{r.returncode}] {r.stdout}{r.stderr}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        out = t / "conf_head_f32.bin"
        c = checkpoint(t / "f32", {other[0]: other[1], HEAD: ("F32", [1, N], f32)})
        r = run(c, out)
        check(r.returncode == 0 and out.read_bytes() == f32, "F32 pure cast: the file holds the tensor's own bytes", r)
        before = out.stat().st_mtime_ns
        r = run(c, out)
        check(r.returncode == 0 and out.stat().st_mtime_ns == before, "an equal file is left alone", r)
        r = run(checkpoint(t / "f16", {HEAD: ("F16", [1, N], f16)}), out)
        check(r.returncode == 0 and not out.exists(), "F16: no file, the stale one removed", r)
        r = run(checkpoint(t / "bad", {HEAD: ("F32", [1, N], struct.pack(f"<{N}f", *bad))}), out)
        check(r.returncode == 1 and not out.exists(), "F32 that is not a pure cast of F16: refused, nothing written", r)
        r = run(checkpoint(t / "size", {HEAD: ("F32", [1, N - 1], f32[:-4])}), out)
        check(r.returncode == 1 and not out.exists(), "another shape: refused", r)
        r = run(checkpoint(t / "none", {other[0]: other[1]}), out)
        check(r.returncode == 1, "no confidence head: refused", r)
    print("PASS" if not fails else f"FAIL: {fails} check(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
