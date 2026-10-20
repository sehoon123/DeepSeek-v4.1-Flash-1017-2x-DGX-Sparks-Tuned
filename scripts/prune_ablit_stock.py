#!/usr/bin/env python3
"""Remove only stock HF shards replaced by the verified Mia-2.9 Ablit graft.

The grafted checkpoint hardlinks every unchanged shard and writes each changed shard separately. This helper records an
atomic plan before unlinking anything, removes the stock snapshot links for the 24 changed shards, and removes their blob
only when no other snapshot link still resolves to it. It is resumable and refuses any shape other than 104 wo_b tensors
in layers 10..35 across 24 shards.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import time


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def other_snapshot_ref(model_dir: Path, source: Path, blob: Path) -> bool:
    for p in (model_dir / "snapshots").glob("*/*"):
        if p == source or not p.is_symlink():
            continue
        try:
            if p.resolve(strict=True) == blob:
                return True
        except FileNotFoundError:
            pass
    return False


def make_plan(base: Path, graft: Path, plan_path: Path) -> dict:
    if not (graft / ".ablit-ready").is_file():
        raise RuntimeError(f"not a verified graft: {graft}/.ablit-ready is absent")
    meta = json.loads((graft / "ABLIT_META.json").read_text())
    if meta.get("n_edited") != 104:
        raise RuntimeError(f"expected n_edited=104, got {meta.get('n_edited')!r}")
    index = json.loads((base / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    matched = []
    for key, shard in weight_map.items():
        m = re.search(r"(?:^|\.)layers\.(\d+)\.attn\.wo_b\.", key)
        if m and 10 <= int(m.group(1)) <= 35:
            matched.append((key, shard))
    shards = sorted({shard for _, shard in matched})
    if len(matched) != 104 or len(shards) != 24:
        raise RuntimeError(f"refusing unexpected graft shape: {len(matched)} tensors / {len(shards)} shards")
    model_dir = base.parent.parent
    blob_root = (model_dir / "blobs").resolve()
    rows = []
    for name in shards:
        source = base / name
        target = source.resolve(strict=True)
        try:
            target.relative_to(blob_root)
        except ValueError as exc:
            raise RuntimeError(f"stock shard is outside this model's blob directory: {source} -> {target}") from exc
        changed = graft / name
        if not changed.is_file():
            raise RuntimeError(f"grafted shard missing: {changed}")
        if os.stat(source).st_ino == os.stat(changed).st_ino and os.stat(source).st_dev == os.stat(changed).st_dev:
            raise RuntimeError(f"grafted shard still shares the stock inode: {name}")
        rows.append({"name": name, "snapshot_link": str(source), "blob": str(target), "bytes": target.stat().st_size})
    plan = {
        "schema": 1,
        "created": time.strftime("%FT%T%z"),
        "base": str(base),
        "graft": str(graft),
        "matched_tensors": len(matched),
        "shards": rows,
        "planned_bytes": sum(r["bytes"] for r in rows),
    }
    atomic_json(plan_path, plan)
    return plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("graft", type=Path)
    ap.add_argument("marker", type=Path)
    args = ap.parse_args()
    base, graft, marker = (p.resolve() for p in (args.base, args.graft, args.marker))
    if marker.exists():
        done = json.loads(marker.read_text())
        print(f"stock graft sources already pruned: {done['removed_shards']} shards, {done['unlinked_bytes']} bytes")
        return
    plan_path = marker.with_suffix(marker.suffix + ".plan")
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else make_plan(base, graft, plan_path)
    model_dir = Path(plan["base"]).parent.parent
    removed = 0
    unlinked_bytes = 0
    kept_shared = []
    for row in plan["shards"]:
        source, blob = Path(row["snapshot_link"]), Path(row["blob"])
        source.unlink(missing_ok=True)
        if blob.exists() and not other_snapshot_ref(model_dir, source, blob):
            blob.unlink()
            unlinked_bytes += int(row["bytes"])
        elif blob.exists():
            kept_shared.append(str(blob))
        removed += 1
    done = {
        **plan,
        "finished": time.strftime("%FT%T%z"),
        "removed_shards": removed,
        "unlinked_bytes": unlinked_bytes,
        "shared_blobs_kept": kept_shared,
    }
    atomic_json(marker, done)
    print(f"pruned {removed} replaced stock shards; unlinked {unlinked_bytes} bytes")


if __name__ == "__main__":
    main()
