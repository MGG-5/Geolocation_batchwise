#!/usr/bin/env python3
"""Reconstruct the exact 30,000-image OSV-5M training subset from HuggingFace.

The subset is already fully determined: which images are in it is recorded in
metadata.csv, so this script does NOT redo the pipeline that produced it (no
train.csv, no perceptual hashing, no sampling). It downloads the three source
shards, extracts only the listed IDs, and verifies the result byte-for-byte.

The images are written verbatim out of the source zips, never re-encoded, so a
correct run reproduces the dataset bit-for-bit and the fingerprint below will
match exactly.

    pip install huggingface_hub
    python fetch_osv5m_subset.py --metadata metadata.csv --out osv5m_train_30k_b

Downloads ~7.6 GB, peaks at ~4.1 GB on disk (one shard zip at a time, deleted
immediately after extraction), and leaves ~1.5 GB of images.

Dataset: osv5m/osv5m, CC-BY-SA 4.0. Attribute accordingly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("huggingface_hub is required:  pip install huggingface_hub")

REPO_ID = "osv5m/osv5m"
REPO_TYPE = "dataset"

# Pinned so this is reproducible even if the dataset is updated upstream.
REVISION = "cff33609b56b54d8743b7ee7a416eb8433e9a681"
SHARDS = ("08", "48", "88")

EXPECTED_COUNT = 30_000
# sha256 over "<image_id>:<md5 of file bytes>\n" for all ids, sorted by id.
EXPECTED_FINGERPRINT = \
    "5af5a9aa4ac5b399e075ed876466fa5100c7a0d39933e33c329bcc8b9c7b9eea"

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)


def tree_size(*paths: Path) -> int:
    total = 0
    for root in paths:
        if root.is_file():
            total += root.stat().st_size
        elif root.is_dir():
            for dirpath, _, names in os.walk(root):
                for n in names:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, n))
                    except OSError:
                        pass
    return total


def fingerprint(img_dir: Path) -> tuple[str, dict[str, str]]:
    per_file = {}
    for p in sorted(img_dir.glob("*.jpg")):
        per_file[p.stem] = hashlib.md5(p.read_bytes()).hexdigest()
    agg = hashlib.sha256()
    for i in sorted(per_file):
        agg.update(f"{i}:{per_file[i]}\n".encode())
    return agg.hexdigest(), per_file


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", default="metadata.csv",
                    help="CSV with an image_id column (default: metadata.csv)")
    ap.add_argument("--out", default="osv5m_train_30k_b",
                    help="output folder; images land in <out>/images")
    ap.add_argument("--work", default="osv5m_fetch_tmp",
                    help="scratch dir for shard zips, removed on success")
    ap.add_argument("--verify-only", action="store_true",
                    help="skip downloading; just verify an existing --out folder")
    args = ap.parse_args()

    meta_path = Path(args.metadata).resolve()
    out = Path(args.out).resolve()
    work = Path(args.work).resolve()
    img_dir = out / "images"

    if not meta_path.exists():
        sys.exit(f"metadata not found: {meta_path}")

    with open(meta_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if "image_id" not in (reader.fieldnames or []):
            sys.exit(f"'image_id' column not found; got {reader.fieldnames}")
        wanted = {row["image_id"].strip() for row in reader}
    wanted.discard("")
    log(f"metadata lists {len(wanted):,} image ids")
    if len(wanted) != EXPECTED_COUNT:
        log(f"WARNING: expected {EXPECTED_COUNT:,} ids, metadata has {len(wanted):,}. "
            f"The fingerprint check will not match unless this is the original file.")

    img_dir.mkdir(parents=True, exist_ok=True)
    peak = 0

    if not args.verify_only:
        work.mkdir(parents=True, exist_ok=True)
        have = {p.stem for p in img_dir.glob("*.jpg")}
        if have:
            log(f"resuming: {len(have):,} images already present")
        todo = wanted - have

        for shard in SHARDS:
            if not todo:
                log("all images already present, no further downloads needed")
                break
            log(f"shard {shard}: downloading images/train/{shard}.zip (~2.52 GB)")
            zpath = hf_hub_download(REPO_ID, f"images/train/{shard}.zip",
                                    repo_type=REPO_TYPE, revision=REVISION,
                                    local_dir=str(work))
            peak = max(peak, tree_size(work, out))

            n = 0
            with zipfile.ZipFile(zpath) as zf:
                for name in zf.namelist():
                    if not name.endswith(".jpg"):
                        continue
                    img_id = name.rsplit("/", 1)[-1][:-4]
                    if img_id not in todo:
                        continue
                    # written verbatim - no decode, no re-encode
                    with open(img_dir / f"{img_id}.jpg", "wb") as dst:
                        dst.write(zf.read(name))
                    todo.discard(img_id)
                    n += 1
            peak = max(peak, tree_size(work, out))
            os.remove(zpath)
            shutil.rmtree(work / ".cache", ignore_errors=True)
            log(f"shard {shard}: extracted {n:,} images, zip deleted "
                f"({len(todo):,} still outstanding)")

        shutil.rmtree(work, ignore_errors=True)
        if todo:
            log(f"ERROR: {len(todo):,} ids were not found in shards {SHARDS}. "
                f"First few: {sorted(todo)[:5]}")
            return 1

    # ---- verification -------------------------------------------------------
    log("verifying...")
    got = {p.stem for p in img_dir.glob("*.jpg")}
    missing, extra = wanted - got, got - wanted
    fp, per_file = fingerprint(img_dir)

    print()
    print(f"  images on disk : {len(got):,}")
    print(f"  missing        : {len(missing):,}")
    print(f"  unexpected     : {len(extra):,}")
    print(f"  fingerprint    : {fp}")
    print(f"  expected       : {EXPECTED_FINGERPRINT}")
    if peak:
        print(f"  peak disk      : {peak / 1e9:.2f} GB")
    print(f"  final size     : {tree_size(out) / 1e9:.2f} GB")

    ok = not missing and not extra and fp == EXPECTED_FINGERPRINT
    if ok:
        print("\nOK - byte-for-byte identical to the original 30,000-image subset.")
        return 0

    print("\nMISMATCH - this is NOT the same dataset.")
    ref = Path(__file__).parent / "checksums.md5"
    if ref.exists() and not missing and not extra:
        expected = {}
        for line in ref.read_text().splitlines():
            m, f = line.split("  ", 1)
            expected[f[:-4]] = m
        bad = [i for i, m in per_file.items() if expected.get(i) != m]
        print(f"  {len(bad)} file(s) differ in content; first few: {sorted(bad)[:5]}")
        print("  Re-run to re-download, or delete those files and re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
