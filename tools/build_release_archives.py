#!/usr/bin/env python3
"""Build the release archives for the PHOEBI dataset.

Produces three gzipped tar archives plus a copy of splits.json under
data/release/, ready to upload to Hugging Face (or any of the four NeurIPS
preferred hosts):

    data/release/frames.tar.gz   — 40 combo folders, ~40,000 JPEGs
    data/release/videos.tar.gz   — 40 primary MP4 videos
    data/release/retakes.tar.gz  — 16 retake videos + 16,000 retake frames
    data/release/splits.json     — copy of data/splits.json

Each archive root contains the original folder name (frames/, videos/,
retakes/) so that extraction with `tar xzf` produces the conventional
on-disk layout.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "real"
OUT = ROOT / "data" / "release"


def _humansize(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _build_archive(src: Path, archive_path: Path, arcname: str,
                   compresslevel: int = 1) -> None:
    if not src.exists():
        print(f"  skip — {src} does not exist")
        return
    print(f"  packing {src} -> {archive_path.name} (gzip level {compresslevel})",
          flush=True)
    t0 = time.time()
    with tarfile.open(archive_path, "w:gz", compresslevel=compresslevel) as tf:
        tf.add(src, arcname=arcname)
    dt = time.time() - t0
    sz = archive_path.stat().st_size
    print(f"    {_humansize(sz)} in {dt:.1f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT,
                    help="Destination directory for release archives.")
    ap.add_argument("--skip-frames", action="store_true",
                    help="Skip building frames.tar.gz (it is the largest).")
    ap.add_argument("--skip-videos", action="store_true")
    ap.add_argument("--skip-retakes", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits_src = DATA / "splits.json"
    splits_dst = args.out_dir / "splits.json"
    if splits_src.exists():
        shutil.copy2(splits_src, splits_dst)
        print(f"copied {splits_src.relative_to(ROOT)} -> {splits_dst.relative_to(ROOT)}")
    else:
        print(f"WARN: {splits_src} missing")

    print("\nBuilding archives:")
    if not args.skip_frames:
        _build_archive(DATA / "frames", args.out_dir / "frames.tar.gz", "frames")
    if not args.skip_videos:
        _build_archive(DATA / "videos", args.out_dir / "videos.tar.gz", "videos")
    if not args.skip_retakes:
        # Retakes ship as a single archive containing both subfolders so users
        # can re-extract under different sampling.
        retakes_root = args.out_dir / "_retakes_staging"
        if retakes_root.exists():
            shutil.rmtree(retakes_root)
        retakes_root.mkdir()
        if (DATA / "videos_retakes").exists():
            shutil.copytree(DATA / "videos_retakes", retakes_root / "videos_retakes")
        if (DATA / "frames_retakes").exists():
            shutil.copytree(DATA / "frames_retakes", retakes_root / "frames_retakes")
        if any(retakes_root.iterdir()):
            t0 = time.time()
            with tarfile.open(args.out_dir / "retakes.tar.gz", "w:gz",
                              compresslevel=1) as tf:
                for sub in sorted(retakes_root.iterdir()):
                    tf.add(sub, arcname=sub.name)
            sz = (args.out_dir / "retakes.tar.gz").stat().st_size
            print(f"  packed retakes -> retakes.tar.gz "
                  f"({_humansize(sz)} in {time.time()-t0:.1f}s)")
        else:
            print("  skip — no retake data found")
        shutil.rmtree(retakes_root)

    print("\nDone. Run `python tools/build_croissant.py` next to refresh "
          "hashes in croissant.json.")


if __name__ == "__main__":
    main()
