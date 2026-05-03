#!/usr/bin/env python3
"""Refresh data/release/croissant.json with real SHA-256 hashes and URLs.

Reads the existing croissant.json, walks the listed FileObjects, computes
the SHA-256 of any file present under data/release/, and rewrites the
contentUrl field by joining --hosting-url-base with the file's basename.

Usage:

    python tools/build_croissant.py \
        --hosting-url-base https://huggingface.co/datasets/PLACEHOLDER_ANON_USER/BIDS/resolve/main \
        --release-dir data/release

If a FileObject's file is missing locally the hash is left as the
PLACEHOLDER value and a warning is printed; this lets you regenerate
incrementally as archives finish building.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "data" / "release"
DEFAULT_CROISSANT = DEFAULT_RELEASE / "croissant.json"

CHUNK = 1 << 20  # 1 MiB


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE,
                    help="Directory containing the release files.")
    ap.add_argument("--croissant", type=Path, default=DEFAULT_CROISSANT,
                    help="Path to the Croissant JSON-LD file to rewrite in place.")
    ap.add_argument("--hosting-url-base", required=True,
                    help="Base URL where the release files will live, "
                         "e.g. https://huggingface.co/datasets/PLACEHOLDER_ANON_USER/BIDS/resolve/main")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.croissant.exists():
        sys.exit(f"croissant.json not found at {args.croissant}")

    cr = json.loads(args.croissant.read_text())

    base = args.hosting_url_base.rstrip("/")

    n_updated = 0
    n_missing = 0
    for obj in cr.get("distribution", []):
        if obj.get("@type") != "cr:FileObject":
            continue
        name = obj.get("name")
        if not name:
            continue
        local = args.release_dir / name
        new_url = f"{base}/{name}"
        if obj.get("contentUrl") != new_url:
            obj["contentUrl"] = new_url
        if local.exists():
            digest = sha256_of(local)
            if obj.get("sha256") != digest:
                obj["sha256"] = digest
                n_updated += 1
                print(f"  updated {name}: sha256={digest[:16]}... ({local.stat().st_size} bytes)")
        else:
            n_missing += 1
            print(f"  WARN  {name}: file not present at {local}; sha256 left as placeholder")

    cr["url"] = base.rsplit("/resolve/", 1)[0] if "/resolve/" in base else base

    if args.dry_run:
        print("\n[dry-run] would write", args.croissant)
        print(json.dumps(cr, indent=2)[:1000], "...")
    else:
        args.croissant.write_text(json.dumps(cr, indent=2) + "\n")
        print(f"\nWrote {args.croissant} ({n_updated} hashes refreshed, "
              f"{n_missing} files missing)")
        if n_missing:
            print("Run tools/build_release_archives.py first to materialise "
                  "the missing archives, then re-run this script.")


if __name__ == "__main__":
    main()
