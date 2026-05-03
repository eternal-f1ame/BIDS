#!/usr/bin/env python3
"""Build train/val/test splits for the real bacterial dataset.

Splitting rule (temporal split within each video folder)
--------------------------------------------------------
For each video folder under `--frames_dir`, sort frames by filename (zero-padded
so lexicographic equals numeric) and slice:
    train: first  `--train_frac` of the frames           (default 0.80)
    val:   next   `--val_frac`   of the frames           (default 0.10)
    test:  remainder                                     (default 0.10)

Every video contributes to every split, so every species combination is
represented. The temporal split keeps adjacent frames out of train+val and avoids
the autocorrelation leak that random per-frame splits suffer from.

Label parsing
-------------
Folder names encode the multi-label via underscore separation:
`bs_ka_fj` -> {bs, ka, fj}. The canonical species list is the sorted union of
all tokens seen across all folders, so dropping in a new species video (e.g.
`xy.mp4` -> `xy/`) extends the class list on the next run.

`bs_ka` and `ka_bs` are treated as the same combination (label vector
identical) but different videos (different folder names, different splits) —
that is the right semantics for two recording sessions of the same mix.

Output
------
Writes `--output` (default `data/real/splits.json`):
{
    "class_names": ["bs", "bt", "fj", "ka", "mx", "pf"],
    "train_frac": 0.80, "val_frac": 0.10, "test_frac": 0.10,
    "splits": {
        "train": [{"path": "...", "label": [1,0,0,1,0,0], "video": "bs_ka"}, ...],
        "val":   [...],
        "test":  [...]
    },
    "video_to_label": {"bs": [1,0,0,0,0,0], "bs_ka": [1,0,0,1,0,0], ...}
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}

# Retake suffix like "take2", "take3" -- not a species, discard when parsing labels.
_TAKE_RE = re.compile(r"^take\d+$", re.IGNORECASE)


def parse_label_tokens(folder_name: str) -> List[str]:
    """Split a folder name like 'b_f_k_p' into ['b','f','k','p']. Empty parts stripped.
    Retake suffixes like 'take2' are stripped so a retake folder name
    'bs_bt_ka_fj_take2' yields the same tokens as 'bs_bt_ka_fj'."""
    return [tok for tok in folder_name.split("_") if tok and not _TAKE_RE.match(tok)]


def discover_class_names(frames_dir: Path) -> List[str]:
    """Sorted union of all tokens across all video folders."""
    classes = set()
    for folder in frames_dir.iterdir():
        if not folder.is_dir():
            continue
        classes.update(parse_label_tokens(folder.name))
    return sorted(classes)


def label_vector(folder_name: str, class_names: List[str]) -> List[int]:
    tokens = set(parse_label_tokens(folder_name))
    return [1 if c in tokens else 0 for c in class_names]


def list_frames(folder: Path) -> List[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def temporal_split(
    frames: List[Path],
    train_frac: float,
    val_frac: float,
) -> Tuple[List[Path], List[Path], List[Path]]:
    n = len(frames)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = frames[:n_train]
    val = frames[n_train : n_train + n_val]
    test = frames[n_train + n_val :]
    return train, val, test


def build_splits(
    frames_dir: Path,
    output: Path,
    train_frac: float,
    val_frac: float,
) -> None:
    if not frames_dir.exists():
        raise SystemExit(f"frames_dir does not exist: {frames_dir}")
    if abs((train_frac + val_frac) - 1.0) > 1.0 and not (0 < train_frac + val_frac <= 1.0):
        raise SystemExit("train_frac + val_frac must be in (0, 1]")
    test_frac = 1.0 - train_frac - val_frac

    class_names = discover_class_names(frames_dir)
    if not class_names:
        raise SystemExit(f"No class folders found in {frames_dir}")
    print(f"Discovered {len(class_names)} species: {class_names}")

    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    video_to_label: Dict[str, List[int]] = {}

    folders = sorted(p for p in frames_dir.iterdir() if p.is_dir())
    for folder in folders:
        frames = list_frames(folder)
        if not frames:
            print(f"  WARN: {folder.name} has no images, skipping")
            continue
        label = label_vector(folder.name, class_names)
        video_to_label[folder.name] = label

        train, val, test = temporal_split(frames, train_frac, val_frac)
        for path in train:
            splits["train"].append({"path": str(path), "label": label, "video": folder.name})
        for path in val:
            splits["val"].append({"path": str(path), "label": label, "video": folder.name})
        for path in test:
            splits["test"].append({"path": str(path), "label": label, "video": folder.name})

        print(f"  {folder.name:15s} ({sum(label)} sp.) "
              f"-> train {len(train):4d}  val {len(val):3d}  test {len(test):3d}")

    payload = {
        "class_names": class_names,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "splits": splits,
        "video_to_label": video_to_label,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    n_train, n_val, n_test = len(splits["train"]), len(splits["val"]), len(splits["test"])
    n_total = n_train + n_val + n_test
    print(f"\nWrote {output}")
    print(f"  total frames: {n_total}")
    print(f"  train:        {n_train} ({n_train/max(n_total,1):.1%})")
    print(f"  val:          {n_val} ({n_val/max(n_total,1):.1%})")
    print(f"  test:         {n_test} ({n_test/max(n_total,1):.1%})")
    print(f"  videos:       {len(video_to_label)}")
    print(f"  class_names:  {class_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build temporal train/val/test splits for real data.")
    parser.add_argument("--frames_dir", type=Path, default=Path("data/real/frames"))
    parser.add_argument("--output", type=Path, default=Path("data/real/splits.json"))
    parser.add_argument("--train_frac", type=float, default=0.80)
    parser.add_argument("--val_frac", type=float, default=0.10)
    args = parser.parse_args()

    build_splits(args.frames_dir, args.output, args.train_frac, args.val_frac)


if __name__ == "__main__":
    main()
