import json
import os
from typing import List, Tuple

import numpy as np


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_real_splits(splits_path: str) -> dict:
    """Load the full splits JSON produced by tools/build_splits.py."""
    return load_json(splits_path)


def load_real_split(
    splits_path: str,
    split: str,
) -> Tuple[List[str], np.ndarray, List[str], List[str]]:
    """Load a single split (train/val/test) of the real dataset.

    Returns
    -------
    image_paths : list of str
    labels      : (N, K) int64 multilabel array, columns ordered by class_names
    class_names : list of K species names (canonical sorted order)
    video_ids   : list of N strings — folder/video name each frame came from.
                  Useful for video-level cross-validation later.
    """
    payload = load_real_splits(splits_path)
    if split not in payload["splits"]:
        raise ValueError(f"split must be one of {list(payload['splits'].keys())}; got {split!r}")

    entries = payload["splits"][split]
    class_names: List[str] = payload["class_names"]
    image_paths: List[str] = []
    labels: List[List[int]] = []
    video_ids: List[str] = []
    for e in entries:
        image_paths.append(e["path"])
        labels.append(e["label"])
        video_ids.append(e["video"])

    return image_paths, np.asarray(labels, dtype=np.int64), class_names, video_ids
