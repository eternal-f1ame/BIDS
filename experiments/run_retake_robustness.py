#!/usr/bin/env python3
"""Retake robustness: evaluate primary-trained models on the 16 retake videos.

Report per-sample F1 on the retake frames for both Method A (simplex unmix) and
Method B (prototype matching), and compare against the primary-test F1 to
quantify distribution shift from slide preparation, focus, and culture density
variation.

Reads trained models from:
  outputs/simplex_unmixing/6class/
  outputs/prototype_matching/6class/

Writes:
  outputs/retake_robustness/results.json
  outputs/retake_robustness/summary.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.features import (
    extract_features_multicrop_gpu,
    scatter_mean_by_image,
)
from src.common.tiling import TileConfig
from src.common.metrics import per_sample_f1, macro_f1_per_class, exact_match_accuracy
from src.simplex_unmixing.model import UnmixerModel, ModelConfig as UnmixerConfig
from src.prototype_matching.model import PrototypeMatchingModel, ProtoConfig
from tools.build_splits import parse_label_tokens


def enumerate_retake_frames(retakes_dir: Path) -> Tuple[List[Path], List[str]]:
    """Return (paths, folder_stems) for every .jpg under retakes_dir."""
    paths: List[Path] = []
    folders: List[str] = []
    for folder in sorted(retakes_dir.iterdir()):
        if not folder.is_dir():
            continue
        for frame in sorted(folder.glob("*.jpg")):
            paths.append(frame)
            folders.append(folder.name)
    return paths, folders


def build_labels(folders: List[str], class_names: List[str]) -> np.ndarray:
    K = len(class_names)
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    out = np.zeros((len(folders), K), dtype=np.int64)
    for i, folder in enumerate(folders):
        for tok in parse_label_tokens(folder):
            if tok in cls_to_idx:
                out[i, cls_to_idx[tok]] = 1
    return out


def score_simplex(cfg_path: Path, model_path: Path, tile_feats: torch.Tensor,
                  image_index: torch.Tensor, num_images: int,
                  device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (image-level weights (N, K), image-level residual norms (N,))."""
    cfg = json.loads(cfg_path.read_text())
    mcfg = UnmixerConfig(
        embedding_dim=cfg["embedding_dim"],
        num_prototypes=cfg["num_prototypes"],
        temperature=cfg["temperature"],
    )
    model = UnmixerModel(mcfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        _, w_tile, r_tile = model(tile_feats.to(device))
    w_tile = w_tile.cpu()
    r_norm_tile = r_tile.cpu().norm(p=2, dim=1)
    w_img = scatter_mean_by_image(w_tile, image_index, num_images).numpy()
    r_img = scatter_mean_by_image(r_norm_tile.unsqueeze(-1), image_index,
                                  num_images).squeeze(-1).numpy()
    return w_img, r_img


def score_proto(cfg_path: Path, model_path: Path, tile_feats: torch.Tensor,
                image_index: torch.Tensor, num_images: int,
                device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (image-level similarities (N, K), image-level max-sim (N,))."""
    cfg = json.loads(cfg_path.read_text())
    mcfg = ProtoConfig(
        embedding_dim=cfg["embedding_dim"],
        num_prototypes=cfg["num_prototypes"],
    )
    model = PrototypeMatchingModel(mcfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        sims_tile, max_tile = model(tile_feats.to(device))
    sims_tile = sims_tile.cpu()
    max_tile = max_tile.cpu()
    s_img = scatter_mean_by_image(sims_tile, image_index, num_images).numpy()
    m_img = scatter_mean_by_image(max_tile.unsqueeze(-1), image_index,
                                  num_images).squeeze(-1).numpy()
    return s_img, m_img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retakes_dir", type=str,
                    default="data/real/augmented_retakes")
    ap.add_argument("--simplex_dir", type=str,
                    default="outputs/simplex_unmixing/6class")
    ap.add_argument("--proto_dir", type=str,
                    default="outputs/prototype_matching/6class")
    ap.add_argument("--output_dir", type=str,
                    default="outputs/retake_robustness")
    ap.add_argument("--frame_batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    retakes_dir = ROOT / args.retakes_dir
    simplex_dir = ROOT / args.simplex_dir
    proto_dir = ROOT / args.proto_dir
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- load configs ---------------------------------------------------------
    simplex_cfg = json.loads((simplex_dir / "config.json").read_text())
    proto_cfg = json.loads((proto_dir / "config.json").read_text())
    assert simplex_cfg["class_names"] == proto_cfg["class_names"], \
        "Methods A and B trained on different class sets"
    class_names = simplex_cfg["class_names"]
    K = len(class_names)
    print(f"Class names: {class_names}")

    # ---- enumerate retakes ----------------------------------------------------
    paths, folders = enumerate_retake_frames(retakes_dir)
    if not paths:
        raise SystemExit(f"No frames under {retakes_dir}")
    labels = build_labels(folders, class_names)
    print(f"Retake frames: {len(paths)} across {len(set(folders))} folders")
    # Per-folder counts for the markdown summary.
    per_folder_counts: Dict[str, int] = {}
    for f in folders:
        per_folder_counts[f] = per_folder_counts.get(f, 0) + 1

    # ---- extract tile features (GPU illumination path) ------------------------
    tile_cfg_dict = simplex_cfg["tile_config"]
    tile_cfg = TileConfig(
        tile_size=tile_cfg_dict["tile_size"],
        train_tiles_per_image=tile_cfg_dict["train_tiles_per_image"],
        eval_grid_size=tile_cfg_dict["eval_grid_size"],
    )
    cache_path = out_dir / "retake_features_cache.pt"
    print(f"Extracting features (cache: {cache_path})...")
    tile_feats, image_index = extract_features_multicrop_gpu(
        image_paths=[str(p) for p in paths],
        tile_config=tile_cfg,
        backbone=simplex_cfg["backbone"],
        frame_batch_size=args.frame_batch_size,
        num_workers=args.num_workers,
        device=device,
        illum_method=simplex_cfg["illumination"],
        illum_sigma=simplex_cfg["illum_sigma"],
        cache_path=str(cache_path),
    )
    tile_feats = torch.nn.functional.normalize(tile_feats, dim=1)
    print(f"  features: {tuple(tile_feats.shape)}  image_index: {tuple(image_index.shape)}")

    N = len(paths)

    # ---- Method A: simplex unmix ---------------------------------------------
    print("Scoring Method A...")
    w_img, r_img = score_simplex(
        simplex_dir / "config.json",
        simplex_dir / "bids_model.pt",
        tile_feats, image_index, N, device,
    )
    simplex_thresh = np.asarray(simplex_cfg["thresholds"])
    simplex_pred = (w_img > simplex_thresh).astype(np.int64)

    # ---- Method B: prototype matching ----------------------------------------
    print("Scoring Method B...")
    s_img, m_img = score_proto(
        proto_dir / "config.json",
        proto_dir / "proto_model.pt",
        tile_feats, image_index, N, device,
    )
    proto_thresh = np.asarray(proto_cfg["thresholds"])
    proto_pred = (s_img > proto_thresh).astype(np.int64)

    # ---- metrics --------------------------------------------------------------
    simplex_metrics = {
        "per_sample_f1": per_sample_f1(labels, simplex_pred),
        "macro_f1": macro_f1_per_class(labels, simplex_pred, class_names),
        "exact_match": exact_match_accuracy(labels, simplex_pred),
    }
    proto_metrics = {
        "per_sample_f1": per_sample_f1(labels, proto_pred),
        "macro_f1": macro_f1_per_class(labels, proto_pred, class_names),
        "exact_match": exact_match_accuracy(labels, proto_pred),
    }

    # ---- compare against primary test -----------------------------------------
    primary_simplex = json.loads((simplex_dir / "presence_test" /
                                  "summary.json").read_text())
    primary_proto = json.loads((proto_dir / "presence_test" /
                                "summary.json").read_text())
    primary_A_f1 = primary_simplex["per_sample_f1"]
    primary_B_f1 = primary_proto["per_sample_f1"]

    results = {
        "num_frames": int(N),
        "num_folders": len(set(folders)),
        "folders": sorted(set(folders)),
        "per_folder_counts": per_folder_counts,
        "class_names": class_names,
        "simplex_retake": simplex_metrics,
        "simplex_primary_test_per_sample_f1": primary_A_f1,
        "simplex_delta_pp": 100 * (simplex_metrics["per_sample_f1"] - primary_A_f1),
        "prototype_retake": proto_metrics,
        "prototype_primary_test_per_sample_f1": primary_B_f1,
        "prototype_delta_pp": 100 * (proto_metrics["per_sample_f1"] - primary_B_f1),
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    # Markdown summary for quick scanning.
    md = []
    md.append("# Retake robustness\n")
    md.append(f"Evaluated on {N} retake frames across {len(set(folders))} "
              f"retake videos. Models are trained on the 40 primary videos.\n")
    md.append("| Method | Primary test F1 | Retake F1 | Delta (pp) |")
    md.append("|--------|-----------------|-----------|------------|")
    md.append(f"| A (simplex unmix) | {primary_A_f1:.4f} | "
              f"{simplex_metrics['per_sample_f1']:.4f} | "
              f"{results['simplex_delta_pp']:+.2f} |")
    md.append(f"| B (proto match)   | {primary_B_f1:.4f} | "
              f"{proto_metrics['per_sample_f1']:.4f} | "
              f"{results['prototype_delta_pp']:+.2f} |")
    md.append("")
    md.append("Per-class F1 (retake):")
    md.append("| Class | Method A | Method B |")
    md.append("|-------|----------|----------|")
    for c in class_names:
        md.append(f"| {c} | {simplex_metrics['macro_f1'].get(c, float('nan')):.4f} | "
                  f"{proto_metrics['macro_f1'].get(c, float('nan')):.4f} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")

    print("\n=== Retake robustness ===")
    print(f"  Method A: primary {primary_A_f1:.4f}  retake "
          f"{simplex_metrics['per_sample_f1']:.4f}  "
          f"delta {results['simplex_delta_pp']:+.2f} pp")
    print(f"  Method B: primary {primary_B_f1:.4f}  retake "
          f"{proto_metrics['per_sample_f1']:.4f}  "
          f"delta {results['prototype_delta_pp']:+.2f} pp")
    print(f"\nWrote {out_dir / 'results.json'}")
    print(f"Wrote {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
