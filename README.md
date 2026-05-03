# BIDS — Bacterial IDentification under Spatial homogeneity

Code for the BIDS framework: open-world multi-label identification of mixed
bacterial cultures from phase-contrast microscopy images. Three lightweight
decoders sit on identical frozen DINOv2-S/14 tile features and share one
geometric scaffold, the **Spatial Homogeneity assumption (H)**: for any image
`x` and any sufficiently large crop `c ⊂ x`, `Pr(y | c) = Pr(y | x) = y(x)`.
Under H, every crop is an i.i.d. sample of the whole-image label and per-tile
predictions can be mean-aggregated to image level with `O(1/T)` variance.

The repository contains:

- Three decoders over a shared frozen-feature pool (`src/`).
- The supervised baselines used to expose the compositional collapse on the
  leave-combinations-out (LCO) protocol (`baselines/`).
- The experiment drivers for LCO, leave-one-class-out (LOOCV) open-set
  rejection, and SK K=1 novel-class discovery (`experiments/`).
- The tooling that ingests raw video into the splits the paper uses, packages
  the dataset for release, and regenerates the paper figures (`tools/`).

The dataset (~22 GB, CC BY 4.0) is hosted separately; the URL is in the paper.

## Install

```bash
conda env create -f environment.yml
conda activate bact
```

## Methods

| Module                    | Method                  | Mechanism                                                                          |
|---------------------------|-------------------------|------------------------------------------------------------------------------------|
| `src/simplex_unmixing/`   | A — simplex unmixing    | Sparsemax projection onto a learned prototype simplex; residual norm is the open-set substrate. |
| `src/prototype_matching/` | B — cosine matching     | Closed-form cosine similarity to per-class prototypes; thresholds calibrated on val. |
| `src/mc_channel/`         | C — channel-grouped UFG | 390-parameter discriminative head splitting the 384-dim embedding into K=6 groups of 64 channels with CRA dropout. |

All three reuse the same shared front-end:

- `src/common/tiling.py` — multi-crop pipeline (4×4 deterministic eval grid, random crops at training).
- `src/common/illumination.py` — divide-by-Gaussian hotspot correction (CPU + GPU paths).
- `src/common/features.py` — frozen DINOv2-S/14 tile-feature extractor with on-disk cache.
- `src/common/prototypes.py` — pure-culture-mean prototype init.
- `src/common/sinkhorn.py` — Sinkhorn-Knopp doubly-stochastic primitive used by SK K=1 discovery.
- `src/common/metrics.py` — multi-label per-sample F1, macro F1, exact match, open-set AUROC/AUPR/FPR@95TPR.
- `src/common/io.py` — `load_real_split()`, the canonical `splits.json` entry point.

## Data layout

```
data/real/
├── videos/                          # 40 primary .mp4
├── videos_retakes/                  # 16 cross-session retakes (filenames end _takeN)
├── frames/<combo>/frame_XXXX.jpg    # extracted primary frames
├── frames_retakes/<combo>_takeN/... # extracted retake frames
└── splits.json                      # video-level random 80/10/10
```

Filenames encode the label: `bs_ka_fj.mp4` → `[bs, ka, fj]` present. Six
species in 40 combinations of orders 1 to 6; three pairwise combinations
(`bs+bt`, `bt+fj`, `ka+pf`) failed to plate stably and are excluded by biology.

| Token | Species                       | Token | Species                  |
|-------|-------------------------------|-------|--------------------------|
| `bs`  | *Bacillus subtilis*           | `mx`  | *Myxococcus xanthus*     |
| `bt`  | *Bacillus thermoamylovorans*  | `ka`  | *Klebsiella aerogenes*   |
| `fj`  | *Flavobacterium johnsoniae*   | `pf`  | *Pseudomonas fluorescens*|

Once the dataset archives are unpacked, build the splits with:

```bash
python tools/prepare_real_data.py
```

## Quick start

Train the three decoders on the random 80/10/10 split:

```bash
python -m src.simplex_unmixing.train  --output_dir outputs/simplex_unmixing/run1
python -m src.prototype_matching.train --output_dir outputs/prototype_matching/run1
python -m src.mc_channel.train         --output_dir outputs/mc_channel/run1
```

Score the test split:

```bash
python experiments/run_presence_detection.py --method simplex   --model_dir outputs/simplex_unmixing/run1
python experiments/run_presence_detection.py --method prototype --model_dir outputs/prototype_matching/run1
python -m src.mc_channel.test_eval                              --model_dir outputs/mc_channel/run1
```

## Reproducing the paper

| Table / Figure                      | Driver                                                                                     |
|-------------------------------------|--------------------------------------------------------------------------------------------|
| LCO compositional collapse (Table)  | `python experiments/run_bids_heldout.py --output_dir outputs/bids_heldout`                 |
| Supervised LCO baselines            | `python baselines/supervised_multilabel_heldout.py --backbone resnet50.a1_in1k --output_dir outputs/supervised_multilabel_heldout/resnet50` |
| 13-encoder linear probe             | `python -m baselines.multilabel_probe --epochs 10` and `python -m baselines.multilabel_probe_bio --epochs 10` |
| LOOCV open-set                      | `python experiments/run_openset_detection.py`                                              |
| LOOCV discovery (SK K=1, canonical) | `python experiments/run_discovery.py --output_dir outputs/discovery_loocv_sk_k1 --cluster_method sinkhorn --sinkhorn_k 1 --residual_threshold 0.15` |
| 5-way OSR score sweep               | `python experiments/run_osr_score_sweep.py`                                                |
| Cross-session retake test           | `python experiments/run_retake_robustness.py`                                              |
| Tile-count and ablation grids       | `python experiments/run_ablations.py`                                                      |
| Per-class learnable temperature     | `python experiments/run_learned_tau.py`                                                    |
| Inter-prototype repulsion           | `python experiments/run_repulsion.py`                                                      |
| Reliability diagrams                | `python experiments/run_calibration_analysis.py`                                           |
| Per-order F1 breakdown              | `python experiments/run_per_order_breakdown.py`                                            |
| Boundary-tile robustness check      | `python experiments/run_boundary_tile_check.py`                                            |
| Isotonic recalibration              | `python experiments/run_isotonic_ablation.py`                                              |

The LCO split is canonical at seed 1337. The held-out 9 combinations are
`bt`, `bs_pf`, `ka_fj`, `bs_mx_fj`, `bs_ka_pf`, `mx_ka_fj`, `bs_bt_ka_fj`,
`bs_mx_fj_pf`, and `bs_bt_mx_ka_fj_pf`.

## Design constraints

These are non-negotiable design rules of BIDS, not preferences:

1. **No proportion or composition estimation.** There is no ground truth for
   mixture ratios (species grow at different rates, so plated proportions do
   not equal cell-count proportions). All metrics are presence / absence.
2. **Real microscopy only.** All reported numbers come from real
   microscopy; there is no synthetic-data path.
3. **Video-level splits.** Adjacent-frame autocorrelation in a multi-minute
   video is high. Splits never mix frames from the same video across
   train / val / test.

## Repository layout

```
src/
├── common/             # tiling, illumination, features, prototypes, metrics, io, sinkhorn
├── simplex_unmixing/   # method A
├── prototype_matching/ # method B
└── mc_channel/         # method C

experiments/            # LCO, LOOCV open-set + discovery, ablations, calibration, retake
baselines/              # 13-encoder probe, supervised LCO fine-tunes, attention-MIL
tools/                  # data ingest, splits builder, dataset packaging
```

Outputs land under `outputs/<run>/` (gitignored). Feature caches are keyed on
`(paths, tile_config, backbone, illum_method, illum_sigma)`; copy a cache
between runs to skip re-extraction when the key matches.

## Citation

```bibtex
@misc{bids2026,
  title  = {BIDS: Bacterial IDentification under Spatial homogeneity},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Under review}
}
```

Code released under the MIT license (`LICENSE`). The dataset is released
separately under CC BY 4.0.
