"""Shared utilities for BIDS (Bacterial IDentification under Spatial homogeneity).

All three decoders (A simplex unmixing, B prototype matching, C channel-grouped
UFG) sit on the spatial homogeneity assumption H: for any sufficiently large
crop c of a microscopy frame x, P(y|c) = P(y|x). Random crops are therefore
label-preserving augmentations and per-tile predictions can be aggregated to
image level by simple averaging. See `tiling.py` for the implementation and the
per-method `model.py` files for the math each method runs on top.
"""
