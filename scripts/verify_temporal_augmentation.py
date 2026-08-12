"""Print augmentation shape summary and verify leakage-free CV splits.

Run: ``uv run python scripts/verify_temporal_augmentation.py``

Builds a small synthetic subject, constructs the AugmentationContext used by
the pipeline, then walks through 5 stratified folds and asserts the
origin-based train/test split never mixes sibling windows of the same trial.
"""

from __future__ import annotations

import numpy as np

from src.data_loader import CHANNELS
from src.epochs import sliding_window_offsets
from src.pipeline import _build_augmentation_context, _pairs_to_features
from src.validation import make_cv_splits


def main() -> None:
    sfreq = 250.0
    aug_window_sec = 2.5
    aug_stride_sec = 0.25
    window_samples = int(aug_window_sec * sfreq)
    stride_samples = int(aug_stride_sec * sfreq)
    task_samples = int(3.0 * sfreq)

    offsets = sliding_window_offsets(task_samples, window_samples, stride_samples)
    print("== Sliding window geometry ==")
    print(f"  task length: {task_samples} samples (3.0s @ {sfreq:.0f} Hz)")
    print(f"  window: {window_samples} samples ({aug_window_sec}s)")
    print(f"  stride: {stride_samples} samples ({aug_stride_sec}s)")
    print(f"  offsets: {offsets}")
    print(f"  windows per epoch: {len(offsets)}")

    rng = np.random.default_rng(0)
    n_left, n_right = 15, 15

    def _task(label: int) -> np.ndarray:
        t = np.arange(task_samples) / sfreq
        freq = 11.0 if label == 0 else 19.0
        return 10.0 * np.sin(2 * np.pi * freq * t) + rng.standard_normal(task_samples)

    left = {
        ch: [(rng.standard_normal(250), _task(0)) for _ in range(n_left)]
        for ch in CHANNELS
    }
    right = {
        ch: [(rng.standard_normal(250), _task(1)) for _ in range(n_right)]
        for ch in CHANNELS
    }

    X_feat, X_mc, y = _pairs_to_features(left, right, sfreq)
    ctx = _build_augmentation_context(
        left, right, sfreq, aug_window_sec, aug_stride_sec
    )

    print("\n== Per-subject tensors ==")
    print(f"  y shape: {y.shape}  class balance: {np.bincount(y).tolist()}")
    print(f"  X_features (canonical 3.0s path): {X_feat.shape}")
    print(f"  X_multichannel (canonical 3.0s path): {X_mc.shape}")
    print(f"  aug features pool: {ctx.X_features_pool.shape}")
    print(f"  aug multichannel pool: {ctx.X_multichannel_pool.shape}")
    print(f"  aug origins (len): {ctx.origins.shape[0]}")
    print(f"  aug canonical features (centered crop): {ctx.X_features_canonical.shape}")
    print(
        f"  aug canonical multichannel (centered crop): {ctx.X_multichannel_canonical.shape}"
    )
    print(
        f"  ratio pool/original: {ctx.X_features_pool.shape[0] / X_feat.shape[0]:.2f}x"
    )

    print("\n== Leakage check across 5 stratified folds ==")
    splits = make_cv_splits(y, n_splits=5, strategy="stratified", random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        feat_tr, mc_tr, origins_tr = ctx.expand_train(train_idx)
        feat_te = ctx.X_features_canonical[test_idx]
        mc_te = ctx.X_multichannel_canonical[test_idx]

        train_origin_set = set(origins_tr.tolist())
        test_origin_set = set(int(i) for i in test_idx)
        overlap = train_origin_set & test_origin_set
        assert not overlap, f"Fold {fold_idx}: origin leakage {overlap}"

        print(
            f"  fold {fold_idx}: train augmented={feat_tr.shape[0]} rows "
            f"({mc_tr.shape[0]} mc) from {len(train_origin_set)} trials; "
            f"test canonical={feat_te.shape[0]} rows ({mc_te.shape[0]} mc) "
            f"from {len(test_origin_set)} trials; leakage=none"
        )

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
