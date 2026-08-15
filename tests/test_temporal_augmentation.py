"""Tests for sliding-window temporal augmentation.

Verifies:
1. Helper shapes / offsets / origins behave correctly.
2. AugmentationContext.expand_train contains no leakage: no original trial
   contributes windows to both train and test.
3. Running the inner CV with augmentation yields the augmented training
   tensor lengths we expect (n_train_trials * n_windows).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data_loader import CHANNELS
from src.epochs import (
    center_crop_offset,
    expand_trial_windows,
    sliding_window_offsets,
    slice_epoch_windows,
)
from src.pipeline import _build_augmentation_context
from src.train import AugmentationContext


SFREQ = 250.0


def _make_pairs(
    n_left: int,
    n_right: int,
    task_samples: int = 750,
    baseline_samples: int = 250,
    seed: int = 0,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    left = {
        ch: [
            (rng.standard_normal(baseline_samples), rng.standard_normal(task_samples))
            for _ in range(n_left)
        ]
        for ch in CHANNELS
    }
    right = {
        ch: [
            (rng.standard_normal(baseline_samples), rng.standard_normal(task_samples))
            for _ in range(n_right)
        ]
        for ch in CHANNELS
    }
    return left, right


def test_sliding_window_offsets_defaults():
    # 3.0s epoch at 250 Hz = 750 samples; 2.5s window = 625; 0.25s stride = 62
    offsets = sliding_window_offsets(750, 625, 62)
    assert offsets[0] == 0
    # Last offset must place the window against the tail.
    assert offsets[-1] + 625 == 750
    # Window fits at least three times (plus tail clamp).
    assert len(offsets) >= 3


def test_sliding_window_offsets_short_signal():
    # Window larger than signal → single offset [0].
    assert sliding_window_offsets(100, 200, 10) == [0]


def test_center_crop_offset():
    # 750 → 625 centered leaves 125 samples, half on each side.
    assert center_crop_offset(750, 625) == 62
    # Window equal to signal → offset 0.
    assert center_crop_offset(625, 625) == 0


def test_slice_epoch_windows_shape():
    signal = np.arange(750)
    windows = slice_epoch_windows(signal, 625, 62)
    assert windows.shape == (4, 625)
    # First window starts at 0, second at 62, etc. Last is clamped to tail.
    np.testing.assert_array_equal(windows[0], np.arange(625))
    assert windows[-1, 0] == 750 - 625


def test_expand_trial_windows_per_trial_ordering():
    # 3 trials, 2 channels, 10 samples; window 5, stride 2.
    X = np.arange(3 * 2 * 10).reshape(3, 2, 10)
    X_exp, origins = expand_trial_windows(X, 5, 2)
    # Origins must be grouped per trial in the output ordering.
    assert origins.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    # Last window per trial should match the trailing slice.
    n_windows = 4
    for trial in range(3):
        assert X_exp[trial * n_windows + n_windows - 1, 0, -1] == X[trial, 0, -1]


def test_build_augmentation_context_shapes():
    left, right = _make_pairs(n_left=6, n_right=4)
    ctx = _build_augmentation_context(left, right, SFREQ, 2.5, 0.25)

    n_total = 10
    n_windows = ctx.X_features_pool.shape[0] // n_total
    assert ctx.X_features_pool.shape == (n_total * n_windows, 45)
    assert ctx.X_multichannel_pool.shape == (n_total * n_windows, len(CHANNELS), 625)
    assert ctx.X_features_canonical.shape == (n_total, 45)
    assert ctx.X_multichannel_canonical.shape == (n_total, len(CHANNELS), 625)
    assert set(ctx.origins.tolist()) == set(range(n_total))
    # Right-class rows must map to origins n_left..n_total - 1.
    right_origins = ctx.origins[-4 * n_windows :]
    assert set(right_origins.tolist()) == set(range(6, 10))


def test_expand_train_has_no_leakage():
    left, right = _make_pairs(n_left=6, n_right=4)
    ctx = _build_augmentation_context(left, right, SFREQ, 2.5, 0.25)

    train_idx = np.array([0, 1, 2, 5, 6, 7])
    test_idx = np.array([3, 4, 8, 9])
    feat_tr, mc_tr, origins_tr = ctx.expand_train(train_idx)
    # Augmented training rows = len(train_idx) * n_windows
    n_windows = ctx.X_features_pool.shape[0] // 10
    assert feat_tr.shape[0] == len(train_idx) * n_windows
    assert mc_tr.shape[0] == len(train_idx) * n_windows
    assert set(origins_tr.tolist()) == set(train_idx.tolist())
    # No overlap with test trials: canonical test rows come from test_idx,
    # and augmented training rows never reference them.
    assert set(origins_tr.tolist()).isdisjoint(set(test_idx.tolist()))


def test_augmentation_context_types():
    left, right = _make_pairs(n_left=3, n_right=3)
    ctx = _build_augmentation_context(left, right, SFREQ, 2.5, 0.25)
    assert isinstance(ctx, AugmentationContext)
    # Canonical tensors must have one row per original trial.
    assert ctx.X_features_canonical.shape[0] == 6
    assert ctx.X_multichannel_canonical.shape[0] == 6


def test_config_defaults_are_off():
    from src.config import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.temporal_augmentation is False
    assert cfg.use_composite_csp is True
    assert cfg.session_level_ea is False
    assert cfg.aug_window_sec == 2.5
    assert cfg.aug_stride_sec == 0.25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
