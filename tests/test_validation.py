"""Tests for leakage-resistant cross-validation helpers."""

import numpy as np
import pytest

from src.validation import (
    adaptive_fold_count,
    build_classwise_trial_groups,
    make_cv_splits,
)


def test_build_classwise_trial_groups_separates_classes():
    """Auto-built groups are class-wise and deterministic."""
    y = np.array([0] * 10 + [1] * 10)

    groups = build_classwise_trial_groups(y, group_size=3)

    assert groups.shape == y.shape

    # No group should mix classes.
    for group_id in np.unique(groups):
        labels = np.unique(y[groups == group_id])
        assert len(labels) == 1


def test_make_cv_splits_stratified_group_keeps_groups_intact():
    """Grouped CV never places the same group in train and test of one split."""
    y = np.array([0] * 12 + [1] * 12)
    groups = build_classwise_trial_groups(y, group_size=4)

    splits = make_cv_splits(
        y,
        n_splits=3,
        strategy="stratified_group",
        groups=groups,
        random_state=42,
    )

    assert len(splits) >= 2
    for train_idx, test_idx in splits:
        train_groups = set(groups[train_idx].tolist())
        test_groups = set(groups[test_idx].tolist())
        assert train_groups.isdisjoint(test_groups)


def test_make_cv_splits_falls_back_when_grouped_not_feasible():
    """When grouped stratification is infeasible, we fall back to plain stratified CV."""
    y = np.array([0, 0, 1, 1, 0, 1])
    # Too few groups for grouped CV with n_splits=3
    groups = np.array([0, 0, 0, 1, 1, 1])

    splits = make_cv_splits(
        y,
        n_splits=3,
        strategy="stratified_group",
        groups=groups,
        random_state=42,
    )

    assert len(splits) == 2 or len(splits) == 3
    # Ensure we still get usable splits with class presence in each test fold.
    for _, test_idx in splits:
        assert set(np.unique(y[test_idx])) == {0, 1}


def test_make_cv_splits_warns_when_grouped_fallback_is_used():
    """Grouped fallback emits a warning so leakage risk is visible to callers."""
    # Each class appears in only one group => grouped stratification is impossible.
    y = np.array([0, 0, 1, 1])
    groups = np.array([0, 0, 1, 1])

    with pytest.warns(RuntimeWarning, match="falling back to stratified K-fold"):
        splits = make_cv_splits(
            y,
            n_splits=2,
            strategy="stratified_group",
            groups=groups,
            random_state=42,
        )

    assert len(splits) >= 2


def test_make_cv_splits_rejects_group_length_mismatch():
    """Group labels must align one-to-one with the label array."""
    y = np.array([0, 1, 0, 1])
    groups = np.array([0, 1, 0])

    with pytest.raises(ValueError, match="same length as y"):
        make_cv_splits(y, n_splits=2, strategy="stratified_group", groups=groups)


def test_adaptive_fold_count_large_dataset():
    """100 trials, max 10 folds -> 10 folds."""
    assert adaptive_fold_count(100, max_folds=10) == 10


def test_adaptive_fold_count_small_dataset():
    """32 trials, max 10 folds -> 6 folds (32 // 5 = 6)."""
    assert adaptive_fold_count(32, max_folds=10) == 6


def test_adaptive_fold_count_medium_dataset():
    """64 trials, max 10 folds -> 10 folds (64 // 5 = 12, capped at 10)."""
    assert adaptive_fold_count(64, max_folds=10) == 10


def test_adaptive_fold_count_inner():
    """27 training trials, max 7 inner folds -> 5 folds (27 // 5 = 5)."""
    assert adaptive_fold_count(27, max_folds=7) == 5


def test_adaptive_fold_count_minimum():
    """Very few trials still returns at least 2."""
    assert adaptive_fold_count(8, max_folds=10) == 2


def test_split_epoch_pairs_respects_fraction():
    """split_epoch_pairs splits into train/test with approximately correct proportions."""
    import numpy as np

    from src.validation import split_epoch_pairs

    n_epochs = 10
    channels = ["C3", "C4"]
    left = {
        ch: [(np.zeros(100), np.zeros(300)) for _ in range(n_epochs)] for ch in channels
    }
    right = {
        ch: [(np.zeros(100), np.zeros(300)) for _ in range(n_epochs)] for ch in channels
    }

    train_l, train_r, test_l, test_r = split_epoch_pairs(
        left, right, test_fraction=0.2, random_state=42
    )

    assert len(train_l["C3"]) == 8
    assert len(test_l["C3"]) == 2
    assert len(train_r["C3"]) == 8
    assert len(test_r["C3"]) == 2
