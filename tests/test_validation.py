"""Tests for leakage-resistant cross-validation helpers."""

import numpy as np

from src.validation import build_classwise_trial_groups, make_cv_splits


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
