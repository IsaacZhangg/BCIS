"""Cross-validation utilities focused on leakage-resistant evaluation."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.config import SplitStrategy


def build_classwise_trial_groups(y: np.ndarray, group_size: int = 5) -> np.ndarray:
    """Create deterministic trial groups while preserving class balance.

    Trials are grouped *within each class* into contiguous chunks of size
    ``group_size``. This allows grouped CV without introducing class imbalance.

    Args:
        y: Label array of shape (n_trials,).
        group_size: Number of same-class trials per group.

    Returns:
        Integer group labels of shape (n_trials,).
    """
    if group_size < 1:
        raise ValueError("group_size must be >= 1")

    groups = np.empty(len(y), dtype=int)
    next_group_id = 0

    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        cls_groups = np.arange(len(cls_idx), dtype=int) // group_size
        groups[cls_idx] = cls_groups + next_group_id
        next_group_id += int(cls_groups.max() + 1) if len(cls_groups) > 0 else 0

    return groups


def _effective_n_splits(
    y: np.ndarray,
    requested: int,
    groups: np.ndarray | None,
) -> int:
    """Compute the maximum feasible split count for the label/group constraints."""
    if requested < 2 or len(y) == 0:
        return 0

    _, class_counts = np.unique(y, return_counts=True)
    n_splits = min(requested, int(class_counts.min()))

    if groups is not None:
        unique_groups = np.unique(groups)
        n_splits = min(n_splits, len(unique_groups))

        # Each class should appear in at least n_splits groups for grouped stratification.
        per_class_group_counts = [
            len(np.unique(groups[y == cls])) for cls in np.unique(y)
        ]
        n_splits = min(n_splits, min(per_class_group_counts))

    return n_splits if n_splits >= 2 else 0


def adaptive_fold_count(n_trials: int, max_folds: int, min_per_fold: int = 5) -> int:
    """Compute fold count ensuring at least min_per_fold trials per fold.

    Returns at least 2 (minimum for CV) and at most max_folds.
    """
    return max(2, min(max_folds, n_trials // min_per_fold))


def make_cv_splits(
    y: np.ndarray,
    n_splits: int,
    strategy: SplitStrategy = "stratified",
    groups: np.ndarray | None = None,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return CV train/test index splits with safe fallbacks.

    If ``strategy='stratified_group'`` is requested but grouped splitting is not
    feasible (e.g., too few groups), the function falls back to stratified K-fold.
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")

    X_dummy = np.zeros((len(y), 1))

    if strategy == "stratified_group" and groups is not None:
        groups = np.asarray(groups)
        if groups.shape[0] != y.shape[0]:
            raise ValueError("groups must have the same length as y")
        grouped_splits = _effective_n_splits(y, n_splits, groups)
        if grouped_splits >= 2:
            try:
                cv = StratifiedGroupKFold(
                    n_splits=grouped_splits,
                    shuffle=True,
                    random_state=random_state,
                )
                return list(cv.split(X_dummy, y, groups=groups))
            except ValueError as exc:
                warnings.warn(
                    "Grouped CV was requested but StratifiedGroupKFold failed; "
                    "falling back to stratified K-fold without group constraints. "
                    f"Original error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                "Grouped CV was requested but is infeasible for this subject/fold "
                "configuration; falling back to stratified K-fold without group "
                "constraints.",
                RuntimeWarning,
                stacklevel=2,
            )

    plain_splits = _effective_n_splits(y, n_splits, groups=None)
    if plain_splits < 2:
        return []

    cv = StratifiedKFold(
        n_splits=plain_splits,
        shuffle=True,
        random_state=random_state,
    )
    return list(cv.split(X_dummy, y))
