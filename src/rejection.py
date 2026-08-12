"""Artifact rejection for EEG trials.

Two-layer API:
- reject_in_fold: Fold-time rejection using per-trial max PTP + index filtering.
  Threshold computed from training indices only, applied to both splits.
- Epoch-time multi-criteria rejection (PTP, gradient, HF-power) remains in
  src/epochs.py (reject_bad_epochs). Re-exported here for discoverability.
"""

import numpy as np

from src.epochs import reject_bad_epochs  # noqa: F401 – re-export for discoverability


def reject_in_fold(
    trial_ptps: np.ndarray | None,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_mad: float = 3.5,
    flat_uv: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter train/test indices using an adaptive amplitude threshold.

    The threshold is computed from training indices only, then applied to both
    splits.  When *trial_ptps* is ``None``, indices are returned unchanged.
    """
    if trial_ptps is None:
        return train_idx, test_idx

    train_ptps = trial_ptps[train_idx]
    median = float(np.median(train_ptps))
    mad = float(np.median(np.abs(train_ptps - median)))
    threshold = median + n_mad * mad

    def _keep(idx: np.ndarray) -> np.ndarray:
        ptps = trial_ptps[idx]
        mask = (ptps >= flat_uv) & (ptps <= threshold)
        return idx[mask]

    return _keep(train_idx), _keep(test_idx)
