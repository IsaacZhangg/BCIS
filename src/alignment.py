"""Euclidean Alignment utility for cross-subject covariance normalization."""

from __future__ import annotations

import numpy as np
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import fractional_matrix_power


def compute_ea_transform(covariances: np.ndarray) -> np.ndarray:
    """Compute R^{-1/2} from the Riemannian geometric mean of trial covariances.

    Args:
        covariances: (n_trials, n_channels, n_channels) SPD matrices.

    Returns:
        (n_channels, n_channels) inverse square root of the reference matrix.
    """
    ref = mean_covariance(covariances, metric="riemann")
    return fractional_matrix_power(ref, -0.5).real


def apply_ea_transform(covariances: np.ndarray, ref_inv_sqrt: np.ndarray) -> np.ndarray:
    """Apply a precomputed EA transform: R^{-1/2} C R^{-T/2}.

    Args:
        covariances: (n_trials, n_channels, n_channels) SPD matrices.
        ref_inv_sqrt: (n_channels, n_channels) from compute_ea_transform.

    Returns:
        (n_trials, n_channels, n_channels) aligned covariances.
    """
    return ref_inv_sqrt @ covariances @ ref_inv_sqrt.T


def euclidean_align(
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute EA transform and apply it.

    Args:
        covariances: (n_trials, n_channels, n_channels) SPD matrices.

    Returns:
        Tuple of (aligned_covariances, ref_inv_sqrt).
    """
    ref_inv_sqrt = compute_ea_transform(covariances)
    aligned = apply_ea_transform(covariances, ref_inv_sqrt)
    return aligned, ref_inv_sqrt


def compute_trial_ea_transform(X: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Fit He & Wu Euclidean Alignment from EEG trials (train-only).

    Uses the arithmetic mean of sample covariances, matching the original EA
    paper. Never pass test trials into this function.

    Args:
        X: EEG trials of shape (n_trials, n_channels, n_samples).
        ridge: Diagonal jitter for numerical stability.

    Returns:
        (n_channels, n_channels) matrix R^{-1/2}.
    """
    n_trials, n_channels, n_samples = X.shape
    if n_trials < 1 or n_samples < 2:
        raise ValueError("Need at least one trial with >= 2 samples for EA")
    ref = np.einsum("nct,ndt->cd", X, X, dtype=np.float64)
    ref /= float(n_trials * n_samples)
    ref = 0.5 * (ref + ref.T)
    ref = ref + ridge * np.eye(n_channels)
    return fractional_matrix_power(ref, -0.5).real


def apply_ea_to_trials(X: np.ndarray, ref_inv_sqrt: np.ndarray) -> np.ndarray:
    """Whiten EEG trials: X' = R^{-1/2} X.

    Args:
        X: (n_trials, n_channels, n_samples).
        ref_inv_sqrt: (n_channels, n_channels) from compute_trial_ea_transform.

    Returns:
        Aligned trials with the same shape as *X*.
    """
    return np.einsum("ij,njt->nit", ref_inv_sqrt, X)


def align_trials_ea(
    X: np.ndarray, ridge: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Fit and apply trial-level Euclidean Alignment.

    Returns:
        Tuple of (aligned_trials, ref_inv_sqrt).
    """
    ref_inv_sqrt = compute_trial_ea_transform(X, ridge=ridge)
    return apply_ea_to_trials(X, ref_inv_sqrt), ref_inv_sqrt
