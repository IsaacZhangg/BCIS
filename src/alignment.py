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
