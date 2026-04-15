"""Tests for Euclidean Alignment shared utility."""

import numpy as np
from pyriemann.estimation import Covariances

from src.alignment import apply_ea_transform, compute_ea_transform, euclidean_align


def _make_spd_trials(n_trials: int = 20, n_channels: int = 8, seed: int = 42):
    """Generate synthetic multichannel EEG and its covariances."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_channels, n_channels))
    cov_mean = A @ A.T + np.eye(n_channels)
    X = rng.standard_normal((n_trials, n_channels, 375))
    X = np.einsum("ij,njt->nit", np.linalg.cholesky(cov_mean), X)
    covs = Covariances(estimator="lwf").fit_transform(X)
    return covs


def test_compute_ea_transform_shape():
    covs = _make_spd_trials()
    transform = compute_ea_transform(covs)
    assert transform.shape == (8, 8)


def test_apply_ea_transform_shape():
    covs = _make_spd_trials()
    transform = compute_ea_transform(covs)
    aligned = apply_ea_transform(covs, transform)
    assert aligned.shape == covs.shape


def test_euclidean_align_centers_to_identity():
    covs = _make_spd_trials()
    aligned, _ = euclidean_align(covs)
    mean_cov = np.mean(aligned, axis=0)
    np.testing.assert_allclose(mean_cov, np.eye(8), atol=0.5)


def test_apply_precomputed_transform():
    covs = _make_spd_trials()
    aligned_full, transform = euclidean_align(covs)
    aligned_manual = apply_ea_transform(covs, transform)
    np.testing.assert_allclose(aligned_full, aligned_manual, atol=1e-10)
