"""Tests for Euclidean Alignment shared utility."""

import numpy as np
from pyriemann.estimation import Covariances

from src.alignment import (
    align_trials_ea,
    apply_ea_to_trials,
    apply_ea_transform,
    compute_ea_transform,
    compute_trial_ea_transform,
    euclidean_align,
)


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


def test_trial_ea_centers_mean_covariance():
    rng = np.random.default_rng(7)
    n_channels = 8
    A = rng.standard_normal((n_channels, n_channels))
    cov_mean = A @ A.T + np.eye(n_channels)
    X = rng.standard_normal((24, n_channels, 375))
    X = np.einsum("ij,njt->nit", np.linalg.cholesky(cov_mean), X)

    aligned, transform = align_trials_ea(X)
    assert aligned.shape == X.shape
    assert transform.shape == (n_channels, n_channels)

    sample_covs = np.einsum("nct,ndt->ncd", aligned, aligned) / aligned.shape[2]
    mean_cov = sample_covs.mean(axis=0)
    np.testing.assert_allclose(mean_cov, np.eye(n_channels), atol=0.35)


def test_trial_ea_transform_is_fit_on_train_only():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((20, 8, 200))
    train, test = X[:12], X[12:]
    transform = compute_trial_ea_transform(train)
    aligned_test = apply_ea_to_trials(test, transform)
    assert aligned_test.shape == test.shape
    # Fitting on train+test must not be required for a finite transform.
    assert np.isfinite(aligned_test).all()
