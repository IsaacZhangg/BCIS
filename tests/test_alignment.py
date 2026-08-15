"""Tests for Euclidean Alignment shared utility."""

import numpy as np
from pyriemann.estimation import Covariances

from src.alignment import (
    align_trials_ea,
    apply_ea_to_trials,
    apply_ea_transform,
    apply_session_ea_train_only,
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


def test_session_ea_is_noop_for_single_session():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((16, 8, 200))
    session_ids = np.zeros(16, dtype=int)
    train_idx = np.arange(12)
    test_idx = np.arange(12, 16)
    X_tr, X_te = apply_session_ea_train_only(X, session_ids, train_idx, test_idx)
    np.testing.assert_allclose(X_tr, X[train_idx])
    np.testing.assert_allclose(X_te, X[test_idx])


def test_session_ea_uses_train_only_per_session():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((20, 8, 200))
    session_ids = np.array([0] * 10 + [1] * 10)
    train_idx = np.array([0, 1, 2, 3, 4, 10, 11, 12, 13, 14])
    test_idx = np.array([5, 6, 7, 8, 9, 15, 16, 17, 18, 19])
    X_tr, X_te = apply_session_ea_train_only(X, session_ids, train_idx, test_idx)
    assert X_tr.shape == (10, 8, 200)
    assert X_te.shape == (10, 8, 200)
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_te).all()
    # Train-only transform for session 0 should match a direct fit.
    ref0 = compute_trial_ea_transform(X[train_idx[session_ids[train_idx] == 0]])
    expected = apply_ea_to_trials(X[test_idx[session_ids[test_idx] == 0]], ref0)
    np.testing.assert_allclose(X_te[session_ids[test_idx] == 0], expected, atol=1e-10)
