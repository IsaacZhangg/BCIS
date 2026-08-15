"""Tests for Composite CSP (Lotte & Guan) mixing."""

import numpy as np

from src.band_cache import FBCSP_BANDS, precompute_bandpassed
from src.composite_csp import (
    apply_csp_logvar,
    composite_class_covariances,
    csp_filters_from_covariances,
    epoch_covariance,
    extract_composite_fbcsp,
    pool_source_class_covariances,
)
from src.train import train_nested_model_selection_cv

N_CHANNELS = 8
N_SAMPLES = 375
SFREQ = 250.0


def _class_conditional_trials(
    n_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((n_per_class, N_CHANNELS, N_SAMPLES))
    right = rng.standard_normal((n_per_class, N_CHANNELS, N_SAMPLES))
    left[:, 2] *= 1.8
    right[:, 5] *= 1.8
    X = np.vstack([left, right])
    y = np.array([0] * n_per_class + [1] * n_per_class)
    return X, y


def test_epoch_covariance_is_spd_and_trace_normalized():
    X, _ = _class_conditional_trials(12, seed=0)
    cov = epoch_covariance(X)
    assert cov.shape == (N_CHANNELS, N_CHANNELS)
    np.testing.assert_allclose(cov, cov.T, atol=1e-10)
    np.testing.assert_allclose(np.trace(cov), 1.0, atol=1e-6)
    evals = np.linalg.eigvalsh(cov)
    assert np.all(evals > -1e-8)


def test_lam_zero_matches_target_covariance():
    X, y = _class_conditional_trials(10, seed=1)
    source = {0: np.eye(N_CHANNELS), 1: np.eye(N_CHANNELS) * 2}
    C0, C1 = composite_class_covariances(X, y, source, lam=0.0)
    np.testing.assert_allclose(C0, epoch_covariance(X[y == 0]), atol=1e-10)
    np.testing.assert_allclose(C1, epoch_covariance(X[y == 1]), atol=1e-10)


def test_extract_composite_fbcsp_shapes():
    X_tgt, y_tgt = _class_conditional_trials(16, seed=2)
    X_src, y_src = _class_conditional_trials(16, seed=3)
    bands = FBCSP_BANDS[:3]
    src_cache = precompute_bandpassed(X_src, SFREQ, bands=bands)
    pooled = pool_source_class_covariances([(src_cache, y_src)], bands)
    train, test = extract_composite_fbcsp(
        X_tgt[:20],
        X_tgt[20:],
        y_tgt[:20],
        pooled,
        bands=bands,
        lam=0.3,
        sfreq=SFREQ,
    )
    assert train.shape[0] == 20
    assert test.shape[0] == 12
    assert train.shape[1] == test.shape[1]
    assert train.shape[1] > 0


def test_csp_filters_and_logvar():
    X, y = _class_conditional_trials(14, seed=4)
    C0 = epoch_covariance(X[y == 0])
    C1 = epoch_covariance(X[y == 1])
    filters = csp_filters_from_covariances(C0, C1, n_components=3)
    feats = apply_csp_logvar(X, filters)
    assert filters.shape == (3, N_CHANNELS)
    assert feats.shape == (len(y), 3)
    assert np.all(np.isfinite(feats))


def test_nested_composite_csp_stays_near_chance_on_random_data():
    rng = np.random.default_rng(11)
    n_trials, n_feat = 40, 10
    X_by_subject = []
    y_by_subject = []
    for i in range(3):
        local = np.random.default_rng(100 + i)
        X_feat = local.standard_normal((n_trials, n_feat))
        X_mc = rng.standard_normal((n_trials, N_CHANNELS, N_SAMPLES))
        y = np.array([0] * 20 + [1] * 20)
        X_by_subject.append((X_feat, X_mc))
        y_by_subject.append(y)

    scores, mean_acc, _, methods, _, _ = train_nested_model_selection_cv(
        X_by_subject,
        y_by_subject,
        sfreq=SFREQ,
        n_outer_folds=4,
        n_inner_folds=3,
        n_jobs=1,
        use_composite_csp=True,
        composite_csp_lam=0.3,
    )
    assert len(scores) == 3
    assert mean_acc < 0.70
    assert all(m in {"lda", "riemann", "svm", "ensemble", "ccsp"} for m in methods)
