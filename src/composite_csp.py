"""Composite CSP (Lotte & Guan): mix target and source class covariances.

Source covariances are computed from other subjects only. Target test trials
never enter the mix. Filters are applied to the target subject alone, so the
classifier is not trained on foreign trials.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.covariance import OAS

from src.band_cache import BandCache, FBCSP_BANDS, N_CSP_COMPONENTS


def epoch_covariance(X: np.ndarray) -> np.ndarray:
    """OAS covariance of concatenated samples, shape (n_channels, n_channels)."""
    n_trials, n_channels, n_times = X.shape
    if n_trials < 1:
        raise ValueError("Need at least one trial to estimate a covariance")
    samples = X.transpose(0, 2, 1).reshape(n_trials * n_times, n_channels)
    cov = OAS().fit(samples).covariance_
    cov = 0.5 * (cov + cov.T)
    trace = float(np.trace(cov))
    if trace > 0:
        cov = cov / trace
    return cov


def composite_class_covariances(
    X_target: np.ndarray,
    y_target: np.ndarray,
    source_covs: dict[int, np.ndarray],
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return mixed (C0, C1) = (1-λ) C_target + λ C_source."""
    lam = float(np.clip(lam, 0.0, 1.0))
    mixed: dict[int, np.ndarray] = {}
    for cls in (0, 1):
        target_cls = X_target[y_target == cls]
        if len(target_cls) == 0:
            raise ValueError(f"Target training set missing class {cls}")
        ct = epoch_covariance(target_cls)
        cs = source_covs[cls]
        mixed[cls] = (1.0 - lam) * ct + lam * cs
    return mixed[0], mixed[1]


def csp_filters_from_covariances(
    C0: np.ndarray,
    C1: np.ndarray,
    n_components: int = N_CSP_COMPONENTS,
) -> np.ndarray:
    """Return (n_components, n_channels) CSP filters from two class covariances."""
    evals, evecs = eigh(C1, C0 + C1)
    order = np.argsort(np.abs(evals - 0.5))[::-1]
    return evecs[:, order[:n_components]].T


def apply_csp_logvar(X: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """Log-variance CSP features, shape (n_trials, n_components)."""
    projected = np.einsum("dc,nct->ndt", filters, X)
    var = np.var(projected, axis=2)
    var = np.maximum(var, 1e-10)
    return np.log(var)


def extract_composite_fbcsp(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    source_covs_by_band: dict[tuple[float, float], dict[int, np.ndarray]],
    bands: list[tuple[float, float]] = FBCSP_BANDS,
    lam: float = 0.3,
    n_components: int = N_CSP_COMPONENTS,
    train_cache: BandCache | None = None,
    test_cache: BandCache | None = None,
    sfreq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite FBCSP features for train/test using precomputed source covariances."""
    from src.band_cache import precompute_bandpassed

    if train_cache is None or test_cache is None:
        if sfreq is None:
            return (
                np.empty((X_train.shape[0], 0)),
                np.empty((X_test.shape[0], 0)),
            )
        train_cache = precompute_bandpassed(X_train, sfreq, bands=bands)
        test_cache = precompute_bandpassed(X_test, sfreq, bands=bands)

    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for band in bands:
        if band not in source_covs_by_band:
            continue
        if band not in train_cache or band not in test_cache:
            continue
        X_tr = train_cache[band]
        X_te = test_cache[band]
        try:
            C0, C1 = composite_class_covariances(
                X_tr, y_train, source_covs_by_band[band], lam
            )
            filters = csp_filters_from_covariances(C0, C1, n_components=n_components)
            train_parts.append(apply_csp_logvar(X_tr, filters))
            test_parts.append(apply_csp_logvar(X_te, filters))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if not train_parts:
        return (
            np.empty((X_train.shape[0], 0)),
            np.empty((X_test.shape[0], 0)),
        )
    return np.hstack(train_parts), np.hstack(test_parts)


def pool_source_class_covariances(
    source_caches: list[tuple[BandCache, np.ndarray]],
    bands: list[tuple[float, float]],
) -> dict[tuple[float, float], dict[int, np.ndarray]]:
    """Pool other subjects' band-filtered trials into one covariance per class."""
    pooled: dict[tuple[float, float], dict[int, np.ndarray]] = {}
    for band in bands:
        by_class: dict[int, list[np.ndarray]] = {0: [], 1: []}
        for cache, y in source_caches:
            if band not in cache:
                continue
            X = cache[band]
            for cls in (0, 1):
                trials = X[y == cls]
                if len(trials) > 0:
                    by_class[cls].append(trials)
        if not by_class[0] or not by_class[1]:
            continue
        pooled[band] = {
            0: epoch_covariance(np.vstack(by_class[0])),
            1: epoch_covariance(np.vstack(by_class[1])),
        }
    return pooled
