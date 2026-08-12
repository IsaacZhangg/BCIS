"""Classifier registry with protocol-based adapters.

Each classifier family (FBCSP-based, Riemannian, Ensemble) exposes fit, score,
predict_proba, and export methods. Adding a new classifier = one adapter class +
one @register_classifier decorator.

This module is a NEW parallel interface — the existing CV orchestration in
train.py continues to use its own inline classifier construction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import mne
import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.band_cache import (
    FBCSP_BANDS,
    N_CSP_COMPONENTS,
    BandCache,
    precompute_bandpassed,
)

DEFAULT_K_CANDIDATES = (3, 5, 8, 10, 15, 20, 25)


class ClassifierAdapter(ABC):
    """Base class for classifier adapters."""

    name: str

    @abstractmethod
    def fit(
        self,
        X_feat_train: np.ndarray,
        y_train: np.ndarray,
        X_mc_train: np.ndarray,
        sfreq: float,
        config: Any | None = None,
        band_cache: BandCache | None = None,
    ) -> None: ...

    @abstractmethod
    def score(
        self,
        X_feat_test: np.ndarray,
        y_test: np.ndarray,
        X_mc_test: np.ndarray,
    ) -> float: ...

    @abstractmethod
    def predict_proba(
        self,
        X_feat_test: np.ndarray,
        X_mc_test: np.ndarray,
    ) -> np.ndarray: ...

    @abstractmethod
    def export(self) -> dict[str, Any]: ...


CLASSIFIER_REGISTRY: dict[str, type[ClassifierAdapter]] = {}


def register_classifier(name: str):
    """Decorator to register a classifier adapter class."""

    def decorator(cls: type[ClassifierAdapter]) -> type[ClassifierAdapter]:
        CLASSIFIER_REGISTRY[name] = cls
        return cls

    return decorator


def get_classifier(name: str) -> ClassifierAdapter:
    """Instantiate a registered classifier by name."""
    if name not in CLASSIFIER_REGISTRY:
        raise KeyError(
            f"Unknown classifier: {name!r}. Available: {list(CLASSIFIER_REGISTRY)}"
        )
    return CLASSIFIER_REGISTRY[name]()


def _safe_k_values(
    n_features: int,
    n_train_trials: int,
    k_best: int = 10,
    k_candidates: tuple[int, ...] = DEFAULT_K_CANDIDATES,
) -> list[int]:
    """Return conservative K values to reduce overfitting on small folds."""
    if n_features <= 0:
        return []
    k_cap = min(n_features, max(2, n_train_trials // 3))
    candidates = sorted(k for k in k_candidates if 1 <= k <= k_cap)
    return candidates or [min(k_best, k_cap)]


def _extract_fbcsp(
    X_mc_train: np.ndarray,
    X_mc_test: np.ndarray | None,
    y_train: np.ndarray,
    sfreq: float,
    bands: Sequence[tuple[float, float]] = FBCSP_BANDS,
    n_components: int = N_CSP_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray | None, list[tuple[CSP, tuple[float, float]]]]:
    """Extract FBCSP features for train (and optionally test) data."""
    train_filtered = precompute_bandpassed(X_mc_train, sfreq, bands=list(bands))

    if X_mc_test is not None and X_mc_test is not X_mc_train:
        test_filtered = precompute_bandpassed(X_mc_test, sfreq, bands=list(bands))
    else:
        test_filtered = train_filtered

    csp_models: list[tuple[CSP, tuple[float, float]]] = []
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for low, high in bands:
        band = (low, high)
        if band not in train_filtered or band not in test_filtered:
            continue
        try:
            csp = CSP(n_components=n_components, reg="oas", log=True, norm_trace=True)
            train_parts.append(csp.fit_transform(train_filtered[band], y_train))
            test_parts.append(csp.transform(test_filtered[band]))
            csp_models.append((csp, band))
        except (ValueError, np.linalg.LinAlgError):
            continue

    if not train_parts:
        n_train = X_mc_train.shape[0]
        n_test = X_mc_test.shape[0] if X_mc_test is not None else n_train
        return np.empty((n_train, 0)), np.empty((n_test, 0)), []

    fbcsp_train = np.hstack(train_parts)
    fbcsp_test = np.hstack(test_parts) if X_mc_test is not None else None
    return fbcsp_train, fbcsp_test, csp_models


def _transform_test_fbcsp(
    X_mc_test: np.ndarray,
    sfreq: float,
    csp_models: list[tuple[CSP, tuple[float, float]]],
) -> np.ndarray:
    """Apply pre-fitted CSP models to test data."""
    parts = []
    for csp, (low, high) in csp_models:
        X_filt = mne.filter.filter_data(X_mc_test, sfreq, low, high, verbose=False)
        parts.append(csp.transform(X_filt))
    return np.hstack(parts) if parts else np.empty((X_mc_test.shape[0], 0))


class _FBCSPBase(ClassifierAdapter):
    """Shared FBCSP feature extraction for LDA/SVM adapters."""

    def __init__(self):
        self._csp_models: list[tuple[CSP, tuple[float, float]]] = []
        self._selector: SelectKBest | None = None
        self._scaler: StandardScaler | None = None
        self._sfreq: float = 250.0

    def _prepare_train(
        self,
        X_feat_train: np.ndarray,
        y_train: np.ndarray,
        X_mc_train: np.ndarray,
        sfreq: float,
    ) -> np.ndarray:
        """Extract FBCSP, concat handcrafted, select K best, scale. Returns scaled training features."""
        self._sfreq = sfreq
        fbcsp_train, _, self._csp_models = _extract_fbcsp(
            X_mc_train, X_mc_train, y_train, sfreq
        )
        X_combined = np.hstack([fbcsp_train, X_feat_train])

        k_values = _safe_k_values(
            n_features=X_combined.shape[1],
            n_train_trials=len(y_train),
        )
        k = k_values[-1] if k_values else min(10, max(1, X_combined.shape[1]))

        self._selector = SelectKBest(f_classif, k=k)
        X_selected = self._selector.fit_transform(X_combined, y_train)
        self._scaler = StandardScaler()
        return self._scaler.fit_transform(X_selected)

    def _prepare_test(
        self,
        X_feat_test: np.ndarray,
        X_mc_test: np.ndarray,
    ) -> np.ndarray:
        """Transform test data using fitted CSP/selector/scaler."""
        fbcsp_test = _transform_test_fbcsp(X_mc_test, self._sfreq, self._csp_models)
        X_combined = np.hstack([fbcsp_test, X_feat_test])
        X_selected = self._selector.transform(X_combined)
        return self._scaler.transform(X_selected)

    def _base_export(self) -> dict[str, Any]:
        return {
            "csp_models": self._csp_models,
            "selector": self._selector,
            "scaler": self._scaler,
            "sfreq": self._sfreq,
        }


@register_classifier("lda")
class LDAClassifier(_FBCSPBase):
    name = "lda"

    def __init__(self):
        super().__init__()
        self._clf: LinearDiscriminantAnalysis | None = None

    def fit(
        self, X_feat_train, y_train, X_mc_train, sfreq, config=None, band_cache=None
    ):
        X_scaled = self._prepare_train(X_feat_train, y_train, X_mc_train, sfreq)
        _, counts = np.unique(y_train, return_counts=True)
        priors = counts / counts.sum()
        self._clf = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=priors
        )
        self._clf.fit(X_scaled, y_train)

    def score(self, X_feat_test, y_test, X_mc_test):
        X_scaled = self._prepare_test(X_feat_test, X_mc_test)
        return float(self._clf.score(X_scaled, y_test))

    def predict_proba(self, X_feat_test, X_mc_test):
        X_scaled = self._prepare_test(X_feat_test, X_mc_test)
        return self._clf.predict_proba(X_scaled)

    def export(self):
        result = self._base_export()
        result["classifier"] = self._clf
        return result


@register_classifier("riemann")
class RiemannianClassifier(ClassifierAdapter):
    name = "riemann"

    def __init__(self):
        self._pipeline = None
        self._sfreq: float = 250.0

    def fit(
        self, X_feat_train, y_train, X_mc_train, sfreq, config=None, band_cache=None
    ):
        self._sfreq = sfreq
        self._pipeline = make_pipeline(
            Covariances(estimator="lwf"),
            TangentSpace(metric="riemann"),
            LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000),
        )
        self._pipeline.fit(X_mc_train, y_train)

    def score(self, X_feat_test, y_test, X_mc_test):
        return float(self._pipeline.score(X_mc_test, y_test))

    def predict_proba(self, X_feat_test, X_mc_test):
        return self._pipeline.predict_proba(X_mc_test)

    def export(self):
        return {"pipeline": self._pipeline, "sfreq": self._sfreq}


@register_classifier("svm")
class SVMClassifier(_FBCSPBase):
    name = "svm"

    def __init__(self):
        super().__init__()
        self._clf: SVC | None = None

    def fit(
        self, X_feat_train, y_train, X_mc_train, sfreq, config=None, band_cache=None
    ):
        X_scaled = self._prepare_train(X_feat_train, y_train, X_mc_train, sfreq)
        self._clf = SVC(
            kernel="rbf", C=20.0, gamma="scale", probability=True, random_state=42
        )
        self._clf.fit(X_scaled, y_train)

    def score(self, X_feat_test, y_test, X_mc_test):
        X_scaled = self._prepare_test(X_feat_test, X_mc_test)
        return float(self._clf.score(X_scaled, y_test))

    def predict_proba(self, X_feat_test, X_mc_test):
        X_scaled = self._prepare_test(X_feat_test, X_mc_test)
        return self._clf.predict_proba(X_scaled)

    def export(self):
        result = self._base_export()
        result["classifier"] = self._clf
        return result


@register_classifier("ensemble")
class EnsembleClassifier(ClassifierAdapter):
    """Soft-voting ensemble of LDA + SVM on shared FBCSP features."""

    name = "ensemble"

    def __init__(self):
        self._lda = LDAClassifier()
        self._svm = SVMClassifier()

    def fit(
        self, X_feat_train, y_train, X_mc_train, sfreq, config=None, band_cache=None
    ):
        self._lda.fit(X_feat_train, y_train, X_mc_train, sfreq, config, band_cache)
        self._svm.fit(X_feat_train, y_train, X_mc_train, sfreq, config, band_cache)

    def score(self, X_feat_test, y_test, X_mc_test):
        proba = self.predict_proba(X_feat_test, X_mc_test)
        preds = np.argmax(proba, axis=1)
        return float(np.mean(preds == y_test))

    def predict_proba(self, X_feat_test, X_mc_test):
        proba_lda = self._lda.predict_proba(X_feat_test, X_mc_test)
        proba_svm = self._svm.predict_proba(X_feat_test, X_mc_test)
        return (proba_lda + proba_svm) / 2

    def export(self):
        return self._lda.export()
