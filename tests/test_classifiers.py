"""Tests for classifier registry and adapters."""

import numpy as np
import pytest

from src.classifiers import (
    CLASSIFIER_REGISTRY,
    ClassifierAdapter,
    get_classifier,
    register_classifier,
)

N_TRIALS = 40
N_FEATURES = 10
N_CHANNELS = 8
N_SAMPLES = 375
SFREQ = 250.0


def _make_data(n: int = N_TRIALS):
    rng = np.random.default_rng(42)
    X_feat = rng.standard_normal((n, N_FEATURES))
    X_mc = rng.standard_normal((n, N_CHANNELS, N_SAMPLES))
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X_feat, X_mc, y


class TestRegistry:
    def test_builtin_classifiers_registered(self):
        assert "lda" in CLASSIFIER_REGISTRY
        assert "riemann" in CLASSIFIER_REGISTRY
        assert "svm" in CLASSIFIER_REGISTRY
        assert "ensemble" in CLASSIFIER_REGISTRY

    def test_get_classifier_returns_instance(self):
        clf = get_classifier("lda")
        assert isinstance(clf, ClassifierAdapter)

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError):
            get_classifier("nonexistent")

    def test_custom_registration(self):
        @register_classifier("test_dummy")
        class DummyClassifier(ClassifierAdapter):
            name = "test_dummy"

            def fit(
                self,
                X_feat_train,
                y_train,
                X_mc_train,
                sfreq,
                config=None,
                band_cache=None,
            ):
                pass

            def score(self, X_feat_test, y_test, X_mc_test):
                return 0.5

            def predict_proba(self, X_feat_test, X_mc_test):
                n = X_feat_test.shape[0]
                return np.full((n, 2), 0.5)

            def export(self):
                return {"type": "dummy"}

        assert "test_dummy" in CLASSIFIER_REGISTRY
        del CLASSIFIER_REGISTRY["test_dummy"]


class TestLDAClassifier:
    def test_fit_score_roundtrip(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("lda")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0

    def test_export_has_required_keys(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("lda")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        exported = clf.export()
        assert "csp_models" in exported
        assert "classifier" in exported


class TestRiemannClassifier:
    def test_fit_score_roundtrip(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("riemann")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0

    def test_export_has_pipeline(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("riemann")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        exported = clf.export()
        assert "pipeline" in exported


class TestSVMClassifier:
    def test_fit_score_roundtrip(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("svm")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0

    def test_predict_proba_shape(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("svm")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        proba = clf.predict_proba(X_feat[30:], X_mc[30:])
        assert proba.shape == (10, 2)


class TestEnsembleClassifier:
    def test_fit_score_roundtrip(self):
        X_feat, X_mc, y = _make_data()
        clf = get_classifier("ensemble")
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0
