"""Integration tests for the refactored pipeline.

These tests verify that the module extractions (band_cache, rejection,
classifiers, experiments) work correctly when composed together through
the existing train.py orchestration functions.
"""

import numpy as np
import pytest

from src.band_cache import precompute_bandpassed, subset_band_cache
from src.classifiers import get_classifier
from src.rejection import reject_in_fold
from src.train import (
    train_within_subject_cv_all_models,
    train_nested_model_selection_cv,
)

N_TRIALS = 40
N_FEATURES = 10
N_CHANNELS = 8
N_SAMPLES = 375
SFREQ = 250.0


def _make_subject_data(n_trials: int = N_TRIALS, seed: int = 42):
    rng = np.random.default_rng(seed)
    X_feat = rng.standard_normal((n_trials, N_FEATURES))
    X_mc = rng.standard_normal((n_trials, N_CHANNELS, N_SAMPLES))
    y = np.array([0] * (n_trials // 2) + [1] * (n_trials // 2))
    return X_feat, X_mc, y


class TestBandCacheIntegration:
    def test_cache_subset_in_cv_fold(self):
        """Band cache subsetting works correctly in a CV fold context."""
        _, X_mc, y = _make_subject_data()
        cache = precompute_bandpassed(X_mc, SFREQ)

        train_idx = np.arange(30)
        test_idx = np.arange(30, 40)

        train_cache = subset_band_cache(cache, train_idx)
        test_cache = subset_band_cache(cache, test_idx)

        for band in cache:
            np.testing.assert_array_equal(train_cache[band], cache[band][train_idx])
            np.testing.assert_array_equal(test_cache[band], cache[band][test_idx])


class TestRejectionIntegration:
    def test_rejection_removes_outlier(self):
        """Rejection + classifier pipeline works end-to-end."""
        X_feat, X_mc, y = _make_subject_data()
        ptps = np.abs(X_mc).max(axis=(1, 2))
        ptps[5] = ptps.max() * 100  # Make one trial an outlier

        train_idx = np.arange(30)
        test_idx = np.arange(30, 40)
        clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)

        assert 5 not in clean_train
        assert len(clean_train) < len(train_idx)


class TestClassifierIntegration:
    @pytest.mark.parametrize("name", ["lda", "riemann", "svm", "ensemble"])
    def test_classifier_fit_score(self, name):
        """Each registered classifier can fit and score."""
        X_feat, X_mc, y = _make_subject_data()
        clf = get_classifier(name)
        clf.fit(X_feat[:30], y[:30], X_mc[:30], SFREQ)
        score = clf.score(X_feat[30:], y[30:], X_mc[30:])
        assert 0.0 <= score <= 1.0


class TestCVIntegration:
    def test_all_models_cv_still_works(self):
        """train_within_subject_cv_all_models produces valid results after refactor."""
        X_feat, X_mc, y = _make_subject_data()
        results = train_within_subject_cv_all_models(
            [(X_feat, X_mc)],
            [y],
            sfreq=SFREQ,
            n_folds=3,
            enable_band_cache=True,
        )
        for name in ("lda", "riemann", "svm", "ensemble"):
            scores, mean, std = results[name]
            assert len(scores) == 1
            assert 0.0 <= mean <= 1.0

    def test_nested_cv_still_works(self):
        """train_nested_model_selection_cv produces valid results after refactor."""
        X_feat, X_mc, y = _make_subject_data()
        scores, mean, std, methods = train_nested_model_selection_cv(
            [(X_feat, X_mc)],
            [y],
            sfreq=SFREQ,
            n_outer_folds=3,
            n_inner_folds=2,
            enable_band_cache=True,
        )
        assert len(scores) == 1
        assert 0.0 <= mean <= 1.0
        assert len(methods) == 1
        assert methods[0] in ("lda", "riemann", "svm", "ensemble")


class TestNoLeakage:
    def test_random_data_below_70_percent(self):
        """Random data should not exceed 70% accuracy (leakage check)."""
        rng = np.random.default_rng(99)
        X_feat = rng.standard_normal((60, N_FEATURES))
        X_mc = rng.standard_normal((60, N_CHANNELS, N_SAMPLES))
        y = np.array([0] * 30 + [1] * 30)

        results = train_within_subject_cv_all_models(
            [(X_feat, X_mc)],
            [y],
            sfreq=SFREQ,
            n_folds=5,
            enable_band_cache=True,
        )
        for name in ("lda", "riemann", "svm", "ensemble"):
            _, mean, _ = results[name]
            assert mean < 0.70, (
                f"{name} scored {mean:.1%} on random data — possible leakage"
            )
