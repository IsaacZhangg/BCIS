"""Integration tests for the CV entry points the pipeline calls."""

import numpy as np

from src.train import (
    train_nested_model_selection_cv,
    train_within_subject_cv_all_models,
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


class TestCVIntegration:
    def test_all_models_cv_still_works(self):
        """train_within_subject_cv_all_models produces valid results."""
        X_feat, X_mc, y = _make_subject_data()
        results = train_within_subject_cv_all_models(
            [(X_feat, X_mc)],
            [y],
            sfreq=SFREQ,
            n_folds=3,
        )
        for name in ("lda", "riemann", "svm", "ensemble"):
            scores, mean, std = results[name]
            assert len(scores) == 1
            assert 0.0 <= mean <= 1.0

    def test_nested_cv_still_works(self):
        """train_nested_model_selection_cv produces valid results."""
        X_feat, X_mc, y = _make_subject_data()
        scores, mean, std, methods, _band_configs, _stacking = (
            train_nested_model_selection_cv(
                [(X_feat, X_mc)],
                [y],
                sfreq=SFREQ,
                n_outer_folds=3,
                n_inner_folds=2,
            )
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
        )
        for name in ("lda", "riemann", "svm", "ensemble"):
            _, mean, _ = results[name]
            assert mean < 0.70, (
                f"{name} scored {mean:.1%} on random data — possible leakage"
            )
