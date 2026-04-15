"""Tests for artifact rejection module."""

import numpy as np

from src.rejection import reject_in_fold


class TestRejectInFold:
    def test_none_ptps_returns_unchanged(self):
        train_idx = np.array([0, 1, 2, 3])
        test_idx = np.array([4, 5])
        clean_train, clean_test = reject_in_fold(None, train_idx, test_idx)
        np.testing.assert_array_equal(clean_train, train_idx)
        np.testing.assert_array_equal(clean_test, test_idx)

    def test_removes_high_amplitude_trials(self):
        ptps = np.array([50, 60, 500, 55, 65, 45, 70, 50, 55, 60], dtype=float)
        train_idx = np.arange(8)
        test_idx = np.array([8, 9])
        clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)
        assert 2 not in clean_train

    def test_removes_flat_trials(self):
        ptps = np.array([50, 60, 0.5, 55, 65, 45, 70, 50, 55, 60], dtype=float)
        train_idx = np.arange(8)
        test_idx = np.array([8, 9])
        clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)
        assert 2 not in clean_train

    def test_threshold_from_training_only(self):
        ptps = np.array([10, 12, 11, 13, 10, 200], dtype=float)
        train_idx = np.array([0, 1, 2, 3, 4])
        test_idx = np.array([5])
        clean_train, clean_test = reject_in_fold(ptps, train_idx, test_idx)
        assert len(clean_test) == 0

    def test_custom_n_mad(self):
        ptps = np.array([50, 60, 120, 55, 65, 45, 70, 50, 55, 60], dtype=float)
        train_idx = np.arange(8)
        test_idx = np.array([8, 9])
        clean_tight, _ = reject_in_fold(ptps, train_idx, test_idx, n_mad=1.0)
        clean_loose, _ = reject_in_fold(ptps, train_idx, test_idx, n_mad=10.0)
        assert len(clean_tight) <= len(clean_loose)
