"""Tests for band-filtered data caching."""

import numpy as np

from src.band_cache import (
    FBCSP_BANDS,
    N_CSP_COMPONENTS,
    precompute_bandpassed,
    subset_band_cache,
)

N_TRIALS = 10
N_CHANNELS = 8
N_SAMPLES = 375  # 1.5s at 250Hz
SFREQ = 250.0


def _make_multichannel(n_trials: int = N_TRIALS) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n_trials, N_CHANNELS, N_SAMPLES))


class TestPrecomputeBandpassed:
    def test_returns_dict_keyed_by_band_tuples(self):
        X = _make_multichannel()
        cache = precompute_bandpassed(X, SFREQ)
        assert isinstance(cache, dict)
        for key in cache:
            assert isinstance(key, tuple)
            assert len(key) == 2

    def test_each_band_preserves_shape(self):
        X = _make_multichannel()
        cache = precompute_bandpassed(X, SFREQ)
        for band, X_band in cache.items():
            assert X_band.shape == X.shape

    def test_custom_bands(self):
        X = _make_multichannel()
        bands = [(8, 12), (12, 30)]
        cache = precompute_bandpassed(X, SFREQ, bands=bands)
        assert set(cache.keys()) == {(8, 12), (12, 30)}

    def test_default_bands_match_constant(self):
        X = _make_multichannel()
        cache = precompute_bandpassed(X, SFREQ)
        assert set(cache.keys()) == set(FBCSP_BANDS)


class TestSubsetBandCache:
    def test_subsets_by_trial_indices(self):
        X = _make_multichannel(20)
        cache = precompute_bandpassed(X, SFREQ)
        indices = np.array([0, 5, 10, 15])
        subset = subset_band_cache(cache, indices)
        for band in cache:
            assert subset[band].shape[0] == 4
            np.testing.assert_array_equal(subset[band], cache[band][indices])

    def test_preserves_all_bands(self):
        X = _make_multichannel()
        cache = precompute_bandpassed(X, SFREQ)
        indices = np.array([0, 1, 2])
        subset = subset_band_cache(cache, indices)
        assert set(subset.keys()) == set(cache.keys())


class TestConstants:
    def test_fbcsp_bands_are_8_bands(self):
        assert len(FBCSP_BANDS) == 8

    def test_bands_cover_8_to_30_hz(self):
        assert FBCSP_BANDS[0][0] == 8
        assert FBCSP_BANDS[-1][1] == 30

    def test_n_csp_components(self):
        assert N_CSP_COMPONENTS == 3
