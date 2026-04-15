"""Band-filtered data caching for FBCSP pipelines."""

from collections.abc import Sequence

import mne
import numpy as np

FBCSP_BANDS: list[tuple[float, float]] = [
    (8, 10),
    (10, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 24),
    (24, 30),
]

N_CSP_COMPONENTS: int = 3

BandCache = dict[tuple[float, float], np.ndarray]


def precompute_bandpassed(
    X: np.ndarray,
    sfreq: float,
    bands: Sequence[tuple[float, float]] = FBCSP_BANDS,
) -> BandCache:
    """Bandpass every trial for each FBCSP band once.

    Filtering is trial-wise along the time axis, so cached per-band tensors can be
    safely indexed for CV train/test splits without changing leakage boundaries.
    """
    filtered_by_band: BandCache = {}
    for low, high in bands:
        try:
            filtered_by_band[(low, high)] = mne.filter.filter_data(
                X, sfreq, low, high, verbose=False
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    return filtered_by_band


def subset_band_cache(cache: BandCache, indices: np.ndarray) -> BandCache:
    """Slice a precomputed band cache by trial indices."""
    return {band: X_band[indices] for band, X_band in cache.items()}
