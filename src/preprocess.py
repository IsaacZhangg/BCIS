"""EEG preprocessing: filtering and artifact removal."""

import numpy as np
import mne
from asrpy import ASR


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 1.0,
    h_freq: float = 40.0,
) -> np.ndarray:
    """Apply bandpass filter to a 1D signal."""
    return mne.filter.filter_data(
        data.reshape(1, -1), sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False
    ).flatten()


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    freq: float = 60.0,
) -> np.ndarray:
    """Apply notch filter to remove power line interference from a 1D signal."""
    return mne.filter.notch_filter(
        data.reshape(1, -1), sfreq, freqs=freq, verbose=False
    ).flatten()


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """Apply Common Average Reference (CAR) spatial filter."""
    return data - data.mean(axis=0, keepdims=True)


def _patch_asrpy_numpy2() -> None:
    """Patch asrpy for numpy 2.x: squeeze np.diff() results to scalars."""
    import asrpy.asr as _asr
    import asrpy.asr_utils as _au
    from scipy.special import gamma, gammaincinv

    _orig = _au.fit_eeg_distribution
    if getattr(_orig, "_patched", False):
        return

    def _fixed(
        X,
        min_clean_fraction=0.25,
        max_dropout_fraction=0.1,
        fit_quantiles=(0.022, 0.6),
        step_sizes=(0.01, 0.01),
        shape_range=np.arange(1.7, 3.5, 0.15),
    ):
        X = np.sort(X)
        n = len(X)
        quants = np.array(fit_quantiles)
        zbounds, rescale = [], []
        for b in range(len(shape_range)):
            gam = gammaincinv(
                1 / shape_range[b], np.sign(quants - 0.5) * (2 * quants - 1)
            )
            zbounds.append(np.sign(quants - 0.5) * gam ** (1 / shape_range[b]))
            rescale.append(shape_range[b] / (2 * gamma(1 / shape_range[b])))

        lower_min = float(np.min(quants))
        max_width = np.diff(quants).item()  # ← fix: squeeze to scalar
        min_width = min_clean_fraction * max_width

        cols = np.arange(
            lower_min,
            lower_min + max_dropout_fraction + step_sizes[0] * 1e-9,
            step_sizes[0],
        )
        cols = np.round(n * cols).astype(int)
        rows = np.arange(0, int(np.round(n * max_width)))
        newX = np.zeros((len(rows), len(cols)))
        for i, c in enumerate(range(len(rows))):
            newX[i] = X[c + cols]

        X1 = newX[0, :]
        newX = newX - X1

        opt_val = np.inf
        opt_lu = np.inf
        opt_bounds = np.inf
        opt_beta = np.inf
        gridsearch = np.round(n * np.arange(max_width, min_width, -step_sizes[1]))
        for m in gridsearch.astype(int):
            mcurr = m - 1
            nbins = int(np.round(3 * np.log2(1 + m / 2)))
            c = nbins / newX[mcurr]
            H = newX[:m] * c
            hist_all = []
            for ih in range(len(c)):
                hist_all.append(np.histogram(H[:, ih], bins=np.arange(0, nbins + 1))[0])
            hist_all = np.array(hist_all, dtype=int).T
            hist_all = np.vstack((hist_all, np.zeros(len(c), dtype=int)))
            logq = np.log(hist_all + 0.01)

            for k, b in enumerate(shape_range):
                bounds = zbounds[k]
                x = bounds[0] + np.arange(0.5, nbins + 0.5) / nbins * np.diff(bounds)
                p = np.exp(-(np.abs(x) ** b)) * rescale[k]
                p = p / np.sum(p)
                kl = np.sum(p * (np.log(p) - logq[:-1, :].T), axis=1) + np.log(m)
                min_val = np.min(kl)
                idx = np.argmin(kl)
                if min_val < opt_val:
                    opt_val = min_val
                    opt_beta = shape_range[k]
                    opt_bounds = bounds
                    opt_lu = [X1[idx], X1[idx] + newX[m - 1, idx]]

        alpha = (opt_lu[1] - opt_lu[0]) / np.diff(opt_bounds).item()
        mu = opt_lu[0] - opt_bounds[0] * alpha
        beta = opt_beta
        sig = np.sqrt((alpha**2) * gamma(3 / beta) / gamma(1 / beta))
        return mu, sig, alpha, beta

    _fixed._patched = True
    _au.fit_eeg_distribution = _fixed
    _asr.fit_eeg_distribution = _fixed


_patch_asrpy_numpy2()


def apply_asr(
    data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    cutoff: float = 20.0,
) -> np.ndarray:
    """Apply Artifact Subspace Reconstruction to multichannel EEG."""
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    asr = ASR(sfreq=sfreq, cutoff=cutoff)
    asr.fit(raw)
    raw_clean = asr.transform(raw)

    return raw_clean.get_data()


def preprocess_eeg(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Full preprocessing pipeline: bandpass + notch filter on a single channel."""
    filtered = bandpass_filter(data, sfreq)
    return notch_filter(filtered, sfreq)
