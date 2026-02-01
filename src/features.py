"""Feature extraction: multi-channel, multi-band power computation with ERD."""

import numpy as np
from scipy.signal import welch, hilbert, butter, filtfilt
from scipy.stats import kurtosis, skew
from scipy.linalg import eigh


# Frequency bands for feature extraction
FREQUENCY_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
}


def compute_band_power(
    epoch: np.ndarray,
    sfreq: float,
    low_freq: float,
    high_freq: float,
) -> float:
    """
    Compute power in a frequency band using Welch's method.

    Args:
        epoch: 1D signal array (single epoch)
        sfreq: Sampling frequency in Hz
        low_freq: Lower bound of frequency band (Hz)
        high_freq: Upper bound of frequency band (Hz)

    Returns:
        Mean power in the frequency band
    """
    # nperseg must be <= len(epoch); use half the epoch for good frequency resolution
    nperseg = min(int(sfreq / 2), len(epoch) // 2)
    nperseg = max(nperseg, 64)  # Minimum for reasonable FFT
    freqs, psd = welch(epoch, sfreq, nperseg=nperseg, noverlap=nperseg // 2)
    mask = (freqs >= low_freq) & (freqs <= high_freq)
    return float(np.mean(psd[mask]))


def compute_theta_power(
    epoch: np.ndarray,
    sfreq: float,
    theta_low: float = 4.0,
    theta_high: float = 8.0,
) -> float:
    """Compute theta band power (backward compatible)."""
    return compute_band_power(epoch, sfreq, theta_low, theta_high)


def compute_hjorth_parameters(epoch: np.ndarray) -> tuple[float, float, float]:
    """
    Compute Hjorth parameters: Activity, Mobility, Complexity.

    These are efficient time-domain features for EEG.
    """
    # Activity = variance
    activity = np.var(epoch)

    # First derivative
    diff1 = np.diff(epoch)
    var_diff1 = np.var(diff1)

    # Second derivative
    diff2 = np.diff(diff1)
    var_diff2 = np.var(diff2)

    # Mobility = sqrt(var(diff1) / var(signal))
    mobility = np.sqrt(var_diff1 / (activity + 1e-10))

    # Complexity = mobility(diff1) / mobility(signal)
    mobility_diff1 = np.sqrt(var_diff2 / (var_diff1 + 1e-10))
    complexity = mobility_diff1 / (mobility + 1e-10)

    return activity, mobility, complexity


def compute_envelope_features(epoch: np.ndarray) -> tuple[float, float, float]:
    """
    Compute features from the signal envelope using Hilbert transform.
    """
    analytic_signal = hilbert(epoch)
    envelope = np.abs(analytic_signal)

    env_mean = np.mean(envelope)
    env_std = np.std(envelope)
    env_max = np.max(envelope)

    return env_mean, env_std, env_max


class CSP:
    """Common Spatial Patterns for EEG feature extraction."""

    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.filters_ = None
        self.mean_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit CSP filters.

        Args:
            X: EEG data of shape (n_trials, n_channels, n_samples)
            y: Labels (0 or 1)
        """
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly 2 classes")

        # Compute covariance matrices for each class
        X0 = X[y == classes[0]]
        X1 = X[y == classes[1]]

        cov0 = self._compute_covariance(X0)
        cov1 = self._compute_covariance(X1)

        # Solve generalized eigenvalue problem
        eigvals, eigvecs = eigh(cov0, cov0 + cov1)

        # Sort by eigenvalue (most discriminative first and last)
        ix = np.argsort(eigvals)
        eigvecs = eigvecs[:, ix]

        # Select filters (n_components/2 from each end)
        n = self.n_components // 2
        self.filters_ = np.concatenate([
            eigvecs[:, :n],
            eigvecs[:, -n:]
        ], axis=1)

        return self

    def _compute_covariance(self, X: np.ndarray) -> np.ndarray:
        """Compute average covariance matrix."""
        n_trials, n_channels, n_samples = X.shape
        covs = np.zeros((n_channels, n_channels))

        for trial in X:
            cov = np.dot(trial, trial.T) / n_samples
            cov /= np.trace(cov)
            covs += cov

        return covs / n_trials

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply CSP filters and compute log-variance features.

        Args:
            X: EEG data of shape (n_trials, n_channels, n_samples)

        Returns:
            Features of shape (n_trials, n_components)
        """
        if self.filters_ is None:
            raise ValueError("CSP not fitted")

        n_trials = X.shape[0]
        features = np.zeros((n_trials, self.n_components))

        for i, trial in enumerate(X):
            # Apply spatial filters
            filtered = np.dot(self.filters_.T, trial)
            # Compute log-variance
            var = np.var(filtered, axis=1)
            features[i] = np.log(var + 1e-10)

        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


def compute_time_domain_features(epoch: np.ndarray) -> list[float]:
    """
    Compute comprehensive time-domain features from an epoch.
    """
    features = []

    # Basic statistics
    features.append(np.mean(epoch))
    features.append(np.std(epoch))
    features.append(np.var(epoch))
    features.append(skew(epoch))
    features.append(kurtosis(epoch))

    # Range and percentiles
    features.append(np.max(epoch) - np.min(epoch))
    features.append(np.percentile(epoch, 75) - np.percentile(epoch, 25))

    # Zero crossings
    zero_crossings = np.sum(np.diff(np.sign(epoch)) != 0)
    features.append(zero_crossings / len(epoch))

    # Hjorth parameters
    activity, mobility, complexity = compute_hjorth_parameters(epoch)
    features.extend([activity, mobility, complexity])

    # Envelope features
    env_mean, env_std, env_max = compute_envelope_features(epoch)
    features.extend([env_mean, env_std, env_max])

    return features


def extract_multichannel_features(
    epochs_by_channel: dict[str, list[np.ndarray]],
    sfreq: float,
    log_transform: bool = True,
) -> np.ndarray:
    """
    Extract mu band features from motor cortex channels.

    Focus on mu (8-12 Hz) power which shows strong ERD during motor imagery.

    Args:
        epochs_by_channel: Dict mapping channel names to list of epochs
        sfreq: Sampling frequency in Hz
        log_transform: Whether to apply log transform

    Returns:
        Feature array of shape (n_epochs, n_features)
    """
    # Motor-imagery relevant bands - focus on mu and beta only
    mi_bands = {
        "mu": (8, 12),
        "beta": (13, 30),
        "theta": (4, 8),
    }

    # Motor cortex channels - most important for motor imagery
    motor_channels = ["C3", "C4", "Cz"]

    channel_names = list(epochs_by_channel.keys())
    n_epochs = len(epochs_by_channel[channel_names[0]])

    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []
        channel_band_powers = {}

        for ch_name in motor_channels:
            if ch_name not in epochs_by_channel:
                continue

            epoch = epochs_by_channel[ch_name][epoch_idx]

            # Compute total power for normalization
            total_power = compute_band_power(epoch, sfreq, 1, 40)

            # Compute band powers
            band_powers = {}
            for band_name, (low, high) in mi_bands.items():
                power = compute_band_power(epoch, sfreq, low, high)
                band_powers[band_name] = power

                # Log power
                if log_transform:
                    epoch_features.append(np.log(power + 1e-10))
                else:
                    epoch_features.append(power)

                # Relative power (normalized by total)
                epoch_features.append(power / (total_power + 1e-10))

            channel_band_powers[ch_name] = band_powers

            # Mu/Beta ratio (key ERD marker)
            mu_beta_ratio = band_powers["mu"] / (band_powers["beta"] + 1e-10)
            epoch_features.append(np.log(mu_beta_ratio + 1e-10))

        # C3-C4 mu asymmetry (lateralization)
        if "C3" in channel_band_powers and "C4" in channel_band_powers:
            c3_mu = channel_band_powers["C3"]["mu"]
            c4_mu = channel_band_powers["C4"]["mu"]
            mu_asymmetry = np.log((c3_mu + 1e-10) / (c4_mu + 1e-10))
            epoch_features.append(mu_asymmetry)

            # Beta asymmetry
            c3_beta = channel_band_powers["C3"]["beta"]
            c4_beta = channel_band_powers["C4"]["beta"]
            beta_asymmetry = np.log((c3_beta + 1e-10) / (c4_beta + 1e-10))
            epoch_features.append(beta_asymmetry)

        all_features.append(epoch_features)

    return np.array(all_features)


def extract_features(
    epochs: list[np.ndarray],
    sfreq: float,
    log_transform: bool = True,
) -> np.ndarray:
    """
    Extract theta power features from a list of epochs (backward compatible).

    Args:
        epochs: List of 1D epoch arrays
        sfreq: Sampling frequency in Hz
        log_transform: Whether to apply log transform to power values

    Returns:
        Feature array of shape (n_epochs, 1)
    """
    features = []

    for epoch in epochs:
        theta_power = compute_theta_power(epoch, sfreq)
        if log_transform:
            theta_power = np.log(theta_power + 1e-10)
        features.append([theta_power])

    return np.array(features)


def extract_erd_features(
    epoch_pairs_by_channel: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    sfreq: float,
) -> np.ndarray:
    """
    Extract comprehensive ERD features from all motor cortex channels.

    ERD = (baseline_power - task_power) / baseline_power * 100

    Args:
        epoch_pairs_by_channel: Dict mapping channel names to list of (baseline, task) pairs
        sfreq: Sampling frequency in Hz

    Returns:
        Feature array of shape (n_epochs, n_features)
    """
    # More bands for better discrimination
    bands = {
        "theta": (4, 8),
        "low_alpha": (8, 10),
        "high_alpha": (10, 13),
        "mu": (8, 12),
        "low_beta": (13, 20),
        "high_beta": (20, 30),
        "beta": (13, 30),
        "gamma": (30, 40),
    }

    # All relevant channels
    motor_channels = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]

    channel_names = list(epoch_pairs_by_channel.keys())
    n_epochs = len(epoch_pairs_by_channel[channel_names[0]])

    all_features = []

    for epoch_idx in range(n_epochs):
        epoch_features = []
        channel_powers = {}

        for ch_name in motor_channels:
            if ch_name not in epoch_pairs_by_channel:
                continue

            baseline, task = epoch_pairs_by_channel[ch_name][epoch_idx]
            channel_powers[ch_name] = {}

            # Compute total power for normalization
            baseline_total = compute_band_power(baseline, sfreq, 1, 40)
            task_total = compute_band_power(task, sfreq, 1, 40)

            for band_name, (low, high) in bands.items():
                baseline_power = compute_band_power(baseline, sfreq, low, high)
                task_power = compute_band_power(task, sfreq, low, high)

                # ERD percentage
                erd = (baseline_power - task_power) / (baseline_power + 1e-10) * 100
                epoch_features.append(erd)

                # Log power ratio
                log_ratio = np.log((task_power + 1e-10) / (baseline_power + 1e-10))
                epoch_features.append(log_ratio)

                # Relative power (normalized)
                baseline_rel = baseline_power / (baseline_total + 1e-10)
                task_rel = task_power / (task_total + 1e-10)
                epoch_features.append(task_rel - baseline_rel)

                # Absolute log power (often more discriminative)
                epoch_features.append(np.log(task_power + 1e-10))
                epoch_features.append(np.log(baseline_power + 1e-10))

                # Store for inter-channel features
                channel_powers[ch_name][band_name] = {
                    'baseline': baseline_power,
                    'task': task_power,
                    'erd': erd,
                }

            # Add time-domain features from task epoch
            activity, mobility, complexity = compute_hjorth_parameters(task)
            epoch_features.extend([activity, mobility, complexity])

            # Envelope features
            env_mean, env_std, env_max = compute_envelope_features(task)
            epoch_features.extend([env_mean, env_std, env_max])

        # Inter-channel features: C3-C4 asymmetry for each band
        if "C3" in channel_powers and "C4" in channel_powers:
            for band_name in bands:
                c3_erd = channel_powers["C3"][band_name]['erd']
                c4_erd = channel_powers["C4"][band_name]['erd']
                epoch_features.append(c3_erd - c4_erd)

                # Power ratio
                c3_task = channel_powers["C3"][band_name]['task']
                c4_task = channel_powers["C4"][band_name]['task']
                epoch_features.append(np.log((c3_task + 1e-10) / (c4_task + 1e-10)))

        # Frontal-Parietal connectivity proxy (Fz vs Pz)
        if "Fz" in channel_powers and "Pz" in channel_powers:
            for band_name in ["theta", "mu", "beta"]:
                fz_task = channel_powers["Fz"][band_name]['task']
                pz_task = channel_powers["Pz"][band_name]['task']
                epoch_features.append(np.log((fz_task + 1e-10) / (pz_task + 1e-10)))

        # Central vs occipital (Cz vs Oz)
        if "Cz" in channel_powers and "Oz" in channel_powers:
            for band_name in ["mu", "beta"]:
                cz_task = channel_powers["Cz"][band_name]['task']
                oz_task = channel_powers["Oz"][band_name]['task']
                epoch_features.append(np.log((cz_task + 1e-10) / (oz_task + 1e-10)))

        all_features.append(epoch_features)

    return np.array(all_features)
