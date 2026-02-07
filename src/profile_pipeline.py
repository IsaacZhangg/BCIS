"""Profile the EEG pipeline to identify bottlenecks and measure feature dimensionality."""

import time
from pathlib import Path

import numpy as np

from src.data_loader import load_recording, get_complete_recordings, CHANNELS
from src.preprocess import (
    bandpass_filter,
    notch_filter,
    apply_car,
    preprocess_eeg_multichannel,
    preprocess_eeg,
)
from src.epochs import extract_augmented_epochs
from src.features import (
    compute_band_power,
    compute_hjorth_parameters,
    compute_envelope_features,
    compute_spectral_entropy,
    compute_filter_bank_features,
    compute_temporal_features,
    compute_asymmetry_features,
    extract_erd_features,
    extract_realtime_features,
)


def time_it(func, *args, n_runs=1, **kwargs):
    """Time a function, return (result, elapsed_ms)."""
    times = []
    result = None
    for _ in range(n_runs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return result, np.mean(times), np.std(times)


def profile_preprocessing(data, sample_rate):
    """Profile each preprocessing step."""
    print("\n" + "=" * 60)
    print("PREPROCESSING PROFILING")
    print("=" * 60)

    single_channel = data[0]
    n_channels = data.shape[0]

    # Bandpass filter - single channel
    _, bp_ms, bp_std = time_it(
        bandpass_filter, single_channel, sample_rate, 1.0, 40.0, n_runs=3
    )
    print(f"  Bandpass filter (1 channel):    {bp_ms:8.2f} ms (+/- {bp_std:.2f})")
    print(f"  Bandpass filter (8 channels):   {bp_ms * n_channels:8.2f} ms (estimated)")

    # Notch filter - single channel
    _, notch_ms, notch_std = time_it(
        notch_filter, single_channel, sample_rate, 60.0, n_runs=3
    )
    print(f"  Notch filter (1 channel):       {notch_ms:8.2f} ms (+/- {notch_std:.2f})")
    print(
        f"  Notch filter (8 channels):      {notch_ms * n_channels:8.2f} ms (estimated)"
    )

    # CAR spatial filter
    _, car_ms, car_std = time_it(apply_car, data, n_runs=3)
    print(f"  CAR spatial filter:             {car_ms:8.2f} ms (+/- {car_std:.2f})")

    # Full multichannel preprocessing
    _, full_ms, full_std = time_it(
        preprocess_eeg_multichannel, data, sample_rate, CHANNELS, "car", n_runs=3
    )
    print(f"  Full preprocess (8 ch):         {full_ms:8.2f} ms (+/- {full_std:.2f})")

    # Single channel preprocess (used in realtime simulation)
    _, single_ms, single_std = time_it(
        preprocess_eeg, single_channel, sample_rate, n_runs=3
    )
    print(
        f"  preprocess_eeg (1 ch):          {single_ms:8.2f} ms (+/- {single_std:.2f})"
    )
    print(
        f"  preprocess_eeg (8 ch):          {single_ms * n_channels:8.2f} ms (estimated)"
    )

    return full_ms


def profile_feature_extraction_components(epoch, sfreq):
    """Profile individual feature extraction components on a single epoch."""
    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPONENTS (per single epoch, single channel)")
    print("=" * 60)

    n_runs = 10

    # Welch PSD / band power
    _, bp_ms, bp_std = time_it(
        compute_band_power, epoch, sfreq, 8.0, 13.0, n_runs=n_runs
    )
    print(f"  Welch PSD (1 band):             {bp_ms:8.4f} ms (+/- {bp_std:.4f})")
    # In extract_erd_features: 9 bands * 8 channels * 2 (baseline+task) = 144 calls
    # Plus total power calls: 8 channels * 2 = 16 more
    n_welch_calls_erd = (9 * 8 * 2) + (8 * 2)
    print(
        f"  Welch PSD ({n_welch_calls_erd} calls in ERD): {bp_ms * n_welch_calls_erd:8.2f} ms (estimated)"
    )

    # Filter bank features (16 sub-bands with bandpass + variance)
    _, fb_ms, fb_std = time_it(
        compute_filter_bank_features, epoch, sfreq, n_runs=n_runs
    )
    print(f"  Filter bank (16 sub-bands):     {fb_ms:8.4f} ms (+/- {fb_std:.4f})")
    print(f"  Filter bank (8 channels):       {fb_ms * 8:8.2f} ms (estimated)")

    # Hjorth parameters
    _, hj_ms, hj_std = time_it(compute_hjorth_parameters, epoch, n_runs=n_runs)
    print(f"  Hjorth parameters:              {hj_ms:8.4f} ms (+/- {hj_std:.4f})")

    # Envelope features (Hilbert transform)
    _, env_ms, env_std = time_it(compute_envelope_features, epoch, n_runs=n_runs)
    print(f"  Envelope (Hilbert):             {env_ms:8.4f} ms (+/- {env_std:.4f})")

    # Spectral entropy
    _, se_ms, se_std = time_it(compute_spectral_entropy, epoch, sfreq, n_runs=n_runs)
    print(f"  Spectral entropy:               {se_ms:8.4f} ms (+/- {se_std:.4f})")

    # Temporal features
    _, tf_ms, tf_std = time_it(compute_temporal_features, epoch, sfreq, n_runs=n_runs)
    print(f"  Temporal features:              {tf_ms:8.4f} ms (+/- {tf_std:.4f})")

    # Asymmetry features (needs multichannel)
    dummy_channels = {ch: epoch for ch in CHANNELS}
    _, asym_ms, asym_std = time_it(
        compute_asymmetry_features, dummy_channels, sfreq, n_runs=n_runs
    )
    print(f"  Asymmetry features:             {asym_ms:8.4f} ms (+/- {asym_std:.4f})")

    return {
        "welch_per_call": bp_ms,
        "filter_bank_per_channel": fb_ms,
        "hjorth_per_channel": hj_ms,
        "envelope_per_channel": env_ms,
        "spectral_entropy_per_channel": se_ms,
        "temporal_per_channel": tf_ms,
        "asymmetry": asym_ms,
    }


def profile_erd_feature_extraction(epoch_pairs_by_channel, sfreq):
    """Profile full ERD feature extraction."""
    print("\n" + "=" * 60)
    print("FULL ERD FEATURE EXTRACTION")
    print("=" * 60)

    n_epochs = len(epoch_pairs_by_channel[CHANNELS[0]])
    _, erd_ms, erd_std = time_it(
        extract_erd_features, epoch_pairs_by_channel, sfreq, n_runs=3
    )
    print(
        f"  extract_erd_features ({n_epochs} epochs): {erd_ms:8.2f} ms (+/- {erd_std:.2f})"
    )
    print(f"  Per epoch:                      {erd_ms / n_epochs:8.2f} ms")

    # Get feature count
    features = extract_erd_features(epoch_pairs_by_channel, sfreq)
    print(f"  Feature vector size:            {features.shape[1]} features per epoch")
    print(f"  Output shape:                   {features.shape}")

    return erd_ms, features.shape[1]


def profile_realtime_features(channel_signals, sfreq):
    """Profile realtime feature extraction."""
    print("\n" + "=" * 60)
    print("REALTIME FEATURE EXTRACTION")
    print("=" * 60)

    # Create a 5-second window
    window_samples = int(5.0 * sfreq)
    window = {ch: signal[:window_samples] for ch, signal in channel_signals.items()}

    n_runs = 10
    features, rt_ms, rt_std = time_it(
        extract_realtime_features, window, sfreq, n_runs=n_runs
    )
    print(f"  extract_realtime_features:      {rt_ms:8.2f} ms (+/- {rt_std:.2f})")
    print(f"  Feature vector size:            {len(features)} features")

    # Check if this meets real-time constraint (5s window, should process in < 100ms)
    budget_ms = 100.0
    print(f"  Real-time budget:               {budget_ms:.0f} ms (for 5s window)")
    print(f"  Budget usage:                   {rt_ms / budget_ms * 100:.1f}%")

    return rt_ms, len(features)


def profile_realtime_simulation_window(channel_signals, sfreq):
    """Profile a single window of real-time simulation (preprocess + features)."""
    print("\n" + "=" * 60)
    print("REALTIME SIMULATION: FULL WINDOW PROCESSING")
    print("=" * 60)

    window_samples = int(5.0 * sfreq)

    # Simulate what simulate_realtime does: preprocess each channel then extract features
    raw_window = {ch: signal[:window_samples] for ch, signal in channel_signals.items()}

    n_runs = 5

    def process_one_window():
        # Preprocess each channel
        processed = {}
        for ch_name, signal in raw_window.items():
            processed[ch_name] = preprocess_eeg(signal, sfreq)
        # Extract features
        features = extract_realtime_features(processed, sfreq)
        return features

    _, total_ms, total_std = time_it(process_one_window, n_runs=n_runs)

    # Break down: preprocessing portion
    def preprocess_only():
        processed = {}
        for ch_name, signal in raw_window.items():
            processed[ch_name] = preprocess_eeg(signal, sfreq)
        return processed

    _, preproc_ms, _ = time_it(preprocess_only, n_runs=n_runs)

    # Features portion
    preprocessed_window = preprocess_only()
    _, feat_ms, _ = time_it(
        extract_realtime_features, preprocessed_window, sfreq, n_runs=n_runs
    )

    print(f"  Total per-window processing:    {total_ms:8.2f} ms (+/- {total_std:.2f})")
    print(
        f"    Preprocessing (8 channels):   {preproc_ms:8.2f} ms ({preproc_ms / total_ms * 100:.1f}%)"
    )
    print(
        f"    Feature extraction:           {feat_ms:8.2f} ms ({feat_ms / total_ms * 100:.1f}%)"
    )
    print("  Window duration:                5000.00 ms")
    print(f"  Processing/Window ratio:        {total_ms / 5000 * 100:.2f}%")

    return total_ms, preproc_ms, feat_ms


def profile_model_inference(features_array, sfreq):
    """Profile model inference (Random Forest predict)."""
    print("\n" + "=" * 60)
    print("MODEL INFERENCE")
    print("=" * 60)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    n_samples = features_array.shape[0]
    labels = np.array([1] * (n_samples // 2) + [0] * (n_samples - n_samples // 2))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_array)

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_scaled, labels)

    # Time single prediction
    single_sample = X_scaled[0:1]
    _, pred_ms, pred_std = time_it(model.predict, single_sample, n_runs=50)
    print(f"  RF predict (1 sample):          {pred_ms:8.4f} ms (+/- {pred_std:.4f})")

    _, proba_ms, proba_std = time_it(model.predict_proba, single_sample, n_runs=50)
    print(f"  RF predict_proba (1 sample):    {proba_ms:8.4f} ms (+/- {proba_std:.4f})")

    # Time batch prediction
    _, batch_ms, batch_std = time_it(model.predict, X_scaled, n_runs=10)
    print(
        f"  RF predict ({n_samples} samples):     {batch_ms:8.2f} ms (+/- {batch_std:.2f})"
    )

    # Time scaler transform
    _, scale_ms, scale_std = time_it(scaler.transform, single_sample, n_runs=50)
    print(f"  Scaler transform (1 sample):    {scale_ms:8.4f} ms (+/- {scale_std:.4f})")

    return pred_ms, proba_ms


def analyze_feature_dimensionality(epoch_pairs_by_channel, sfreq, channel_signals):
    """Analyze feature dimensionality: computed vs actually used."""
    print("\n" + "=" * 60)
    print("FEATURE DIMENSIONALITY ANALYSIS")
    print("=" * 60)

    # ERD features
    erd_features = extract_erd_features(epoch_pairs_by_channel, sfreq)
    n_erd = erd_features.shape[1]

    # Break down ERD features per channel
    bands = [
        "theta",
        "low_alpha",
        "high_alpha",
        "mu",
        "low_beta",
        "high_beta",
        "beta",
        "gamma",
    ]
    n_bands = len(bands)
    n_channels = 8

    # Per channel: 5 features per band (ERD, log_ratio, rel_diff, log_task, log_baseline) = 5 * 8 = 40
    per_band_features = 5
    # Plus per channel: Hjorth(3) + Envelope(3) + SpectralEntropy(1) + FilterBank(16) + Temporal(9) = 32
    per_channel_extra = 3 + 3 + 1 + 16 + 9
    per_channel_total = (per_band_features * n_bands) + per_channel_extra
    all_channel_features = per_channel_total * n_channels

    # Asymmetry: C3-C4 (4 bands * 2 features) + Fz-Pz (2 bands * 1 feature) = 10
    asym_features = (4 * 2) + (2 * 1)
    # Inter-channel: C3-C4 (8 bands * 2) + Fz-Pz (3 bands * 1) + Cz-Oz (2 bands * 1) = 21
    inter_channel_features = (n_bands * 2) + (3 * 1) + (2 * 1)

    expected_total = all_channel_features + asym_features + inter_channel_features

    print("\n  ERD Feature Breakdown:")
    print(
        f"    Per band per channel:         {per_band_features} features x {n_bands} bands = {per_band_features * n_bands}"
    )
    print("    Per channel extras:")
    print("      Hjorth parameters:          3")
    print("      Envelope features:          3")
    print("      Spectral entropy:           1")
    print("      Filter bank (16 sub-bands): 16")
    print("      Temporal features:          9")
    print(f"    Per channel total:            {per_channel_total}")
    print(f"    All channels (x{n_channels}):          {all_channel_features}")
    print(f"    Asymmetry features:           {asym_features}")
    print(f"    Inter-channel features:       {inter_channel_features}")
    print(f"    Expected total:               {expected_total}")
    print(f"    Actual total:                 {n_erd}")

    # Realtime features
    window_samples = int(5.0 * sfreq)
    window = {ch: sig[:window_samples] for ch, sig in channel_signals.items()}
    rt_features = extract_realtime_features(window, sfreq)
    n_rt = len(rt_features)

    # Realtime breakdown
    rt_bands = 6
    rt_per_band = 2  # log_power + relative_power
    rt_per_channel_band = rt_bands * rt_per_band  # 12
    rt_ratios = 2  # theta/alpha + theta/beta
    rt_hjorth = 3
    rt_envelope = 3
    rt_per_channel = rt_per_channel_band + rt_ratios + rt_hjorth + rt_envelope  # 20
    rt_all_channels = rt_per_channel * n_channels
    rt_inter = 6 + 3  # C3-C4 (6 bands) + Fz-Pz (3 bands)

    print("\n  Realtime Feature Breakdown:")
    print(
        f"    Per channel ({rt_per_channel} features):       {rt_per_channel} x {n_channels} channels = {rt_all_channels}"
    )
    print(f"    Inter-channel features:       {rt_inter}")
    print(f"    Expected total:               {rt_all_channels + rt_inter}")
    print(f"    Actual total:                 {n_rt}")

    for pct in [0.3, 0.5, 0.7]:
        k = max(10, int(pct * n_erd))
        print(
            f"\n  SelectKBest at {pct:.0%}: {k}/{n_erd} features selected ({k / n_erd * 100:.1f}% used)"
        )

    print("\n  SUMMARY:")
    print(f"    ERD features computed:        {n_erd}")
    print(f"    Realtime features computed:   {n_rt}")
    print(
        f"    Max features actually used:   {max(10, int(0.7 * n_erd))} (70% SelectKBest)"
    )
    print(
        f"    Min features actually used:   {max(10, int(0.3 * n_erd))} (30% SelectKBest)"
    )
    print(
        f"    Features never used (worst):  {n_erd - max(10, int(0.3 * n_erd))} ({(n_erd - max(10, int(0.3 * n_erd))) / n_erd * 100:.1f}%)"
    )


def main():
    data_dir = Path("unicorn-data")
    recordings = get_complete_recordings(data_dir)

    if not recordings:
        print("ERROR: No complete recordings found in unicorn-data/")
        return

    print(f"Found {len(recordings)} complete recordings. Using first for profiling.")
    recording_path = recordings[0]
    print(f"Recording: {recording_path}")

    # Load data
    data, events, sample_rate = load_recording(recording_path)
    print(f"Data shape: {data.shape}, Sample rate: {sample_rate} Hz")
    print(f"Duration: {data.shape[1] / sample_rate:.1f}s, Events: {len(events)}")

    # 1. Profile preprocessing
    preproc_ms = profile_preprocessing(data, sample_rate)

    # 2. Preprocess for feature extraction
    preprocessed = preprocess_eeg_multichannel(data, sample_rate, CHANNELS, "car")
    channel_signals = {ch: preprocessed[i] for i, ch in enumerate(CHANNELS)}

    # 3. Extract epochs for profiling
    engaged_epochs_by_channel = {}
    disengaged_epochs_by_channel = {}
    for ch_name, signal in channel_signals.items():
        engaged_pairs, disengaged_pairs = extract_augmented_epochs(
            signal,
            events,
            sample_rate,
            window_duration=2.0,
            n_windows=2,
            class1_phase=3,
            class2_phase=5,
        )
        engaged_epochs_by_channel[ch_name] = engaged_pairs
        disengaged_epochs_by_channel[ch_name] = disengaged_pairs

    print(f"\nExtracted {len(engaged_epochs_by_channel[CHANNELS[0]])} engaged epochs")
    print(
        f"Extracted {len(disengaged_epochs_by_channel[CHANNELS[0]])} disengaged epochs"
    )

    # 4. Profile individual feature components
    _, task_epoch = engaged_epochs_by_channel[CHANNELS[0]][0]
    component_times = profile_feature_extraction_components(task_epoch, sample_rate)

    # 5. Profile full ERD feature extraction
    erd_ms, n_erd_features = profile_erd_feature_extraction(
        engaged_epochs_by_channel, sample_rate
    )

    # 6. Profile realtime features
    rt_ms, n_rt_features = profile_realtime_features(channel_signals, sample_rate)

    # 7. Profile realtime simulation window
    raw_channel_signals = {ch: data[i] for i, ch in enumerate(CHANNELS)}
    total_window_ms, preproc_window_ms, feat_window_ms = (
        profile_realtime_simulation_window(raw_channel_signals, sample_rate)
    )

    # 8. Profile model inference
    erd_features = extract_erd_features(engaged_epochs_by_channel, sample_rate)
    pred_ms, proba_ms = profile_model_inference(erd_features, sample_rate)

    # 9. Feature dimensionality analysis
    analyze_feature_dimensionality(
        engaged_epochs_by_channel, sample_rate, channel_signals
    )

    # 10. Summary
    print("\n" + "=" * 60)
    print("BOTTLENECK SUMMARY")
    print("=" * 60)

    n_epochs = len(engaged_epochs_by_channel[CHANNELS[0]])

    print("\n  Pipeline stage timings (for 1 subject):")
    print(f"    Preprocessing:                {preproc_ms:8.2f} ms")
    print(f"    ERD feature extraction:       {erd_ms:8.2f} ms ({n_epochs} epochs)")
    print(f"    Per-epoch ERD features:       {erd_ms / n_epochs:8.2f} ms")
    print(f"    RF single prediction:         {pred_ms:8.4f} ms")

    print("\n  Real-time simulation per window:")
    print(f"    Total:                        {total_window_ms:8.2f} ms")
    print(f"    Preprocessing:                {preproc_window_ms:8.2f} ms")
    print(f"    Feature extraction:           {feat_window_ms:8.2f} ms")
    print(f"    Model inference:              {proba_ms:8.4f} ms")

    print("\n  Component breakdown (per epoch, per channel):")
    for name, ms in sorted(component_times.items(), key=lambda x: -x[1]):
        print(f"    {name:35s} {ms:8.4f} ms")

    print("\n  Top bottlenecks:")

    bottlenecks = [
        ("Preprocessing (bandpass+notch, 8ch)", preproc_ms),
        ("ERD features (all epochs)", erd_ms),
        (
            "Filter bank (16 bands x 8 ch x N epochs)",
            component_times["filter_bank_per_channel"] * 8 * n_epochs,
        ),
        (
            "Welch PSD (160 calls per epoch x N epochs)",
            component_times["welch_per_call"] * 160 * n_epochs,
        ),
        ("Realtime window total", total_window_ms),
        ("Realtime preprocess per window", preproc_window_ms),
        ("Realtime features per window", feat_window_ms),
    ]

    for i, (name, ms) in enumerate(sorted(bottlenecks, key=lambda x: -x[1]), 1):
        print(f"    {i}. {name}: {ms:.2f} ms")


if __name__ == "__main__":
    main()
