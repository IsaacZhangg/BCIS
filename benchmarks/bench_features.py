"""Benchmark: compare original vs optimized feature extraction.

Measures:
1. Speed (wall-clock time per window)
2. Feature count
3. Numerical correlation between old and new features where applicable
4. Classification accuracy on real labeled data
"""

import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_recording, get_complete_recordings, CHANNELS, SFREQ
from src.preprocess import preprocess_eeg
from src.features import extract_realtime_features
from src.features_fast import (
    extract_realtime_features_fast,
    extract_realtime_features_fast_v2,
)


def load_test_windows(n_windows: int = 200) -> list[dict[str, np.ndarray]]:
    """Load real EEG windows from the first complete recording."""
    data_dir = Path("unicorn-data")
    recordings = get_complete_recordings(data_dir)
    if not recordings:
        raise FileNotFoundError("No complete recordings found")

    recording_path = recordings[0]
    print(f"Loading data from: {recording_path}")

    data, events, sfreq = load_recording(recording_path)

    # Preprocess each channel
    processed = {}
    for ch_idx, ch_name in enumerate(CHANNELS):
        processed[ch_name] = preprocess_eeg(data[ch_idx], sfreq)

    # Extract 5-second windows
    window_samples = int(5.0 * sfreq)
    total_samples = len(processed[CHANNELS[0]])
    step_samples = window_samples  # non-overlapping

    windows = []
    start = 0
    while start + window_samples <= total_samples and len(windows) < n_windows:
        window = {ch: processed[ch][start : start + window_samples] for ch in CHANNELS}
        windows.append(window)
        start += step_samples

    print(f"Extracted {len(windows)} windows ({window_samples} samples each)")
    return windows


def load_labeled_data() -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    """Load labeled windows from complete recordings (phase 3 vs phase 5)."""
    data_dir = Path("unicorn-data")
    recordings = get_complete_recordings(data_dir)

    all_windows = []
    all_labels = []

    for recording_path in recordings[:3]:  # Use first 3 subjects for speed
        subject_id = recording_path.parent.parent.name
        data, events, sfreq = load_recording(recording_path)

        # Preprocess
        processed = {}
        for ch_idx, ch_name in enumerate(CHANNELS):
            processed[ch_name] = preprocess_eeg(data[ch_idx], sfreq)

        total_samples = len(processed[CHANNELS[0]])
        window_samples = int(5.0 * sfreq)

        for sample_idx, phase, movement in events:
            if phase not in [3, 5]:
                continue
            if sample_idx + window_samples > total_samples:
                continue

            window = {
                ch: processed[ch][sample_idx : sample_idx + window_samples]
                for ch in CHANNELS
            }
            all_windows.append(window)
            all_labels.append(1 if phase == 3 else 0)

        print(
            f"  {subject_id}: {sum(1 for _, p, _ in events if p == 3)} engaged, "
            f"{sum(1 for _, p, _ in events if p == 5)} disengaged events"
        )

    labels = np.array(all_labels)
    print(
        f"Total labeled windows: {len(labels)} "
        f"(engaged={np.sum(labels == 1)}, disengaged={np.sum(labels == 0)})"
    )
    return all_windows, labels


def benchmark_function(func, windows, name, warmup=3, repeats=5):
    """Time a feature extraction function over all windows."""
    n = len(windows)
    # Warmup
    for w in windows[: min(warmup, n)]:
        func(w, SFREQ)

    # Timed runs
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for w in windows:
            func(w, SFREQ)
        elapsed = time.perf_counter() - t0
        times.append(elapsed)

    best = min(times)
    mean = np.mean(times)
    per_window_ms = (best / n) * 1000

    # Get feature count
    feat = func(windows[0], SFREQ)
    n_features = len(feat)

    print(f"\n--- {name} ---")
    print(f"  Features: {n_features}")
    print(f"  Best total ({n} windows): {best * 1000:.1f} ms")
    print(f"  Mean total ({n} windows): {mean * 1000:.1f} ms")
    print(f"  Per window (best): {per_window_ms:.2f} ms")
    print(f"  Per window (mean): {(mean / n) * 1000:.2f} ms")

    return per_window_ms, n_features


def evaluate_classification(windows, labels, func, name):
    """Evaluate classification accuracy with real labels."""
    features = np.array([func(w, SFREQ) for w in windows])

    # Replace NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    n_folds = min(5, min(np.sum(labels == 0), np.sum(labels == 1)))
    if n_folds < 2:
        print(f"  {name}: Not enough samples for CV")
        return 0.0

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_scores = []
    for train_idx, test_idx in cv.split(features, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(features[train_idx])
        X_test = scaler.transform(features[test_idx])
        rf_fold = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf_fold.fit(X_train, labels[train_idx])
        acc = rf_fold.score(X_test, labels[test_idx])
        fold_scores.append(acc)

    mean_acc = np.mean(fold_scores)
    std_acc = np.std(fold_scores)
    print(
        f"  {name}: {mean_acc:.1%} +/- {std_acc:.1%} ({features.shape[1]} features, {n_folds}-fold CV)"
    )
    return mean_acc


def main():
    print("=" * 60)
    print("Feature Extraction Benchmark")
    print("=" * 60)

    # --- Speed benchmark ---
    print("\n[1] Speed benchmark (unlabeled windows)")
    windows = load_test_windows(n_windows=200)

    old_ms, old_n = benchmark_function(
        extract_realtime_features, windows, "Original (features.py)"
    )
    fast_ms, fast_n = benchmark_function(
        extract_realtime_features_fast, windows, "Fast v1 (single PSD, no envelope/FB)"
    )
    fast2_ms, fast2_n = benchmark_function(
        extract_realtime_features_fast_v2,
        windows,
        "Fast v2 (single PSD + PSD filter bank)",
    )

    print("\n" + "=" * 60)
    print("Speedup Summary")
    print("=" * 60)
    print(f"  Original:  {old_ms:.2f} ms/window  ({old_n} features)")
    print(
        f"  Fast v1:   {fast_ms:.2f} ms/window  ({fast_n} features)  -> {old_ms / fast_ms:.1f}x speedup"
    )
    print(
        f"  Fast v2:   {fast2_ms:.2f} ms/window  ({fast2_n} features)  -> {old_ms / fast2_ms:.1f}x speedup"
    )

    # --- Classification accuracy ---
    print("\n" + "=" * 60)
    print("[2] Classification accuracy (real labels: Phase 3 vs Phase 5)")
    print("=" * 60)

    labeled_windows, labels = load_labeled_data()

    evaluate_classification(
        labeled_windows, labels, extract_realtime_features, "Original"
    )
    evaluate_classification(
        labeled_windows, labels, extract_realtime_features_fast, "Fast v1"
    )
    evaluate_classification(
        labeled_windows, labels, extract_realtime_features_fast_v2, "Fast v2"
    )

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
