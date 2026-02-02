"""Main pipeline: load data, preprocess, extract features, train, evaluate."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import load_recording, get_complete_recordings, CHANNELS
from src.preprocess import preprocess_eeg, preprocess_eeg_multichannel
from src.epochs import extract_erd_epochs, extract_augmented_epochs
from src.features import extract_erd_features, extract_realtime_features
from src.train import train_within_subject_cv_riemannian, train_final_model


def run_pipeline(data_dir: Path, output_dir: Path) -> dict:
    """
    Run the full training pipeline.

    Args:
        data_dir: Path to unicorn-data directory
        output_dir: Path to save model and results

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("Engagement Classifier - Training Pipeline")
    print("=" * 60)

    # Step 1: Find complete recordings
    print("\n[1/5] Finding complete recordings...")
    recordings = get_complete_recordings(data_dir)
    print(f"Found {len(recordings)} complete recordings")

    # Step 2: Load and preprocess data for each subject
    print("\n[2/5] Loading and preprocessing data...")
    X_by_subject = []
    y_by_subject = []
    subject_ids = []
    processed_data_cache = {}  # Cache processed data for realtime model training

    for rec_path in recordings:
        subject_id = rec_path.parent.parent.name
        print(f"  Processing {subject_id}...")

        # Load recording
        data, events, sfreq = load_recording(rec_path)

        # Process ALL channels with CAR spatial filtering
        data_multichannel = preprocess_eeg_multichannel(data, sfreq, CHANNELS, spatial_filter="car")

        processed_channels = {}
        for ch_idx, ch_name in enumerate(CHANNELS):
            processed_channels[ch_name] = data_multichannel[ch_idx]

        # Cache for realtime model training
        processed_data_cache[rec_path] = (processed_channels, events, sfreq)

        # Extract ERD epochs (baseline + task pairs) for each channel
        # Use augmented epochs to increase training data
        engaged_pairs_by_channel = {}
        disengaged_pairs_by_channel = {}

        for ch_name, signal in processed_channels.items():
            # Compare phase 3 (engaged/imagery) vs phase 5 (disengaged/rest)
            # Use augmented epochs with 2 overlapping windows per trial
            engaged_pairs, disengaged_pairs = extract_augmented_epochs(
                signal, events, sfreq,
                window_duration=2.0,
                n_windows=2,
                class1_phase=3,
                class2_phase=5,
            )
            engaged_pairs_by_channel[ch_name] = engaged_pairs
            disengaged_pairs_by_channel[ch_name] = disengaged_pairs

        # Check counts
        n_engaged = len(engaged_pairs_by_channel[CHANNELS[0]])
        n_disengaged = len(disengaged_pairs_by_channel[CHANNELS[0]])

        # Balance classes
        if n_disengaged > n_engaged:
            np.random.seed(42)
            indices = np.random.choice(n_disengaged, n_engaged, replace=False)
            for ch_name in CHANNELS:
                disengaged_pairs_by_channel[ch_name] = [disengaged_pairs_by_channel[ch_name][i] for i in indices]
            n_disengaged = n_engaged

        print(f"    Engaged epochs: {n_engaged}, Disengaged epochs: {n_disengaged}")

        # Extract ERD features
        engaged_features = extract_erd_features(engaged_pairs_by_channel, sfreq)
        disengaged_features = extract_erd_features(disengaged_pairs_by_channel, sfreq)

        # Also create multichannel arrays for Riemannian classification
        # Shape: (n_trials, n_channels, n_samples)
        n_channels = len(CHANNELS)
        n_engaged = len(engaged_pairs_by_channel[CHANNELS[0]])
        n_disengaged = len(disengaged_pairs_by_channel[CHANNELS[0]])
        n_samples = engaged_pairs_by_channel[CHANNELS[0]][0][1].shape[0]

        engaged_multichannel = np.zeros((n_engaged, n_channels, n_samples))
        disengaged_multichannel = np.zeros((n_disengaged, n_channels, n_samples))

        for ch_idx, ch_name in enumerate(CHANNELS):
            for trial_idx in range(n_engaged):
                engaged_multichannel[trial_idx, ch_idx, :] = engaged_pairs_by_channel[ch_name][trial_idx][1]
            for trial_idx in range(n_disengaged):
                disengaged_multichannel[trial_idx, ch_idx, :] = disengaged_pairs_by_channel[ch_name][trial_idx][1]

        X_multichannel = np.vstack([engaged_multichannel, disengaged_multichannel])

        # Combine ERD features
        X = np.vstack([engaged_features, disengaged_features])
        y = np.array([1] * len(engaged_features) + [0] * len(disengaged_features))

        X_by_subject.append((X, X_multichannel))  # tuple of (features, multichannel)
        y_by_subject.append(y)
        subject_ids.append(subject_id)

    # Step 3: Within-subject Cross-validation (realistic BCI evaluation)
    print("\n[3/5] Running within-subject cross-validation...")
    print("(Using Riemannian geometry + feature ensemble)")
    scores, mean_acc, std_acc = train_within_subject_cv_riemannian(X_by_subject, y_by_subject)

    print("\nPer-subject accuracy (within-subject CV):")
    for sid, score in zip(subject_ids, scores):
        status = "PASS" if score >= 0.9 else "FAIL"
        print(f"  {sid}: {score:.1%} [{status}]")

    print(f"\nMean accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    # Step 4: Check if target met
    target_met = mean_acc >= 0.90
    print(f"\nTarget (>90%): {'MET' if target_met else 'NOT MET'}")

    # Step 5: Train final model on all data
    print("\n[4/5] Training final model on all data...")
    X_all = np.vstack([x[0] for x in X_by_subject])  # Just features for final model
    y_all = np.concatenate(y_by_subject)

    final_model, scaler = train_final_model(X_all, y_all)

    # Also train realtime model using only realtime-compatible features
    print("\n[4b/5] Training realtime model...")

    # Extract realtime features from cached processed data
    X_realtime_all = []
    for rec_path, (processed, events, sfreq) in processed_data_cache.items():
        # Get phase 3 and phase 5 event indices
        for sample_idx, phase, movement in events:
            if phase not in [3, 5]:
                continue

            # Extract 5-second window (1250 samples at 250 Hz)
            window_samples = int(5.0 * sfreq)
            if sample_idx + window_samples > len(processed[CHANNELS[0]]):
                continue

            window = {ch: processed[ch][sample_idx:sample_idx + window_samples]
                     for ch in CHANNELS}

            features = extract_realtime_features(window, sfreq)
            X_realtime_all.append((features, 1 if phase == 3 else 0))

    if X_realtime_all:
        X_rt = np.array([x[0] for x in X_realtime_all])
        y_rt = np.array([x[1] for x in X_realtime_all])

        # Balance classes
        n_engaged = np.sum(y_rt == 1)
        n_disengaged = np.sum(y_rt == 0)
        print(f"    Realtime - Engaged windows: {n_engaged}, Disengaged windows: {n_disengaged}")

        if n_disengaged > n_engaged:
            np.random.seed(42)
            engaged_idx = np.where(y_rt == 1)[0]
            disengaged_idx = np.where(y_rt == 0)[0]
            sampled_disengaged_idx = np.random.choice(disengaged_idx, n_engaged, replace=False)
            balanced_idx = np.concatenate([engaged_idx, sampled_disengaged_idx])
            X_rt = X_rt[balanced_idx]
            y_rt = y_rt[balanced_idx]
            print(f"    Balanced to {len(y_rt)} windows")

        realtime_model, realtime_scaler = train_final_model(X_rt, y_rt)

        realtime_model_path = output_dir / "engagement_realtime_model.joblib"
        realtime_scaler_path = output_dir / "engagement_realtime_scaler.joblib"

        joblib.dump(realtime_model, realtime_model_path)
        joblib.dump(realtime_scaler, realtime_scaler_path)

        print(f"    Realtime model saved to: {realtime_model_path}")
    else:
        print("    Warning: No realtime training data extracted")

    # Step 6: Save model and scaler
    print("\n[5/5] Saving model...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "engagement_classifier_model.joblib"
    scaler_path = output_dir / "engagement_classifier_scaler.joblib"

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Save results as JSON
    results = {
        "n_subjects": len(recordings),
        "per_subject_scores": dict(zip(subject_ids, [float(s) for s in scores])),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "target_met": bool(target_met),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "realtime_model_path": str(output_dir / "engagement_realtime_model.joblib"),
        "realtime_scaler_path": str(output_dir / "engagement_realtime_scaler.joblib"),
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    data_dir = Path("unicorn-data")
    output_dir = Path("models")

    results = run_pipeline(data_dir, output_dir)
