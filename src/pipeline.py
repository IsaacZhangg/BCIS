"""Main pipeline: load data, preprocess, extract features, train, evaluate."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import load_recording, get_complete_recordings, CHANNELS
from src.preprocess import preprocess_eeg
from src.epochs import extract_labeled_epochs, extract_motor_imagery_epochs, extract_erd_epochs
from src.features import extract_multichannel_features, extract_erd_features, CSP
from src.train import train_loso_cv, train_within_subject_cv, train_within_subject_cv_riemannian, train_final_model


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
    print("Theta Focus Classifier - Training Pipeline")
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

    for rec_path in recordings:
        subject_id = rec_path.parent.parent.name
        print(f"  Processing {subject_id}...")

        # Load recording
        data, events, sfreq = load_recording(rec_path)

        # Process ALL channels
        processed_channels = {}
        for ch_idx, ch_name in enumerate(CHANNELS):
            processed_channels[ch_name] = preprocess_eeg(data[ch_idx], sfreq)

        # Extract ERD epochs (baseline + task pairs) for each channel
        focused_pairs_by_channel = {}
        rest_pairs_by_channel = {}

        for ch_name, signal in processed_channels.items():
            # Compare phase 3 vs phase 4 (different imagery task types)
            focused_pairs, rest_pairs = extract_erd_epochs(
                signal, events, sfreq,
                task_duration=1.5,
                baseline_duration=1.0,
                class1_phase=3,
                class2_phase=4,
            )
            focused_pairs_by_channel[ch_name] = focused_pairs
            rest_pairs_by_channel[ch_name] = rest_pairs

        # Check counts
        n_focused = len(focused_pairs_by_channel[CHANNELS[0]])
        n_rest = len(rest_pairs_by_channel[CHANNELS[0]])

        # Balance classes
        if n_rest > n_focused:
            np.random.seed(42)
            indices = np.random.choice(n_rest, n_focused, replace=False)
            for ch_name in CHANNELS:
                rest_pairs_by_channel[ch_name] = [rest_pairs_by_channel[ch_name][i] for i in indices]
            n_rest = n_focused

        print(f"    Focused epochs: {n_focused}, Not-focused epochs: {n_rest}")

        # Extract ERD features
        focused_features = extract_erd_features(focused_pairs_by_channel, sfreq)
        rest_features = extract_erd_features(rest_pairs_by_channel, sfreq)

        # Also create multichannel arrays for Riemannian classification
        # Shape: (n_trials, n_channels, n_samples)
        n_channels = len(CHANNELS)
        n_focused = len(focused_pairs_by_channel[CHANNELS[0]])
        n_rest = len(rest_pairs_by_channel[CHANNELS[0]])
        n_samples = focused_pairs_by_channel[CHANNELS[0]][0][1].shape[0]

        focused_multichannel = np.zeros((n_focused, n_channels, n_samples))
        rest_multichannel = np.zeros((n_rest, n_channels, n_samples))

        for ch_idx, ch_name in enumerate(CHANNELS):
            for trial_idx in range(n_focused):
                focused_multichannel[trial_idx, ch_idx, :] = focused_pairs_by_channel[ch_name][trial_idx][1]
            for trial_idx in range(n_rest):
                rest_multichannel[trial_idx, ch_idx, :] = rest_pairs_by_channel[ch_name][trial_idx][1]

        X_multichannel = np.vstack([focused_multichannel, rest_multichannel])

        # Combine ERD features
        X = np.vstack([focused_features, rest_features])
        y = np.array([1] * len(focused_features) + [0] * len(rest_features))

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

    # Step 6: Save model and scaler
    print("\n[5/5] Saving model...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "theta_classifier_model.joblib"
    scaler_path = output_dir / "theta_classifier_scaler.joblib"

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
