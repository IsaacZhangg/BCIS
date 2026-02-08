"""Main pipeline: load data, preprocess, extract features, train left/right classifier."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import load_recording, get_complete_recordings, CHANNELS
from src.preprocess import preprocess_eeg
from src.epochs import extract_left_right_epochs
from src.features import extract_lateralization_features
from src.train import (
    train_left_right_within_subject,
    train_final_model,
    train_final_model_cv,
)


def run_pipeline(data_dir: Path, output_dir: Path) -> dict:
    """
    Run the full training pipeline for left/right motor imagery classification.

    Args:
        data_dir: Path to unicorn-data directory
        output_dir: Path to save model and results

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("Left/Right Motor Imagery Classifier - Training Pipeline")
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
        processed_channels = {
            ch_name: preprocess_eeg(data[ch_idx], sfreq)
            for ch_idx, ch_name in enumerate(CHANNELS)
        }

        # Extract left/right epochs from phase 3
        epoch_kwargs = dict(task_duration=1.8, baseline_duration=1.0, skip_duration=0.5)
        epoch_pairs = {
            ch_name: extract_left_right_epochs(signal, events, sfreq, **epoch_kwargs)
            for ch_name, signal in processed_channels.items()
        }
        left_pairs_by_channel = {ch: pairs[0] for ch, pairs in epoch_pairs.items()}
        right_pairs_by_channel = {ch: pairs[1] for ch, pairs in epoch_pairs.items()}

        n_left = len(left_pairs_by_channel[CHANNELS[0]])
        n_right = len(right_pairs_by_channel[CHANNELS[0]])
        print(f"    Left epochs: {n_left}, Right epochs: {n_right}")

        if n_left == 0 or n_right == 0:
            print(f"    Skipping {subject_id} - no epochs")
            continue

        # Extract lateralization features
        left_features = extract_lateralization_features(left_pairs_by_channel, sfreq)
        right_features = extract_lateralization_features(right_pairs_by_channel, sfreq)

        # Create multichannel arrays for CSP/Riemannian
        left_multichannel = np.array(
            [[trial[1] for trial in left_pairs_by_channel[ch]] for ch in CHANNELS]
        ).transpose(1, 0, 2)  # (n_trials, n_channels, n_samples)
        right_multichannel = np.array(
            [[trial[1] for trial in right_pairs_by_channel[ch]] for ch in CHANNELS]
        ).transpose(1, 0, 2)

        X_multichannel = np.vstack([left_multichannel, right_multichannel])
        X_features = np.vstack([left_features, right_features])
        y = np.array([0] * n_left + [1] * n_right)  # 0=left, 1=right

        X_by_subject.append((X_features, X_multichannel))
        y_by_subject.append(y)
        subject_ids.append(subject_id)

    if not X_by_subject:
        raise ValueError("No valid subjects found")

    # Step 3: Within-Subject Cross-validation
    print("\n[3/6] Running ensemble within-subject cross-validation...")
    print("(10-fold CV per subject with probability-averaged ensemble)")
    scores, mean_acc, std_acc = train_left_right_within_subject(
        X_by_subject, y_by_subject
    )

    print("\nPer-subject accuracy (Ensemble 10-fold CV):")
    for sid, score in zip(subject_ids, scores):
        print(f"  {sid}: {score:.1%}")

    print(f"\nEnsemble mean accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    # Step 4: Final model CV (LGBMClassifier on handcrafted features)
    print("\n[4/6] Running final-model within-subject cross-validation...")
    print("(10-fold CV per subject with LGBMClassifier on handcrafted features)")
    fm_scores, fm_mean_acc, fm_std_acc = train_final_model_cv(
        X_by_subject, y_by_subject
    )

    print("\nPer-subject accuracy (LGBMClassifier 10-fold CV):")
    for sid, score in zip(subject_ids, fm_scores):
        print(f"  {sid}: {score:.1%}")

    print(f"\nLGBMClassifier mean accuracy: {fm_mean_acc:.1%} (+/- {fm_std_acc:.1%})")

    # Step 5: Check if target met (using final model accuracy)
    target_met = fm_mean_acc >= 0.90
    print(f"\nTarget (>90% final model): {'MET' if target_met else 'NOT MET'}")

    # Step 6: Train final model on all data
    print("\n[5/6] Training final model on all data...")
    X_all_features = np.vstack([x[0] for x in X_by_subject])
    y_all = np.concatenate(y_by_subject)

    final_model, scaler = train_final_model(X_all_features, y_all)

    # Step 7: Save model and scaler
    print("\n[6/6] Saving model...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "left_right_classifier_model.joblib"
    scaler_path = output_dir / "left_right_classifier_scaler.joblib"

    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Save results as JSON
    results = {
        "task": "left_right_motor_imagery",
        "n_subjects": len(subject_ids),
        "ensemble_per_subject_scores": dict(
            zip(subject_ids, [float(s) for s in scores])
        ),
        "ensemble_mean_accuracy": float(mean_acc),
        "ensemble_std_accuracy": float(std_acc),
        "final_model_per_subject_scores": dict(
            zip(subject_ids, [float(s) for s in fm_scores])
        ),
        "final_model_mean_accuracy": float(fm_mean_acc),
        "final_model_std_accuracy": float(fm_std_acc),
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
