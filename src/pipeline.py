"""Main pipeline: load data, preprocess, extract features, train left/right classifier."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import CHANNELS, get_complete_recordings, load_recording
from src.epochs import extract_left_right_epochs
from src.features import extract_lateralization_features
from src.preprocess import preprocess_eeg
from src.train import train_final_model, train_within_subject_cv


def run_pipeline(data_dir: Path, output_dir: Path) -> dict:
    """Run the full training pipeline for left/right motor imagery classification.

    Args:
        data_dir: Path to unicorn-data directory
        output_dir: Path to save models and results

    Returns:
        Dictionary with results
    """
    print("=" * 60)
    print("Left/Right Motor Imagery Classifier - Training Pipeline")
    print("=" * 60)

    # Step 1: Find complete recordings
    print("\n[1/4] Finding complete recordings...")
    recordings = get_complete_recordings(data_dir)
    print(f"Found {len(recordings)} complete recordings")

    # Step 2: Load and preprocess data for each subject
    print("\n[2/4] Loading and preprocessing data...")
    X_by_subject: list[tuple[np.ndarray, np.ndarray]] = []
    y_by_subject: list[np.ndarray] = []
    subject_ids: list[str] = []

    for rec_path in recordings:
        subject_id = rec_path.parent.parent.name
        print(f"  Processing {subject_id}...")

        data, events, sfreq = load_recording(rec_path)

        processed_channels = {
            ch_name: preprocess_eeg(data[ch_idx], sfreq)
            for ch_idx, ch_name in enumerate(CHANNELS)
        }

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

        left_features = extract_lateralization_features(left_pairs_by_channel, sfreq)
        right_features = extract_lateralization_features(right_pairs_by_channel, sfreq)

        left_multichannel = np.array(
            [[trial[1] for trial in left_pairs_by_channel[ch]] for ch in CHANNELS]
        ).transpose(1, 0, 2)
        right_multichannel = np.array(
            [[trial[1] for trial in right_pairs_by_channel[ch]] for ch in CHANNELS]
        ).transpose(1, 0, 2)

        X_multichannel = np.vstack([left_multichannel, right_multichannel])
        X_features = np.vstack([left_features, right_features])
        y = np.array([0] * n_left + [1] * n_right)

        X_by_subject.append((X_features, X_multichannel))
        y_by_subject.append(y)
        subject_ids.append(subject_id)

    if not X_by_subject:
        raise ValueError("No valid subjects found")

    # Step 3: Within-Subject Cross-validation
    print("\n[3/4] Running FBCSP + LDA within-subject cross-validation...")
    print("(10-fold CV per subject)")
    scores, mean_acc, std_acc = train_within_subject_cv(
        X_by_subject, y_by_subject, sfreq=sfreq
    )

    print("\nPer-subject accuracy (FBCSP + LDA 10-fold CV):")
    for sid, score in zip(subject_ids, scores):
        status = "signal" if score >= 0.60 else "chance"
        print(f"  {sid}: {score:.1%}  [{status}]")

    print(f"\nMean accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    # Step 4: Train & save per-subject models
    print("\n[4/4] Training and saving per-subject models...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = {}
    for subj_idx, sid in enumerate(subject_ids):
        X_features, X_multichannel = X_by_subject[subj_idx]
        y = y_by_subject[subj_idx]

        model = train_final_model(X_features, X_multichannel, y, sfreq=sfreq)
        model_path = output_dir / f"{sid}_fbcsp_lda.joblib"
        joblib.dump(model, model_path)
        model_paths[sid] = str(model_path)
        print(f"  Saved {model_path}")

    # Diagnostic summary
    above_chance = [sid for sid, s in zip(subject_ids, scores) if s >= 0.60]
    at_chance = [sid for sid, s in zip(subject_ids, scores) if s < 0.60]

    print("\n" + "=" * 60)
    print("Diagnostic Summary")
    print("=" * 60)
    print(f"Subjects with signal (>=60%): {', '.join(above_chance) or 'none'}")
    print(f"Subjects at chance  (<60%):  {', '.join(at_chance) or 'none'}")
    print(f"Mean accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")

    # Save results
    results = {
        "task": "left_right_motor_imagery",
        "model": "FBCSP + LDA",
        "n_subjects": len(subject_ids),
        "per_subject_scores": dict(zip(subject_ids, [float(s) for s in scores])),
        "mean_accuracy": float(mean_acc),
        "std_accuracy": float(std_acc),
        "subjects_with_signal": above_chance,
        "subjects_at_chance": at_chance,
        "model_paths": model_paths,
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    data_dir = Path("unicorn-data")
    output_dir = Path("models")

    results = run_pipeline(data_dir, output_dir)
