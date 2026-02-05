"""Main pipeline: load data, preprocess, extract features, train, evaluate."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import load_recording, get_complete_recordings, CHANNELS
from src.preprocess import preprocess_eeg_multichannel
from src.epochs import extract_augmented_epochs
from src.features import extract_erd_features, extract_realtime_features
from src.train import train_within_subject_cv_riemannian, train_final_model


def create_multichannel_arrays(
    engaged_epochs_by_channel: dict,
    disengaged_epochs_by_channel: dict,
    channels: list[str],
) -> np.ndarray:
    """Create multichannel arrays for Riemannian classification."""
    num_channels = len(channels)
    num_engaged = len(engaged_epochs_by_channel[channels[0]])
    num_disengaged = len(disengaged_epochs_by_channel[channels[0]])
    num_samples = engaged_epochs_by_channel[channels[0]][0][1].shape[0]

    engaged_array = np.zeros((num_engaged, num_channels, num_samples))
    disengaged_array = np.zeros((num_disengaged, num_channels, num_samples))

    for channel_idx, channel_name in enumerate(channels):
        for trial_idx in range(num_engaged):
            _, task_epoch = engaged_epochs_by_channel[channel_name][trial_idx]
            engaged_array[trial_idx, channel_idx, :] = task_epoch
        for trial_idx in range(num_disengaged):
            _, task_epoch = disengaged_epochs_by_channel[channel_name][trial_idx]
            disengaged_array[trial_idx, channel_idx, :] = task_epoch

    return np.vstack([engaged_array, disengaged_array])


def extract_realtime_training_data(
    processed_cache: dict, channels: list[str]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extract realtime features from processed data cache."""
    all_features = []
    all_labels = []

    for recording_path, (
        channel_signals,
        events,
        sample_rate,
    ) in processed_cache.items():
        for sample_idx, phase, movement in events:
            if phase not in [3, 5]:
                continue

            window_samples = int(5.0 * sample_rate)
            if sample_idx + window_samples > len(channel_signals[channels[0]]):
                continue

            window = {
                channel: channel_signals[channel][
                    sample_idx : sample_idx + window_samples
                ]
                for channel in channels
            }

            features = extract_realtime_features(window, sample_rate)
            label = 1 if phase == 3 else 0

            all_features.append(features)
            all_labels.append(label)

    if not all_features:
        return None

    return np.array(all_features), np.array(all_labels)


def balance_realtime_data(
    features: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Balance realtime training data to prevent class imbalance."""
    engaged_count = np.sum(labels == 1)
    disengaged_count = np.sum(labels == 0)

    print(
        f"    Realtime - Engaged windows: {engaged_count}, Disengaged windows: {disengaged_count}"
    )

    if disengaged_count <= engaged_count:
        return features, labels

    # Downsample majority class
    np.random.seed(42)
    engaged_indices = np.where(labels == 1)[0]
    disengaged_indices = np.where(labels == 0)[0]
    sampled_disengaged = np.random.choice(
        disengaged_indices, engaged_count, replace=False
    )

    balanced_indices = np.concatenate([engaged_indices, sampled_disengaged])
    balanced_features = features[balanced_indices]
    balanced_labels = labels[balanced_indices]

    print(f"    Balanced to {len(balanced_labels)} windows")
    return balanced_features, balanced_labels


def save_realtime_model(model: object, scaler: object, output_dir: Path) -> None:
    """Save realtime model and scaler to disk."""
    model_path = output_dir / "engagement_realtime_model.joblib"
    scaler_path = output_dir / "engagement_realtime_scaler.joblib"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"    Realtime model saved to: {model_path}")


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
    subject_data = []
    subject_labels = []
    subject_ids = []
    processed_data_cache = {}

    for recording_path in recordings:
        subject_id = recording_path.parent.parent.name
        print(f"  Processing {subject_id}...")

        # Load and preprocess recording
        data, events, sample_rate = load_recording(recording_path)
        preprocessed_data = preprocess_eeg_multichannel(
            data, sample_rate, CHANNELS, spatial_filter="car"
        )

        # Create channel dictionary
        channel_signals = {
            channel_name: preprocessed_data[channel_index]
            for channel_index, channel_name in enumerate(CHANNELS)
        }

        # Cache for realtime model training
        processed_data_cache[recording_path] = (channel_signals, events, sample_rate)

        # Extract augmented epochs for each channel
        engaged_epochs_by_channel = {}
        disengaged_epochs_by_channel = {}

        for channel_name, signal in channel_signals.items():
            engaged_pairs, disengaged_pairs = extract_augmented_epochs(
                signal,
                events,
                sample_rate,
                window_duration=2.0,
                n_windows=2,
                class1_phase=3,  # Phase 3: Motor imagery (engaged)
                class2_phase=5,  # Phase 5: Rest (disengaged)
            )
            engaged_epochs_by_channel[channel_name] = engaged_pairs
            disengaged_epochs_by_channel[channel_name] = disengaged_pairs

        # Balance classes to prevent bias
        engaged_count = len(engaged_epochs_by_channel[CHANNELS[0]])
        disengaged_count = len(disengaged_epochs_by_channel[CHANNELS[0]])

        if disengaged_count > engaged_count:
            np.random.seed(42)
            selected_indices = np.random.choice(
                disengaged_count, engaged_count, replace=False
            )
            for channel_name in CHANNELS:
                disengaged_epochs_by_channel[channel_name] = [
                    disengaged_epochs_by_channel[channel_name][i]
                    for i in selected_indices
                ]
            disengaged_count = engaged_count

        print(
            f"    Engaged epochs: {engaged_count}, Disengaged epochs: {disengaged_count}"
        )

        # Extract ERD features
        engaged_features = extract_erd_features(engaged_epochs_by_channel, sample_rate)
        disengaged_features = extract_erd_features(
            disengaged_epochs_by_channel, sample_rate
        )

        # Create multichannel arrays for Riemannian classification
        multichannel_data = create_multichannel_arrays(
            engaged_epochs_by_channel, disengaged_epochs_by_channel, CHANNELS
        )

        # Combine features and labels
        features = np.vstack([engaged_features, disengaged_features])
        labels = np.array([1] * len(engaged_features) + [0] * len(disengaged_features))

        subject_data.append((features, multichannel_data))
        subject_labels.append(labels)
        subject_ids.append(subject_id)

    # Step 3: Within-subject Cross-validation
    print("\n[3/5] Running within-subject cross-validation...")
    print("(Using Riemannian geometry + feature ensemble)")
    scores, mean_accuracy, std_accuracy = train_within_subject_cv_riemannian(
        subject_data, subject_labels
    )

    print("\nPer-subject accuracy (within-subject CV):")
    for subject_id, score in zip(subject_ids, scores):
        status = "PASS" if score >= 0.9 else "FAIL"
        print(f"  {subject_id}: {score:.1%} [{status}]")

    print(f"\nMean accuracy: {mean_accuracy:.1%} (+/- {std_accuracy:.1%})")

    # Check if target met
    target_met = mean_accuracy >= 0.90
    print(f"\nTarget (>90%): {'MET' if target_met else 'NOT MET'}")

    # Step 4: Train final model on all data
    print("\n[4/5] Training final model on all data...")
    all_features = np.vstack([data[0] for data in subject_data])
    all_labels = np.concatenate(subject_labels)

    final_model, scaler = train_final_model(all_features, all_labels)

    # Step 4b: Train realtime model using only realtime-compatible features
    print("\n[4b/5] Training realtime model...")
    realtime_data = extract_realtime_training_data(processed_data_cache, CHANNELS)

    if realtime_data:
        features, labels = realtime_data
        balanced_features, balanced_labels = balance_realtime_data(features, labels)
        realtime_model, realtime_scaler = train_final_model(
            balanced_features, balanced_labels
        )
        save_realtime_model(realtime_model, realtime_scaler, output_dir)
    else:
        print("    Warning: No realtime training data extracted")

    # Step 5: Save models and results
    print("\n[5/5] Saving models and results...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save main model
    model_path = output_dir / "engagement_classifier_model.joblib"
    scaler_path = output_dir / "engagement_classifier_scaler.joblib"
    joblib.dump(final_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")

    # Create and save results
    results = create_results_dict(
        recordings=recordings,
        subject_ids=subject_ids,
        scores=scores,
        mean_accuracy=mean_accuracy,
        std_accuracy=std_accuracy,
        target_met=target_met,
        output_dir=output_dir,
    )

    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return results


def create_results_dict(
    recordings: list,
    subject_ids: list,
    scores: list,
    mean_accuracy: float,
    std_accuracy: float,
    target_met: bool,
    output_dir: Path,
) -> dict:
    """Create results dictionary for saving."""
    return {
        "n_subjects": len(recordings),
        "per_subject_scores": dict(
            zip(subject_ids, [float(score) for score in scores])
        ),
        "mean_accuracy": float(mean_accuracy),
        "std_accuracy": float(std_accuracy),
        "target_met": bool(target_met),
        "model_path": str(output_dir / "engagement_classifier_model.joblib"),
        "scaler_path": str(output_dir / "engagement_classifier_scaler.joblib"),
        "realtime_model_path": str(output_dir / "engagement_realtime_model.joblib"),
        "realtime_scaler_path": str(output_dir / "engagement_realtime_scaler.joblib"),
    }


if __name__ == "__main__":
    data_dir = Path("unicorn-data")
    output_dir = Path("models")

    results = run_pipeline(data_dir, output_dir)
