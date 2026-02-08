"""Main pipeline: load data, preprocess, extract features, train left/right classifier."""

import json
from pathlib import Path

import joblib
import numpy as np

from src.data_loader import CHANNELS, get_complete_recordings, load_recording
from src.epochs import extract_left_right_epochs, reject_bad_epochs
from src.features import extract_lateralization_features
from src.preprocess import preprocess_eeg
from src.train import (
    train_final_model,
    train_final_model_riemann,
    train_final_model_svm,
    train_within_subject_cv,
    train_within_subject_cv_ensemble,
    train_within_subject_cv_riemann,
    train_within_subject_cv_svm,
)


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

    def _task_epochs(pairs_by_ch):
        # (n_channels, n_trials, n_samples) -> (n_trials, n_channels, n_samples)
        return np.array(
            [[trial[1] for trial in pairs_by_ch[ch]] for ch in CHANNELS]
        ).transpose(1, 0, 2)

    for rec_path in recordings:
        subject_id = rec_path.parent.parent.name
        print(f"  Processing {subject_id}...")

        data, events, sfreq = load_recording(rec_path)

        left_pairs_by_channel = {}
        right_pairs_by_channel = {}
        for ch_idx, ch_name in enumerate(CHANNELS):
            signal = preprocess_eeg(data[ch_idx], sfreq)
            left, right = extract_left_right_epochs(
                signal,
                events,
                sfreq,
                task_duration=1.8,
                baseline_duration=1.0,
                skip_duration=0.5,
            )
            left_pairs_by_channel[ch_name] = left
            right_pairs_by_channel[ch_name] = right

        n_left_raw = len(left_pairs_by_channel[CHANNELS[0]])
        n_right_raw = len(right_pairs_by_channel[CHANNELS[0]])

        # Artifact rejection
        left_pairs_by_channel, right_pairs_by_channel, rej_l, rej_r = reject_bad_epochs(
            left_pairs_by_channel, right_pairs_by_channel
        )

        n_left = len(left_pairs_by_channel[CHANNELS[0]])
        n_right = len(right_pairs_by_channel[CHANNELS[0]])
        print(
            f"    Epochs: {n_left_raw}L/{n_right_raw}R → "
            f"rejected {rej_l}L/{rej_r}R → kept {n_left}L/{n_right}R"
        )

        if n_left == 0 or n_right == 0:
            print(f"    Skipping {subject_id} - no epochs after rejection")
            continue

        left_features = extract_lateralization_features(left_pairs_by_channel, sfreq)
        right_features = extract_lateralization_features(right_pairs_by_channel, sfreq)

        X_multichannel = np.vstack(
            [
                _task_epochs(left_pairs_by_channel),
                _task_epochs(right_pairs_by_channel),
            ]
        )
        X_features = np.vstack([left_features, right_features])
        y = np.array([0] * n_left + [1] * n_right)

        X_by_subject.append((X_features, X_multichannel))
        y_by_subject.append(y)
        subject_ids.append(subject_id)

    if not X_by_subject:
        raise ValueError("No valid subjects found")

    # Separate multichannel arrays for Riemannian pipeline
    X_multi_by_subject = [X_mc for _, X_mc in X_by_subject]

    # Step 3: Cross-validation (four classifiers + ensemble)
    print("\n[3/4] Running cross-validation...")
    print("(10-fold CV per subject: FBCSP+LDA, Riemannian, SVM, Ensemble)")

    fbcsp_scores, fbcsp_mean, fbcsp_std = train_within_subject_cv(
        X_by_subject, y_by_subject, sfreq=sfreq
    )
    riemann_scores, riemann_mean, riemann_std = train_within_subject_cv_riemann(
        X_multi_by_subject, y_by_subject, sfreq=sfreq
    )
    svm_scores, svm_mean, svm_std = train_within_subject_cv_svm(
        X_by_subject, y_by_subject, sfreq=sfreq
    )
    ensemble_scores, ensemble_mean, ensemble_std = train_within_subject_cv_ensemble(
        X_by_subject, y_by_subject, sfreq=sfreq
    )

    # Per-subject best score across all four classifiers
    best_scores = []
    best_methods = []
    for fs, rs, ss, es in zip(
        fbcsp_scores, riemann_scores, svm_scores, ensemble_scores
    ):
        candidates = [("FBCSP", fs), ("Riemann", rs), ("SVM", ss), ("Ensemble", es)]
        best_method, best_score = max(candidates, key=lambda x: x[1])
        best_scores.append(best_score)
        best_methods.append(best_method)

    best_mean = float(np.mean(best_scores))
    best_std = float(np.std(best_scores))

    print(
        f"\n{'Subject':<14} {'FBCSP+LDA':>10} {'Riemann':>10} {'SVM':>10} {'Ensemble':>10} {'Best':>10}"
    )
    print("-" * 74)
    for sid, fs, rs, ss, es, bs, bm in zip(
        subject_ids,
        fbcsp_scores,
        riemann_scores,
        svm_scores,
        ensemble_scores,
        best_scores,
        best_methods,
    ):
        status = "signal" if bs >= 0.60 else "chance"
        print(
            f"  {sid:<12} {fs:>9.1%} {rs:>9.1%} {ss:>9.1%} {es:>9.1%} {bs:>9.1%}  [{bm}, {status}]"
        )

    print(f"\nFBCSP+LDA mean: {fbcsp_mean:.1%} (+/- {fbcsp_std:.1%})")
    print(f"Riemann mean:   {riemann_mean:.1%} (+/- {riemann_std:.1%})")
    print(f"SVM mean:       {svm_mean:.1%} (+/- {svm_std:.1%})")
    print(f"Ensemble mean:  {ensemble_mean:.1%} (+/- {ensemble_std:.1%})")
    print(f"Best-of mean:   {best_mean:.1%} (+/- {best_std:.1%})")

    # Step 4: Train & save per-subject models (best method per subject)
    print("\n[4/4] Training and saving per-subject models...")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = {}
    for i, (sid, (X_features, X_multichannel), y, method) in enumerate(
        zip(subject_ids, X_by_subject, y_by_subject, best_methods)
    ):
        if method == "SVM":
            model = train_final_model_svm(X_features, X_multichannel, y, sfreq=sfreq)
            model_path = output_dir / f"{sid}_svm.joblib"
        elif method == "Riemann":
            model = train_final_model_riemann(X_multichannel, y, sfreq=sfreq)
            model_path = output_dir / f"{sid}_riemann.joblib"
        elif method == "Ensemble":
            # Ensemble is CV-only; save FBCSP+LDA as deployable model
            model = train_final_model(X_features, X_multichannel, y, sfreq=sfreq)
            model_path = output_dir / f"{sid}_ensemble_lda.joblib"
        else:
            model = train_final_model(X_features, X_multichannel, y, sfreq=sfreq)
            model_path = output_dir / f"{sid}_fbcsp_lda.joblib"
        joblib.dump(model, model_path)
        model_paths[sid] = str(model_path)
        print(f"  Saved {model_path} ({method})")

    above_chance, at_chance = [], []
    for sid, s in zip(subject_ids, best_scores):
        (above_chance if s >= 0.60 else at_chance).append(sid)

    print("\n" + "=" * 60)
    print("Diagnostic Summary")
    print("=" * 60)
    print(f"Subjects with signal (>=60%): {', '.join(above_chance) or 'none'}")
    print(f"Subjects at chance  (<60%):  {', '.join(at_chance) or 'none'}")
    print(f"Ensemble mean accuracy: {ensemble_mean:.1%} (+/- {ensemble_std:.1%})")
    print(f"Best-of mean accuracy:  {best_mean:.1%} (+/- {best_std:.1%})")

    # Save results
    results = {
        "task": "left_right_motor_imagery",
        "models": [
            "FBCSP+LDA (nested k)",
            "Riemannian (8-30Hz + OAS+TangentSpace+LR)",
            "FBCSP+SVM (nested k+C)",
            "Ensemble (soft voting)",
        ],
        "n_subjects": len(subject_ids),
        "fbcsp_scores": {sid: float(s) for sid, s in zip(subject_ids, fbcsp_scores)},
        "riemann_scores": {
            sid: float(s) for sid, s in zip(subject_ids, riemann_scores)
        },
        "svm_scores": {sid: float(s) for sid, s in zip(subject_ids, svm_scores)},
        "ensemble_scores": {
            sid: float(s) for sid, s in zip(subject_ids, ensemble_scores)
        },
        "best_scores": {sid: float(s) for sid, s in zip(subject_ids, best_scores)},
        "best_methods": {sid: m for sid, m in zip(subject_ids, best_methods)},
        "fbcsp_mean_accuracy": float(fbcsp_mean),
        "riemann_mean_accuracy": float(riemann_mean),
        "svm_mean_accuracy": float(svm_mean),
        "ensemble_mean_accuracy": float(ensemble_mean),
        "best_mean_accuracy": float(best_mean),
        "best_std_accuracy": float(best_std),
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
