"""Cross-subject Riemannian transfer learning with Euclidean Alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_covariance
from scipy.linalg import fractional_matrix_power
from sklearn.linear_model import LogisticRegression

from src.data_loader import CHANNELS, get_recordings_by_subject, load_recording
from src.epochs import (
    compute_rejection_threshold,
    extract_left_right_epochs,
    reject_bad_epochs,
)
from src.preprocess import preprocess_multichannel_eeg


def _task_epochs(pairs_by_ch: dict) -> np.ndarray:
    """Convert per-channel epoch pairs to (n_trials, n_channels, n_samples)."""
    return np.array(
        [[trial[1] for trial in pairs_by_ch[ch]] for ch in CHANNELS]
    ).transpose(1, 0, 2)


def load_all_subjects(
    data_dirs: list[Path],
    sfreq: float = 250.0,
    subject_merge: dict[str, str] | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load and preprocess data from multiple directories, merging by subject.

    Args:
        data_dirs: List of directories to scan for recordings.
        sfreq: Sampling frequency.
        subject_merge: Optional mapping of subject_id -> canonical_id for merging
            (e.g., {"subject0100_2": "subject0100"}).

    Returns:
        Dict mapping subject_id to (X_multichannel, y) where
        X_multichannel is (n_trials, 8, n_samples) and y is (n_trials,).
    """
    if subject_merge is None:
        subject_merge = {}

    all_recordings: dict[str, list[Path]] = {}
    for data_dir in data_dirs:
        for sid, paths in get_recordings_by_subject(data_dir).items():
            canonical = subject_merge.get(sid, sid)
            all_recordings.setdefault(canonical, []).extend(paths)

    subjects: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for sid, rec_paths in sorted(all_recordings.items()):
        all_left: dict[str, list] = {ch: [] for ch in CHANNELS}
        all_right: dict[str, list] = {ch: [] for ch in CHANNELS}

        for rec_path in rec_paths:
            data, events, rec_sfreq = load_recording(rec_path)
            if abs(rec_sfreq - sfreq) > 1e-6:
                continue
            preprocessed = preprocess_multichannel_eeg(data, sfreq)

            left_pairs: dict[str, list] = {}
            right_pairs: dict[str, list] = {}
            for ch_idx, ch_name in enumerate(CHANNELS):
                signal = preprocessed[ch_idx]
                left, right = extract_left_right_epochs(
                    signal,
                    events,
                    sfreq,
                    task_duration=3.0,
                    baseline_duration=1.0,
                    skip_duration=0.25,
                )
                left_pairs[ch_name] = left
                right_pairs[ch_name] = right

            threshold = compute_rejection_threshold(left_pairs, right_pairs)
            left_pairs, right_pairs, _, _ = reject_bad_epochs(
                left_pairs,
                right_pairs,
                threshold_uv=threshold,
            )

            for ch in CHANNELS:
                all_left[ch].extend(left_pairs[ch])
                all_right[ch].extend(right_pairs[ch])

        n_left = len(all_left[CHANNELS[0]])
        n_right = len(all_right[CHANNELS[0]])
        if n_left == 0 or n_right == 0:
            continue

        X_mc = np.vstack([_task_epochs(all_left), _task_epochs(all_right)])
        y = np.array([0] * n_left + [1] * n_right)
        subjects[sid] = (X_mc, y)

    return subjects


def align_subjects(
    subjects: dict[str, tuple[np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Compute covariance matrices and apply Euclidean Alignment across subjects.

    EA re-centers each subject's covariance distribution to the identity matrix,
    removing inter-subject variability from electrode impedance and placement.

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency (unused, kept for API consistency).

    Returns:
        Tuple of:
        - aligned_covs: (total_trials, n_channels, n_channels) aligned SPD matrices
        - labels: (total_trials,) class labels
        - subject_ids: list of subject_id per trial (for LOSO indexing)
    """
    cov_estimator = Covariances(estimator="lwf")

    all_covs = []
    all_labels = []
    all_subject_ids: list[str] = []

    for sid in sorted(subjects.keys()):
        X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        ref = mean_covariance(covs, metric="riemann")
        ref_inv_sqrt = fractional_matrix_power(ref, -0.5).real
        covs_aligned = ref_inv_sqrt @ covs @ ref_inv_sqrt.T

        all_covs.append(covs_aligned)
        all_labels.append(y)
        all_subject_ids.extend([sid] * len(y))

    return np.vstack(all_covs), np.concatenate(all_labels), all_subject_ids


def loso_cv(
    subjects: dict[str, tuple[np.ndarray, np.ndarray]],
    sfreq: float = 250.0,
    fine_tune: bool = False,
) -> dict[str, float]:
    """Leave-one-subject-out CV with Euclidean Alignment.

    For each held-out subject:
    1. Align all subjects independently (EA per subject)
    2. Train TangentSpace + LR on aligned data from all other subjects
    3. Test on held-out subject's aligned data
    4. (If fine_tune) Re-center tangent space reference on held-out subject's mean

    Args:
        subjects: Dict mapping subject_id to (X_multichannel, y).
        sfreq: Sampling frequency.
        fine_tune: If True, re-center the tangent space reference point
            to the held-out subject's Riemannian mean before testing.

    Returns:
        Dict mapping subject_id to LOSO accuracy.
    """
    cov_estimator = Covariances(estimator="lwf")

    # Pre-compute aligned covariances per subject
    aligned_per_subject: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sid in sorted(subjects.keys()):
        X_mc, y = subjects[sid]
        covs = cov_estimator.fit_transform(X_mc)
        ref = mean_covariance(covs, metric="riemann")
        ref_inv_sqrt = fractional_matrix_power(ref, -0.5).real
        covs_aligned = ref_inv_sqrt @ covs @ ref_inv_sqrt.T
        aligned_per_subject[sid] = (covs_aligned, y)

    subject_ids = sorted(subjects.keys())
    scores: dict[str, float] = {}

    for held_out in subject_ids:
        train_covs = []
        train_labels = []
        for sid in subject_ids:
            if sid == held_out:
                continue
            covs, y = aligned_per_subject[sid]
            train_covs.append(covs)
            train_labels.append(y)

        X_train = np.vstack(train_covs)
        y_train = np.concatenate(train_labels)

        X_test, y_test = aligned_per_subject[held_out]

        if fine_tune:
            # Re-center tangent space on held-out subject's mean
            test_ref = mean_covariance(X_test, metric="riemann")
            # Train projection uses pooled training data reference
            train_ts = TangentSpace(metric="riemann")
            X_train_ts = train_ts.fit_transform(X_train)
            # Test projection uses held-out subject's own reference
            test_ts = TangentSpace(metric="riemann")
            test_ts.fit(X_train)  # fit to get consistent dimensionality
            test_ts.reference_ = test_ref  # override reference point
            X_test_ts = test_ts.transform(X_test)
        else:
            pipe_ts = TangentSpace(metric="riemann")
            X_train_ts = pipe_ts.fit_transform(X_train)
            X_test_ts = pipe_ts.transform(X_test)

        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=1000)
        clf.fit(X_train_ts, y_train)
        scores[held_out] = float(clf.score(X_test_ts, y_test))

    return scores


def run_transfer_evaluation(
    data_dirs: list[Path] | None = None,
    sfreq: float = 250.0,
    subject_merge: dict[str, str] | None = None,
    within_subject_scores: dict[str, float] | None = None,
) -> dict:
    """Run cross-subject transfer learning evaluation and print comparison.

    Args:
        data_dirs: Directories to scan. Defaults to Data/unicorn-data + Data/MI_DATA_NEW.
        sfreq: Sampling frequency.
        subject_merge: Subject ID merging map.
        within_subject_scores: Optional dict of subject_id -> nested CV accuracy
            for comparison table.

    Returns:
        Results dict with LOSO and LOSO+FT scores.
    """
    if data_dirs is None:
        data_dirs = [Path("Data/unicorn-data"), Path("Data/MI_DATA_NEW")]
    if subject_merge is None:
        subject_merge = {"subject0100_2": "subject0100"}

    print("=" * 68)
    print("Cross-Subject Transfer Learning (Euclidean Alignment + LOSO)")
    print("=" * 68)

    print("\nLoading subjects from all data directories...")
    subjects = load_all_subjects(data_dirs, sfreq=sfreq, subject_merge=subject_merge)
    print(f"Loaded {len(subjects)} subjects: {', '.join(sorted(subjects.keys()))}")
    for sid in sorted(subjects.keys()):
        X_mc, y = subjects[sid]
        n_left = int(np.sum(y == 0))
        n_right = int(np.sum(y == 1))
        print(f"  {sid}: {len(y)} trials ({n_left}L/{n_right}R)")

    print("\nRunning LOSO CV (no fine-tuning)...")
    loso_scores = loso_cv(subjects, sfreq=sfreq, fine_tune=False)

    print("Running LOSO CV (with fine-tuning)...")
    loso_ft_scores = loso_cv(subjects, sfreq=sfreq, fine_tune=True)

    # Print comparison table
    loso_mean = float(np.mean(list(loso_scores.values())))
    loso_ft_mean = float(np.mean(list(loso_ft_scores.values())))

    header = f"\n{'Subject':<16}"
    if within_subject_scores:
        header += f"{'Within-Subj':>12}"
    header += f"{'LOSO':>10}{'LOSO+FT':>10}"
    print(header)
    print("-" * len(header))

    for sid in sorted(loso_scores.keys()):
        row = f"  {sid:<14}"
        if within_subject_scores and sid in within_subject_scores:
            row += f"{within_subject_scores[sid]:>11.1%}"
        elif within_subject_scores:
            row += f"{'n/a':>12}"
        row += f"{loso_scores[sid]:>9.1%}{loso_ft_scores[sid]:>9.1%}"
        print(row)

    print(f"\n  LOSO mean:       {loso_mean:.1%}")
    print(f"  LOSO+FT mean:    {loso_ft_mean:.1%}")
    if within_subject_scores:
        ws_mean = float(np.mean(list(within_subject_scores.values())))
        print(f"  Within-subj mean: {ws_mean:.1%}")

    return {
        "loso_scores": loso_scores,
        "loso_ft_scores": loso_ft_scores,
        "loso_mean": loso_mean,
        "loso_ft_mean": loso_ft_mean,
        "n_subjects": len(subjects),
    }


if __name__ == "__main__":
    results = run_transfer_evaluation()
