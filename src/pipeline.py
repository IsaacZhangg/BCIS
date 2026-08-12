"""Main pipeline: load data, preprocess, extract features, train left/right classifier."""

import json
import time
from pathlib import Path

import joblib
import numpy as np

from src.config import DEFAULT_SUBJECT_MERGE, MI_DATA_NEW_DIR, TrainingConfig
from src.data_loader import (
    CHANNELS,
    get_complete_recordings,
    get_recordings_by_subject,
    get_recordings_flexible,
    load_recording,
)
from src.epochs import (
    adaptive_threshold,
    compute_rejection_threshold,
    compute_trial_max_ptp,
    extract_left_right_epochs,
    reject_bad_epochs,
    task_epochs,
)
from src.features import extract_lateralization_features
from src.preprocess import preprocess_multichannel_eeg
from src.runtime_output import configure_console_output
from src.transfer import run_transfer_evaluation
from src.train import (
    FBCSP_BAND_CANDIDATES,
    FBCSP_BANDS,
    cross_session_evaluate,
    train_final_model,
    train_final_model_riemann,
    train_final_model_svm,
    train_nested_model_selection_cv,
    train_within_subject_cv_all_models,
)
from src.validation import build_classwise_trial_groups

_NESTED_TO_DISPLAY = {
    "lda": "FBCSP",
    "riemann": "Riemann",
    "svm": "SVM",
    "ensemble": "Ensemble",
}

_LINE_WIDTH = 68


def _print_header(title: str) -> None:
    """Print a consistent run header."""
    print("=" * _LINE_WIDTH)
    print(title)
    print("=" * _LINE_WIDTH)


def _print_step(step_idx: int, total_steps: int, title: str) -> None:
    """Print a clean step marker."""
    print(f"\n[{step_idx}/{total_steps}] {title}")


def _process_recording(
    rec_path: Path,
    sfreq: float,
    threshold_uv: float | None = None,
) -> tuple | None:
    """Load, preprocess, epoch, and artifact-reject a single recording."""
    data, events, _ = load_recording(rec_path)
    preprocessed = preprocess_multichannel_eeg(data, sfreq)

    left_pairs_by_channel: dict = {}
    right_pairs_by_channel: dict = {}
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
        left_pairs_by_channel[ch_name] = left
        right_pairs_by_channel[ch_name] = right

    n_left_raw = len(left_pairs_by_channel[CHANNELS[0]])
    n_right_raw = len(right_pairs_by_channel[CHANNELS[0]])

    left_pairs_by_channel, right_pairs_by_channel, rej_l, rej_r = reject_bad_epochs(
        left_pairs_by_channel,
        right_pairs_by_channel,
        threshold_uv=threshold_uv,
    )

    n_left = len(left_pairs_by_channel[CHANNELS[0]])
    n_right = len(right_pairs_by_channel[CHANNELS[0]])

    if n_left == 0 or n_right == 0:
        return None

    return (
        left_pairs_by_channel,
        right_pairs_by_channel,
        n_left_raw,
        n_right_raw,
        rej_l,
        rej_r,
    )


def _split_epoch_pairs(
    left_pairs_by_channel: dict,
    right_pairs_by_channel: dict,
    test_fraction: float,
    random_state: int = 42,
) -> tuple[dict, dict, dict, dict]:
    """Stratified split of epoch pairs into train/test before artifact rejection."""
    rng = np.random.default_rng(random_state)
    channels = list(left_pairs_by_channel.keys())

    def _split_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
        n_test = max(1, int(n * test_fraction))
        indices = rng.permutation(n)
        return indices[n_test:], indices[:n_test]

    n_left = len(left_pairs_by_channel[channels[0]])
    n_right = len(right_pairs_by_channel[channels[0]])

    left_train_idx, left_test_idx = _split_indices(n_left)
    right_train_idx, right_test_idx = _split_indices(n_right)

    def _select(pairs_by_ch: dict, indices: np.ndarray) -> dict:
        return {ch: [pairs_by_ch[ch][i] for i in indices] for ch in channels}

    return (
        _select(left_pairs_by_channel, left_train_idx),
        _select(right_pairs_by_channel, right_train_idx),
        _select(left_pairs_by_channel, left_test_idx),
        _select(right_pairs_by_channel, right_test_idx),
    )


def _pairs_to_features(
    left_pairs: dict,
    right_pairs: dict,
    sfreq: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cleaned epoch pairs to feature arrays and labels."""
    n_left = len(left_pairs[CHANNELS[0]])
    n_right = len(right_pairs[CHANNELS[0]])

    left_features = extract_lateralization_features(left_pairs, sfreq)
    right_features = extract_lateralization_features(right_pairs, sfreq)

    X_multichannel = np.vstack(
        [task_epochs(left_pairs, CHANNELS), task_epochs(right_pairs, CHANNELS)]
    )
    X_features = np.vstack([left_features, right_features])
    y = np.array([0] * n_left + [1] * n_right)
    return X_features, X_multichannel, y


def _augmented_nested_cv_subject(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    donor_feat: np.ndarray,
    donor_mc: np.ndarray,
    donor_y: np.ndarray,
    sfreq: float,
    n_outer_folds: int,
    n_inner_folds: int,
    k_best: int,
    trial_ptps: np.ndarray | None,
    split_strategy: str,
    groups: np.ndarray | None,
    random_state: int,
) -> float:
    """Leakage-free augmented nested CV for a single subject.

    Outer/inner CV splits are on TARGET data only. Donor data augments
    training portions only — never appears in validation or test folds.
    """
    from src.train import (
        ALL_CLASSIFIERS,
        _evaluate_classifier,
        _evaluate_classifiers_batch,
        _reject_in_fold,
    )
    from src.validation import make_cv_splits

    classifier_names = list(ALL_CLASSIFIERS)
    outer_splits = make_cv_splits(
        y,
        n_splits=n_outer_folds,
        strategy=split_strategy,
        groups=groups,
        random_state=random_state,
    )
    fold_scores: list[float] = []

    for outer_train_idx, outer_test_idx in outer_splits:
        outer_train_idx, outer_test_idx = _reject_in_fold(
            trial_ptps, outer_train_idx, outer_test_idx
        )
        if len(outer_train_idx) < 2 or len(outer_test_idx) < 1:
            continue

        X_feat_otest = X_features[outer_test_idx]
        X_mc_otest = X_multichannel[outer_test_idx]
        y_otest = y[outer_test_idx]

        X_feat_otrain = np.vstack([X_features[outer_train_idx], donor_feat])
        X_mc_otrain = np.vstack([X_multichannel[outer_train_idx], donor_mc])
        y_otrain = np.concatenate([y[outer_train_idx], donor_y])

        inner_groups = groups[outer_train_idx] if groups is not None else None
        inner_splits = make_cv_splits(
            y[outer_train_idx],
            n_splits=n_inner_folds,
            strategy=split_strategy,
            groups=inner_groups,
            random_state=random_state,
        )

        inner_scores_by_clf: dict[str, list[float]] = {n: [] for n in classifier_names}
        for inner_train_idx, inner_val_idx in inner_splits:
            # Inner train: target inner train + ALL donor data
            inner_feat_train = np.vstack(
                [X_features[outer_train_idx[inner_train_idx]], donor_feat]
            )
            inner_mc_train = np.vstack(
                [X_multichannel[outer_train_idx[inner_train_idx]], donor_mc]
            )
            inner_y_train = np.concatenate(
                [y[outer_train_idx[inner_train_idx]], donor_y]
            )
            # Inner val: TARGET only (no donor)
            inner_feat_val = X_features[outer_train_idx[inner_val_idx]]
            inner_mc_val = X_multichannel[outer_train_idx[inner_val_idx]]
            inner_y_val = y[outer_train_idx[inner_val_idx]]

            split_scores = _evaluate_classifiers_batch(
                classifier_names,
                inner_feat_train,
                inner_feat_val,
                inner_y_train,
                inner_y_val,
                inner_mc_train,
                inner_mc_val,
                sfreq,
                k_best,
            )
            for clf_name, score in split_scores.items():
                inner_scores_by_clf[clf_name].append(score)

        inner_means = {
            name: float(np.mean(scores)) if scores else 0.5
            for name, scores in inner_scores_by_clf.items()
        }
        best_clf = max(inner_means, key=inner_means.get)  # type: ignore[arg-type]

        outer_score = _evaluate_classifier(
            best_clf,
            X_feat_otrain,
            X_feat_otest,
            y_otrain,
            y_otest,
            X_mc_otrain,
            X_mc_otest,
            sfreq,
            k_best,
        )
        fold_scores.append(outer_score)

    return float(np.mean(fold_scores)) if fold_scores else 0.5


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    holdout_fraction: float = 0.0,
    training_config: TrainingConfig | None = None,
    quiet_output: bool = True,
) -> dict:
    """Run the full training pipeline for left/right motor imagery classification."""
    configure_console_output(quiet_output)
    _print_header("Left/Right Motor Imagery Classifier - Training Pipeline")
    cfg = training_config or TrainingConfig()
    sfreq = cfg.sfreq
    runtime_seconds: dict[str, float] = {}
    total_start = time.perf_counter()

    # Step 1: Find complete recordings
    step_start = time.perf_counter()
    _print_step(1, 5, "Finding complete recordings")
    recordings = get_complete_recordings(data_dir)
    print(f"Found {len(recordings)} complete recordings")

    if holdout_fraction > 0:
        print(f"  Held-out fraction: {holdout_fraction:.0%}")
    print(
        f"  CV split strategy: {cfg.split_strategy} (group_size={cfg.trial_group_size})"
    )
    print(
        f"  Parallelism: n_jobs={cfg.n_jobs}, backend={cfg.parallel_backend}, "
        f"blas_threads/worker={cfg.max_blas_threads_per_worker}"
    )
    print(f"  Bandpass cache: {'on' if cfg.enable_band_cache else 'off'}")
    runtime_seconds["find_recordings"] = time.perf_counter() - step_start

    # Step 2: Load and preprocess data for each subject
    step_start = time.perf_counter()
    _print_step(2, 5, "Loading and preprocessing data")
    X_by_subject: list[tuple[np.ndarray, np.ndarray]] = []
    y_by_subject: list[np.ndarray] = []
    subject_ids: list[str] = []
    trial_ptps_by_subject: list[np.ndarray] = []
    trial_groups_by_subject: list[np.ndarray] = []

    X_holdout_by_subject: list[tuple[np.ndarray, np.ndarray]] = []
    y_holdout_by_subject: list[np.ndarray] = []

    for rec_path in recordings:
        subject_id = rec_path.parent.parent.name
        print(f"  Processing {subject_id}...")

        data, events, rec_sfreq = load_recording(rec_path)
        if abs(rec_sfreq - sfreq) > 1e-6:
            raise ValueError(
                f"Sampling-rate mismatch for {rec_path}: expected {sfreq}, got {rec_sfreq}"
            )
        preprocessed = preprocess_multichannel_eeg(data, sfreq)

        left_pairs_by_channel: dict = {}
        right_pairs_by_channel: dict = {}
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
            left_pairs_by_channel[ch_name] = left
            right_pairs_by_channel[ch_name] = right

        n_left_raw = len(left_pairs_by_channel[CHANNELS[0]])
        n_right_raw = len(right_pairs_by_channel[CHANNELS[0]])

        if holdout_fraction > 0:
            # Split before artifact rejection
            train_left, train_right, test_left, test_right = _split_epoch_pairs(
                left_pairs_by_channel,
                right_pairs_by_channel,
                test_fraction=holdout_fraction,
            )

            # Compute threshold from training data only
            threshold = compute_rejection_threshold(train_left, train_right)

            # Apply same threshold to both splits
            train_left, train_right, rej_l_tr, rej_r_tr = reject_bad_epochs(
                train_left, train_right, threshold_uv=threshold
            )
            test_left, test_right, rej_l_te, rej_r_te = reject_bad_epochs(
                test_left, test_right, threshold_uv=threshold
            )

            n_train_l = len(train_left[CHANNELS[0]])
            n_train_r = len(train_right[CHANNELS[0]])
            n_test_l = len(test_left[CHANNELS[0]])
            n_test_r = len(test_right[CHANNELS[0]])

            print(
                f"    Epochs: {n_left_raw}L/{n_right_raw}R → "
                f"train {n_train_l}L/{n_train_r}R (rej {rej_l_tr}/{rej_r_tr}), "
                f"holdout {n_test_l}L/{n_test_r}R (rej {rej_l_te}/{rej_r_te})"
            )

            if n_train_l == 0 or n_train_r == 0:
                print(f"    Skipping {subject_id} - no training epochs after rejection")
                continue

            X_feat, X_mc, y = _pairs_to_features(train_left, train_right, sfreq)
            X_by_subject.append((X_feat, X_mc))
            y_by_subject.append(y)
            subject_ids.append(subject_id)
            trial_ptps_by_subject.append(compute_trial_max_ptp(X_mc))
            trial_groups_by_subject.append(
                build_classwise_trial_groups(y, group_size=cfg.trial_group_size)
            )

            if n_test_l > 0 and n_test_r > 0:
                X_feat_ho, X_mc_ho, y_ho = _pairs_to_features(
                    test_left, test_right, sfreq
                )
                X_holdout_by_subject.append((X_feat_ho, X_mc_ho))
                y_holdout_by_subject.append(y_ho)
            else:
                # Empty holdout — still track so indices match
                X_holdout_by_subject.append((np.empty((0, 0)), np.empty((0, 0, 0))))
                y_holdout_by_subject.append(np.array([]))
        else:
            n_left = len(left_pairs_by_channel[CHANNELS[0]])
            n_right = len(right_pairs_by_channel[CHANNELS[0]])

            if n_left == 0 or n_right == 0:
                print(f"    Skipping {subject_id} - no epochs")
                continue

            X_feat, X_mc, y = _pairs_to_features(
                left_pairs_by_channel, right_pairs_by_channel, sfreq
            )

            trial_ptps = compute_trial_max_ptp(X_mc)

            global_thresh = adaptive_threshold(trial_ptps, 4.0)
            n_flagged = int(np.sum((trial_ptps > global_thresh) | (trial_ptps < 1.0)))
            print(
                f"    Epochs: {n_left_raw}L/{n_right_raw}R "
                f"(~{n_flagged} artifact trials, rejected per CV fold)"
            )

            X_by_subject.append((X_feat, X_mc))
            y_by_subject.append(y)
            subject_ids.append(subject_id)
            trial_ptps_by_subject.append(trial_ptps)
            trial_groups_by_subject.append(
                build_classwise_trial_groups(y, group_size=cfg.trial_group_size)
            )

    if not X_by_subject:
        raise ValueError("No valid subjects found")
    runtime_seconds["load_preprocess"] = time.perf_counter() - step_start

    # Load additional subjects from MI_DATA_NEW (subject-level merge)
    if MI_DATA_NEW_DIR.exists():
        mi_new_grouped = get_recordings_flexible(MI_DATA_NEW_DIR, min_trials=20)

        # Apply subject merge mapping
        merged_grouped: dict[str, list[Path]] = {}
        for sid, paths in mi_new_grouped.items():
            canonical = DEFAULT_SUBJECT_MERGE.get(sid, sid)
            merged_grouped.setdefault(canonical, []).extend(paths)

        # Skip subjects already loaded from unicorn-data
        for sid in sorted(merged_grouped.keys()):
            if sid in subject_ids:
                continue

            rec_paths = merged_grouped[sid]
            print(f"  Processing {sid} (MI_DATA_NEW, {len(rec_paths)} recording(s))...")

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

                # Per-recording artifact rejection
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
                print(f"    Skipping {sid} — no clean epochs")
                continue

            print(
                f"    Epochs: {n_left}L/{n_right}R (merged from {len(rec_paths)} recording(s))"
            )

            X_feat, X_mc, y = _pairs_to_features(all_left, all_right, sfreq)
            trial_ptps = compute_trial_max_ptp(X_mc)

            X_by_subject.append((X_feat, X_mc))
            y_by_subject.append(y)
            subject_ids.append(sid)
            trial_ptps_by_subject.append(trial_ptps)
            trial_groups_by_subject.append(
                build_classwise_trial_groups(y, group_size=cfg.trial_group_size)
            )

    # Step 3: Cross-validation (four classifiers + nested model selection)
    step_start = time.perf_counter()
    _print_step(3, 5, "Running cross-validation")
    print(f"({cfg.n_folds}-fold CV per subject: FBCSP+LDA, Riemannian, SVM, Ensemble)")

    cv_results = train_within_subject_cv_all_models(
        X_by_subject,
        y_by_subject,
        sfreq=sfreq,
        n_folds=cfg.n_folds,
        trial_ptps_by_subject=trial_ptps_by_subject,
        split_strategy=cfg.split_strategy,
        trial_groups_by_subject=trial_groups_by_subject,
        trial_group_size=cfg.trial_group_size,
        random_state=cfg.random_state,
        k_best=cfg.k_best,
        k_candidates=cfg.k_candidates,
        n_jobs=cfg.n_jobs,
        parallel_backend=cfg.parallel_backend,
        max_blas_threads_per_worker=cfg.max_blas_threads_per_worker,
        enable_band_cache=cfg.enable_band_cache,
    )
    fbcsp_scores, fbcsp_mean, fbcsp_std = cv_results["lda"]
    riemann_scores, riemann_mean, riemann_std = cv_results["riemann"]
    svm_scores, svm_mean, svm_std = cv_results["svm"]
    ensemble_scores, ensemble_mean, ensemble_std = cv_results["ensemble"]

    best_scores = []
    best_methods_posthoc = []
    for fs, rs, ss, es in zip(
        fbcsp_scores, riemann_scores, svm_scores, ensemble_scores
    ):
        candidates = [("FBCSP", fs), ("Riemann", rs), ("SVM", ss), ("Ensemble", es)]
        best_method, best_score = max(candidates, key=lambda x: x[1])
        best_scores.append(best_score)
        best_methods_posthoc.append(best_method)

    best_mean = float(np.mean(best_scores))
    best_std = float(np.std(best_scores))

    print("\nRunning nested model-selection CV (unbiased best-of estimate)...")
    nested_scores, nested_mean, nested_std, nested_methods, nested_band_configs = (
        train_nested_model_selection_cv(
            X_by_subject,
            y_by_subject,
            sfreq=sfreq,
            n_outer_folds=cfg.n_outer_folds,
            n_inner_folds=cfg.n_inner_folds,
            k_best=cfg.k_best,
            trial_ptps_by_subject=trial_ptps_by_subject,
            split_strategy=cfg.split_strategy,
            trial_groups_by_subject=trial_groups_by_subject,
            trial_group_size=cfg.trial_group_size,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
            parallel_backend=cfg.parallel_backend,
            max_blas_threads_per_worker=cfg.max_blas_threads_per_worker,
            enable_band_cache=cfg.enable_band_cache,
            cache_scope=cfg.cache_scope,
            band_candidates=FBCSP_BAND_CANDIDATES,
        )
    )
    runtime_seconds["cross_validation"] = time.perf_counter() - step_start
    nested_methods_display = [_NESTED_TO_DISPLAY[m] for m in nested_methods]

    print(
        f"\n{'Subject':<14} {'FBCSP+LDA':>10} {'Riemann':>10} {'SVM':>10} "
        f"{'Ensemble':>10} {'Best':>10} {'Nested':>10}"
    )
    print("-" * 84)
    for sid, fs, rs, ss, es, bs, bm, ns, nm, bc in zip(
        subject_ids,
        fbcsp_scores,
        riemann_scores,
        svm_scores,
        ensemble_scores,
        best_scores,
        best_methods_posthoc,
        nested_scores,
        nested_methods_display,
        nested_band_configs,
    ):
        status = "signal" if ns >= 0.60 else "chance"
        print(
            f"  {sid:<12} {fs:>9.1%} {rs:>9.1%} {ss:>9.1%} {es:>9.1%} "
            f"{bs:>9.1%} {ns:>9.1%}  [{nm}, {bc}, {status}]"
        )

    print(f"\nFBCSP+LDA mean: {fbcsp_mean:.1%} (+/- {fbcsp_std:.1%})")
    print(f"Riemann mean:   {riemann_mean:.1%} (+/- {riemann_std:.1%})")
    print(f"SVM mean:       {svm_mean:.1%} (+/- {svm_std:.1%})")
    print(f"Ensemble mean:  {ensemble_mean:.1%} (+/- {ensemble_std:.1%})")
    print(f"Best-of mean (optimistic):     {best_mean:.1%} (+/- {best_std:.1%})")
    print(f"Nested selection mean (unbiased): {nested_mean:.1%} (+/- {nested_std:.1%})")

    # Step 3b: Augmented nested CV for weak subjects (leakage-free)
    aug_nested_scores = None
    aug_nested_mean = None
    step_start_aug = time.perf_counter()
    if MI_DATA_NEW_DIR.exists():
        print("\n[3b/5] Augmented nested CV (cross-subject data for weak subjects)")
        from pyriemann.estimation import Covariances as _Cov
        from pyriemann.utils.distance import distance_riemann as _dist_riemann
        from pyriemann.utils.mean import mean_covariance as _mean_cov

        from src.transfer import load_all_subjects

        transfer_data_dirs = [data_dir, MI_DATA_NEW_DIR]
        donor_subjects = load_all_subjects(
            transfer_data_dirs,
            sfreq=sfreq,
            subject_merge=DEFAULT_SUBJECT_MERGE,
        )
        print(f"  Donor pool: {len(donor_subjects)} subjects")

        # Compute Riemannian means for donor selection
        _cov_est = _Cov(estimator="lwf")
        donor_means = {
            sid: _mean_cov(_cov_est.fit_transform(d[1]), metric="riemann")
            for sid, d in donor_subjects.items()
        }
        target_means = {
            subject_ids[i]: _mean_cov(
                _cov_est.fit_transform(X_by_subject[i][1]), metric="riemann"
            )
            for i in range(len(subject_ids))
        }

        weakness_threshold = cfg.augmentation_weakness_threshold
        aug_nested_scores = list(nested_scores)  # start with original scores

        for i, (sid, score) in enumerate(zip(subject_ids, nested_scores)):
            if score >= weakness_threshold:
                continue

            # Find closest donor by Riemannian distance
            dists = {
                dsid: float(_dist_riemann(target_means[sid], donor_means[dsid]))
                for dsid in donor_subjects
                if dsid != sid
            }
            best_donor = min(dists, key=dists.get)

            donor_feat, donor_mc, donor_y = donor_subjects[best_donor]
            target_feat, target_mc = X_by_subject[i]

            aug_score = _augmented_nested_cv_subject(
                target_feat,
                target_mc,
                y_by_subject[i],
                donor_feat,
                donor_mc,
                donor_y,
                sfreq=sfreq,
                n_outer_folds=cfg.n_outer_folds,
                n_inner_folds=cfg.n_inner_folds,
                k_best=cfg.k_best,
                trial_ptps=trial_ptps_by_subject[i],
                split_strategy=cfg.split_strategy,
                groups=trial_groups_by_subject[i],
                random_state=cfg.random_state,
            )
            aug_nested_scores[i] = aug_score
            print(
                f"    {sid}: +{best_donor} (d={dists[best_donor]:.1f}) "
                f"{score:.1%} -> {aug_score:.1%} ({aug_score - score:+.1%})"
            )

        aug_nested_mean = float(np.mean(aug_nested_scores))

        print(f"\n  {'Subject':<14} {'Original':>10} {'Augmented':>10} {'Delta':>8}")
        print("  " + "-" * 42)
        for sid, orig, aug in zip(subject_ids, nested_scores, aug_nested_scores):
            delta = aug - orig
            sign = "+" if delta >= 0 else ""
            weak = " *" if orig < weakness_threshold else ""
            print(f"  {sid:<14} {orig:>9.1%} {aug:>9.1%} {sign}{delta:>6.1%}{weak}")
        print(f"\n  Original nested mean:  {nested_mean:.1%}")
        print(f"  Augmented nested mean: {aug_nested_mean:.1%}")
        print(f"  Delta:                 {aug_nested_mean - nested_mean:+.1%}")
        print("  (* = weak subject, augmented with closest neighbor)")
    runtime_seconds["augmented_nested_cv"] = time.perf_counter() - step_start_aug

    holdout_results: dict[str, float] = {}
    if holdout_fraction > 0 and X_holdout_by_subject:
        print("\n[3b/5] Evaluating held-out test sets")
        from src.train import _evaluate_classifier

        for i, (sid, (X_feat, X_mc), y_tr) in enumerate(
            zip(subject_ids, X_by_subject, y_by_subject)
        ):
            X_feat_ho, X_mc_ho = X_holdout_by_subject[i]
            y_ho = y_holdout_by_subject[i]
            if len(y_ho) == 0:
                print(f"  {sid}: no held-out trials")
                continue
            method_name = nested_methods[i]
            acc = _evaluate_classifier(
                method_name,
                X_feat,
                X_feat_ho,
                y_tr,
                y_ho,
                X_mc,
                X_mc_ho,
                sfreq,
                k_best=cfg.k_best,
            )
            holdout_results[sid] = acc
            print(
                f"  {sid}: holdout acc = {acc:.1%} "
                f"({len(y_ho)} trials, method={nested_methods_display[i]})"
            )

        if holdout_results:
            ho_mean = float(np.mean(list(holdout_results.values())))
            print(f"\n  Held-out mean: {ho_mean:.1%}")

    # Step 4: Cross-session evaluation
    step_start = time.perf_counter()
    _print_step(4, 5, "Cross-session evaluation")
    recordings_by_subject = get_recordings_by_subject(data_dir)
    multi_session_subjects = {
        sid: paths for sid, paths in recordings_by_subject.items() if len(paths) > 1
    }

    cross_session_results: dict[str, dict[str, float]] = {}
    if multi_session_subjects:
        for sid, paths in multi_session_subjects.items():
            print(f"  {sid}: {len(paths)} sessions, evaluating cross-session...")
            result_a = _process_recording(paths[0], sfreq)
            result_b = _process_recording(paths[1], sfreq)
            if result_a is None or result_b is None:
                print(f"    Skipping {sid} — insufficient epochs in one session")
                continue
            left_a, right_a, *_ = result_a
            left_b, right_b, *_ = result_b
            X_feat_a, X_mc_a, y_a = _pairs_to_features(left_a, right_a, sfreq)
            X_feat_b, X_mc_b, y_b = _pairs_to_features(left_b, right_b, sfreq)
            cs_results = cross_session_evaluate(
                (X_feat_a, X_mc_a), y_a, (X_feat_b, X_mc_b), y_b, sfreq=sfreq
            )
            cross_session_results[sid] = cs_results
            for method, acc in cs_results.items():
                print(f"    {method}: {acc:.1%}")
    else:
        print(
            "  WARNING: All subjects have a single recording. "
            "Cross-session evaluation requires >=2 recordings per subject."
        )
    runtime_seconds["cross_session_eval"] = time.perf_counter() - step_start

    # Step 4b: Cross-subject transfer evaluation
    step_start = time.perf_counter()
    print("\n[4b/5] Cross-subject transfer evaluation (Euclidean Alignment + LOSO)")
    transfer_results: dict | None = None
    try:
        transfer_data_dirs = [data_dir]
        if MI_DATA_NEW_DIR.exists():
            transfer_data_dirs.append(MI_DATA_NEW_DIR)
        within_scores = {sid: float(s) for sid, s in zip(subject_ids, nested_scores)}
        transfer_results = run_transfer_evaluation(
            data_dirs=transfer_data_dirs,
            sfreq=sfreq,
            within_subject_scores=within_scores,
        )
    except Exception as e:
        print(f"  Transfer evaluation failed: {e}")
    runtime_seconds["transfer_eval"] = time.perf_counter() - step_start

    # Step 5: Train & save per-subject models (nested CV method)
    step_start = time.perf_counter()
    _print_step(5, 5, "Training and saving per-subject models")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = {}
    for i, (sid, (X_features, X_multichannel), y, method) in enumerate(
        zip(subject_ids, X_by_subject, y_by_subject, nested_methods_display)
    ):
        if trial_ptps_by_subject:
            ptps = trial_ptps_by_subject[i]
            thresh = adaptive_threshold(ptps, 4.0)
            clean_mask = (ptps >= 1.0) & (ptps <= thresh)
            X_features = X_features[clean_mask]
            X_multichannel = X_multichannel[clean_mask]
            y = y[clean_mask]
            if len(y) < 2:
                print(f"  Skipping {sid} - too few clean trials for deployment model")
                continue

        # Look up the selected band config for this subject
        selected_band_name = nested_band_configs[i]
        selected_bands = FBCSP_BAND_CANDIDATES.get(selected_band_name, FBCSP_BANDS)

        if method == "SVM":
            model = train_final_model_svm(
                X_features,
                X_multichannel,
                y,
                sfreq=sfreq,
                k_best=cfg.k_best,
                bands=selected_bands,
            )
            model_path = output_dir / f"{sid}_svm.joblib"
        elif method == "Riemann":
            model = train_final_model_riemann(X_multichannel, y, sfreq=sfreq)
            model_path = output_dir / f"{sid}_riemann.joblib"
        elif method == "Ensemble":
            # Ensemble is CV-only; save FBCSP+LDA as deployable model
            model = train_final_model(
                X_features,
                X_multichannel,
                y,
                sfreq=sfreq,
                k_best=cfg.k_best,
                bands=selected_bands,
            )
            model_path = output_dir / f"{sid}_ensemble_lda.joblib"
        else:
            model = train_final_model(
                X_features,
                X_multichannel,
                y,
                sfreq=sfreq,
                k_best=cfg.k_best,
                bands=selected_bands,
            )
            model_path = output_dir / f"{sid}_fbcsp_lda.joblib"
        joblib.dump(model, model_path)
        model_paths[sid] = str(model_path)
        print(f"  Saved {model_path} ({method})")
    runtime_seconds["train_and_save_models"] = time.perf_counter() - step_start

    above_chance, at_chance = [], []
    for sid, s in zip(subject_ids, nested_scores):
        (above_chance if s >= 0.60 else at_chance).append(sid)

    print("\n" + "=" * _LINE_WIDTH)
    print("Diagnostic Summary")
    print("=" * _LINE_WIDTH)
    print(f"Subjects with signal (>=60%): {', '.join(above_chance) or 'none'}")
    print(f"Subjects at chance  (<60%):  {', '.join(at_chance) or 'none'}")
    print(f"Best-of mean (optimistic):     {best_mean:.1%} (+/- {best_std:.1%})")
    print(f"Nested selection mean (unbiased): {nested_mean:.1%} (+/- {nested_std:.1%})")
    if holdout_results:
        ho_mean = float(np.mean(list(holdout_results.values())))
        print(f"Held-out mean:                 {ho_mean:.1%}")

    results: dict = {
        "task": "left_right_motor_imagery",
        "training_config": cfg.to_dict(),
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
        "best_methods_posthoc": {
            sid: m for sid, m in zip(subject_ids, best_methods_posthoc)
        },
        "nested_scores": {sid: float(s) for sid, s in zip(subject_ids, nested_scores)},
        "nested_methods": {
            sid: m for sid, m in zip(subject_ids, nested_methods_display)
        },
        "nested_band_configs": {
            sid: bc for sid, bc in zip(subject_ids, nested_band_configs)
        },
        "fbcsp_mean_accuracy": float(fbcsp_mean),
        "riemann_mean_accuracy": float(riemann_mean),
        "svm_mean_accuracy": float(svm_mean),
        "ensemble_mean_accuracy": float(ensemble_mean),
        "best_mean_accuracy": float(best_mean),
        "best_std_accuracy": float(best_std),
        "nested_mean_accuracy": float(nested_mean),
        "nested_std_accuracy": float(nested_std),
        "subjects_with_signal": above_chance,
        "subjects_at_chance": at_chance,
        "model_paths": model_paths,
        "runtime_seconds": runtime_seconds,
    }

    if holdout_fraction > 0:
        results["holdout_fraction"] = holdout_fraction
        results["holdout_scores"] = holdout_results

    if cross_session_results:
        results["cross_session_results"] = cross_session_results

    if transfer_results is not None:
        results["transfer_results"] = transfer_results

    if aug_nested_scores is not None:
        results["augmented_nested_scores"] = {
            sid: float(s) for sid, s in zip(subject_ids, aug_nested_scores)
        }
        results["augmented_nested_mean"] = float(aug_nested_mean)

    results_path = output_dir / "training_results.json"
    runtime_seconds["total"] = time.perf_counter() - total_start
    results["runtime_seconds"] = runtime_seconds

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    print("Runtime summary (s):")
    for stage, seconds in runtime_seconds.items():
        print(f"  {stage}: {seconds:.2f}")
    print("\n" + "=" * _LINE_WIDTH)
    print("Pipeline complete!")
    print("=" * _LINE_WIDTH)

    return results


if __name__ == "__main__":
    data_dir = Path("Data/unicorn-data")
    output_dir = Path("models")

    results = run_pipeline(data_dir, output_dir)
