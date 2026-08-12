"""Extract per-fold CV accuracies per subject per classifier.

Reproduces the 10-fold within-subject CV from src/train.py
(_evaluate_subject_all_models), but records each fold's score
instead of returning only the subject mean. Writes a JSON
report to .context/failure_fold_scores.json for downstream stats.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import TrainingConfig

MI_DATA_DIR = Path("data/unicorn-data")
from src.data_loader import CHANNELS, get_complete_recordings, load_recording
from src.epochs import (
    compute_trial_max_ptp,
    extract_left_right_epochs,
    task_epochs,
)
from src.features import extract_lateralization_features
from src.preprocess import preprocess_multichannel_eeg
from src.train import (
    _extract_fbcsp_features_prefiltered,
    _make_riemann_pipeline,
    _precompute_bandpassed,
    _reject_in_fold,
    _safe_k_values,
    _subset_band_cache,
)
from src.validation import build_classwise_trial_groups, make_cv_splits

OUT_PATH = Path(".context/failure_fold_scores.json")


def _pairs_to_arrays(rec_path: Path, sfreq: float):
    data, events, rec_sfreq = load_recording(rec_path)
    if abs(rec_sfreq - sfreq) > 1e-6:
        raise ValueError(f"sfreq mismatch: {rec_path}")
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

    n_left = len(left_pairs_by_channel[CHANNELS[0]])
    n_right = len(right_pairs_by_channel[CHANNELS[0]])

    left_feat = extract_lateralization_features(left_pairs_by_channel, sfreq)
    right_feat = extract_lateralization_features(right_pairs_by_channel, sfreq)
    X_mc = np.vstack(
        [
            task_epochs(left_pairs_by_channel, CHANNELS),
            task_epochs(right_pairs_by_channel, CHANNELS),
        ]
    )
    X_feat = np.vstack([left_feat, right_feat])
    y = np.array([0] * n_left + [1] * n_right)
    return X_feat, X_mc, y


def _evaluate_subject_per_fold(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    cfg: TrainingConfig,
) -> dict:
    groups = build_classwise_trial_groups(y, group_size=cfg.trial_group_size)
    trial_ptps = compute_trial_max_ptp(X_multichannel)

    splits = make_cv_splits(
        y,
        n_splits=cfg.n_folds,
        strategy=cfg.split_strategy,
        groups=groups,
        random_state=cfg.random_state,
    )

    subject_band_cache = _precompute_bandpassed(X_multichannel, sfreq)

    fold_records = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        rec = {
            "fold": fold_idx,
            "n_train_raw": int(len(train_idx)),
            "n_test_raw": int(len(test_idx)),
        }
        train_idx, test_idx = _reject_in_fold(trial_ptps, train_idx, test_idx)
        rec["n_train_clean"] = int(len(train_idx))
        rec["n_test_clean"] = int(len(test_idx))
        if len(train_idx) < 2 or len(test_idx) < 1:
            rec["skipped"] = True
            fold_records.append(rec)
            continue

        X_feat_tr = X_features[train_idx]
        X_feat_te = X_features[test_idx]
        X_mc_tr = X_multichannel[train_idx]
        X_mc_te = X_multichannel[test_idx]
        y_tr = y[train_idx]
        y_te = y[test_idx]
        rec["n_test_left"] = int(np.sum(y_te == 0))
        rec["n_test_right"] = int(np.sum(y_te == 1))

        # Riemann
        riemann_pipe = _make_riemann_pipeline()
        riemann_pipe.fit(X_mc_tr, y_tr)
        rec["riemann_acc"] = float(riemann_pipe.score(X_mc_te, y_te))
        preds_r = riemann_pipe.predict(X_mc_te)
        rec["riemann_correct"] = [int(p) for p in (preds_r == y_te).astype(int)]

        # FBCSP
        train_band_cache = _subset_band_cache(subject_band_cache, train_idx)
        test_band_cache = _subset_band_cache(subject_band_cache, test_idx)
        fbcsp_tr, fbcsp_te, _ = _extract_fbcsp_features_prefiltered(
            train_band_cache,
            test_band_cache,
            y_tr,
            n_train_trials=len(train_idx),
            n_test_trials=len(test_idx),
        )
        X_tr_comb = np.hstack([fbcsp_tr, X_feat_tr])
        X_te_comb = np.hstack([fbcsp_te, X_feat_te])

        k_values = _safe_k_values(
            n_features=X_tr_comb.shape[1],
            n_train_trials=len(y_tr),
            k_best=cfg.k_best,
            k_candidates=tuple(cfg.k_candidates),
        )
        if not k_values:
            rec["skipped"] = True
            fold_records.append(rec)
            continue

        _, outer_counts = np.unique(y_tr, return_counts=True)
        outer_priors = outer_counts / outer_counts.sum()

        # LDA inner-CV k selection (reproduces train.py logic)
        best_k = k_values[0]
        best_inner_score = -1.0
        inner_groups = groups[train_idx]
        inner_splits = make_cv_splits(
            y_tr,
            n_splits=3,
            strategy=cfg.split_strategy,
            groups=inner_groups,
            random_state=cfg.random_state,
        )
        for k_actual in k_values:
            if not inner_splits:
                break
            inner_scores = []
            for inner_train, inner_val in inner_splits:
                sel = SelectKBest(f_classif, k=k_actual)
                X_it = sel.fit_transform(X_tr_comb[inner_train], y_tr[inner_train])
                X_iv = sel.transform(X_tr_comb[inner_val])
                sc = StandardScaler()
                X_it = sc.fit_transform(X_it)
                X_iv = sc.transform(X_iv)
                _, inner_counts = np.unique(y_tr[inner_train], return_counts=True)
                inner_priors = inner_counts / inner_counts.sum()
                clf = LinearDiscriminantAnalysis(
                    solver="lsqr", shrinkage="auto", priors=inner_priors
                )
                clf.fit(X_it, y_tr[inner_train])
                inner_scores.append(float(clf.score(X_iv, y_tr[inner_val])))
            mean_inner = float(np.mean(inner_scores))
            if mean_inner > best_inner_score:
                best_inner_score = mean_inner
                best_k = k_actual
        rec["best_k_lda"] = int(best_k)

        lda_selector = SelectKBest(f_classif, k=best_k)
        X_train_lda = lda_selector.fit_transform(X_tr_comb, y_tr)
        X_test_lda = lda_selector.transform(X_te_comb)
        lda_scaler = StandardScaler()
        X_train_lda = lda_scaler.fit_transform(X_train_lda)
        X_test_lda = lda_scaler.transform(X_test_lda)
        lda = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=outer_priors
        )
        lda.fit(X_train_lda, y_tr)
        rec["lda_acc"] = float(lda.score(X_test_lda, y_te))
        rec["lda_correct"] = [
            int(p) for p in (lda.predict(X_test_lda) == y_te).astype(int)
        ]

        # SVM (k_values[-1], same as train.py)
        k_fixed = k_values[-1]
        selector = SelectKBest(f_classif, k=k_fixed)
        X_tr_sel = selector.fit_transform(X_tr_comb, y_tr)
        X_te_sel = selector.transform(X_te_comb)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_sel)
        X_te_scaled = scaler.transform(X_te_sel)

        svm = SVC(
            kernel="rbf",
            C=20.0,
            gamma="scale",
            probability=True,
            random_state=42,
        )
        svm.fit(X_tr_scaled, y_tr)
        rec["svm_acc"] = float(svm.score(X_te_scaled, y_te))
        rec["svm_correct"] = [
            int(p) for p in (svm.predict(X_te_scaled) == y_te).astype(int)
        ]

        lda_ens = LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=outer_priors
        )
        lda_ens.fit(X_tr_scaled, y_tr)
        proba_lda = lda_ens.predict_proba(X_te_scaled)
        proba_svm = svm.predict_proba(X_te_scaled)
        avg = (proba_lda + proba_svm) / 2
        preds_e = lda_ens.classes_[np.argmax(avg, axis=1)]
        rec["ensemble_acc"] = float(np.mean(preds_e == y_te))
        rec["ensemble_correct"] = [int(p) for p in (preds_e == y_te).astype(int)]
        rec["y_test"] = [int(v) for v in y_te]

        fold_records.append(rec)

    return {"folds": fold_records}


def main() -> None:
    cfg = TrainingConfig()
    sfreq = cfg.sfreq
    data_dir = MI_DATA_DIR

    recordings = get_complete_recordings(data_dir)
    print(f"Found {len(recordings)} complete recordings")

    report: dict = {
        "config": {
            "n_folds": cfg.n_folds,
            "trial_group_size": cfg.trial_group_size,
            "split_strategy": cfg.split_strategy,
            "random_state": cfg.random_state,
            "k_best": cfg.k_best,
            "k_candidates": list(cfg.k_candidates),
        },
        "subjects": {},
    }

    for rec_path in recordings:
        subject_id = rec_path.parent.parent.name
        t0 = time.perf_counter()
        X_feat, X_mc, y = _pairs_to_arrays(rec_path, sfreq)
        print(
            f"{subject_id}: shape X_mc={X_mc.shape} y={Counter(y.tolist())}", flush=True
        )
        subj = _evaluate_subject_per_fold(X_feat, X_mc, y, sfreq, cfg)
        subj["n_trials"] = int(len(y))
        subj["n_left"] = int(np.sum(y == 0))
        subj["n_right"] = int(np.sum(y == 1))
        subj["runtime_s"] = float(time.perf_counter() - t0)
        report["subjects"][subject_id] = subj
        print(
            f"  {subject_id} done in {subj['runtime_s']:.1f}s, n_folds={len(subj['folds'])}",
            flush=True,
        )

        # Incremental save after each subject
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2))

    print(f"Wrote {OUT_PATH}")


from collections import Counter  # noqa: E402

if __name__ == "__main__":
    main()
