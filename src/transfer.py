"""Leakage-free donor augmentation for weak subjects (EA-gated nested CV)."""

from __future__ import annotations

from collections import Counter

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_covariance

from src.alignment import (
    align_trials_ea,
    apply_ea_to_trials,
    compute_trial_ea_transform,
)
from src.train import (
    ALL_CLASSIFIERS,
    _evaluate_classifier,
    _evaluate_classifiers_batch,
    _reject_in_fold,
)
from src.validation import adaptive_fold_count, make_cv_splits


def _precompute_ea_aligned_donors(
    donor_subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Whiten each donor independently so pooled CSP sees a shared identity mean."""

    aligned: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid, (feat, mc, y) in donor_subjects.items():
        mc_ea, _ = align_trials_ea(mc)
        aligned[sid] = (feat, mc_ea, y)
    return aligned


def gated_ea_augmented_nested_cv(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    donor_subjects: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    donor_means: dict[str, np.ndarray],
    target_sid: str,
    sfreq: float,
    n_outer_folds: int,
    n_inner_folds: int,
    k_best: int,
    trial_ptps: np.ndarray | None,
    split_strategy: str,
    groups: np.ndarray | None,
    random_state: int,
    signal_donor_ids: set[str] | None = None,
    pool_k: int = 3,
) -> tuple[float, str, float]:
    """Nested CV with leakage-free EA donor pooling and an inner-CV gate.

    Per outer fold, Euclidean Alignment is fit on the *target training trials
    only* and applied to the target test fold. Donors are whitened with their
    own transforms (independent of the target). Inner CV chooses among:

    - ``none``: target training data only (no EA)
    - ``nearest_ea``: target + closest donor, both EA-whitened
    - ``pool_ea``: target + up to ``pool_k`` nearest *signal* donors, EA-whitened

    Ties prefer ``none`` so negative transfer cannot inflate the score. Test
    trials never enter donor selection, EA fitting, or inner model selection.
    """
    classifier_names = list(ALL_CLASSIFIERS)
    n_outer_folds = adaptive_fold_count(len(y), n_outer_folds)
    outer_splits = make_cv_splits(
        y,
        n_splits=n_outer_folds,
        strategy=split_strategy,
        groups=groups,
        random_state=random_state,
    )

    donor_ea = _precompute_ea_aligned_donors(
        {sid: data for sid, data in donor_subjects.items() if sid != target_sid}
    )
    if signal_donor_ids is None:
        signal_ids = set(donor_ea)
    else:
        signal_ids = {sid for sid in signal_donor_ids if sid in donor_ea}

    fold_scores: list[float] = []
    fold_strategies: list[str] = []

    for outer_train_idx, outer_test_idx in outer_splits:
        outer_train_idx, outer_test_idx = _reject_in_fold(
            trial_ptps, outer_train_idx, outer_test_idx
        )
        if len(outer_train_idx) < 8 or len(outer_test_idx) < 1:
            continue

        y_otrain_target = y[outer_train_idx]
        y_otest = y[outer_test_idx]
        feat_otrain_target = X_features[outer_train_idx]
        feat_otest = X_features[outer_test_idx]
        mc_otrain_raw = X_multichannel[outer_train_idx]
        mc_otest_raw = X_multichannel[outer_test_idx]

        try:
            target_ea = compute_trial_ea_transform(mc_otrain_raw)
            mc_otrain_ea = apply_ea_to_trials(mc_otrain_raw, target_ea)
            mc_otest_ea = apply_ea_to_trials(mc_otest_raw, target_ea)
        except (ValueError, np.linalg.LinAlgError):
            target_ea = None
            mc_otrain_ea = mc_otrain_raw
            mc_otest_ea = mc_otest_raw

        train_mean = mean_covariance(
            Covariances(estimator="lwf").fit_transform(mc_otrain_raw),
            metric="riemann",
        )
        dists = {
            dsid: float(distance_riemann(train_mean, donor_means[dsid]))
            for dsid in donor_ea
            if dsid in donor_means
        }
        ranked = sorted(dists, key=dists.get)

        strategies: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
            "none": (feat_otrain_target, mc_otrain_raw, y_otrain_target, mc_otest_raw),
        }
        if ranked and target_ea is not None:
            nearest = ranked[0]
            d_feat, d_mc, d_y = donor_ea[nearest]
            strategies["nearest_ea"] = (
                np.vstack([feat_otrain_target, d_feat]),
                np.vstack([mc_otrain_ea, d_mc]),
                np.concatenate([y_otrain_target, d_y]),
                mc_otest_ea,
            )
            pool_ids = [sid for sid in ranked if sid in signal_ids][:pool_k]
            if len(pool_ids) >= 2:
                pool_feat = np.vstack(
                    [feat_otrain_target, *[donor_ea[s][0] for s in pool_ids]]
                )
                pool_mc = np.vstack([mc_otrain_ea, *[donor_ea[s][1] for s in pool_ids]])
                pool_y = np.concatenate(
                    [y_otrain_target, *[donor_ea[s][2] for s in pool_ids]]
                )
                strategies["pool_ea"] = (pool_feat, pool_mc, pool_y, mc_otest_ea)

        n_inner_actual = adaptive_fold_count(len(outer_train_idx), n_inner_folds)
        inner_groups = groups[outer_train_idx] if groups is not None else None
        inner_splits = make_cv_splits(
            y[outer_train_idx],
            n_splits=n_inner_actual,
            strategy=split_strategy,
            groups=inner_groups,
            random_state=random_state,
        )

        best_strategy = "none"
        best_inner = -1.0
        best_clf = "lda"
        strategy_order = ("none", "nearest_ea", "pool_ea")

        for strategy in strategy_order:
            if strategy not in strategies:
                continue
            feat_tr, mc_tr, y_tr_full, _mc_te = strategies[strategy]
            n_target_train = len(outer_train_idx)
            n_donor = len(y_tr_full) - n_target_train

            inner_scores_by_clf: dict[str, list[float]] = {
                n: [] for n in classifier_names
            }
            for inner_train_idx, inner_val_idx in inner_splits:
                if strategy == "none":
                    inner_feat_train = X_features[outer_train_idx[inner_train_idx]]
                    inner_mc_train = X_multichannel[outer_train_idx[inner_train_idx]]
                    inner_y_train = y[outer_train_idx[inner_train_idx]]
                    inner_feat_val = X_features[outer_train_idx[inner_val_idx]]
                    inner_mc_val = X_multichannel[outer_train_idx[inner_val_idx]]
                    inner_y_val = y[outer_train_idx[inner_val_idx]]
                else:
                    donor_slice = slice(n_target_train, n_target_train + n_donor)
                    inner_feat_train = np.vstack(
                        [feat_tr[inner_train_idx], feat_tr[donor_slice]]
                    )
                    inner_mc_train = np.vstack(
                        [mc_tr[inner_train_idx], mc_tr[donor_slice]]
                    )
                    inner_y_train = np.concatenate(
                        [y_tr_full[inner_train_idx], y_tr_full[donor_slice]]
                    )
                    inner_feat_val = feat_otrain_target[inner_val_idx]
                    inner_mc_val = mc_otrain_ea[inner_val_idx]
                    inner_y_val = y_otrain_target[inner_val_idx]

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
            clf = max(inner_means, key=inner_means.get)
            mean_score = inner_means[clf]
            # Donor strategies must beat target-only by a margin so inner-CV
            # noise cannot select negative transfer.
            margin = 0.0 if strategy == "none" else 0.02
            if mean_score > best_inner + margin:
                best_inner = mean_score
                best_strategy = strategy
                best_clf = clf

        feat_tr, mc_tr, y_tr_full, mc_te = strategies[best_strategy]
        outer_score = _evaluate_classifier(
            best_clf,
            feat_tr,
            feat_otest,
            y_tr_full,
            y_otest,
            mc_tr,
            mc_te,
            sfreq,
            k_best,
        )
        fold_scores.append(outer_score)
        fold_strategies.append(best_strategy)

    accuracy = float(np.mean(fold_scores)) if fold_scores else 0.5
    strategy_counts = Counter(fold_strategies)
    most_common = strategy_counts.most_common(1)[0][0] if strategy_counts else "none"
    return accuracy, most_common, accuracy
