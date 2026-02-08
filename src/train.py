"""Training pipeline: FBCSP + LDA and Riemannian classifiers for left/right MI."""

import mne
import numpy as np
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FBCSP_BANDS = [
    (4, 8),
    (8, 10),
    (10, 12),
    (12, 14),
    (14, 16),
    (16, 18),
    (18, 20),
    (20, 24),
    (24, 30),
    (30, 40),
]

N_CSP_COMPONENTS = 4


def _extract_fbcsp_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    sfreq: float,
    bands: list[tuple[float, float]] = FBCSP_BANDS,
    n_components: int = N_CSP_COMPONENTS,
) -> tuple[np.ndarray, np.ndarray, list[tuple[CSP, tuple[float, float]]]]:
    """Extract filter-bank CSP features for train/test splits.

    For each frequency band, bandpass-filters the data and fits CSP spatial
    filters on the training set, then transforms both train and test.

    Returns:
        Tuple of (train_features, test_features, list of (fitted_csp, band) pairs).
    """
    csp_models = []
    train_parts = []
    test_parts = []

    for low, high in bands:
        try:
            X_train_filt = mne.filter.filter_data(
                X_train, sfreq, low, high, verbose=False
            )
            X_test_filt = mne.filter.filter_data(
                X_test, sfreq, low, high, verbose=False
            )
            csp = CSP(
                n_components=n_components,
                reg="ledoit_wolf",
                log=True,
                norm_trace=True,
            )
            train_parts.append(csp.fit_transform(X_train_filt, y_train))
            test_parts.append(csp.transform(X_test_filt))
            csp_models.append((csp, (low, high)))
        except (ValueError, np.linalg.LinAlgError):
            continue

    if not train_parts:
        return np.empty((X_train.shape[0], 0)), np.empty((X_test.shape[0], 0)), []

    return np.hstack(train_parts), np.hstack(test_parts), csp_models


def _extract_fbcsp_features_pretrained(
    X: np.ndarray,
    sfreq: float,
    csp_models: list[tuple[CSP, tuple[float, float]]],
) -> np.ndarray:
    """Apply pre-fitted CSP models to new data at inference time.

    Returns:
        Feature array of shape (n_trials, n_csp_components * n_bands).
    """
    parts = []
    for csp, (low, high) in csp_models:
        X_filt = mne.filter.filter_data(X, sfreq, low, high, verbose=False)
        parts.append(csp.transform(X_filt))

    if not parts:
        return np.empty((X.shape[0], 0))

    return np.hstack(parts)


def train_within_subject_cv(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    sfreq: float = 250.0,
    n_folds: int = 10,
    k_best: int = 15,
) -> tuple[list[float], float, float]:
    """Within-subject stratified k-fold CV using FBCSP + LDA.

    Per fold:
    1. FBCSP: 10 frequency bands -> CSP (4 components each) -> log-variance
    2. Concatenate FBCSP features (up to 40) with handcrafted features (39) = ~79
    3. SelectKBest(f_classif, k=k_best)
    4. StandardScaler
    5. LDA(solver='lsqr', shrinkage='auto')

    Args:
        X_by_subject: List of (handcrafted_features, multichannel_eeg) per subject.
        y_by_subject: List of label arrays per subject.
        sfreq: Sampling frequency in Hz.
        n_folds: Number of CV folds.
        k_best: Number of features to select via SelectKBest.

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy).
    """
    scores = []

    for (X_features, X_multichannel), y in zip(X_by_subject, y_by_subject):
        n_samples = len(y)
        actual_folds = min(n_folds, n_samples // 2)
        if actual_folds < 2:
            actual_folds = n_samples

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, test_idx in skf.split(X_features, y):
            y_train, y_test = y[train_idx], y[test_idx]

            fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features(
                X_multichannel[train_idx], X_multichannel[test_idx], y_train, sfreq
            )

            X_train_combined = np.hstack([fbcsp_train, X_features[train_idx]])
            X_test_combined = np.hstack([fbcsp_test, X_features[test_idx]])

            k = min(k_best, X_train_combined.shape[1])
            selector = SelectKBest(f_classif, k=k)
            X_train_sel = selector.fit_transform(X_train_combined, y_train)
            X_test_sel = selector.transform(X_test_combined)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
            lda.fit(X_train_scaled, y_train)
            fold_scores.append(lda.score(X_test_scaled, y_test))

        scores.append(float(np.mean(fold_scores)))

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return scores, mean_acc, std_acc


def train_final_model(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float = 250.0,
    k_best: int = 15,
) -> dict:
    """Train a deployable FBCSP + LDA model on all data for a single subject.

    Returns:
        Dict with keys: csp_models, selector, scaler, classifier, sfreq, k_best.
    """
    fbcsp_features, _, csp_models = _extract_fbcsp_features(
        X_multichannel, X_multichannel, y, sfreq
    )

    X_combined = np.hstack([fbcsp_features, X_features])

    k = min(k_best, X_combined.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X_combined, y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    classifier.fit(X_scaled, y)

    return {
        "csp_models": csp_models,
        "selector": selector,
        "scaler": scaler,
        "classifier": classifier,
        "sfreq": sfreq,
        "k_best": k,
    }


def predict(
    model: dict,
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
) -> np.ndarray:
    """Run inference with a saved FBCSP + LDA model.

    Args:
        model: Dict returned by train_final_model.
        X_features: Handcrafted features of shape (n_trials, n_features).
        X_multichannel: Multichannel EEG of shape (n_trials, n_channels, n_samples).

    Returns:
        Predicted class labels.
    """
    fbcsp = _extract_fbcsp_features_pretrained(
        X_multichannel, model["sfreq"], model["csp_models"]
    )
    X_combined = np.hstack([fbcsp, X_features])
    X_selected = model["selector"].transform(X_combined)
    X_scaled = model["scaler"].transform(X_selected)
    return model["classifier"].predict(X_scaled)


# ---------------------------------------------------------------------------
# Riemannian geometry classifier (Covariances → TangentSpace → LogisticRegression)
# ---------------------------------------------------------------------------


def train_within_subject_cv_riemann(
    X_by_subject: list[np.ndarray],
    y_by_subject: list[np.ndarray],
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """Within-subject stratified k-fold CV using Riemannian tangent-space classifier.

    Per fold:
    1. Covariances(estimator='oas') — shrinkage covariance estimation
    2. TangentSpace(metric='riemann') — project SPD matrices to tangent space
    3. LogisticRegression(C=1.0, solver='lbfgs') — classify

    Args:
        X_by_subject: List of multichannel EEG arrays (n_trials, n_channels, n_samples).
        y_by_subject: List of label arrays per subject.
        n_folds: Number of CV folds.

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy).
    """
    scores = []

    for X_multichannel, y in zip(X_by_subject, y_by_subject):
        n_samples = len(y)
        actual_folds = min(n_folds, n_samples // 2)
        if actual_folds < 2:
            actual_folds = n_samples

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, test_idx in skf.split(X_multichannel, y):
            pipe = make_pipeline(
                Covariances(estimator="oas"),
                TangentSpace(metric="riemann"),
                LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
            )
            pipe.fit(X_multichannel[train_idx], y[train_idx])
            fold_scores.append(pipe.score(X_multichannel[test_idx], y[test_idx]))

        scores.append(float(np.mean(fold_scores)))

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return scores, mean_acc, std_acc


def train_final_model_riemann(
    X_multichannel: np.ndarray,
    y: np.ndarray,
) -> dict:
    """Train a deployable Riemannian model on all data for a single subject.

    Returns:
        Dict with key 'pipeline' containing the fitted sklearn Pipeline.
    """
    pipe = make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )
    pipe.fit(X_multichannel, y)
    return {"pipeline": pipe}


# ---------------------------------------------------------------------------
# Transfer learning: Riemannian domain adaptation across subjects
# ---------------------------------------------------------------------------


def train_transfer_cv(
    X_by_subject: list[np.ndarray],
    y_by_subject: list[np.ndarray],
    subject_ids: list[str],
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """Leave-one-subject-out CV with Riemannian domain adaptation.

    For each target subject, all other subjects serve as source data.
    Covariances are computed, domains are encoded, and TLCenter re-centers
    each subject's covariances to identity in Riemannian space before
    projecting to tangent space and classifying with Logistic Regression.

    Within each target subject, k-fold CV is applied (source data is always
    fully included in the training set).

    Args:
        X_by_subject: List of multichannel EEG arrays (n_trials, n_channels, n_samples).
        y_by_subject: List of label arrays per subject.
        subject_ids: List of subject identifier strings.
        n_folds: Number of CV folds within each target subject.

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy).
    """
    from pyriemann.transfer import TLCenter, encode_domains

    # Pre-compute covariances per subject
    cov_estimator = Covariances(estimator="oas")
    covs_by_subject = [cov_estimator.fit_transform(X) for X in X_by_subject]

    scores = []

    for target_idx in range(len(subject_ids)):
        target_covs = covs_by_subject[target_idx]
        target_y = y_by_subject[target_idx]
        n_target = len(target_y)

        # Pool source subjects
        source_covs = np.concatenate(
            [covs_by_subject[j] for j in range(len(subject_ids)) if j != target_idx]
        )
        source_y = np.concatenate(
            [y_by_subject[j] for j in range(len(subject_ids)) if j != target_idx]
        )

        actual_folds = min(n_folds, n_target // 2)
        if actual_folds < 2:
            actual_folds = n_target

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, test_idx in skf.split(target_covs, target_y):
            # Combine source + target train
            train_covs = np.concatenate([source_covs, target_covs[train_idx]])
            train_y = np.concatenate([source_y, target_y[train_idx]])

            # Domain labels
            source_domain = np.array(["source"] * len(source_covs))
            target_train_domain = np.array(["target"] * len(train_idx))
            train_domain = np.concatenate([source_domain, target_train_domain])

            test_covs = target_covs[test_idx]
            test_domain = np.array(["target"] * len(test_idx))
            test_y = target_y[test_idx]

            # Encode domains
            X_train_enc, y_train_enc = encode_domains(train_covs, train_y, train_domain)
            X_test_enc, _y_test_enc = encode_domains(test_covs, test_y, test_domain)

            # Re-center to target domain
            tlc = TLCenter(target_domain="target")
            X_train_centered = tlc.fit_transform(X_train_enc, y_train_enc)
            X_test_centered = tlc.transform(X_test_enc)

            # Tangent space projection
            ts = TangentSpace(metric="riemann")
            X_train_ts = ts.fit_transform(X_train_centered)
            X_test_ts = ts.transform(X_test_centered)

            # Classify with original numeric labels (not encoded domain/label strings)
            clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            clf.fit(X_train_ts, train_y)
            fold_scores.append(clf.score(X_test_ts, test_y))

        scores.append(float(np.mean(fold_scores)))

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return scores, mean_acc, std_acc


def train_final_model_transfer(
    X_all: list[np.ndarray],
    y_all: list[np.ndarray],
    subject_ids: list[str],
    target_idx: int,
) -> dict:
    """Train a deployable transfer-learning Riemannian model for a target subject.

    Uses all subjects' data (source + target) to fit the pipeline.

    Returns:
        Dict with keys: cov_estimator, tlc, ts, classifier.
    """
    from pyriemann.transfer import TLCenter, encode_domains

    cov_estimator = Covariances(estimator="oas")
    covs_list = [cov_estimator.fit_transform(X) for X in X_all]

    all_covs = np.concatenate(covs_list)
    all_y = np.concatenate(y_all)
    all_domain = np.concatenate(
        [
            np.array(["target" if i == target_idx else "source"] * len(y_all[i]))
            for i in range(len(subject_ids))
        ]
    )

    X_enc, y_enc = encode_domains(all_covs, all_y, all_domain)

    tlc = TLCenter(target_domain="target")
    X_centered = tlc.fit_transform(X_enc, y_enc)

    ts = TangentSpace(metric="riemann")
    X_ts = ts.fit_transform(X_centered)

    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    clf.fit(X_ts, all_y)

    return {
        "cov_estimator": cov_estimator,
        "tlc": tlc,
        "ts": ts,
        "classifier": clf,
    }


def predict_transfer(
    model: dict,
    X_multichannel: np.ndarray,
) -> np.ndarray:
    """Run inference with a saved transfer-learning model.

    Args:
        model: Dict returned by train_final_model_transfer.
        X_multichannel: Multichannel EEG of shape (n_trials, n_channels, n_samples).

    Returns:
        Predicted class labels.
    """
    from pyriemann.transfer import encode_domains

    covs = model["cov_estimator"].transform(X_multichannel)
    # Encode as target domain for centering
    dummy_y = np.zeros(len(covs))
    domain = np.array(["target"] * len(covs))
    X_enc, _y_enc = encode_domains(covs, dummy_y, domain)

    X_centered = model["tlc"].transform(X_enc)
    X_ts = model["ts"].transform(X_centered)
    return model["classifier"].predict(X_ts)


def predict_riemann(
    model: dict,
    X_multichannel: np.ndarray,
) -> np.ndarray:
    """Run inference with a saved Riemannian model.

    Args:
        model: Dict returned by train_final_model_riemann.
        X_multichannel: Multichannel EEG of shape (n_trials, n_channels, n_samples).

    Returns:
        Predicted class labels.
    """
    return model["pipeline"].predict(X_multichannel)
