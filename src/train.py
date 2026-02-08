"""Training pipeline: FBCSP + LDA, FBCSP + SVM, and Riemannian classifiers for left/right MI."""

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
from sklearn.svm import SVC

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
    k_best: int = 10,
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

            # Class priors from training fold
            classes, counts = np.unique(y_train, return_counts=True)
            priors = counts / counts.sum()

            # Nested CV to select k_best
            best_k = min(k_best, X_train_combined.shape[1])
            best_inner_score = -1.0
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            for k_candidate in [5, 8, 10, 15, 20]:
                k_actual = min(k_candidate, X_train_combined.shape[1])
                inner_scores = []
                for inner_train, inner_val in inner_cv.split(X_train_combined, y_train):
                    sel = SelectKBest(f_classif, k=k_actual)
                    X_it = sel.fit_transform(
                        X_train_combined[inner_train], y_train[inner_train]
                    )
                    X_iv = sel.transform(X_train_combined[inner_val])
                    sc = StandardScaler()
                    X_it = sc.fit_transform(X_it)
                    X_iv = sc.transform(X_iv)
                    clf = LinearDiscriminantAnalysis(
                        solver="lsqr", shrinkage="auto", priors=priors
                    )
                    clf.fit(X_it, y_train[inner_train])
                    inner_scores.append(clf.score(X_iv, y_train[inner_val]))
                mean_inner = float(np.mean(inner_scores))
                if mean_inner > best_inner_score:
                    best_inner_score = mean_inner
                    best_k = k_actual

            selector = SelectKBest(f_classif, k=best_k)
            X_train_sel = selector.fit_transform(X_train_combined, y_train)
            X_test_sel = selector.transform(X_test_combined)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            lda = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=priors
            )
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
    k_best: int = 10,
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
    sfreq: float = 250.0,
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """Within-subject stratified k-fold CV using Riemannian tangent-space classifier.

    Per fold:
    1. Bandpass filter 8-30Hz (mu/beta bands)
    2. Covariances(estimator='oas') — shrinkage covariance estimation
    3. TangentSpace(metric='riemann') — project SPD matrices to tangent space
    4. LogisticRegression(C=1.0, solver='lbfgs') — classify

    Args:
        X_by_subject: List of multichannel EEG arrays (n_trials, n_channels, n_samples).
        y_by_subject: List of label arrays per subject.
        sfreq: Sampling frequency in Hz.
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
    sfreq: float = 250.0,
) -> dict:
    """Train a deployable Riemannian model on all data for a single subject.

    Returns:
        Dict with keys 'pipeline', 'sfreq'.
    """
    pipe = make_pipeline(
        Covariances(estimator="oas"),
        TangentSpace(metric="riemann"),
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000),
    )
    pipe.fit(X_multichannel, y)
    return {"pipeline": pipe, "sfreq": sfreq}


# ---------------------------------------------------------------------------
# FBCSP + SVM classifier
# ---------------------------------------------------------------------------


def train_within_subject_cv_svm(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    sfreq: float = 250.0,
    n_folds: int = 10,
    k_best: int = 10,
) -> tuple[list[float], float, float]:
    """Within-subject stratified k-fold CV using FBCSP + SVM.

    Per fold:
    1. FBCSP: 10 frequency bands -> CSP (4 components each) -> log-variance
    2. Concatenate FBCSP features (up to 40) with handcrafted features
    3. SelectKBest(f_classif, k=k_best)
    4. StandardScaler
    5. SVC(kernel='rbf', C=1.0, gamma='scale')

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

            svm = SVC(kernel="rbf", C=1.0, gamma="scale")
            svm.fit(X_train_scaled, y_train)
            fold_scores.append(svm.score(X_test_scaled, y_test))

        scores.append(float(np.mean(fold_scores)))

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return scores, mean_acc, std_acc


def train_final_model_svm(
    X_features: np.ndarray,
    X_multichannel: np.ndarray,
    y: np.ndarray,
    sfreq: float = 250.0,
    k_best: int = 10,
) -> dict:
    """Train a deployable FBCSP + SVM model on all data for a single subject.

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

    classifier = SVC(kernel="rbf", C=1.0, gamma="scale")
    classifier.fit(X_scaled, y)

    return {
        "csp_models": csp_models,
        "selector": selector,
        "scaler": scaler,
        "classifier": classifier,
        "sfreq": sfreq,
        "k_best": k,
    }


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


# ---------------------------------------------------------------------------
# Soft-voting ensemble (FBCSP+LDA + Riemannian + FBCSP+SVM)
# ---------------------------------------------------------------------------


def train_within_subject_cv_ensemble(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    sfreq: float = 250.0,
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """Within-subject CV using soft-voting ensemble of FBCSP+LDA and FBCSP+SVM.

    Per fold, trains both FBCSP classifiers on the same features, averages
    their predicted probabilities, and picks the argmax class.  Uses only
    LDA + SVM since they provide genuine diversity (linear vs nonlinear)
    on the same FBCSP feature space.

    Args:
        X_by_subject: List of (handcrafted_features, multichannel_eeg) per subject.
        y_by_subject: List of label arrays per subject.
        sfreq: Sampling frequency in Hz.
        n_folds: Number of CV folds.

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

            # --- Shared FBCSP features ---
            fbcsp_train, fbcsp_test, _ = _extract_fbcsp_features(
                X_multichannel[train_idx], X_multichannel[test_idx], y_train, sfreq
            )
            X_train_combined = np.hstack([fbcsp_train, X_features[train_idx]])
            X_test_combined = np.hstack([fbcsp_test, X_features[test_idx]])

            k = min(10, X_train_combined.shape[1])
            selector = SelectKBest(f_classif, k=k)
            X_train_sel = selector.fit_transform(X_train_combined, y_train)
            X_test_sel = selector.transform(X_test_combined)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)

            # --- Classifier 1: LDA ---
            classes, counts = np.unique(y_train, return_counts=True)
            priors = counts / counts.sum()
            lda = LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=priors
            )
            lda.fit(X_train_scaled, y_train)
            proba_lda = lda.predict_proba(X_test_scaled)

            # --- Classifier 2: SVM ---
            svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True)
            svm.fit(X_train_scaled, y_train)
            proba_svm = svm.predict_proba(X_test_scaled)

            # --- Soft voting (LDA + SVM) ---
            avg_proba = (proba_lda + proba_svm) / 2
            preds = np.argmax(avg_proba, axis=1)

            # Map predictions back to original class labels
            classes_sorted = lda.classes_
            preds_labels = classes_sorted[preds]

            fold_scores.append(float(np.mean(preds_labels == y_test)))

        scores.append(float(np.mean(fold_scores)))

    mean_acc = float(np.mean(scores))
    std_acc = float(np.std(scores))
    return scores, mean_acc, std_acc
