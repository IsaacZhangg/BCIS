"""Training pipeline: FBCSP + LDA for left/right motor imagery classification."""

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
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
    k_best: int = 20,
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
    k_best: int = 20,
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
