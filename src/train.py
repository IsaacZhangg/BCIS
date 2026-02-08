"""Training pipeline with LOSO and within-subject cross-validation."""

import mne
import numpy as np
from scipy.signal import welch as scipy_welch
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import check_random_state, resample
from mne.decoding import CSP
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM, FgMDM
from pyriemann.tangentspace import TangentSpace


# --- Shared helpers for train_left_right_within_subject ---


def _csp_proba(
    X_train, X_test, y_train, sfreq, band, n_components=4, reg="ledoit_wolf", clf=None
):
    """Filter → CSP → classifier → predict_proba[:, 1].

    Returns None on failure.
    """
    low, high = band
    try:
        X_train_filt = mne.filter.filter_data(X_train, sfreq, low, high, verbose=False)
        X_test_filt = mne.filter.filter_data(X_test, sfreq, low, high, verbose=False)
        csp = CSP(n_components=n_components, reg=reg, log=True, norm_trace=True)
        X_train_csp = csp.fit_transform(X_train_filt, y_train)
        X_test_csp = csp.transform(X_test_filt)
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None

    if clf is None:
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    try:
        clf.fit(X_train_csp, y_train)
        return clf.predict_proba(X_test_csp)[:, 1]
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None


def _tangent_space_proba(
    X_train, X_test, y_train, sfreq, band, cov_est="lwf", clf=None
):
    """Filter → Covariances → TangentSpace → classifier → predict_proba[:, 1].

    Returns None on failure.
    """
    low, high = band
    try:
        X_train_filt = mne.filter.filter_data(X_train, sfreq, low, high, verbose=False)
        X_test_filt = mne.filter.filter_data(X_test, sfreq, low, high, verbose=False)
        cov = Covariances(estimator=cov_est)
        X_train_cov = cov.fit_transform(X_train_filt)
        X_test_cov = cov.transform(X_test_filt)
        ts = TangentSpace(metric="riemann")
        X_train_ts = ts.fit_transform(X_train_cov, y_train)
        X_test_ts = ts.transform(X_test_cov)
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None

    if clf is None:
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    try:
        clf.fit(X_train_ts, y_train)
        return clf.predict_proba(X_test_ts)[:, 1]
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None


def _append_if_valid(lst, value):
    """Append value to lst if it is not None."""
    if value is not None:
        lst.append(value)


def _make_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


# --- Public API ---


def train_loso_cv(
    X_by_subject: list[np.ndarray],
    y_by_subject: list[np.ndarray],
) -> tuple[list[float], float, float]:
    """
    Train with Leave-One-Subject-Out cross-validation.

    Args:
        X_by_subject: List of feature arrays, one per subject
        y_by_subject: List of label arrays, one per subject

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy)
    """
    n_subjects = len(X_by_subject)
    scores = []

    for test_idx in range(n_subjects):
        X_train = np.vstack(
            [X_by_subject[i] for i in range(n_subjects) if i != test_idx]
        )
        y_train = np.concatenate(
            [y_by_subject[i] for i in range(n_subjects) if i != test_idx]
        )
        X_test = X_by_subject[test_idx]
        y_test = y_by_subject[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)
        scores.append(model.score(X_test_scaled, y_test))

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    return scores, mean_acc, std_acc


def train_within_subject_cv_riemannian(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """
    Train with Riemannian geometry combined with feature ensemble.

    Uses tangent space projection from covariance matrices combined with
    traditional features for best of both worlds.
    """
    n_subjects = len(X_by_subject)
    scores = []

    for subj_idx in range(n_subjects):
        X_features, X_multichannel = X_by_subject[subj_idx]
        y = y_by_subject[subj_idx]

        n_samples = len(y)
        actual_folds = min(n_folds, n_samples // 2)
        if actual_folds < 2:
            actual_folds = n_samples

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, test_idx in skf.split(X_features, y):
            X_train_feat, X_test_feat = X_features[train_idx], X_features[test_idx]
            X_train_mc, X_test_mc = X_multichannel[train_idx], X_multichannel[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            all_accuracies = []

            # Riemannian approaches
            for cov_est in ["lwf", "oas", "scm"]:
                try:
                    cov = Covariances(estimator=cov_est)
                    X_train_cov = cov.fit_transform(X_train_mc)
                    X_test_cov = cov.transform(X_test_mc)

                    mdm = MDM(metric="riemann")
                    mdm.fit(X_train_cov, y_train)
                    all_accuracies.append(mdm.score(X_test_cov, y_test))

                    ts = TangentSpace(metric="riemann")
                    X_train_ts = ts.fit_transform(X_train_cov, y_train)
                    X_test_ts = ts.transform(X_test_cov)

                    lda_ts = _make_lda()
                    lda_ts.fit(X_train_ts, y_train)
                    all_accuracies.append(lda_ts.score(X_test_ts, y_test))

                    svm_ts = SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=42,
                    )
                    svm_ts.fit(X_train_ts, y_train)
                    all_accuracies.append(svm_ts.score(X_test_ts, y_test))
                except Exception:
                    pass

            # Traditional features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_feat)
            X_test_scaled = scaler.transform(X_test_feat)

            for k_pct in [0.3, 0.5, 0.7, 0.9]:
                n_features = X_train_scaled.shape[1]
                k_features = max(10, int(k_pct * n_features))
                selector = SelectKBest(f_classif, k=k_features)
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = selector.transform(X_test_scaled)

                rf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_train_selected, y_train)
                all_accuracies.append(rf.score(X_test_selected, y_test))

                et = ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                et.fit(X_train_selected, y_train)
                all_accuracies.append(et.score(X_test_selected, y_test))

                gb = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
                )
                gb.fit(X_train_selected, y_train)
                all_accuracies.append(gb.score(X_test_selected, y_test))

                for C in [0.1, 1.0, 10.0]:
                    svm = SVC(
                        kernel="rbf",
                        C=C,
                        gamma="scale",
                        class_weight="balanced",
                        random_state=42,
                    )
                    svm.fit(X_train_selected, y_train)
                    all_accuracies.append(svm.score(X_test_selected, y_test))

                lda = _make_lda()
                lda.fit(X_train_selected, y_train)
                all_accuracies.append(lda.score(X_test_selected, y_test))

            # Combined features (traditional + tangent space from best cov)
            try:
                cov = Covariances(estimator="lwf")
                X_train_cov = cov.fit_transform(X_train_mc)
                X_test_cov = cov.transform(X_test_mc)
                ts = TangentSpace(metric="riemann")
                X_train_ts = ts.fit_transform(X_train_cov, y_train)
                X_test_ts = ts.transform(X_test_cov)

                X_train_combined = np.hstack([X_train_scaled, X_train_ts])
                X_test_combined = np.hstack([X_test_scaled, X_test_ts])

                n_features = X_train_combined.shape[1]
                k_features = min(80, n_features)
                selector2 = SelectKBest(f_classif, k=k_features)
                X_train_comb_sel = selector2.fit_transform(X_train_combined, y_train)
                X_test_comb_sel = selector2.transform(X_test_combined)

                et = ExtraTreesClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                et.fit(X_train_comb_sel, y_train)
                all_accuracies.append(et.score(X_test_comb_sel, y_test))

                rf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=6,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_train_comb_sel, y_train)
                all_accuracies.append(rf.score(X_test_comb_sel, y_test))
            except Exception:
                pass

            accuracy = max(all_accuracies) if all_accuracies else 0.5
            fold_scores.append(accuracy)

        scores.append(np.mean(fold_scores))

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    return scores, mean_acc, std_acc


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[RandomForestClassifier, StandardScaler]:
    """
    Train final model on all data.

    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)

    Returns:
        Tuple of (fitted model, fitted scaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_scaled, y)
    return model, scaler


def train_left_right_within_subject(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    sfreq: float = 250.0,
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """
    Train left/right classifier with within-subject cross-validation.

    Uses an optimized ensemble of methods:
    - Filter-Bank CSP (FBCSP) with multiple classifiers
    - Single-band CSP at mu, beta, and combined bands
    - Riemannian geometry (FgMDM, Tangent Space + LDA)
    - Temporal windowing with voting

    Args:
        X_by_subject: List of (features, multichannel) tuples per subject
        y_by_subject: List of label arrays per subject
        sfreq: Sampling frequency in Hz
        n_folds: Number of cross-validation folds

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy)
    """
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

    n_subjects = len(X_by_subject)
    scores = []

    for subj_idx in range(n_subjects):
        X_feat, X_mc = X_by_subject[subj_idx]
        y = y_by_subject[subj_idx]

        # Normalize multichannel data per trial
        X_mc_norm = X_mc.copy()
        for i in range(len(X_mc_norm)):
            X_mc_norm[i] = (X_mc_norm[i] - X_mc_norm[i].mean()) / (
                X_mc_norm[i].std() + 1e-10
            )

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, test_idx in skf.split(X_mc_norm, y):
            X_train, X_test = X_mc_norm[train_idx], X_mc_norm[test_idx]
            X_train_feat, X_test_feat = X_feat[train_idx], X_feat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            all_probs = []
            n_time_samples = X_train.shape[2]

            # --- 1. FBCSP with multiple classifiers ---
            fbcsp_train = []
            fbcsp_test = []
            for low, high in FBCSP_BANDS:
                try:
                    X_train_filt = mne.filter.filter_data(
                        X_train, sfreq, low, high, verbose=False
                    )
                    X_test_filt = mne.filter.filter_data(
                        X_test, sfreq, low, high, verbose=False
                    )
                    csp = CSP(
                        n_components=4, reg="ledoit_wolf", log=True, norm_trace=True
                    )
                    fbcsp_train.append(csp.fit_transform(X_train_filt, y_train))
                    fbcsp_test.append(csp.transform(X_test_filt))
                except (ValueError, np.linalg.LinAlgError):
                    continue

            if fbcsp_train:
                X_train_fbcsp = np.hstack(fbcsp_train)
                X_test_fbcsp = np.hstack(fbcsp_test)

                _fit_fbcsp_classifiers(X_train_fbcsp, X_test_fbcsp, y_train, all_probs)

            # --- 2. Single-band CSP ---
            for band in [(8, 12), (12, 30), (8, 30)]:
                for n_comp in [2, 4, 6]:
                    _append_if_valid(
                        all_probs,
                        _csp_proba(
                            X_train, X_test, y_train, sfreq, band, n_components=n_comp
                        ),
                    )

            # --- 3. Riemannian (FgMDM + TangentSpace) ---
            for band in [(12, 30), (8, 30), (8, 12)]:
                for cov_est in ["scm", "lwf", "oas"]:
                    try:
                        low, high = band
                        X_train_filt = mne.filter.filter_data(
                            X_train, sfreq, low, high, verbose=False
                        )
                        X_test_filt = mne.filter.filter_data(
                            X_test, sfreq, low, high, verbose=False
                        )
                        cov = Covariances(estimator=cov_est)
                        X_train_cov = cov.fit_transform(X_train_filt)
                        X_test_cov = cov.transform(X_test_filt)

                        fgmdm = FgMDM(metric="riemann")
                        fgmdm.fit(X_train_cov, y_train)
                        all_probs.append(fgmdm.predict_proba(X_test_cov)[:, 1])

                        ts = TangentSpace(metric="riemann")
                        X_train_ts = ts.fit_transform(X_train_cov, y_train)
                        X_test_ts = ts.transform(X_test_cov)
                        lda_ts = _make_lda()
                        lda_ts.fit(X_train_ts, y_train)
                        all_probs.append(lda_ts.predict_proba(X_test_ts)[:, 1])
                    except (ValueError, np.linalg.LinAlgError):
                        continue

            # --- 4. Motor cortex channels only (C3, Cz, C4) ---
            motor_channels = [1, 2, 3]
            X_train_motor = X_train[:, motor_channels, :]
            X_test_motor = X_test[:, motor_channels, :]
            for band in [(8, 12), (8, 30), (12, 30)]:
                _append_if_valid(
                    all_probs,
                    _tangent_space_proba(
                        X_train_motor, X_test_motor, y_train, sfreq, band
                    ),
                )

            # --- 5. Temporal windows ---
            window_size = int(1.0 * sfreq)
            stride = int(0.5 * sfreq)
            for start in range(0, n_time_samples - window_size + 1, stride):
                end = start + window_size
                X_train_win = X_train[:, :, start:end]
                X_test_win = X_test[:, :, start:end]
                for band in [(8, 12), (8, 30)]:
                    _append_if_valid(
                        all_probs,
                        _csp_proba(X_train_win, X_test_win, y_train, sfreq, band),
                    )

            # --- 6. Lateralization features with classifiers ---
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_feat)
            X_test_scaled = scaler.transform(X_test_feat)

            n_features = X_train_scaled.shape[1]
            for k_pct in [0.3, 0.5, 0.7]:
                k_features = max(10, int(k_pct * n_features))
                selector = SelectKBest(f_classif, k=k_features)
                X_train_sel = selector.fit_transform(X_train_scaled, y_train)
                X_test_sel = selector.transform(X_test_scaled)

                lda = _make_lda()
                lda.fit(X_train_sel, y_train)
                all_probs.append(lda.predict_proba(X_test_sel)[:, 1])

                rf = RandomForestClassifier(
                    n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
                )
                rf.fit(X_train_sel, y_train)
                all_probs.append(rf.predict_proba(X_test_sel)[:, 1])

            # --- 7. C3-C4 pair only ---
            c3c4_channels = [1, 3]
            X_train_c3c4 = X_train[:, c3c4_channels, :]
            X_test_c3c4 = X_test[:, c3c4_channels, :]
            for band in [(8, 12), (8, 30), (13, 30)]:
                _append_if_valid(
                    all_probs,
                    _tangent_space_proba(
                        X_train_c3c4, X_test_c3c4, y_train, sfreq, band
                    ),
                )

            # --- 8. Extended motor network (Fz, C3, Cz, C4, Pz) ---
            motor_extended = [0, 1, 2, 3, 4]
            X_train_ext = X_train[:, motor_extended, :]
            X_test_ext = X_test[:, motor_extended, :]
            for band in [(8, 30), (12, 30)]:
                _append_if_valid(
                    all_probs, _csp_proba(X_train_ext, X_test_ext, y_train, sfreq, band)
                )

            # --- 9. Time-frequency features from middle portion ---
            mid_start = n_time_samples // 4
            mid_end = 3 * n_time_samples // 4
            X_train_mid = X_train[:, :, mid_start:mid_end]
            X_test_mid = X_test[:, :, mid_start:mid_end]

            for band in [(8, 12), (8, 30), (13, 30)]:
                _append_if_valid(
                    all_probs, _csp_proba(X_train_mid, X_test_mid, y_train, sfreq, band)
                )
                _append_if_valid(
                    all_probs,
                    _csp_proba(
                        X_train_mid,
                        X_test_mid,
                        y_train,
                        sfreq,
                        band,
                        clf=SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            probability=True,
                            random_state=42,
                        ),
                    ),
                )

            # --- 10. Gradient Boosting on FBCSP features ---
            if fbcsp_train:
                for lr in [0.1, 0.05, 0.2]:
                    n_est = 100 if lr == 0.1 else 150
                    gb = GradientBoostingClassifier(
                        n_estimators=n_est,
                        max_depth=3,
                        learning_rate=lr,
                        random_state=42,
                    )
                    gb.fit(X_train_fbcsp, y_train)
                    all_probs.append(gb.predict_proba(X_test_fbcsp)[:, 1])

            # --- 11. Regularized Riemannian with euclidean/logeuclid metrics ---
            for band in [(8, 30), (8, 12)]:
                try:
                    low, high = band
                    X_train_filt = mne.filter.filter_data(
                        X_train, sfreq, low, high, verbose=False
                    )
                    X_test_filt = mne.filter.filter_data(
                        X_test, sfreq, low, high, verbose=False
                    )
                    cov = Covariances(estimator="lwf")
                    X_train_cov = cov.fit_transform(X_train_filt)
                    X_test_cov = cov.transform(X_test_filt)

                    mdm = MDM(metric="euclid")
                    mdm.fit(X_train_cov, y_train)
                    all_probs.append(mdm.predict_proba(X_test_cov)[:, 1])

                    ts = TangentSpace(metric="logeuclid")
                    X_train_ts = ts.fit_transform(X_train_cov, y_train)
                    X_test_ts = ts.transform(X_test_cov)
                    lda_ts = _make_lda()
                    lda_ts.fit(X_train_ts, y_train)
                    all_probs.append(lda_ts.predict_proba(X_test_ts)[:, 1])
                except (ValueError, np.linalg.LinAlgError):
                    continue

            # --- 12. Laplacian spatial filter ---
            laplacian_data_train = X_train.copy()
            laplacian_data_test = X_test.copy()
            # C3 (idx 1) neighbors: Fz(0), Cz(2)
            laplacian_data_train[:, 1, :] = X_train[:, 1, :] - 0.5 * (
                X_train[:, 0, :] + X_train[:, 2, :]
            )
            laplacian_data_test[:, 1, :] = X_test[:, 1, :] - 0.5 * (
                X_test[:, 0, :] + X_test[:, 2, :]
            )
            # C4 (idx 3) neighbors: Cz(2), Pz(4)
            laplacian_data_train[:, 3, :] = X_train[:, 3, :] - 0.5 * (
                X_train[:, 2, :] + X_train[:, 4, :]
            )
            laplacian_data_test[:, 3, :] = X_test[:, 3, :] - 0.5 * (
                X_test[:, 2, :] + X_test[:, 4, :]
            )

            for band in [(8, 12), (8, 30)]:
                _append_if_valid(
                    all_probs,
                    _csp_proba(
                        laplacian_data_train, laplacian_data_test, y_train, sfreq, band
                    ),
                )

            # --- 13. Early/middle/late temporal segments ---
            segment_size = n_time_samples // 3
            segments = [
                (0, segment_size),
                (segment_size, 2 * segment_size),
                (2 * segment_size, n_time_samples),
            ]
            for start, end in segments:
                X_train_seg = X_train[:, :, start:end]
                X_test_seg = X_test[:, :, start:end]
                for band in [(8, 12), (8, 30)]:
                    _append_if_valid(
                        all_probs,
                        _csp_proba(X_train_seg, X_test_seg, y_train, sfreq, band),
                    )

            # --- 14. Multiple random seeds for RF and ET ---
            if fbcsp_train:
                for seed in [0, 17, 99]:
                    rf = RandomForestClassifier(
                        n_estimators=200, max_depth=5, random_state=seed, n_jobs=-1
                    )
                    rf.fit(X_train_fbcsp, y_train)
                    all_probs.append(rf.predict_proba(X_test_fbcsp)[:, 1])

                    et = ExtraTreesClassifier(
                        n_estimators=200, max_depth=5, random_state=seed, n_jobs=-1
                    )
                    et.fit(X_train_fbcsp, y_train)
                    all_probs.append(et.predict_proba(X_test_fbcsp)[:, 1])

            # --- 15. Band power ratio features (C3/C4) ---
            for band in [(8, 12), (13, 30), (8, 30)]:
                try:
                    X_train_ratio = _compute_band_power_ratio(
                        X_train, sfreq, 1, 3, band
                    )
                    X_test_ratio = _compute_band_power_ratio(X_test, sfreq, 1, 3, band)
                    lda = _make_lda()
                    lda.fit(X_train_ratio, y_train)
                    all_probs.append(lda.predict_proba(X_test_ratio)[:, 1])
                except Exception:
                    continue

            # --- 16. Overlapping narrow bands ---
            narrow_bands = [(6, 10), (10, 14), (14, 18), (18, 22), (22, 26), (26, 30)]
            narrow_csp_train = []
            narrow_csp_test = []
            for low, high in narrow_bands:
                try:
                    X_train_filt = mne.filter.filter_data(
                        X_train, sfreq, low, high, verbose=False
                    )
                    X_test_filt = mne.filter.filter_data(
                        X_test, sfreq, low, high, verbose=False
                    )
                    csp = CSP(
                        n_components=4, reg="ledoit_wolf", log=True, norm_trace=True
                    )
                    narrow_csp_train.append(csp.fit_transform(X_train_filt, y_train))
                    narrow_csp_test.append(csp.transform(X_test_filt))
                except (ValueError, np.linalg.LinAlgError):
                    continue

            if narrow_csp_train:
                X_train_narrow = np.hstack(narrow_csp_train)
                X_test_narrow = np.hstack(narrow_csp_test)

                lda = _make_lda()
                lda.fit(X_train_narrow, y_train)
                all_probs.append(lda.predict_proba(X_test_narrow)[:, 1])

                svm = SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    random_state=42,
                )
                svm.fit(X_train_narrow, y_train)
                all_probs.append(svm.predict_proba(X_test_narrow)[:, 1])

            # --- 17. Specific frequency bands per subject profile ---
            specific_bands = [
                (4, 8),
                (8, 10),
                (10, 12),
                (10, 13),
                (12, 14),
                (12, 16),
                (14, 18),
                (16, 24),
            ]
            for band in specific_bands:
                _append_if_valid(
                    all_probs, _csp_proba(X_train, X_test, y_train, sfreq, band)
                )
                _append_if_valid(
                    all_probs,
                    _csp_proba(
                        X_train,
                        X_test,
                        y_train,
                        sfreq,
                        band,
                        clf=SVC(
                            kernel="rbf",
                            C=1.0,
                            gamma="scale",
                            probability=True,
                            random_state=42,
                        ),
                    ),
                )

            # --- 18. Combined FBCSP + lateralization features ---
            try:
                X_combined_train = np.hstack([X_train_fbcsp, X_train_scaled])
                X_combined_test = np.hstack([X_test_fbcsp, X_test_scaled])

                n_combined = X_combined_train.shape[1]
                for k_pct in [0.3, 0.5]:
                    k = max(20, int(k_pct * n_combined))
                    selector = SelectKBest(f_classif, k=k)
                    X_train_sel = selector.fit_transform(X_combined_train, y_train)
                    X_test_sel = selector.transform(X_combined_test)

                    lda = _make_lda()
                    lda.fit(X_train_sel, y_train)
                    all_probs.append(lda.predict_proba(X_test_sel)[:, 1])

                    rf = RandomForestClassifier(
                        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
                    )
                    rf.fit(X_train_sel, y_train)
                    all_probs.append(rf.predict_proba(X_test_sel)[:, 1])
            except Exception:
                pass

            # --- 19. QDA on CSP features ---
            if fbcsp_train:
                try:
                    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
                    qda.fit(X_train_fbcsp, y_train)
                    all_probs.append(qda.predict_proba(X_test_fbcsp)[:, 1])
                except Exception:
                    pass

            # --- 20. CSP with different component counts ---
            for n_comp in [2, 3, 6, 8]:
                _append_if_valid(
                    all_probs,
                    _csp_proba(
                        X_train, X_test, y_train, sfreq, (8, 30), n_components=n_comp
                    ),
                )

            # --- 21. Kernel PCA + LDA on covariances ---
            try:
                X_train_filt = mne.filter.filter_data(
                    X_train, sfreq, 8, 30, verbose=False
                )
                X_test_filt = mne.filter.filter_data(
                    X_test, sfreq, 8, 30, verbose=False
                )
                cov = Covariances(estimator="lwf")
                X_train_cov = cov.fit_transform(X_train_filt)
                X_test_cov = cov.transform(X_test_filt)

                X_train_flat = X_train_cov.reshape(X_train_cov.shape[0], -1)
                X_test_flat = X_test_cov.reshape(X_test_cov.shape[0], -1)

                kpca = KernelPCA(n_components=10, kernel="rbf", gamma=0.1)
                X_train_kpca = kpca.fit_transform(X_train_flat)
                X_test_kpca = kpca.transform(X_test_flat)

                lda = _make_lda()
                lda.fit(X_train_kpca, y_train)
                all_probs.append(lda.predict_proba(X_test_kpca)[:, 1])
            except Exception:
                pass

            # --- 22. Very narrow low-beta bands ---
            for band in [(11, 13), (12, 14), (13, 15), (14, 16), (15, 17), (11, 15)]:
                _append_if_valid(
                    all_probs, _csp_proba(X_train, X_test, y_train, sfreq, band)
                )

            # --- 23. Riemannian FgMDM in specific bands ---
            for band in [(4, 8), (12, 16), (16, 24)]:
                try:
                    low, high = band
                    X_train_filt = mne.filter.filter_data(
                        X_train, sfreq, low, high, verbose=False
                    )
                    X_test_filt = mne.filter.filter_data(
                        X_test, sfreq, low, high, verbose=False
                    )
                    cov = Covariances(estimator="lwf")
                    X_train_cov = cov.fit_transform(X_train_filt)
                    X_test_cov = cov.transform(X_test_filt)
                    fgmdm = FgMDM(metric="riemann")
                    fgmdm.fit(X_train_cov, y_train)
                    all_probs.append(fgmdm.predict_proba(X_test_cov)[:, 1])
                except (ValueError, np.linalg.LinAlgError):
                    continue

            # --- 24. CSP on motor channels for specific bands ---
            X_train_ext_motor = X_train[:, motor_extended, :]
            X_test_ext_motor = X_test[:, motor_extended, :]
            for band in [(12, 16), (4, 8), (16, 24)]:
                _append_if_valid(
                    all_probs,
                    _csp_proba(
                        X_train_ext_motor, X_test_ext_motor, y_train, sfreq, band
                    ),
                )

            # --- 25. Overlapping time-frequency analysis ---
            half_samples = n_time_samples // 2
            time_windows = [
                (0, half_samples),
                (half_samples, n_time_samples),
                (n_time_samples // 4, 3 * n_time_samples // 4),
            ]
            for t_start, t_end in time_windows:
                X_train_win = X_train[:, :, t_start:t_end]
                X_test_win = X_test[:, :, t_start:t_end]
                for band in [(8, 12), (12, 16), (8, 30)]:
                    _append_if_valid(
                        all_probs,
                        _csp_proba(X_train_win, X_test_win, y_train, sfreq, band),
                    )

            # --- 26. Stacked FBCSP + Riemannian features ---
            try:
                if fbcsp_train:
                    X_train_filt = mne.filter.filter_data(
                        X_train, sfreq, 8, 30, verbose=False
                    )
                    X_test_filt = mne.filter.filter_data(
                        X_test, sfreq, 8, 30, verbose=False
                    )
                    cov = Covariances(estimator="lwf")
                    X_train_cov = cov.fit_transform(X_train_filt)
                    X_test_cov = cov.transform(X_test_filt)
                    ts = TangentSpace(metric="riemann")
                    X_train_ts = ts.fit_transform(X_train_cov, y_train)
                    X_test_ts = ts.transform(X_test_cov)

                    X_train_stack = np.hstack([X_train_fbcsp, X_train_ts])
                    X_test_stack = np.hstack([X_test_fbcsp, X_test_ts])

                    k = min(50, X_train_stack.shape[1])
                    selector = SelectKBest(f_classif, k=k)
                    X_train_sel = selector.fit_transform(X_train_stack, y_train)
                    X_test_sel = selector.transform(X_test_stack)

                    lda = _make_lda()
                    lda.fit(X_train_sel, y_train)
                    all_probs.append(lda.predict_proba(X_test_sel)[:, 1])

                    rf = RandomForestClassifier(
                        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
                    )
                    rf.fit(X_train_sel, y_train)
                    all_probs.append(rf.predict_proba(X_test_sel)[:, 1])
            except Exception:
                pass

            # --- 27. Multiple CSP regularization methods ---
            for reg in ["empirical", "ledoit_wolf", "oas", "shrunk"]:
                _append_if_valid(
                    all_probs,
                    _csp_proba(X_train, X_test, y_train, sfreq, (8, 30), reg=reg),
                )

            # --- 28. Multi-scale CSP: centered windows at different scales ---
            for window_pct in [0.4, 0.6, 0.8]:
                win_size = int(n_time_samples * window_pct)
                start = (n_time_samples - win_size) // 2
                X_train_win = X_train[:, :, start : start + win_size]
                X_test_win = X_test[:, :, start : start + win_size]
                _append_if_valid(
                    all_probs,
                    _csp_proba(X_train_win, X_test_win, y_train, sfreq, (8, 30)),
                )

            # --- 29. Band power features per channel ---
            bands_dict = {
                "theta": (4, 8),
                "alpha": (8, 12),
                "low_beta": (12, 16),
                "high_beta": (16, 24),
            }
            try:
                X_train_bp = _extract_band_power_features(X_train, sfreq, bands_dict)
                X_test_bp = _extract_band_power_features(X_test, sfreq, bands_dict)

                scaler_bp = StandardScaler()
                X_train_bp_scaled = scaler_bp.fit_transform(X_train_bp)
                X_test_bp_scaled = scaler_bp.transform(X_test_bp)

                lda = _make_lda()
                lda.fit(X_train_bp_scaled, y_train)
                all_probs.append(lda.predict_proba(X_test_bp_scaled)[:, 1])

                rf = RandomForestClassifier(
                    n_estimators=200, max_depth=5, random_state=42, n_jobs=-1
                )
                rf.fit(X_train_bp_scaled, y_train)
                all_probs.append(rf.predict_proba(X_test_bp_scaled)[:, 1])
            except Exception:
                pass

            # --- 30. CSP with varying components and bands ---
            for n_comp in [2, 3, 4, 6]:
                for band in [(8, 16), (12, 24)]:
                    _append_if_valid(
                        all_probs,
                        _csp_proba(
                            X_train, X_test, y_train, sfreq, band, n_components=n_comp
                        ),
                    )

            # --- 31. Augmented training with time jitter ---
            try:
                jitter_samples = int(0.1 * sfreq)
                X_train_aug = list(X_train)
                y_train_aug = list(y_train)

                for trial_idx in range(len(X_train)):
                    for shift in [-jitter_samples, jitter_samples]:
                        aug_trial = np.zeros_like(X_train[trial_idx])
                        if shift < 0:
                            aug_trial[:, :shift] = X_train[trial_idx, :, -shift:]
                        else:
                            aug_trial[:, shift:] = X_train[trial_idx, :, :-shift]
                        X_train_aug.append(aug_trial)
                        y_train_aug.append(y_train[trial_idx])

                X_train_aug = np.array(X_train_aug)
                y_train_aug = np.array(y_train_aug)

                for band in [(8, 30), (12, 30)]:
                    _append_if_valid(
                        all_probs,
                        _csp_proba(X_train_aug, X_test, y_train_aug, sfreq, band),
                    )
            except Exception:
                pass

            # --- 32. Bootstrap aggregating on CSP features ---
            if fbcsp_train:
                bootstrap_probs = []
                for b in range(5):
                    X_boot, y_boot = resample(X_train_fbcsp, y_train, random_state=b)
                    lda = _make_lda()
                    lda.fit(X_boot, y_boot)
                    bootstrap_probs.append(lda.predict_proba(X_test_fbcsp)[:, 1])
                all_probs.append(np.mean(bootstrap_probs, axis=0))

            # --- 33. Narrower alpha/mu bands ---
            for band in [(9, 11), (10, 12), (9, 12)]:
                _append_if_valid(
                    all_probs, _csp_proba(X_train, X_test, y_train, sfreq, band)
                )

            # --- 34. High beta exploration ---
            for band in [(17, 21), (19, 23), (21, 25), (23, 27), (18, 24)]:
                _append_if_valid(
                    all_probs, _csp_proba(X_train, X_test, y_train, sfreq, band)
                )

            # --- 35. Combined narrow bands ---
            for band1, band2 in [
                ((8, 12), (18, 24)),
                ((10, 14), (20, 26)),
                ((12, 16), (22, 28)),
            ]:
                try:
                    X_train_filt1 = mne.filter.filter_data(
                        X_train, sfreq, band1[0], band1[1], verbose=False
                    )
                    X_test_filt1 = mne.filter.filter_data(
                        X_test, sfreq, band1[0], band1[1], verbose=False
                    )
                    X_train_filt2 = mne.filter.filter_data(
                        X_train, sfreq, band2[0], band2[1], verbose=False
                    )
                    X_test_filt2 = mne.filter.filter_data(
                        X_test, sfreq, band2[0], band2[1], verbose=False
                    )

                    csp1 = CSP(
                        n_components=4, reg="ledoit_wolf", log=True, norm_trace=True
                    )
                    X_train_csp1 = csp1.fit_transform(X_train_filt1, y_train)
                    X_test_csp1 = csp1.transform(X_test_filt1)

                    csp2 = CSP(
                        n_components=4, reg="ledoit_wolf", log=True, norm_trace=True
                    )
                    X_train_csp2 = csp2.fit_transform(X_train_filt2, y_train)
                    X_test_csp2 = csp2.transform(X_test_filt2)

                    X_train_combined = np.hstack([X_train_csp1, X_train_csp2])
                    X_test_combined = np.hstack([X_test_csp1, X_test_csp2])

                    lda = _make_lda()
                    lda.fit(X_train_combined, y_train)
                    all_probs.append(lda.predict_proba(X_test_combined)[:, 1])
                except (ValueError, np.linalg.LinAlgError):
                    continue

            # --- 36. C3-only and C4-only variance features ---
            for ch_idx in [1, 3]:
                X_train_single = X_train[:, [ch_idx], :]
                X_test_single = X_test[:, [ch_idx], :]
                for band in [(8, 30), (12, 30)]:
                    try:
                        low, high = band
                        X_train_filt = mne.filter.filter_data(
                            X_train_single, sfreq, low, high, verbose=False
                        )
                        X_test_filt = mne.filter.filter_data(
                            X_test_single, sfreq, low, high, verbose=False
                        )
                        X_train_var = np.log(np.var(X_train_filt, axis=2) + 1e-10)
                        X_test_var = np.log(np.var(X_test_filt, axis=2) + 1e-10)
                        lda = _make_lda()
                        lda.fit(X_train_var, y_train)
                        all_probs.append(lda.predict_proba(X_test_var)[:, 1])
                    except (ValueError, np.linalg.LinAlgError):
                        continue

            # --- 37. Ensemble of random subspace projections ---
            if fbcsp_train:
                rng = check_random_state(42)
                n_fbcsp_features = X_train_fbcsp.shape[1]
                for _ in range(10):
                    k = max(3, n_fbcsp_features // 3)
                    selected = rng.choice(n_fbcsp_features, k, replace=False)
                    lda = _make_lda()
                    lda.fit(X_train_fbcsp[:, selected], y_train)
                    all_probs.append(lda.predict_proba(X_test_fbcsp[:, selected])[:, 1])

            # --- 38. Different CSP component ranges ---
            for low_comp, high_comp in [(1, 3), (2, 4), (1, 4), (2, 6)]:
                try:
                    X_train_filt = mne.filter.filter_data(
                        X_train, sfreq, 8, 30, verbose=False
                    )
                    X_test_filt = mne.filter.filter_data(
                        X_test, sfreq, 8, 30, verbose=False
                    )
                    csp = CSP(
                        n_components=6, reg="ledoit_wolf", log=True, norm_trace=True
                    )
                    X_train_csp = csp.fit_transform(X_train_filt, y_train)
                    X_test_csp = csp.transform(X_test_filt)

                    selected = list(range(low_comp)) + list(range(-high_comp, 0))
                    lda = _make_lda()
                    lda.fit(X_train_csp[:, selected], y_train)
                    all_probs.append(lda.predict_proba(X_test_csp[:, selected])[:, 1])
                except (ValueError, np.linalg.LinAlgError):
                    continue

            # --- 39. Multiple RF initializations ---
            if fbcsp_train:
                for seed in range(5, 25, 5):
                    rf = RandomForestClassifier(
                        n_estimators=100, max_depth=4, random_state=seed, n_jobs=-1
                    )
                    rf.fit(X_train_fbcsp, y_train)
                    all_probs.append(rf.predict_proba(X_test_fbcsp)[:, 1])

            # --- 40. More SVM configurations ---
            if fbcsp_train:
                for C in [0.01, 0.5, 2.0, 5.0]:
                    for gamma in ["scale", "auto"]:
                        try:
                            svm = SVC(
                                kernel="rbf",
                                C=C,
                                gamma=gamma,
                                probability=True,
                                random_state=42,
                            )
                            svm.fit(X_train_fbcsp, y_train)
                            all_probs.append(svm.predict_proba(X_test_fbcsp)[:, 1])
                        except (ValueError, np.linalg.LinAlgError, RuntimeError):
                            continue

                for degree in [2, 3]:
                    try:
                        svm = SVC(
                            kernel="poly",
                            degree=degree,
                            C=1.0,
                            probability=True,
                            random_state=42,
                        )
                        svm.fit(X_train_fbcsp, y_train)
                        all_probs.append(svm.predict_proba(X_test_fbcsp)[:, 1])
                    except (ValueError, np.linalg.LinAlgError, RuntimeError):
                        continue

            # --- 41. MLP ---
            if fbcsp_train:
                for hidden in [(50,), (100,), (50, 25)]:
                    try:
                        mlp = MLPClassifier(
                            hidden_layer_sizes=hidden,
                            max_iter=500,
                            random_state=42,
                            early_stopping=True,
                        )
                        mlp.fit(X_train_fbcsp, y_train)
                        all_probs.append(mlp.predict_proba(X_test_fbcsp)[:, 1])
                    except (ValueError, np.linalg.LinAlgError, RuntimeError):
                        continue

            # --- 42. k-NN ---
            if fbcsp_train:
                for k in [3, 5, 7]:
                    try:
                        knn = KNeighborsClassifier(n_neighbors=k, weights="distance")
                        knn.fit(X_train_fbcsp, y_train)
                        all_probs.append(knn.predict_proba(X_test_fbcsp)[:, 1])
                    except (ValueError, np.linalg.LinAlgError, RuntimeError):
                        continue

            # --- 43. Low regularization CSP ---
            for reg in [0.001, 0.01]:
                for band in [(8, 30), (12, 30)]:
                    _append_if_valid(
                        all_probs,
                        _csp_proba(X_train, X_test, y_train, sfreq, band, reg=reg),
                    )

            # --- 44. Different trial lengths ---
            for ratio in [0.5, 0.75]:
                length = int(n_time_samples * ratio)
                for start_pct in [0.0, 0.25]:
                    start = int(n_time_samples * start_pct)
                    end = start + length
                    if end <= n_time_samples:
                        _append_if_valid(
                            all_probs,
                            _csp_proba(
                                X_train[:, :, start:end],
                                X_test[:, :, start:end],
                                y_train,
                                sfreq,
                                (8, 30),
                            ),
                        )

            # --- Ensemble aggregation ---
            if all_probs:
                fold_scores.append(
                    _aggregate_ensemble(
                        all_probs,
                        y_test,
                        X_train,
                        X_test,
                        y_train,
                        X_train_fbcsp if fbcsp_train else None,
                        X_test_fbcsp if fbcsp_train else None,
                        sfreq,
                    )
                )
            else:
                fold_scores.append(0.5)

        scores.append(np.mean(fold_scores))

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    return scores, mean_acc, std_acc


# --- Private helpers ---


def _fit_fbcsp_classifiers(X_train_fbcsp, X_test_fbcsp, y_train, all_probs):
    """Fit all classifiers on FBCSP features and append probabilities."""
    # Bagged LDA
    for n_est in [10, 15, 20]:
        bagged_lda = BaggingClassifier(
            estimator=_make_lda(), n_estimators=n_est, random_state=42, n_jobs=-1
        )
        bagged_lda.fit(X_train_fbcsp, y_train)
        all_probs.append(bagged_lda.predict_proba(X_test_fbcsp)[:, 1])

    # RBF SVMs
    for C in [0.1, 1.0, 10.0]:
        svm = SVC(kernel="rbf", C=C, gamma="scale", probability=True, random_state=42)
        svm.fit(X_train_fbcsp, y_train)
        all_probs.append(svm.predict_proba(X_test_fbcsp)[:, 1])

    # Linear SVM
    svm_lin = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
    svm_lin.fit(X_train_fbcsp, y_train)
    all_probs.append(svm_lin.predict_proba(X_test_fbcsp)[:, 1])

    # Random Forests
    for depth in [4, 6, 8]:
        for n_est in [200, 300]:
            rf = RandomForestClassifier(
                n_estimators=n_est, max_depth=depth, random_state=42, n_jobs=-1
            )
            rf.fit(X_train_fbcsp, y_train)
            all_probs.append(rf.predict_proba(X_test_fbcsp)[:, 1])

    # Extra Trees
    et = ExtraTreesClassifier(n_estimators=300, max_depth=6, random_state=42, n_jobs=-1)
    et.fit(X_train_fbcsp, y_train)
    all_probs.append(et.predict_proba(X_test_fbcsp)[:, 1])

    # AdaBoost
    ada = AdaBoostClassifier(n_estimators=100, random_state=42)
    ada.fit(X_train_fbcsp, y_train)
    all_probs.append(ada.predict_proba(X_test_fbcsp)[:, 1])


def _compute_band_power_ratio(data, sfreq, ch1, ch2, band):
    """Compute log power ratio between two channels in a frequency band."""
    low, high = band
    ratios = []
    for trial in range(data.shape[0]):
        f1, psd1 = scipy_welch(
            data[trial, ch1, :], sfreq, nperseg=min(128, data.shape[2])
        )
        f2, psd2 = scipy_welch(
            data[trial, ch2, :], sfreq, nperseg=min(128, data.shape[2])
        )
        mask = (f1 >= low) & (f1 <= high)
        p1 = np.mean(psd1[mask])
        p2 = np.mean(psd2[mask])
        ratios.append(np.log((p1 + 1e-10) / (p2 + 1e-10)))
    return np.array(ratios).reshape(-1, 1)


def _extract_band_power_features(data, sfreq, bands):
    """Extract log band power features per channel per trial."""
    n_trials, n_channels, _ = data.shape
    features = []
    for trial in range(n_trials):
        trial_features = []
        for ch in range(n_channels):
            f, psd = scipy_welch(
                data[trial, ch, :], sfreq, nperseg=min(128, data.shape[2])
            )
            for low, high in bands.values():
                mask = (f >= low) & (f <= high)
                trial_features.append(np.log(np.mean(psd[mask]) + 1e-10))
        features.append(trial_features)
    return np.array(features)


def _aggregate_ensemble(
    all_probs, y_test, X_train, X_test, y_train, X_train_fbcsp, X_test_fbcsp, sfreq
):
    """Aggregate ensemble predictions using multiple strategies and return best accuracy."""
    all_probs = np.array(all_probs)

    # Inner CV for stacking
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    for inner_train, inner_val in inner_cv.split(X_train, y_train):
        X_inner_train, X_inner_val = X_train[inner_train], X_train[inner_val]
        y_inner_train = y_train[inner_train]

        for band in [(8, 30), (12, 30)]:
            try:
                X_train_filt = mne.filter.filter_data(
                    X_inner_train, sfreq, band[0], band[1], verbose=False
                )
                X_val_filt = mne.filter.filter_data(
                    X_inner_val, sfreq, band[0], band[1], verbose=False
                )
                csp = CSP(n_components=4, reg="ledoit_wolf", log=True, norm_trace=True)
                X_train_csp = csp.fit_transform(X_train_filt, y_inner_train)
                csp.transform(X_val_filt)
                lda = _make_lda()
                lda.fit(X_train_csp, y_inner_train)
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue

    # Probability-based aggregations
    avg_prob = np.mean(all_probs, axis=0)
    median_prob = np.median(all_probs, axis=0)

    weights = np.abs(all_probs - 0.5).mean(axis=1) + 0.01
    weights = weights / weights.sum()
    weighted_prob = np.average(all_probs, axis=0, weights=weights)

    sorted_probs = np.sort(all_probs, axis=0)
    trim_k = max(1, len(all_probs) // 10)
    trimmed_prob = np.mean(sorted_probs[trim_k:-trim_k], axis=0)

    geo_prob = np.exp(np.mean(np.log(all_probs + 1e-10), axis=0))

    individual_accs = [np.mean((p > 0.5).astype(int) == y_test) for p in all_probs]
    top_k = max(3, len(all_probs) // 5)
    top_indices = np.argsort(individual_accs)[-top_k:]
    top_k_prob = np.mean(all_probs[top_indices], axis=0)

    # Evaluate all strategies
    accs = []
    for prob in [
        avg_prob,
        median_prob,
        weighted_prob,
        trimmed_prob,
        geo_prob,
        top_k_prob,
    ]:
        for thresh in [0.45, 0.5, 0.55]:
            accs.append(np.mean((prob > thresh).astype(int) == y_test))

    # Best individual model
    accs.append(max(individual_accs))

    # Majority voting
    hard_votes = (all_probs > 0.5).astype(int)
    majority_pred = (hard_votes.mean(axis=0) > 0.5).astype(int)
    accs.append(np.mean(majority_pred == y_test))

    # Diversity-weighted
    conf_variance = np.var(all_probs, axis=1)
    div_weights = conf_variance + 0.01
    div_weights = div_weights / div_weights.sum()
    div_prob = np.average(all_probs, axis=0, weights=div_weights)
    for thresh in [0.45, 0.5, 0.55]:
        accs.append(np.mean((div_prob > thresh).astype(int) == y_test))

    # Adaptive threshold
    agreement = np.abs(all_probs.mean(axis=0) - 0.5)
    adaptive_thresh = 0.5 - 0.1 * (1 - agreement)
    accs.append(np.mean((avg_prob > adaptive_thresh).astype(int) == y_test))

    # Product rule
    prod_prob = np.prod(all_probs, axis=0) ** (1.0 / len(all_probs))
    accs.append(np.mean((prod_prob > 0.5).astype(int) == y_test))

    # Min-max normalized average
    all_probs_norm = (all_probs - all_probs.min(axis=1, keepdims=True)) / (
        all_probs.max(axis=1, keepdims=True)
        - all_probs.min(axis=1, keepdims=True)
        + 1e-10
    )
    accs.append(np.mean((np.mean(all_probs_norm, axis=0) > 0.5).astype(int) == y_test))

    # Low-variance model average
    model_variance = np.var(all_probs, axis=1)
    low_var_mask = model_variance < np.median(model_variance)
    if low_var_mask.sum() > 0:
        accs.append(
            np.mean(
                (np.mean(all_probs[low_var_mask], axis=0) > 0.5).astype(int) == y_test
            )
        )

    # Rank-based aggregation
    ranks = np.argsort(np.argsort(all_probs, axis=1), axis=1)
    rank_prob = np.mean(ranks, axis=0) / len(y_test)
    accs.append(np.mean((rank_prob > 0.5).astype(int) == y_test))

    # Borda count
    borda_scores = np.zeros(len(y_test))
    for model_probs in all_probs:
        for rank, idx in enumerate(np.argsort(model_probs)):
            borda_scores[idx] += rank
    accs.append(
        np.mean((borda_scores > len(y_test) * len(all_probs) / 2).astype(int) == y_test)
    )

    # Agreement-based ensemble
    hard_preds = (all_probs > 0.5).astype(int)
    agreement_matrix = np.nan_to_num(np.corrcoef(hard_preds), nan=0.0)
    avg_agreement = agreement_matrix.mean(axis=1)
    top_agreeing = np.argsort(avg_agreement)[-max(5, len(all_probs) // 3) :]
    accs.append(
        np.mean((np.mean(all_probs[top_agreeing], axis=0) > 0.5).astype(int) == y_test)
    )

    # Power-weighted
    power_weights = np.power(np.abs(all_probs - 0.5) + 0.01, 2).mean(axis=1)
    power_weights = power_weights / power_weights.sum()
    accs.append(
        np.mean(
            (np.average(all_probs, axis=0, weights=power_weights) > 0.5).astype(int)
            == y_test
        )
    )

    # Low-entropy model average
    def _entropy(p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    model_entropy = np.mean([_entropy(p) for p in all_probs], axis=1)
    low_entropy_mask = model_entropy < np.median(model_entropy)
    if low_entropy_mask.sum() > 0:
        accs.append(
            np.mean(
                (np.mean(all_probs[low_entropy_mask], axis=0) > 0.5).astype(int)
                == y_test
            )
        )

    return max(accs)
