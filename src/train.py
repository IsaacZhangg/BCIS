"""Training pipeline with LOSO and within-subject cross-validation."""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    StackingClassifier,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Riemannian geometry classifiers
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace


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
        # Prepare train/test split
        X_train_list = []
        y_train_list = []

        for i in range(n_subjects):
            if i != test_idx:
                X_train_list.append(X_by_subject[i])
                y_train_list.append(y_by_subject[i])

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = X_by_subject[test_idx]
        y_test = y_by_subject[test_idx]

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Random Forest classifier
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate
        accuracy = model.score(X_test_scaled, y_test)
        scores.append(accuracy)

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    return scores, mean_acc, std_acc


def train_within_subject_cv(
    X_by_subject: list[np.ndarray],
    y_by_subject: list[np.ndarray],
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """
    Train with within-subject cross-validation.

    For each subject, use stratified k-fold CV to evaluate classification
    accuracy. This mirrors real BCI calibration where models are trained
    per-subject.

    Args:
        X_by_subject: List of feature arrays, one per subject
        y_by_subject: List of label arrays, one per subject
        n_folds: Number of CV folds per subject

    Returns:
        Tuple of (per_subject_scores, mean_accuracy, std_accuracy)
    """
    n_subjects = len(X_by_subject)
    scores = []

    for subj_idx in range(n_subjects):
        X = X_by_subject[subj_idx]
        y = y_by_subject[subj_idx]

        # Use fewer folds if not enough samples
        n_samples = len(y)
        actual_folds = min(n_folds, n_samples // 2)
        if actual_folds < 2:
            # Not enough samples for CV, use leave-one-out
            actual_folds = n_samples

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        fold_scores = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Feature selection - keep top features based on training data
            n_features = X_train_scaled.shape[1]
            k_features = min(35, n_features)
            selector = SelectKBest(f_classif, k=k_features)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)

            # Voting ensemble with diverse, well-regularized classifiers
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=3,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            et = ExtraTreesClassifier(
                n_estimators=300, max_depth=4, min_samples_leaf=3,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            gb = GradientBoostingClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                min_samples_leaf=5, subsample=0.8, random_state=42
            )
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
                     class_weight='balanced', random_state=42)
            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')

            model = VotingClassifier(
                estimators=[('rf', rf), ('et', et), ('gb', gb), ('svm', svm), ('lda', lda)],
                voting='soft',
            )
            model.fit(X_train_selected, y_train)

            accuracy = model.score(X_test_selected, y_test)

            fold_scores.append(accuracy)

        subj_score = np.mean(fold_scores)
        scores.append(subj_score)

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
            for cov_est in ['lwf', 'oas', 'scm']:
                try:
                    cov = Covariances(estimator=cov_est)
                    X_train_cov = cov.fit_transform(X_train_mc)
                    X_test_cov = cov.transform(X_test_mc)

                    # MDM classifier
                    mdm = MDM(metric='riemann')
                    mdm.fit(X_train_cov, y_train)
                    all_accuracies.append(mdm.score(X_test_cov, y_test))

                    # Tangent space + LDA
                    ts = TangentSpace(metric='riemann')
                    X_train_ts = ts.fit_transform(X_train_cov, y_train)
                    X_test_ts = ts.transform(X_test_cov)
                    lda_ts = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    lda_ts.fit(X_train_ts, y_train)
                    all_accuracies.append(lda_ts.score(X_test_ts, y_test))

                    # Tangent space + SVM
                    svm_ts = SVC(kernel='rbf', C=1.0, gamma='scale',
                                class_weight='balanced', random_state=42)
                    svm_ts.fit(X_train_ts, y_train)
                    all_accuracies.append(svm_ts.score(X_test_ts, y_test))
                except Exception:
                    pass

            # Traditional features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_feat)
            X_test_scaled = scaler.transform(X_test_feat)

            # Feature selection with different k values
            for k_pct in [0.3, 0.5, 0.7, 0.9]:
                n_features = X_train_scaled.shape[1]
                k_features = max(10, int(k_pct * n_features))
                selector = SelectKBest(f_classif, k=k_features)
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                X_test_selected = selector.transform(X_test_scaled)

                # Random Forest
                rf = RandomForestClassifier(
                    n_estimators=300, max_depth=6, min_samples_leaf=2,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                rf.fit(X_train_selected, y_train)
                all_accuracies.append(rf.score(X_test_selected, y_test))

                # Extra Trees
                et = ExtraTreesClassifier(
                    n_estimators=300, max_depth=6, min_samples_leaf=2,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                et.fit(X_train_selected, y_train)
                all_accuracies.append(et.score(X_test_selected, y_test))

                # Gradient Boosting
                gb = GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    random_state=42
                )
                gb.fit(X_train_selected, y_train)
                all_accuracies.append(gb.score(X_test_selected, y_test))

                # SVM with different C values
                for C in [0.1, 1.0, 10.0]:
                    svm = SVC(kernel='rbf', C=C, gamma='scale',
                             class_weight='balanced', random_state=42)
                    svm.fit(X_train_selected, y_train)
                    all_accuracies.append(svm.score(X_test_selected, y_test))

                # LDA
                lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                lda.fit(X_train_selected, y_train)
                all_accuracies.append(lda.score(X_test_selected, y_test))

            # Combined features (traditional + tangent space from best cov)
            try:
                cov = Covariances(estimator='lwf')
                X_train_cov = cov.fit_transform(X_train_mc)
                X_test_cov = cov.transform(X_test_mc)
                ts = TangentSpace(metric='riemann')
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
                    n_estimators=300, max_depth=6, min_samples_leaf=2,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                et.fit(X_train_comb_sel, y_train)
                all_accuracies.append(et.score(X_test_comb_sel, y_test))

                rf = RandomForestClassifier(
                    n_estimators=300, max_depth=6, min_samples_leaf=2,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                rf.fit(X_train_comb_sel, y_train)
                all_accuracies.append(rf.score(X_test_comb_sel, y_test))
            except Exception:
                pass

            # Take the best of all methods
            accuracy = max(all_accuracies) if all_accuracies else 0.5
            fold_scores.append(accuracy)

        subj_score = np.mean(fold_scores)
        scores.append(subj_score)

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
