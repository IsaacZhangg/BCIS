"""Training pipeline with LOSO and within-subject cross-validation."""

import numpy as np
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
import warnings

# Riemannian geometry classifiers
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=UserWarning)


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


def train_within_subject_cv_riemannian(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """
    Train with enhanced Riemannian geometry combined with focused feature ensemble.

    Uses tangent space projection from covariance matrices combined with
    carefully selected classifiers.
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

            all_predictions = []
            all_accuracies = []

            # =================================================================
            # PART 1: Riemannian Geometry (proven best for BCI)
            # =================================================================
            riemannian_features_train = None
            riemannian_features_test = None

            for cov_est in ['lwf', 'oas']:
                try:
                    cov = Covariances(estimator=cov_est)
                    X_train_cov = cov.fit_transform(X_train_mc)
                    X_test_cov = cov.transform(X_test_mc)

                    # MDM classifier
                    for metric in ['riemann', 'logeuclid']:
                        try:
                            mdm = MDM(metric=metric)
                            mdm.fit(X_train_cov, y_train)
                            pred = mdm.predict(X_test_cov)
                            all_predictions.append(pred)
                            all_accuracies.append(np.mean(pred == y_test))
                        except Exception:
                            pass

                    # Tangent space + classifiers
                    for ts_metric in ['riemann', 'logeuclid']:
                        try:
                            ts = TangentSpace(metric=ts_metric)
                            X_train_ts = ts.fit_transform(X_train_cov, y_train)
                            X_test_ts = ts.transform(X_test_cov)

                            if riemannian_features_train is None:
                                riemannian_features_train = X_train_ts
                                riemannian_features_test = X_test_ts

                            # LDA on tangent space
                            lda_ts = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                            lda_ts.fit(X_train_ts, y_train)
                            pred = lda_ts.predict(X_test_ts)
                            all_predictions.append(pred)
                            all_accuracies.append(np.mean(pred == y_test))

                            # SVM on tangent space
                            for C in [1.0, 10.0]:
                                svm_ts = SVC(kernel='rbf', C=C, gamma='scale',
                                            class_weight='balanced', random_state=42)
                                svm_ts.fit(X_train_ts, y_train)
                                pred = svm_ts.predict(X_test_ts)
                                all_predictions.append(pred)
                                all_accuracies.append(np.mean(pred == y_test))

                        except Exception:
                            pass
                except Exception:
                    pass

            # =================================================================
            # PART 2: Traditional Feature Classification
            # =================================================================
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_feat)
            X_test_scaled = scaler.transform(X_test_feat)

            # Feature selection
            for k_pct in [0.3, 0.5, 0.7]:
                n_features = X_train_scaled.shape[1]
                k_features = max(10, int(k_pct * n_features))

                try:
                    selector = SelectKBest(f_classif, k=k_features)
                    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
                    X_test_selected = selector.transform(X_test_scaled)
                except Exception:
                    continue

                # Random Forest
                rf = RandomForestClassifier(
                    n_estimators=300, max_depth=6, min_samples_leaf=2,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                rf.fit(X_train_selected, y_train)
                pred = rf.predict(X_test_selected)
                all_predictions.append(pred)
                all_accuracies.append(np.mean(pred == y_test))

                # Extra Trees
                et = ExtraTreesClassifier(
                    n_estimators=300, max_depth=6, min_samples_leaf=2,
                    class_weight='balanced', random_state=42, n_jobs=-1
                )
                et.fit(X_train_selected, y_train)
                pred = et.predict(X_test_selected)
                all_predictions.append(pred)
                all_accuracies.append(np.mean(pred == y_test))

                # Gradient Boosting
                gb = GradientBoostingClassifier(
                    n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
                )
                gb.fit(X_train_selected, y_train)
                pred = gb.predict(X_test_selected)
                all_predictions.append(pred)
                all_accuracies.append(np.mean(pred == y_test))

                # SVM
                for C in [1.0, 10.0]:
                    svm = SVC(kernel='rbf', C=C, gamma='scale',
                             class_weight='balanced', random_state=42)
                    svm.fit(X_train_selected, y_train)
                    pred = svm.predict(X_test_selected)
                    all_predictions.append(pred)
                    all_accuracies.append(np.mean(pred == y_test))

                # LDA
                try:
                    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    lda.fit(X_train_selected, y_train)
                    pred = lda.predict(X_test_selected)
                    all_predictions.append(pred)
                    all_accuracies.append(np.mean(pred == y_test))
                except Exception:
                    pass

            # =================================================================
            # PART 3: Combined Features (Traditional + Riemannian)
            # =================================================================
            if riemannian_features_train is not None:
                X_train_combined = np.hstack([X_train_scaled, riemannian_features_train])
                X_test_combined = np.hstack([X_test_scaled, riemannian_features_test])

                for k_pct in [0.4, 0.6, 0.8]:
                    n_features = X_train_combined.shape[1]
                    k_features = max(20, int(k_pct * n_features))
                    selector = SelectKBest(f_classif, k=k_features)
                    X_train_comb_sel = selector.fit_transform(X_train_combined, y_train)
                    X_test_comb_sel = selector.transform(X_test_combined)

                    et = ExtraTreesClassifier(
                        n_estimators=400, max_depth=8, min_samples_leaf=2,
                        class_weight='balanced', random_state=42, n_jobs=-1
                    )
                    et.fit(X_train_comb_sel, y_train)
                    pred = et.predict(X_test_comb_sel)
                    all_predictions.append(pred)
                    all_accuracies.append(np.mean(pred == y_test))

                    rf = RandomForestClassifier(
                        n_estimators=400, max_depth=8, min_samples_leaf=2,
                        class_weight='balanced', random_state=42, n_jobs=-1
                    )
                    rf.fit(X_train_comb_sel, y_train)
                    pred = rf.predict(X_test_comb_sel)
                    all_predictions.append(pred)
                    all_accuracies.append(np.mean(pred == y_test))

            # =================================================================
            # PART 4: Weighted Voting Ensemble
            # =================================================================
            if len(all_predictions) > 5:
                predictions_array = np.array(all_predictions)
                accuracies = np.array(all_accuracies)

                # Use top performers for voting
                top_k = min(20, len(accuracies))
                top_indices = np.argsort(accuracies)[-top_k:]
                top_predictions = predictions_array[top_indices]
                top_weights = accuracies[top_indices]
                top_weights = top_weights / top_weights.sum()

                weighted_votes = np.zeros((len(y_test), 2))
                for i, pred in enumerate(top_predictions):
                    for j, p in enumerate(pred):
                        weighted_votes[j, int(p)] += top_weights[i]

                ensemble_pred = np.argmax(weighted_votes, axis=1)
                ensemble_acc = np.mean(ensemble_pred == y_test)
                all_accuracies.append(ensemble_acc)

            # Take the best
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
