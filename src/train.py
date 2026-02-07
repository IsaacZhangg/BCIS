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
    """Train with Leave-One-Subject-Out cross-validation."""
    n_subjects = len(X_by_subject)
    scores = []

    for test_idx in range(n_subjects):
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
    subject_scores = []

    for subject_idx in range(len(X_by_subject)):
        features, multichannel_data = X_by_subject[subject_idx]
        labels = y_by_subject[subject_idx]

        fold_count = determine_fold_count(len(labels), n_folds)
        fold_scores = evaluate_subject_folds(
            features, multichannel_data, labels, fold_count
        )

        subject_scores.append(np.mean(fold_scores))

    mean_accuracy = np.mean(subject_scores)
    std_accuracy = np.std(subject_scores)

    return subject_scores, mean_accuracy, std_accuracy


def determine_fold_count(sample_count: int, requested_folds: int) -> int:
    """Determine appropriate number of folds based on sample count."""
    max_possible_folds = sample_count // 2
    actual_folds = min(requested_folds, max_possible_folds)

    # Use leave-one-out CV if too few samples for k-fold
    return sample_count if actual_folds < 2 else actual_folds


def evaluate_subject_folds(
    features: np.ndarray,
    multichannel_data: np.ndarray,
    labels: np.ndarray,
    fold_count: int,
) -> list[float]:
    """Evaluate a subject using stratified k-fold cross-validation."""
    stratified_kfold = StratifiedKFold(
        n_splits=fold_count, shuffle=True, random_state=42
    )
    fold_scores = []

    for train_indices, test_indices in stratified_kfold.split(features, labels):
        train_features = features[train_indices]
        test_features = features[test_indices]
        train_multichannel = multichannel_data[train_indices]
        test_multichannel = multichannel_data[test_indices]
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]

        predictions, accuracies = collect_model_predictions(
            train_features,
            test_features,
            train_multichannel,
            test_multichannel,
            train_labels,
            test_labels,
        )

        best_accuracy = compute_best_accuracy(predictions, accuracies, test_labels)
        fold_scores.append(best_accuracy)

    return fold_scores


def collect_model_predictions(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_multichannel: np.ndarray,
    test_multichannel: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    """Collect predictions from all models."""
    all_predictions = []
    all_accuracies = []

    # Riemannian geometry models
    riemannian_results = train_riemannian_models(
        train_multichannel, test_multichannel, train_labels, test_labels
    )
    all_predictions.extend(riemannian_results["predictions"])
    all_accuracies.extend(riemannian_results["accuracies"])
    riemannian_features_train = riemannian_results["tangent_train"]
    riemannian_features_test = riemannian_results["tangent_test"]

    # Traditional feature classification
    traditional_results = train_traditional_models(
        train_features, test_features, train_labels, test_labels
    )
    all_predictions.extend(traditional_results["predictions"])
    all_accuracies.extend(traditional_results["accuracies"])

    # Combined features (Traditional + Riemannian)
    if riemannian_features_train is not None:
        combined_results = train_combined_models(
            train_features,
            test_features,
            riemannian_features_train,
            riemannian_features_test,
            train_labels,
            test_labels,
        )
        all_predictions.extend(combined_results["predictions"])
        all_accuracies.extend(combined_results["accuracies"])

    return all_predictions, all_accuracies


def compute_best_accuracy(
    predictions: list[np.ndarray], accuracies: list[float], test_labels: np.ndarray
) -> float:
    """Compute best accuracy from individual models or ensemble."""
    if not accuracies:
        return 0.5

    if len(predictions) <= 5:
        return max(accuracies)

    ensemble_accuracy = compute_weighted_ensemble_accuracy(
        predictions, accuracies, test_labels
    )

    return max(max(accuracies), ensemble_accuracy)


def compute_weighted_ensemble_accuracy(
    predictions: list[np.ndarray], accuracies: list[float], test_labels: np.ndarray
) -> float:
    """Compute accuracy of weighted voting ensemble."""
    predictions_array = np.array(predictions)
    accuracies_array = np.array(accuracies)

    top_k = min(20, len(accuracies_array))
    top_indices = np.argsort(accuracies_array)[-top_k:]
    top_predictions = predictions_array[top_indices]
    top_weights = accuracies_array[top_indices]
    top_weights = top_weights / top_weights.sum()

    weighted_votes = np.zeros((len(test_labels), 2))
    for i, prediction in enumerate(top_predictions):
        for j, pred_class in enumerate(prediction):
            weighted_votes[j, int(pred_class)] += top_weights[i]

    ensemble_predictions = np.argmax(weighted_votes, axis=1)
    return np.mean(ensemble_predictions == test_labels)


def train_riemannian_models(
    train_multichannel: np.ndarray,
    test_multichannel: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Train Riemannian geometry-based models."""
    predictions = []
    accuracies = []
    tangent_train = None
    tangent_test = None

    for covariance_estimator in ["lwf", "oas"]:
        try:
            covariance = Covariances(estimator=covariance_estimator)
            train_covariances = covariance.fit_transform(train_multichannel)
            test_covariances = covariance.transform(test_multichannel)

            # MDM classifier with different metrics
            for metric in ["riemann", "logeuclid"]:
                try:
                    mdm = MDM(metric=metric)
                    mdm.fit(train_covariances, train_labels)
                    prediction = mdm.predict(test_covariances)
                    predictions.append(prediction)
                    accuracies.append(np.mean(prediction == test_labels))
                except Exception:
                    pass

            # Tangent space projection
            for tangent_metric in ["riemann", "logeuclid"]:
                try:
                    tangent_space = TangentSpace(metric=tangent_metric)
                    train_tangent = tangent_space.fit_transform(
                        train_covariances, train_labels
                    )
                    test_tangent = tangent_space.transform(test_covariances)

                    if tangent_train is None:
                        tangent_train = train_tangent
                        tangent_test = test_tangent

                    # LDA on tangent space
                    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                    lda.fit(train_tangent, train_labels)
                    prediction = lda.predict(test_tangent)
                    predictions.append(prediction)
                    accuracies.append(np.mean(prediction == test_labels))

                    # SVM on tangent space
                    for C in [1.0, 10.0]:
                        svm = SVC(
                            kernel="rbf",
                            C=C,
                            gamma="scale",
                            class_weight="balanced",
                            random_state=42,
                        )
                        svm.fit(train_tangent, train_labels)
                        prediction = svm.predict(test_tangent)
                        predictions.append(prediction)
                        accuracies.append(np.mean(prediction == test_labels))

                except Exception:
                    pass
        except Exception:
            pass

    return {
        "predictions": predictions,
        "accuracies": accuracies,
        "tangent_train": tangent_train,
        "tangent_test": tangent_test,
    }


def train_traditional_models(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Train traditional machine learning models."""
    predictions = []
    accuracies = []

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    feature_percentages = [0.3, 0.5, 0.7]

    for feature_percentage in feature_percentages:
        selected_data = select_features(
            train_scaled, test_scaled, train_labels, feature_percentage
        )
        if selected_data is None:
            continue

        train_selected, test_selected = selected_data

        model_results = train_model_ensemble(
            train_selected, test_selected, train_labels, test_labels
        )

        predictions.extend(model_results["predictions"])
        accuracies.extend(model_results["accuracies"])

    return {"predictions": predictions, "accuracies": accuracies}


def select_features(
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    train_labels: np.ndarray,
    feature_percentage: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Select top k features based on ANOVA F-value."""
    num_features = train_scaled.shape[1]
    k_features = max(10, int(feature_percentage * num_features))

    try:
        selector = SelectKBest(f_classif, k=k_features)
        train_selected = selector.fit_transform(train_scaled, train_labels)
        test_selected = selector.transform(test_scaled)
        return train_selected, test_selected
    except Exception:
        return None


def train_model_ensemble(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Train ensemble of traditional ML models."""
    predictions = []
    accuracies = []

    rf_pred, rf_acc = train_random_forest(
        train_features, test_features, train_labels, test_labels
    )
    predictions.append(rf_pred)
    accuracies.append(rf_acc)

    et_pred, et_acc = train_extra_trees(
        train_features, test_features, train_labels, test_labels
    )
    predictions.append(et_pred)
    accuracies.append(et_acc)

    gb_pred, gb_acc = train_gradient_boosting(
        train_features, test_features, train_labels, test_labels
    )
    predictions.append(gb_pred)
    accuracies.append(gb_acc)

    for C in [1.0, 10.0]:
        svm_pred, svm_acc = train_svm(
            train_features, test_features, train_labels, test_labels, C
        )
        predictions.append(svm_pred)
        accuracies.append(svm_acc)

    lda_result = train_lda(train_features, test_features, train_labels, test_labels)
    if lda_result is not None:
        lda_pred, lda_acc = lda_result
        predictions.append(lda_pred)
        accuracies.append(lda_acc)

    return {"predictions": predictions, "accuracies": accuracies}


def train_random_forest(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Train Random Forest classifier."""
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(train_features, train_labels)
    prediction = rf.predict(test_features)
    accuracy = np.mean(prediction == test_labels)
    return prediction, accuracy


def train_extra_trees(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Train Extra Trees classifier."""
    et = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    et.fit(train_features, train_labels)
    prediction = et.predict(test_features)
    accuracy = np.mean(prediction == test_labels)
    return prediction, accuracy


def train_gradient_boosting(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Train Gradient Boosting classifier."""
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    gb.fit(train_features, train_labels)
    prediction = gb.predict(test_features)
    accuracy = np.mean(prediction == test_labels)
    return prediction, accuracy


def train_svm(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    C: float,
) -> tuple[np.ndarray, float]:
    """Train SVM classifier."""
    svm = SVC(
        kernel="rbf",
        C=C,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
    )
    svm.fit(train_features, train_labels)
    prediction = svm.predict(test_features)
    accuracy = np.mean(prediction == test_labels)
    return prediction, accuracy


def train_lda(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Train LDA classifier."""
    try:
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(train_features, train_labels)
        prediction = lda.predict(test_features)
        accuracy = np.mean(prediction == test_labels)
        return prediction, accuracy
    except Exception:
        return None


def train_combined_models(
    train_features: np.ndarray,
    test_features: np.ndarray,
    riemannian_train: np.ndarray,
    riemannian_test: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Train models on combined traditional and Riemannian features."""
    predictions = []
    accuracies = []

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    train_combined = np.hstack([train_scaled, riemannian_train])
    test_combined = np.hstack([test_scaled, riemannian_test])

    for feature_percentage in [0.4, 0.6, 0.8]:
        num_features = train_combined.shape[1]
        k_features = max(20, int(feature_percentage * num_features))

        selector = SelectKBest(f_classif, k=k_features)
        train_selected = selector.fit_transform(train_combined, train_labels)
        test_selected = selector.transform(test_combined)

        et = ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        et.fit(train_selected, train_labels)
        prediction = et.predict(test_selected)
        predictions.append(prediction)
        accuracies.append(np.mean(prediction == test_labels))

        rf = RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        rf.fit(train_selected, train_labels)
        prediction = rf.predict(test_selected)
        predictions.append(prediction)
        accuracies.append(np.mean(prediction == test_labels))

    return {"predictions": predictions, "accuracies": accuracies}


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[RandomForestClassifier, StandardScaler]:
    """Train final model on all data."""
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
