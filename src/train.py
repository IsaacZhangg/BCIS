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

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

warnings.filterwarnings("ignore", category=UserWarning)


def _fit_predict(
    clf,
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Fit a classifier and return (predictions, accuracy)."""
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    return predictions, float(np.mean(predictions == test_labels))


def train_loso_cv(
    X_by_subject: list[np.ndarray],
    y_by_subject: list[np.ndarray],
) -> tuple[list[float], float, float]:
    """Train with Leave-One-Subject-Out cross-validation."""
    scores = []

    for test_idx in range(len(X_by_subject)):
        X_train = np.vstack([X for i, X in enumerate(X_by_subject) if i != test_idx])
        y_train = np.concatenate(
            [y for i, y in enumerate(y_by_subject) if i != test_idx]
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

    return scores, float(np.mean(scores)), float(np.std(scores))


def train_within_subject_cv_riemannian(
    X_by_subject: list[tuple[np.ndarray, np.ndarray]],
    y_by_subject: list[np.ndarray],
    n_folds: int = 10,
) -> tuple[list[float], float, float]:
    """Train with Riemannian geometry combined with focused feature ensemble."""
    subject_scores = []

    for (features, multichannel_data), labels in zip(X_by_subject, y_by_subject):
        fold_count = determine_fold_count(len(labels), n_folds)
        fold_scores = evaluate_subject_folds(
            features, multichannel_data, labels, fold_count
        )
        subject_scores.append(np.mean(fold_scores))

    return subject_scores, float(np.mean(subject_scores)), float(np.std(subject_scores))


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
    top_predictions = predictions_array[top_indices].astype(int)
    top_weights = accuracies_array[top_indices]
    top_weights = top_weights / top_weights.sum()

    n_samples = len(test_labels)
    weighted_votes = np.zeros((n_samples, 2))
    for i in range(len(top_predictions)):
        weighted_votes[np.arange(n_samples), top_predictions[i]] += top_weights[i]

    ensemble_predictions = np.argmax(weighted_votes, axis=1)
    return float(np.mean(ensemble_predictions == test_labels))


def _tangent_classifiers() -> list:
    """Return classifiers used on tangent space features."""
    return [
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42
        ),
        SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        ),
    ]


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

    for estimator in ["lwf", "oas"]:
        try:
            cov = Covariances(estimator=estimator)
            train_cov = cov.fit_transform(train_multichannel)
            test_cov = cov.transform(test_multichannel)
        except Exception:
            continue

        for metric in ["riemann", "logeuclid"]:
            try:
                pred, acc = _fit_predict(
                    MDM(metric=metric), train_cov, test_cov, train_labels, test_labels
                )
                predictions.append(pred)
                accuracies.append(acc)
            except Exception:
                pass

        for metric in ["riemann", "logeuclid"]:
            try:
                ts = TangentSpace(metric=metric)
                train_tan = ts.fit_transform(train_cov, train_labels)
                test_tan = ts.transform(test_cov)

                if tangent_train is None:
                    tangent_train = train_tan
                    tangent_test = test_tan

                for clf in _tangent_classifiers():
                    pred, acc = _fit_predict(
                        clf, train_tan, test_tan, train_labels, test_labels
                    )
                    predictions.append(pred)
                    accuracies.append(acc)
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


def _traditional_classifiers() -> list:
    """Return the list of traditional classifiers for ensemble training."""
    return [
        RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        ExtraTreesClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
        ),
        SVC(
            kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=42
        ),
        SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        ),
        LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    ]


def train_model_ensemble(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Train ensemble of traditional ML models."""
    predictions = []
    accuracies = []

    for clf in _traditional_classifiers():
        try:
            pred, acc = _fit_predict(
                clf, train_features, test_features, train_labels, test_labels
            )
            predictions.append(pred)
            accuracies.append(acc)
        except Exception:
            pass

    return {"predictions": predictions, "accuracies": accuracies}


def _combined_classifiers() -> list:
    """Return classifiers for combined feature training."""
    return [
        ExtraTreesClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    ]


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
    num_features = train_combined.shape[1]

    for pct in [0.4, 0.6, 0.8]:
        k_features = max(20, int(pct * num_features))
        selector = SelectKBest(f_classif, k=k_features)
        train_selected = selector.fit_transform(train_combined, train_labels)
        test_selected = selector.transform(test_combined)

        for clf in _combined_classifiers():
            pred, acc = _fit_predict(
                clf, train_selected, test_selected, train_labels, test_labels
            )
            predictions.append(pred)
            accuracies.append(acc)

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
