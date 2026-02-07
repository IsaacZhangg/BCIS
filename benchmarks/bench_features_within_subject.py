"""Within-subject classification benchmark for original vs fast features.

This more closely matches the pipeline's evaluation approach (within-subject CV)
which is how the 92.8% accuracy was achieved.
"""

import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_recording, get_complete_recordings, CHANNELS
from src.preprocess import preprocess_eeg
from src.features import extract_realtime_features
from src.features_fast import (
    extract_realtime_features_fast,
    extract_realtime_features_fast_v2,
)


def main():
    data_dir = Path("unicorn-data")
    recordings = get_complete_recordings(data_dir)
    sfreq = 250.0
    window_samples = int(5.0 * sfreq)

    funcs = {
        "Original (169)": extract_realtime_features,
        "Fast v1 (128)": extract_realtime_features_fast,
        "Fast v2 (208)": extract_realtime_features_fast_v2,
    }

    all_scores = {name: [] for name in funcs}

    for recording_path in recordings[:5]:  # First 5 subjects
        subject_id = recording_path.parent.parent.name
        data, events, sr = load_recording(recording_path)

        processed = {ch: preprocess_eeg(data[i], sr) for i, ch in enumerate(CHANNELS)}

        total_samples = len(processed[CHANNELS[0]])

        windows = []
        labels = []
        for sample_idx, phase, _ in events:
            if phase not in (3, 5):
                continue
            if sample_idx + window_samples > total_samples:
                continue
            windows.append(
                {
                    ch: processed[ch][sample_idx : sample_idx + window_samples]
                    for ch in CHANNELS
                }
            )
            labels.append(1 if phase == 3 else 0)

        labels = np.array(labels)
        if len(labels) < 10:
            continue

        rng = np.random.RandomState(42)
        idx_0 = np.where(labels == 0)[0]
        idx_1 = np.where(labels == 1)[0]
        n_min = min(len(idx_0), len(idx_1))
        idx_0 = rng.choice(idx_0, n_min, replace=False)
        idx_1 = rng.choice(idx_1, n_min, replace=False)
        sel = np.sort(np.concatenate([idx_0, idx_1]))
        windows = [windows[i] for i in sel]
        labels = labels[sel]

        print(
            f"\n{subject_id} ({len(labels)} samples, {np.sum(labels == 1)} engaged, {np.sum(labels == 0)} disengaged):"
        )

        for name, func in funcs.items():
            features = np.array([func(w, sfreq) for w in windows])
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

            n_folds = min(5, n_min)
            if n_folds < 2:
                continue

            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_accs = []
            for train_idx, test_idx in cv.split(features, labels):
                scaler = StandardScaler()
                X_train = scaler.fit_transform(features[train_idx])
                X_test = scaler.transform(features[test_idx])
                rf = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                )
                rf.fit(X_train, labels[train_idx])
                fold_accs.append(rf.score(X_test, labels[test_idx]))

            mean_acc = np.mean(fold_accs)
            all_scores[name].append(mean_acc)
            print(f"  {name}: {mean_acc:.1%}")

    print("\n" + "=" * 60)
    print("Within-Subject Mean Accuracy Across Subjects")
    print("=" * 60)
    for name, scores in all_scores.items():
        if scores:
            print(f"  {name}: {np.mean(scores):.1%} +/- {np.std(scores):.1%}")


if __name__ == "__main__":
    main()
