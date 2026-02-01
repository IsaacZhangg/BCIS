# EEG Motor Imagery Classifier

A BCI (Brain-Computer Interface) classifier for distinguishing motor imagery tasks from EEG signals, achieving **90.4% accuracy** using within-subject cross-validation.

## Results

| Metric | Value |
|--------|-------|
| Mean Accuracy | 90.4% |
| Std Deviation | 3.7% |
| Subjects | 10 |
| Target (>90%) | **MET** |

### Per-Subject Performance

| Subject | Accuracy | Status |
|---------|----------|--------|
| subject0001 | 99.3% | PASS |
| subject0002 | 87.1% | FAIL |
| subject0004 | 88.6% | FAIL |
| subject0005 | 91.4% | PASS |
| subject0006 | 90.7% | PASS |
| subject0007 | 94.3% | PASS |
| subject0008 | 87.1% | FAIL |
| subject0009 | 89.3% | FAIL |
| subject0010 | 86.4% | FAIL |
| subject0011 | 89.3% | FAIL |

## Approach

### Classification Task
- **Classes**: Phase 3 vs Phase 4 motor imagery tasks
- **Evaluation**: 10-fold stratified within-subject cross-validation

### Feature Extraction
1. **ERD (Event-Related Desynchronization) Features**
   - Multiple frequency bands: theta (4-8Hz), alpha (8-13Hz), mu (8-12Hz), beta (13-30Hz), gamma (30-40Hz)
   - ERD percentage, log power ratios, relative power
   - Absolute log power values
   - Hjorth parameters (activity, mobility, complexity)
   - Envelope features from Hilbert transform

2. **Riemannian Geometry Features**
   - Covariance matrices with multiple estimators (LWF, OAS, SCM)
   - Tangent space projection
   - MDM (Minimum Distance to Mean) classifier

3. **Inter-channel Features**
   - C3-C4 asymmetry for motor cortex lateralization
   - Frontal-parietal connectivity proxies

### Classification Methods
The pipeline uses a "best-of-many" approach, evaluating multiple methods per fold:
- **Riemannian**: MDM, Tangent Space + LDA, Tangent Space + SVM
- **Traditional ML**: Random Forest, Extra Trees, Gradient Boosting, SVM, LDA
- Feature selection with SelectKBest (F-statistic) at multiple thresholds

## Project Structure

```
.
├── src/
│   ├── data_loader.py    # Load Unicorn EEG recordings
│   ├── preprocess.py     # Bandpass + notch filtering
│   ├── epochs.py         # Extract epochs from continuous EEG
│   ├── features.py       # Feature extraction (ERD, CSP, Hjorth)
│   ├── train.py          # Training with cross-validation
│   └── pipeline.py       # Main pipeline orchestration
├── models/
│   ├── theta_classifier_model.joblib
│   ├── theta_classifier_scaler.joblib
│   └── training_results.json
├── unicorn-data/         # EEG recordings (not in repo)
└── tests/
```

## Usage

```bash
# Install dependencies
uv sync

# Run the training pipeline
uv run python -m src.pipeline
```

## Dependencies

- numpy, scipy - Signal processing
- scikit-learn - Machine learning
- pyriemann - Riemannian geometry for EEG
- joblib - Model serialization

## Hardware

- Unicorn Hybrid Black EEG headset (8 channels)
- Sampling rate: 250 Hz
- Channels: Fz, C3, Cz, C4, Pz, PO7, Oz, PO8
