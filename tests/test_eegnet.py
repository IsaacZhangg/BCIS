"""Smoke tests for src/eegnet.py — runs forward+train only if torch is available."""

import numpy as np
import pytest

from src.eegnet import (
    EEGNET_N_CHANNELS,
    EEGNET_N_SAMPLES,
    TORCH_AVAILABLE,
    EEGNet,
    predict_eegnet,
    train_eegnet,
)


def test_module_imports_without_torch():
    """TORCH_AVAILABLE should be a bool regardless of whether torch is installed."""
    assert isinstance(TORCH_AVAILABLE, bool)


def test_stub_raises_when_torch_missing():
    if TORCH_AVAILABLE:
        pytest.skip("torch is available — stub path is not exercised")
    with pytest.raises(RuntimeError, match="torch"):
        EEGNet()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_forward_shape():
    import torch

    model = EEGNet()
    x = torch.zeros(2, 1, EEGNET_N_CHANNELS, EEGNET_N_SAMPLES)
    out = model(x)
    assert out.shape == (2, 2)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_train_predict_dummy():
    rng = np.random.default_rng(0)
    n_train, n_val = 16, 8
    X_train = rng.normal(size=(n_train, EEGNET_N_CHANNELS, EEGNET_N_SAMPLES)).astype(
        "float32"
    )
    y_train = rng.integers(0, 2, size=n_train)
    X_val = rng.normal(size=(n_val, EEGNET_N_CHANNELS, EEGNET_N_SAMPLES)).astype(
        "float32"
    )
    y_val = rng.integers(0, 2, size=n_val)

    model = EEGNet()
    trained = train_eegnet(
        model, X_train, y_train, X_val, y_val, max_epochs=3, patience=2
    )
    labels, probs = predict_eegnet(trained, X_val)
    assert labels.shape == (n_val,)
    assert probs.shape == (n_val, 2)
    assert set(np.unique(labels)).issubset({0, 1})
