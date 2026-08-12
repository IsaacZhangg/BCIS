"""EEGNet-8-2 PyTorch classifier for left/right motor-imagery MI.

Graceful degradation: if torch is not installed, module imports cleanly with
``TORCH_AVAILABLE = False``; the torch-dependent class and helpers are skipped.
Callers should guard usage with ``if TORCH_AVAILABLE:`` before invoking
training, fine-tuning, or inference.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    TensorDataset = None  # type: ignore[assignment]


# Architecture defaults for 8 channels, 250 Hz, 3.0 s trials (750 samples).
EEGNET_N_CHANNELS = 8
EEGNET_N_SAMPLES = 750
EEGNET_N_CLASSES = 2
EEGNET_F1 = 8  # temporal filters
EEGNET_D = 2  # depth multiplier
EEGNET_F2 = 16  # F1 * D, separable-conv filters
EEGNET_KERNEL_TEMPORAL = 64  # ~0.25 s at 250 Hz
EEGNET_KERNEL_SEPARABLE = 16


if not TORCH_AVAILABLE:

    class EEGNet:  # type: ignore[no-redef]
        """Stub EEGNet used when torch is not installed.

        Instantiation raises ``RuntimeError`` — callers should guard with
        ``TORCH_AVAILABLE`` and skip EEGNet when it is ``False``.
        """

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "EEGNet requires PyTorch — install with `uv add torch` or guard "
                "with TORCH_AVAILABLE before instantiating."
            )

else:

    class EEGNet(nn.Module):
        """EEGNet-8-2 for binary MI classification.

        Input  : (batch, 1, n_channels, n_samples)
        Output : (batch, n_classes) logits
        """

        def __init__(
            self,
            n_channels: int = EEGNET_N_CHANNELS,
            n_samples: int = EEGNET_N_SAMPLES,
            n_classes: int = EEGNET_N_CLASSES,
            f1: int = EEGNET_F1,
            d: int = EEGNET_D,
            f2: int = EEGNET_F2,
            kernel_temporal: int = EEGNET_KERNEL_TEMPORAL,
            kernel_separable: int = EEGNET_KERNEL_SEPARABLE,
            dropout: float = 0.5,
        ) -> None:
            super().__init__()
            self.n_channels = n_channels
            self.n_samples = n_samples

            # Block 1: temporal conv + depthwise spatial conv.
            self.temporal_conv = nn.Conv2d(
                1, f1, kernel_size=(1, kernel_temporal), padding="same", bias=False
            )
            self.bn_temporal = nn.BatchNorm2d(f1)
            self.depthwise_conv = nn.Conv2d(
                f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False
            )
            self.bn_depthwise = nn.BatchNorm2d(f1 * d)
            self.pool1 = nn.AvgPool2d((1, 4))
            self.dropout1 = nn.Dropout(dropout)

            # Block 2: separable conv = depthwise then pointwise.
            self.separable_depthwise = nn.Conv2d(
                f1 * d,
                f1 * d,
                kernel_size=(1, kernel_separable),
                groups=f1 * d,
                padding="same",
                bias=False,
            )
            self.separable_pointwise = nn.Conv2d(f1 * d, f2, kernel_size=1, bias=False)
            self.bn_separable = nn.BatchNorm2d(f2)
            self.pool2 = nn.AvgPool2d((1, 8))
            self.dropout2 = nn.Dropout(dropout)

            self.elu = nn.ELU()

            # Resolve flatten dim via a dry forward on zeros.
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_samples)
                feat_dim = self._forward_features(dummy).shape[1]
            self.classifier = nn.Linear(feat_dim, n_classes)

        def _forward_features(self, x):
            x = self.temporal_conv(x)
            x = self.bn_temporal(x)
            x = self.depthwise_conv(x)
            x = self.bn_depthwise(x)
            x = self.elu(x)
            x = self.pool1(x)
            x = self.dropout1(x)

            x = self.separable_depthwise(x)
            x = self.separable_pointwise(x)
            x = self.bn_separable(x)
            x = self.elu(x)
            x = self.pool2(x)
            x = self.dropout2(x)

            return x.flatten(start_dim=1)

        def forward(self, x):
            return self.classifier(self._forward_features(x))


def _as_tensor(X):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available")
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, None, :, :]
    return torch.from_numpy(arr)


def _labels_tensor(y):
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available")
    return torch.from_numpy(np.asarray(y, dtype=np.int64))


def train_eegnet(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 20,
    device: str | None = None,
    verbose: bool = False,
):
    """Train an EEGNet with Adam + CE loss and early stopping on val loss."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available — cannot train EEGNet")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Xt = _as_tensor(X_train).to(device)
    yt = _labels_tensor(y_train).to(device)
    Xv = _as_tensor(X_val).to(device)
    yv = _labels_tensor(y_val).to(device)

    loader = DataLoader(
        TensorDataset(Xt, yt),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.train(False)
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = float(loss_fn(val_logits, yv).item())

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and (epoch % 25 == 0 or epoch == max_epochs - 1):
            logger.info(
                "EEGNet epoch %d val_loss=%.4f best=%.4f",
                epoch,
                val_loss,
                best_val_loss,
            )

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def pretrain_eegnet(
    other_subjects_epochs: list[np.ndarray],
    other_subjects_labels: list[np.ndarray],
    val_fraction: float = 0.15,
    random_state: int = 42,
    **train_kwargs: Any,
):
    """Pool epochs across donor subjects and pretrain a single EEGNet."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available — cannot pretrain EEGNet")

    if not other_subjects_epochs:
        raise ValueError("pretrain_eegnet requires at least one donor subject")

    X = np.concatenate(other_subjects_epochs, axis=0)
    y = np.concatenate(other_subjects_labels, axis=0)

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(y))
    n_val = max(1, int(len(y) * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = EEGNet(n_channels=X.shape[1], n_samples=X.shape[2])
    return train_eegnet(
        model,
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        **train_kwargs,
    )


def finetune_eegnet(
    pretrained_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lr: float = 1e-4,
    **train_kwargs: Any,
):
    """Fine-tune a pretrained EEGNet on a single subject at a smaller LR."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available — cannot finetune EEGNet")
    train_kwargs.setdefault("weight_decay", 1e-3)
    return train_eegnet(
        pretrained_model,
        X_train,
        y_train,
        X_val,
        y_val,
        lr=lr,
        **train_kwargs,
    )


def predict_eegnet(model, X: np.ndarray, device: str | None = None):
    """Predict labels and class probabilities from a trained EEGNet."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("torch is not available — cannot run EEGNet inference")

    device = device or next(model.parameters()).device
    model.train(False)
    with torch.no_grad():
        logits = model(_as_tensor(X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    labels = np.argmax(probs, axis=1)
    return labels, probs
