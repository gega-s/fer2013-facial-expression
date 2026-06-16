"""Utilities: reproducibility, metrics, and plotting helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Make a run reproducible across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, class_names, normalize: bool = True):
    """Return a matplotlib Figure for the confusion matrix (for wandb logging)."""
    import matplotlib.pyplot as plt

    mat = cm.astype(np.float64)
    if normalize:
        mat = mat / np.maximum(mat.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    thresh = mat.max() / 2.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            txt = f"{val:.2f}" if normalize else f"{int(val)}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    return cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
