"""FER2013 data loading and preprocessing.

The Kaggle competition "challenges-in-representation-learning-facial-expression-
recognition-challenge" ships the data as a single CSV (``fer2013.csv`` or
``icml_face_data.csv``) with three columns:

    emotion   int in [0, 6]
    pixels    space-separated string of 48*48 = 2304 grayscale values (0-255)
    Usage     one of {Training, PublicTest, PrivateTest}

We use ``Training`` as the train split, ``PublicTest`` as validation and
``PrivateTest`` as the held-out test split (this mirrors the original
competition's public/private leaderboard split, so our numbers are comparable
to historical results).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Canonical FER2013 label order.
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTIONS)
IMG_SIZE = 48

# Dataset-wide grayscale mean/std (computed on the training split, range [0,1]).
# These are stable, well-known constants for FER2013 and let us normalize
# consistently across every experiment.
FER_MEAN = 0.5077
FER_STD = 0.2550


def find_csv(data_dir: str) -> str:
    """Return the path to the FER2013 CSV inside ``data_dir``."""
    candidates = ["fer2013.csv", "icml_face_data.csv", "fer2013/fer2013.csv"]
    for name in candidates:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    # Fall back to the first CSV that has the expected columns.
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith(".csv"):
                path = os.path.join(root, f)
                try:
                    cols = pd.read_csv(path, nrows=1).columns
                    cols = [c.strip().lower() for c in cols]
                    if "pixels" in cols:
                        return path
                except Exception:
                    continue
    raise FileNotFoundError(
        f"Could not find a FER2013 csv in {data_dir}. "
        "Download it with the Kaggle API first (see scripts/download_data.sh)."
    )


def load_dataframe(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Some versions name the usage column differently.
    if "Usage" not in df.columns:
        for alt in ["usage", " Usage"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Usage"})
    if "pixels" not in df.columns and " pixels" in df.columns:
        df = df.rename(columns={" pixels": "pixels"})
    if "emotion" not in df.columns and " emotion" in df.columns:
        df = df.rename(columns={" emotion": "emotion"})
    return df


def _pixels_to_array(pixel_strings: pd.Series) -> np.ndarray:
    """Vectorized conversion of the pixel strings into a (N, 48, 48) uint8 array."""
    arr = np.array(
        [np.asarray(p.split(), dtype=np.uint8) for p in pixel_strings],
        dtype=np.uint8,
    )
    return arr.reshape(-1, IMG_SIZE, IMG_SIZE)


class FER2013Dataset(Dataset):
    """In-memory FER2013 dataset for one split."""

    def __init__(self, images: np.ndarray, labels: Optional[np.ndarray], transform=None):
        self.images = images  # (N, 48, 48) uint8
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        # PIL-free path: build a (1, 48, 48) float tensor and apply tensor transforms.
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        if self.labels is None:
            return img
        return img, int(self.labels[idx])


def build_transforms(augment: bool):
    """Training/eval transforms.

    Inputs are ``(48, 48)`` uint8 numpy arrays. ``ToPILImage`` lets us reuse the
    standard torchvision augmentations (flip, small rotation/translation) that
    are sensible for faces. Augmentation is the main lever we use to fight the
    overfitting we observe in the deeper models.
    """
    norm = T.Normalize(mean=[FER_MEAN], std=[FER_STD])
    if augment:
        return T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            norm,
        ])
    return T.Compose([
        T.ToPILImage(),
        T.ToTensor(),
        norm,
    ])


def get_datasets(data_dir: str, augment: bool = False):
    """Return (train_ds, val_ds, test_ds) using the official Usage splits."""
    df = load_dataframe(find_csv(data_dir))
    train_tf = build_transforms(augment)
    eval_tf = build_transforms(False)

    def subset(usage):
        rows = df[df["Usage"] == usage]
        images = _pixels_to_array(rows["pixels"])
        labels = rows["emotion"].to_numpy()
        return images, labels

    tr_x, tr_y = subset("Training")
    va_x, va_y = subset("PublicTest")
    te_x, te_y = subset("PrivateTest")

    train_ds = FER2013Dataset(tr_x, tr_y, train_tf)
    val_ds = FER2013Dataset(va_x, va_y, eval_tf)
    test_ds = FER2013Dataset(te_x, te_y, eval_tf)
    return train_ds, val_ds, test_ds


def get_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    augment: bool = False,
    num_workers: int = 2,
):
    train_ds, val_ds, test_ds = get_datasets(data_dir, augment=augment)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader


def compute_class_weights(data_dir: str) -> torch.Tensor:
    """Inverse-frequency class weights — FER2013 is imbalanced (Disgust is rare).

    Useful for the experiments where we want to study whether re-weighting the
    loss helps the minority classes.
    """
    df = load_dataframe(find_csv(data_dir))
    counts = df[df["Usage"] == "Training"]["emotion"].value_counts().sort_index()
    counts = counts.reindex(range(NUM_CLASSES), fill_value=0).to_numpy()
    weights = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)
