"""Model zoo for FER2013.

The models are intentionally ordered from tiny to large so the README can tell
a story: we start small (and underfit), grow capacity (and start to overfit),
then add regularization (BatchNorm + Dropout + augmentation) and finally try
transfer learning. Every model is registered in ``MODEL_REGISTRY`` so
``train.py`` can build any of them by name from a config file.

Input to every model: a (B, 1, 48, 48) normalized grayscale tensor.
Output: (B, 7) raw logits.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

from .data import NUM_CLASSES, IMG_SIZE


# --------------------------------------------------------------------------- #
# 0. Logistic regression / linear baseline                                    #
#    The simplest possible classifier — a single linear layer. Establishes the
#    floor: anything that can't beat this is broken. Expected to badly underfit.
# --------------------------------------------------------------------------- #
class LinearClassifier(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(IMG_SIZE * IMG_SIZE, num_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


# --------------------------------------------------------------------------- #
# 1. TinyCNN — 2 conv blocks. Our first real CNN. Low capacity: it should      #
#    learn meaningful features but plateau well below the deeper models.       #
# --------------------------------------------------------------------------- #
class TinyCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 48 -> 24
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 24 -> 12
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# 2. DeeperCNN — 4 conv blocks, NO regularization (no BN, no dropout, no aug). #
#    Built specifically to DEMONSTRATE OVERFITTING: high capacity + no         #
#    regularization should drive training accuracy far above validation.      #
# --------------------------------------------------------------------------- #
class DeeperCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 48 -> 24
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 24 -> 12
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                       # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512), nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _conv_bn_block(in_c, out_c, n_convs=2, dropout=0.0):
    layers = []
    c = in_c
    for _ in range(n_convs):
        layers += [
            nn.Conv2d(c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        c = out_c
    layers.append(nn.MaxPool2d(2))
    if dropout > 0:
        layers.append(nn.Dropout2d(dropout))
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------- #
# 3. RegularizedCNN — same depth as DeeperCNN but with BatchNorm + Dropout.    #
#    Paired with data augmentation in the config, this is our "fix the         #
#    overfitting" model and the strongest from-scratch architecture.          #
# --------------------------------------------------------------------------- #
class RegularizedCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            _conv_bn_block(1, 64, 2, dropout=0.25),     # 48 -> 24
            _conv_bn_block(64, 128, 2, dropout=0.25),   # 24 -> 12
            _conv_bn_block(128, 256, 2, dropout=0.30),  # 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------------------------------- #
# 4. ResNet18 (transfer learning). torchvision ResNet adapted to 1-channel     #
#    48x48 input. We can optionally load ImageNet weights. Tests whether a     #
#    modern residual architecture beats our hand-built CNNs.                   #
# --------------------------------------------------------------------------- #
class ResNet18FER(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        # Adapt the stem for 1-channel 48x48 inputs: a 3x3 stride-1 conv keeps
        # spatial resolution (the default 7x7 stride-2 conv + maxpool throws away
        # too much detail on tiny faces).
        net.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {
    "linear": LinearClassifier,
    "tiny_cnn": TinyCNN,
    "deeper_cnn": DeeperCNN,
    "regularized_cnn": RegularizedCNN,
    "resnet18": ResNet18FER,
}


def build_model(name: str, **kwargs) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
