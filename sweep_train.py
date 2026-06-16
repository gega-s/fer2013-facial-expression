"""Entrypoint used by ``wandb agent`` for hyperparameter sweeps.

wandb injects the sampled hyperparameters into ``wandb.config``; we map them
onto a ``TrainConfig`` and run a single experiment. ``dropout`` is routed into
``model_kwargs`` because it is a model constructor argument.
"""
from __future__ import annotations

import os

import wandb

from src.engine import TrainConfig, train_model


def main():
    run = wandb.init()
    c = dict(wandb.config)

    model_kwargs = {}
    if "dropout" in c:
        model_kwargs["dropout"] = c.pop("dropout")

    cfg = TrainConfig(
        model=c.get("model", "regularized_cnn"),
        data_dir=os.environ.get("FER_DATA_DIR", "data"),
        batch_size=c.get("batch_size", 128),
        augment=c.get("augment", True),
        epochs=c.get("epochs", 40),
        lr=c.get("lr", 1e-3),
        weight_decay=c.get("weight_decay", 0.0),
        optimizer=c.get("optimizer", "adam"),
        scheduler=c.get("scheduler", "none"),
        label_smoothing=c.get("label_smoothing", 0.0),
        model_kwargs=model_kwargs,
        group="sweep_regularized_cnn",
        use_wandb=True,
    )
    # Reuse the already-initialized sweep run instead of starting a new one.
    train_model(cfg)


if __name__ == "__main__":
    main()
