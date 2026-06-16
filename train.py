"""Config-driven training entrypoint.

Each architecture / experiment is described by a YAML file in ``configs/``.
Running one config produces exactly one wandb run:

    python train.py --config configs/02_deeper_cnn.yaml
    python train.py --config configs/03_regularized_cnn.yaml --epochs 60

Any field in TrainConfig can be overridden from the command line.
"""
from __future__ import annotations

import argparse
import dataclasses

import yaml

from src.engine import TrainConfig, train_model


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args():
    p = argparse.ArgumentParser(description="Train a FER2013 model (one wandb run).")
    p.add_argument("--config", type=str, required=True, help="Path to a YAML config.")
    # Common overrides (everything else can be set in the YAML).
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--entity", type=str, default=None)
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_dict = load_config(args.config)

    # CLI overrides win over the YAML.
    overrides = {k: v for k, v in {
        "data_dir": args.data_dir,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "project": args.project,
        "entity": args.entity,
        "run_name": args.run_name,
    }.items() if v is not None}
    cfg_dict.update(overrides)
    if args.no_wandb:
        cfg_dict["use_wandb"] = False

    valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = set(cfg_dict) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    cfg = TrainConfig(**cfg_dict)
    print("Config:", cfg)
    train_model(cfg)


if __name__ == "__main__":
    main()
