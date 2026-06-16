"""Training / evaluation loops and the wandb-integrated trainer.

``train_model`` runs one full experiment = one wandb run. Each architecture
gets its own run (mirroring the "one run per model" structure from the MLFlow
assignment), and every hyperparameter lives in ``wandb.config`` so runs are
comparable and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from .data import EMOTIONS, NUM_CLASSES
from .models import build_model, count_parameters
from .utils import (
    accuracy, confusion_matrix, per_class_accuracy,
    plot_confusion_matrix, set_seed, get_device,
)


@dataclass
class TrainConfig:
    # --- experiment identity ---
    model: str = "tiny_cnn"
    run_name: Optional[str] = None
    group: Optional[str] = None          # wandb group, e.g. "from_scratch"
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    model_kwargs: dict = field(default_factory=dict)

    # --- data ---
    data_dir: str = "data"
    batch_size: int = 128
    augment: bool = False
    num_workers: int = 2

    # --- optimization ---
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"              # adam | sgd | adamw
    momentum: float = 0.9                # for sgd
    scheduler: str = "none"              # none | cosine | step | plateau
    label_smoothing: float = 0.0
    class_weighted_loss: bool = False
    grad_clip: float = 0.0

    # --- regularization / misc ---
    seed: int = 42
    early_stop_patience: int = 0         # 0 = disabled

    # --- wandb ---
    use_wandb: bool = True
    project: str = "fer2013-fer-challenge"
    entity: Optional[str] = None


def build_optimizer(model, cfg: TrainConfig):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay, nesterov=True)
    raise ValueError(f"Unknown optimizer {cfg.optimizer}")


def build_scheduler(optimizer, cfg: TrainConfig):
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    if cfg.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1)
    if cfg.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    return None


def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=0.0):
    model.train()
    running_loss, running_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, y) * bs
        n += bs
    return running_loss / n, running_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_preds=False):
    model.eval()
    running_loss, running_acc, n = 0.0, 0.0, 0
    all_true, all_pred = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        running_loss += loss.item() * bs
        running_acc += accuracy(logits, y) * bs
        n += bs
        if collect_preds:
            all_true.append(y.cpu().numpy())
            all_pred.append(logits.argmax(1).cpu().numpy())
    out = {"loss": running_loss / n, "acc": running_acc / n}
    if collect_preds:
        out["y_true"] = np.concatenate(all_true)
        out["y_pred"] = np.concatenate(all_pred)
    return out


def train_model(cfg: TrainConfig, loaders=None):
    """Run one experiment. Returns a history dict. Logs everything to wandb."""
    from .data import get_dataloaders, compute_class_weights

    set_seed(cfg.seed)
    device = get_device()

    if loaders is None:
        loaders = get_dataloaders(cfg.data_dir, cfg.batch_size, cfg.augment, cfg.num_workers)
    train_loader, val_loader, test_loader = loaders

    model = build_model(cfg.model, **cfg.model_kwargs).to(device)
    n_params = count_parameters(model)

    weight = None
    if cfg.class_weighted_loss:
        weight = compute_class_weights(cfg.data_dir).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    run = None
    own_run = False  # did we create the run (vs. reusing an active sweep run)?
    if cfg.use_wandb:
        import wandb
        if wandb.run is not None:
            # Running under `wandb agent` (sweep): reuse the active run and just
            # record the resolved hyperparameters / param count.
            run = wandb.run
            wandb.config.update(
                {**asdict(cfg), "n_params": n_params, "device": str(device)},
                allow_val_change=True,
            )
        else:
            own_run = True
            run = wandb.init(
                project=cfg.project,
                entity=cfg.entity,
                name=cfg.run_name or cfg.model,
                group=cfg.group,
                notes=cfg.notes,
                tags=cfg.tags or [cfg.model],
                config={**asdict(cfg), "n_params": n_params, "device": str(device)},
                reinit=True,
            )
        wandb.watch(model, log="all", log_freq=200)  # log grads & weights histograms

    history = {k: [] for k in
               ["train_loss", "train_acc", "val_loss", "val_acc", "lr", "gen_gap"]}
    best_val_acc, best_state, epochs_no_improve = 0.0, None, 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, cfg.grad_clip)
        val = evaluate(model, val_loader, criterion, device)
        cur_lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            if cfg.scheduler == "plateau":
                scheduler.step(val["acc"])
            else:
                scheduler.step()

        gen_gap = tr_acc - val["acc"]   # overfitting signal
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["acc"])
        history["lr"].append(cur_lr)
        history["gen_gap"].append(gen_gap)

        if cfg.use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": tr_loss, "train/acc": tr_acc,
                "val/loss": val["loss"], "val/acc": val["acc"],
                "lr": cur_lr,
                "generalization_gap": gen_gap,   # train_acc - val_acc
            }, step=epoch)

        print(f"[{cfg.run_name or cfg.model}] epoch {epoch:02d}/{cfg.epochs} "
              f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
              f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f} gap={gen_gap:+.3f}")

        if val["acc"] > best_val_acc:
            best_val_acc = val["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if cfg.early_stop_patience and epochs_no_improve >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch} (no val improvement for "
                      f"{cfg.early_stop_patience} epochs).")
                break

    # Restore best weights and evaluate on the held-out (PrivateTest) split.
    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, test_loader, criterion, device, collect_preds=True)
    cm = confusion_matrix(test["y_true"], test["y_pred"], NUM_CLASSES)
    pca = per_class_accuracy(cm)

    print(f"\n>>> BEST val_acc={best_val_acc:.4f} | TEST acc={test['acc']:.4f}")

    if cfg.use_wandb:
        import wandb
        wandb.run.summary["best_val_acc"] = best_val_acc
        wandb.run.summary["test_acc"] = test["acc"]
        wandb.run.summary["test_loss"] = test["loss"]
        wandb.run.summary["n_params"] = n_params
        wandb.run.summary["final_gen_gap"] = history["gen_gap"][-1]
        for i, name in enumerate(EMOTIONS):
            wandb.run.summary[f"test_acc/{name}"] = float(pca[i])
        # Confusion matrix as an image + wandb's native plot.
        fig = plot_confusion_matrix(cm, EMOTIONS, normalize=True)
        wandb.log({"test/confusion_matrix": wandb.Image(fig)})
        wandb.log({"test/confusion_matrix_interactive": wandb.plot.confusion_matrix(
            y_true=test["y_true"].tolist(), preds=test["y_pred"].tolist(),
            class_names=EMOTIONS)})
        import matplotlib.pyplot as plt
        plt.close(fig)
        if own_run:
            run.finish()

    return {
        "history": history,
        "best_val_acc": best_val_acc,
        "test_acc": test["acc"],
        "confusion_matrix": cm,
        "per_class_acc": pca,
        "n_params": n_params,
        "model": model,
    }
