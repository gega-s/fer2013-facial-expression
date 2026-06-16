"""Forward / backward sanity checks discussed in the lectures.

Before trusting any training curve, we verify that the model and the training
loop are wired correctly. Two cheap, high-signal checks:

1. FORWARD check
   - output shape is (B, num_classes)
   - the initial loss of a freshly-initialized network is approximately
     ``ln(num_classes)`` (= 1.9459 for 7 classes). A random classifier assigns
     ~uniform probability, so cross-entropy ≈ -ln(1/C). A wildly different value
     means the logits/initialization/loss are wrong.

2. BACKWARD check ("overfit a single batch")
   - every learnable parameter receives a finite, non-zero gradient (no dead
     branches, no detached graph)
   - the model can drive the loss on ONE small batch to ~0 / accuracy to ~100%.
     If a model cannot overfit a handful of examples, it can never learn the
     full dataset — the bug is in the model or the loop, not the data.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn

from .data import NUM_CLASSES, IMG_SIZE


@torch.no_grad()
def forward_check(model: nn.Module, device, batch_size: int = 8) -> Dict[str, float]:
    model.eval()
    x = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    logits = model(x)
    loss = nn.functional.cross_entropy(logits, y).item()
    expected = math.log(NUM_CLASSES)
    result = {
        "output_shape_ok": float(tuple(logits.shape) == (batch_size, NUM_CLASSES)),
        "initial_loss": loss,
        "expected_loss_ln_C": expected,
        "loss_close_to_expected": float(abs(loss - expected) < 0.7),
    }
    return result


def gradient_check(model: nn.Module, device, batch_size: int = 8) -> Dict[str, float]:
    """Verify that one backward pass produces finite gradients everywhere."""
    model.train()
    x = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    model.zero_grad()
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()

    n_params = 0
    n_with_grad = 0
    n_finite = 0
    total_norm = 0.0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n_params += 1
        if p.grad is not None:
            n_with_grad += 1
            g = p.grad
            if torch.isfinite(g).all():
                n_finite += 1
            total_norm += float(g.norm()) ** 2
    return {
        "params": n_params,
        "params_with_grad": n_with_grad,
        "params_finite_grad": n_finite,
        "all_params_have_finite_grad": float(n_params == n_with_grad == n_finite),
        "grad_global_norm": math.sqrt(total_norm),
    }


def overfit_single_batch(
    model: nn.Module,
    device,
    steps: int = 200,
    batch_size: int = 16,
    lr: float = 1e-3,
    target_acc: float = 0.99,
):
    """Train on ONE fixed random batch and confirm the loss collapses.

    Returns a history dict you can plot / log, plus whether the target accuracy
    was reached. This is the single most useful debugging check for a new model.
    """
    model.train()
    x = torch.randn(batch_size, 1, IMG_SIZE, IMG_SIZE, device=device)
    y = torch.randint(0, NUM_CLASSES, (batch_size,), device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    losses, accs = [], []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append((logits.argmax(1) == y).float().mean().item())

    return {
        "losses": losses,
        "accuracies": accs,
        "final_loss": losses[-1],
        "final_acc": accs[-1],
        "reached_target": accs[-1] >= target_acc,
    }


def run_all_checks(model: nn.Module, device, verbose: bool = True) -> Dict:
    fwd = forward_check(model, device)
    grad = gradient_check(model, device)
    overfit = overfit_single_batch(model, device)
    if verbose:
        print("── Forward check ──")
        print(f"  output shape ok      : {bool(fwd['output_shape_ok'])}")
        print(f"  initial loss         : {fwd['initial_loss']:.4f} "
              f"(expected ≈ ln(7) = {fwd['expected_loss_ln_C']:.4f})")
        print(f"  loss ≈ expected      : {bool(fwd['loss_close_to_expected'])}")
        print("── Gradient check ──")
        print(f"  all finite gradients : {bool(grad['all_params_have_finite_grad'])}")
        print(f"  grad global norm     : {grad['grad_global_norm']:.4f}")
        print("── Overfit single batch ──")
        print(f"  final loss           : {overfit['final_loss']:.4f}")
        print(f"  final accuracy       : {overfit['final_acc']:.4f}")
        print(f"  reached ~100% acc    : {overfit['reached_target']}")
    return {"forward": fwd, "gradient": grad, "overfit_batch": overfit}
