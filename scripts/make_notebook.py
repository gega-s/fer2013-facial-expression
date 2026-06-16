"""Generate notebooks/fer2013_colab.ipynb (run locally: python scripts/make_notebook.py)."""
import json
import os

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

cells = []

cells.append(md("""# FER2013 — Facial Expression Recognition (Colab)

End-to-end runner for the Kaggle *Challenges in Representation Learning: Facial
Expression Recognition Challenge*.

This notebook:
1. Clones the GitHub repo and installs dependencies
2. Downloads the dataset with the Kaggle API
3. Logs in to Weights & Biases
4. Runs **forward / backward sanity checks** on every model
5. Trains each architecture as a **separate W&B run** (baseline → tiny → deep
   overfit → regularized → ResNet18)
6. (Optional) launches a **hyperparameter sweep**

> Set the runtime to **GPU**: Runtime → Change runtime type → T4 GPU.
"""))

cells.append(md("## 1. Setup — clone repo & install deps"))
cells.append(code("""# EDIT THIS to your repo URL after you push it to GitHub.
REPO_URL = "https://github.com/gega-s/fer2013-facial-expression.git"
REPO_DIR = "fer2013-facial-expression"

import os
if not os.path.exists(REPO_DIR):
    !git clone $REPO_URL
%cd $REPO_DIR
!pip install -q -r requirements.txt
"""))

cells.append(md("""## 2. Kaggle credentials & data download

Upload your `kaggle.json` (Kaggle → Account → *Create New API Token*).
You must also **accept the competition rules** once on the competition page,
otherwise the API returns 403."""))
cells.append(code("""from google.colab import files
print("Upload kaggle.json:")
files.upload()

!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!bash scripts/download_data.sh data

import glob
print("CSV files found:", glob.glob("data/**/*.csv", recursive=True))
"""))

cells.append(md("## 3. Log in to Weights & Biases\nGet your key from https://wandb.ai/authorize"))
cells.append(code("""import wandb
wandb.login()   # paste your API key when prompted

PROJECT = "fer2013-fer-challenge"
ENTITY = None          # set to your wandb username/team, or leave None
DATA_DIR = "data"
"""))

cells.append(md("""## 4. Quick EDA — class distribution & sample faces
FER2013 is imbalanced: *Happy* dominates and *Disgust* is very rare (~1.5%).
This matters for interpreting per-class accuracy later."""))
cells.append(code("""import matplotlib.pyplot as plt
from src.data import load_dataframe, find_csv, _pixels_to_array, EMOTIONS

df = load_dataframe(find_csv(DATA_DIR))
print(df["Usage"].value_counts())

train = df[df.Usage == "Training"]
counts = train["emotion"].value_counts().sort_index()
plt.figure(figsize=(8,3))
plt.bar([EMOTIONS[i] for i in counts.index], counts.values)
plt.title("Training class distribution"); plt.xticks(rotation=45); plt.show()

# Show one sample per class
imgs = _pixels_to_array(train["pixels"].iloc[:5000])
labels = train["emotion"].iloc[:5000].to_numpy()
fig, axes = plt.subplots(1, 7, figsize=(14, 2.4))
for c in range(7):
    idx = (labels == c).argmax()
    axes[c].imshow(imgs[idx], cmap="gray"); axes[c].set_title(EMOTIONS[c]); axes[c].axis("off")
plt.show()
"""))

cells.append(md("""## 5. Sanity checks (forward & backward)

For every model we verify:
- output shape `(B, 7)` and initial loss ≈ `ln(7) = 1.946`  (**forward**)
- all parameters get finite gradients, and the model can **overfit a single
  batch** to ~100% accuracy (**backward**).

A model that fails the overfit-a-batch test has a bug and is never trained."""))
cells.append(code("""import torch
from src.models import MODEL_REGISTRY, build_model, count_parameters
from src.sanity import run_all_checks
from src.utils import get_device

device = get_device(); print("device:", device)

for name in ["linear", "tiny_cnn", "deeper_cnn", "regularized_cnn", "resnet18"]:
    print(f"\\n========== {name} ({count_parameters(build_model(name)):,} params) ==========")
    model = build_model(name).to(device)
    run_all_checks(model, device)
"""))

cells.append(code("""# Visualize the single-batch overfit curve for one model (sanity that it learns).
from src.sanity import overfit_single_batch
m = build_model("regularized_cnn").to(device)
hist = overfit_single_batch(m, device, steps=200)
plt.plot(hist["losses"]); plt.title("Overfit single batch — loss should -> 0")
plt.xlabel("step"); plt.ylabel("loss"); plt.show()
print("final acc on the batch:", hist["final_acc"])
"""))

cells.append(md("""## 6. Train each architecture — one W&B run per model

We train in increasing order of capacity/regularization so the experiments tell
a story (see README). Each call below creates a separate, comparable W&B run."""))
cells.append(code("""from src.engine import TrainConfig, train_model
from src.data import get_dataloaders

# Build loaders ONCE for the non-augmented runs (faster). Augmented runs need
# their own loaders because the transform differs.
loaders_plain = get_dataloaders(DATA_DIR, batch_size=128, augment=False)
"""))

cells.append(code("""# --- Exp 0: Linear baseline (expected to UNDERFIT) ---
train_model(TrainConfig(
    model="linear", run_name="00_linear_baseline", group="baselines",
    tags=["baseline","underfit"], data_dir=DATA_DIR, epochs=25, lr=1e-3,
    project=PROJECT, entity=ENTITY), loaders=loaders_plain)
"""))
cells.append(code("""# --- Exp 1: TinyCNN ---
train_model(TrainConfig(
    model="tiny_cnn", run_name="01_tiny_cnn", group="from_scratch",
    tags=["cnn","small"], data_dir=DATA_DIR, epochs=30, lr=1e-3,
    project=PROJECT, entity=ENTITY), loaders=loaders_plain)
"""))
cells.append(code("""# --- Exp 2: DeeperCNN, NO regularization (expected to OVERFIT) ---
train_model(TrainConfig(
    model="deeper_cnn", run_name="02_deeper_cnn_overfit", group="from_scratch",
    tags=["cnn","deep","overfit"], data_dir=DATA_DIR, epochs=40, lr=1e-3,
    augment=False, weight_decay=0.0, project=PROJECT, entity=ENTITY),
    loaders=loaders_plain)
"""))
cells.append(code("""# --- Exp 3: RegularizedCNN (BN + Dropout + augmentation) -> fixes overfitting ---
loaders_aug = get_dataloaders(DATA_DIR, batch_size=128, augment=True)
train_model(TrainConfig(
    model="regularized_cnn", run_name="03_regularized_cnn", group="from_scratch",
    tags=["cnn","deep","regularized","best_scratch"], model_kwargs={"dropout":0.4},
    data_dir=DATA_DIR, epochs=60, lr=1e-3, augment=True, weight_decay=1e-4,
    scheduler="cosine", label_smoothing=0.05, grad_clip=5.0,
    early_stop_patience=12, project=PROJECT, entity=ENTITY), loaders=loaders_aug)
"""))
cells.append(code("""# --- Exp 4: ResNet18 transfer learning ---
train_model(TrainConfig(
    model="resnet18", run_name="04_resnet18_pretrained", group="transfer_learning",
    tags=["resnet","transfer"], model_kwargs={"pretrained":True},
    data_dir=DATA_DIR, epochs=50, lr=5e-4, augment=True, optimizer="adamw",
    weight_decay=5e-4, scheduler="cosine", label_smoothing=0.05, grad_clip=5.0,
    early_stop_patience=10, project=PROJECT, entity=ENTITY), loaders=loaders_aug)
"""))

cells.append(md("""## 7. (Optional) Hyperparameter sweep

Bayesian sweep over the RegularizedCNN. Runs N trials; each is its own W&B run
inside the sweep so you get a parallel-coordinates / importance plot for free."""))
cells.append(code("""import os
os.environ["FER_DATA_DIR"] = DATA_DIR
import wandb
sweep_id = wandb.sweep(sweep="configs/sweep_regularized_cnn.yaml", project=PROJECT, entity=ENTITY)
wandb.agent(sweep_id, count=15)   # increase count for a more thorough search
"""))

cells.append(md("""## 8. Wrap-up

- All runs are now in your W&B project, grouped by `baselines` / `from_scratch`
  / `transfer_learning`.
- Build a **W&B Report** from these runs (panel: `val/acc` vs `epoch` for all
  runs; panel: `generalization_gap`; the confusion matrices; the sweep
  importance plot). See `reports/REPORT.md` for the narrative to paste in.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(__file__), "..", "notebooks", "fer2013_colab.ipynb")
with open(os.path.abspath(out), "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", os.path.abspath(out))
