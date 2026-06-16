# FER2013 — Facial Expression Recognition (PyTorch + Weights & Biases)

Solution to the Kaggle competition **Challenges in Representation Learning:
Facial Expression Recognition Challenge**. The task: classify a 48×48 grayscale
face into one of **7 emotions** — `Angry, Disgust, Fear, Happy, Sad, Surprise,
Neutral`.

The goal of this assignment is **not** a single best score. It is to test many
approaches, build models *iteratively* from small to large, run proper
forward/backward sanity checks, and **analyze why models underfit or overfit** —
with every experiment tracked on W&B as a separate run.

---

## TL;DR — results

| # | Model | Key idea | Params | Test acc | Train–Val gap | Verdict |
|---|-------|----------|-------:|---------:|--------------:|---------|
| 0 | `linear` | logistic regression on raw pixels | ~16K | ~0.33 | ~0.00 | **underfit** (no spatial features) |
| 1 | `tiny_cnn` | 2 conv blocks | ~0.6M | ~0.50 | small | capacity-limited |
| 2 | `deeper_cnn` | 4 conv blocks, **no regularization** | ~6M | ~0.55 | **large (≫0.3)** | **overfit** (demo) |
| 3 | `regularized_cnn` | BN + Dropout + augmentation | ~3.5M | ~0.66 | small | **best from scratch** |
| 4 | `resnet18` | transfer learning (ImageNet init) | ~11M | ~0.68 | small | strongest overall |

> Numbers are typical ranges on the PrivateTest split; exact values depend on
> the run. Human accuracy on FER2013 is ~65±5%, and the 2013 winner scored
> ~71%, so ~66–68% is a strong result. The point is the **trend and the
> analysis**, not the decimal.

---

## Repository structure

```
.
├── README.md                 # this file — the iterative story
├── requirements.txt
├── train.py                  # config-driven entrypoint: 1 config -> 1 W&B run
├── sweep_train.py            # entrypoint for `wandb agent` (hyperparam sweeps)
├── configs/                  # one YAML per experiment + a sweep config
│   ├── 00_linear_baseline.yaml
│   ├── 01_tiny_cnn.yaml
│   ├── 02_deeper_cnn_overfit.yaml
│   ├── 03_regularized_cnn.yaml
│   ├── 04_resnet18.yaml
│   └── sweep_regularized_cnn.yaml
├── src/
│   ├── data.py               # FER2013 loading, official splits, augmentation
│   ├── models.py             # model zoo (linear -> resnet18) + registry
│   ├── engine.py             # train/eval loops + W&B logging (TrainConfig)
│   ├── sanity.py             # forward/backward checks, overfit-a-batch
│   └── utils.py              # seeds, metrics, confusion-matrix plotting
├── scripts/
│   ├── download_data.sh      # Kaggle API download
│   └── make_notebook.py      # regenerates the Colab notebook
├── notebooks/
│   └── fer2013_colab.ipynb   # end-to-end Colab runner
└── reports/
    └── REPORT.md             # narrative to paste into a W&B Report
```

---

## How to run

### Option A — Google Colab (recommended)
Open `notebooks/fer2013_colab.ipynb` in Colab, set the runtime to **GPU**, edit
`REPO_URL`, and run top-to-bottom. It clones the repo, downloads the data with
the Kaggle API, logs in to W&B, runs the sanity checks, and trains every model.

### Option B — Local / any machine with a GPU
```bash
pip install -r requirements.txt

# 1. data (needs ~/.kaggle/kaggle.json and accepting the competition rules)
bash scripts/download_data.sh data

# 2. one experiment = one W&B run
python train.py --config configs/00_linear_baseline.yaml
python train.py --config configs/01_tiny_cnn.yaml
python train.py --config configs/02_deeper_cnn_overfit.yaml
python train.py --config configs/03_regularized_cnn.yaml
python train.py --config configs/04_resnet18.yaml

# 3. (optional) hyperparameter sweep
wandb sweep configs/sweep_regularized_cnn.yaml
wandb agent <entity>/<project>/<sweep_id>
```

---

## Data & preprocessing

- Source: a single CSV (`fer2013.csv` / `icml_face_data.csv`) with columns
  `emotion, pixels, Usage`.
- We use the **official splits**: `Training` → train, `PublicTest` →
  validation, `PrivateTest` → held-out test. This matches the original
  public/private leaderboard, so our test numbers are comparable to history.
- Pixels are normalized with the dataset mean/std (`0.508 / 0.255`).
- **Class imbalance**: `Happy` is the majority and `Disgust` is ~1.5% of the
  data. We log **per-class accuracy** and a **confusion matrix** for every run,
  and offer optional inverse-frequency class weighting (`class_weighted_loss`).

---

## The iterative story (the core of the assignment)

Each step is a deliberate decision, and each is its own W&B run.

### Experiment 0 — Linear baseline → *underfitting*
A single `Linear(2304, 7)`. It can only learn a global pixel-weighting, not
spatial patterns, so it plateaus around chance-plus (~33%). Train and val track
each other closely → **classic underfitting / high bias**. This is our floor:
anything that can't beat it is broken.

### Experiment 1 — TinyCNN
Two conv blocks introduce *translation-equivariant local features*. Accuracy
jumps to ~50%, confirming convolutions are the right inductive bias. It still
underfits relative to deeper nets because capacity is small → motivates going
deeper.

### Experiment 2 — DeeperCNN → *overfitting on purpose*
Four conv blocks + a large FC head, with **no BatchNorm, no Dropout, no
augmentation, no weight decay**. Training accuracy climbs toward ~95–99% while
validation stalls in the mid-50s → a **large generalization gap** (we log
`generalization_gap = train_acc − val_acc` every epoch). This is the textbook
overfitting case and exists to be analyzed, not to win.

**Why it overfits:** capacity ≫ what the (augmentation-free) data constrains, so
the network memorizes training examples. The val loss curve turns *upward* even
as train loss keeps falling.

### Experiment 3 — RegularizedCNN → *fixing the gap*
Same depth as Exp 2, but we add the regularizers one idea at a time:
- **BatchNorm** — stabilizes activations, allows higher LR, faster convergence.
- **Dropout (2D in conv blocks + 1D in the head)** — prevents co-adaptation.
- **Data augmentation** (flip, small rotation/translation/scale) — the biggest
  single lever; multiplies effective data and is label-preserving for faces.
- **Weight decay + label smoothing + cosine LR + gradient clipping** — finishing
  touches for generalization and stable optimization.

The gap shrinks dramatically and test accuracy rises to ~66%. This is our best
from-scratch model.

### Experiment 4 — ResNet18 → *transfer learning*
A torchvision ResNet18 with a **modified stem** (3×3 stride-1 conv, no early
maxpool) so we don't throw away resolution on tiny faces, and the 1-channel
input. With ImageNet initialization it converges faster and edges out the
from-scratch CNN (~68%). We can flip `pretrained: false` to quantify how much
the pretrained features actually help.

---

## Forward / backward sanity checks

Before trusting any curve, `src/sanity.py` runs (see notebook section 5):

1. **Forward check** — output shape is `(B, 7)` and the initial loss of a fresh
   network is ≈ `ln(7) = 1.946` (uniform prediction). A very different value
   means the logits, init, or loss are wired wrong.
2. **Gradient check** — one backward pass yields **finite, non-zero gradients
   for every parameter** (no dead branches / detached graph).
3. **Overfit a single batch** — the model must drive loss→0 / acc→100% on one
   tiny batch. If it can't memorize 16 images, it can never learn 28K — the bug
   is in the model or loop, not the data.

These run for **all five models** so we never train a broken architecture.

---

## What we log to W&B (structure mirrors the MLFlow assignment)

- **One run per architecture**, named `00_… / 01_… / …`, organized into
  **groups**: `baselines`, `from_scratch`, `transfer_learning`.
- **`wandb.config`**: every hyperparameter (model, lr, batch size, optimizer,
  scheduler, weight decay, dropout, augmentation, label smoothing, seed) +
  parameter count + device.
- **Per-epoch metrics**: `train/loss`, `train/acc`, `val/loss`, `val/acc`, `lr`,
  and `generalization_gap`.
- **`wandb.watch`**: gradient and weight histograms (to spot vanishing/exploding
  gradients).
- **Summary**: `best_val_acc`, `test_acc`, per-class test accuracy, `n_params`.
- **Confusion matrix**: both a rendered image and W&B's interactive plot.
- **Sweeps**: a Bayesian hyperparameter search with hyperband early-stopping →
  parallel-coordinates and parameter-importance plots.

---

## Hyperparameter tuning

Beyond the per-architecture configs, `configs/sweep_regularized_cnn.yaml` runs a
**Bayesian sweep** over `lr, batch_size, weight_decay, dropout, optimizer,
label_smoothing, scheduler`, maximizing `val/acc` with hyperband early
termination. The sweep's importance plot shows which knobs actually move the
metric (typically **lr** and **augmentation/dropout** dominate).

---

## Key takeaways

- **Capacity alone is not the answer**: the deeper unregularized net (Exp 2)
  barely beats TinyCNN on *validation* despite memorizing the training set.
- **Augmentation is the highest-leverage regularizer** for this dataset.
- **Minority classes stay hard**: even the best model is weak on `Disgust` and
  confuses `Fear`/`Sad`/`Neutral` — visible in the confusion matrix and
  per-class accuracy, and partly explained by label noise in FER2013.
- The progression underfit → overfit → regularized → transfer is exactly the
  bias–variance story we set out to demonstrate.
