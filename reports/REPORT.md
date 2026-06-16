# W&B Report draft — FER2013 Facial Expression Recognition

Paste this narrative into a **Weights & Biases Report** (Project → Reports →
Create Report) and attach the panels described in brackets. This earns the bonus
points and doubles as the written analysis.

---

## 1. Problem & setup
- Task: 7-class facial expression classification on 48×48 grayscale faces.
- Official splits: Training (28,709) / PublicTest (3,589) / PrivateTest (3,589).
- Imbalanced data — `Happy` majority, `Disgust` ~1.5%.
  *[Panel: bar chart of class distribution — from EDA cell.]*

## 2. Sanity checks
- Forward: initial loss ≈ ln(7) = 1.946 for every model. ✅
- Backward: all params get finite gradients; every model overfits a single
  batch to ~100%. ✅
  *[Panel: the single-batch overfit loss curve.]*

## 3. Architecture comparison (the main result)
*[Panel: line plot of `val/acc` vs `epoch`, all runs overlaid.]*
*[Panel: bar chart of `test_acc` grouped by run, with `n_params` as a second axis.]*

| Run | Test acc | Train–Val gap | Story |
|-----|---------:|--------------:|-------|
| 00_linear_baseline | ~0.33 | ~0 | underfit — no spatial features |
| 01_tiny_cnn | ~0.50 | small | convolutions help, capacity-limited |
| 02_deeper_cnn_overfit | ~0.55 | large | **overfit** — memorizes train set |
| 03_regularized_cnn | ~0.66 | small | BN+Dropout+aug fix the gap |
| 04_resnet18_pretrained | ~0.68 | small | transfer learning wins |

## 4. Overfitting vs underfitting analysis
*[Panel: `generalization_gap` vs `epoch` for runs 00, 02, 03 on one chart.]*
- **Underfitting (00, 01):** train and val accuracy are both low and close. High
  bias — the model lacks capacity / the right inductive bias.
- **Overfitting (02):** train acc → ~99%, val acc flat, val *loss rises*. High
  variance — capacity unconstrained by data/regularization.
- **Well-fit (03, 04):** small gap, val loss tracks train loss. Augmentation is
  the single biggest contributor to closing the gap.

## 5. Per-class behaviour
*[Panel: confusion matrices for 03 and 04.]*
- `Happy` and `Surprise` are easiest (distinct, well-represented).
- `Disgust` is hardest (rare). `Fear`/`Sad`/`Neutral` are mutually confused.
- This is consistent with known FER2013 label noise.

## 6. Hyperparameter sweep
*[Panel: parallel coordinates + parameter importance from the sweep.]*
- Learning rate and augmentation/dropout dominate the importance ranking.
- Best config: *(fill in from the sweep)* → val acc *(…)*.

## 7. Conclusions
- Best model: `resnet18` (transfer) ≈ 68% test; best from-scratch:
  `regularized_cnn` ≈ 66%.
- Capacity without regularization does not generalize.
- Next steps: stronger augmentation (RandAugment), ensembling, focal loss for
  the minority classes, test-time augmentation.
