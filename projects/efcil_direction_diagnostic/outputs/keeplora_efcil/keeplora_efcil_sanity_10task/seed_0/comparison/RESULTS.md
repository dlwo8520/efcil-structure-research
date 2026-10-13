# KeepLoRA on Single-Head EFCIL

## Experimental Setting

| Item | Value |
|---|---|
| Dataset | CIFAR-100 |
| Backbone | `vit_base_patch16_224.augreg_in21k` (ViT-B/16, IN21K) |
| Tasks | 10 tasks × 10 classes |
| LoRA | Q/K/V/O, rank 16, alpha 1.0 |
| Thresholds | epsilon_w=0.85, epsilon_f=0.99 |
| Classifier | One expanding cosine head; old rows frozen, current rows trainable |
| Training | 1 epoch(s)/task, seed 0 |
| Replay / Task ID | None / None |
| Initialization | First-batch full-weight gradient; paper Eq. 5-6; direction frozen, B-only optimization |

`vanilla_lora` is the parameter-matched no-projection control: it retains the same gradient-informed frozen direction/B-only parameterization but uses neither Wp nor M.

## Main Results

| Method | Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |
|---|---:|---:|---:|---:|---:|
| Vanilla LoRA (matched control) | 64.24 | 37.64 | 31.27 | 95.00 | 63.58 |
| KeepLoRA Wp-only | 71.77 | 38.17 | 31.82 | 95.30 | 61.62 |
| KeepLoRA M-only | 84.79 | 78.94 | 78.88 | 79.50 | 6.71 |
| Full KeepLoRA | 84.30 | 78.39 | 78.36 | 78.70 | 6.87 |

## Ablation

| Method | Wp | M | Final Acc | Delta vs Vanilla |
|---|:---:|:---:|---:|---:|
| Vanilla | X | X | 37.64 | +0.00 |
| Wp-only | O | X | 38.17 | +0.53 |
| M-only | X | O | 78.94 | +41.30 |
| Full KeepLoRA | O | O | 78.39 | +40.75 |

## Task-wise Accuracy Matrix

Each A[t,j] uses Task-j samples while prediction remains over every class seen through Task t. Future cells are blank; no Task-ID masking is used.

### Vanilla LoRA (matched control)

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 93.30 | - | - | - | - | - | - | - | - | - | 93.30 |
| Task1 | 91.70 | 92.10 | - | - | - | - | - | - | - | - | 91.90 |
| Task2 | 88.80 | 89.60 | 95.40 | - | - | - | - | - | - | - | 91.27 |
| Task3 | 77.30 | 79.70 | 94.10 | 92.60 | - | - | - | - | - | - | 85.92 |
| Task4 | 54.30 | 48.80 | 86.60 | 88.00 | 96.00 | - | - | - | - | - | 74.74 |
| Task5 | 11.60 | 7.80 | 54.40 | 79.50 | 87.40 | 94.70 | - | - | - | - | 55.90 |
| Task6 | 2.70 | 1.70 | 20.70 | 51.90 | 68.50 | 78.00 | 97.40 | - | - | - | 45.84 |
| Task7 | 0.10 | 0.30 | 2.10 | 18.40 | 31.10 | 36.90 | 82.30 | 95.80 | - | - | 33.37 |
| Task8 | 0.20 | 0.20 | 0.50 | 3.60 | 13.60 | 21.20 | 69.50 | 87.50 | 96.30 | - | 32.51 |
| Task9 | 0.30 | 0.50 | 1.00 | 5.90 | 14.20 | 21.20 | 69.00 | 76.00 | 93.30 | 95.00 | 37.64 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 37.64 |

Heatmap: [`task_matrix_vanilla_lora.png`](task_matrix_vanilla_lora.png)

### KeepLoRA Wp-only

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 93.00 | - | - | - | - | - | - | - | - | - | 93.00 |
| Task1 | 91.40 | 91.20 | - | - | - | - | - | - | - | - | 91.30 |
| Task2 | 89.30 | 89.60 | 93.80 | - | - | - | - | - | - | - | 90.90 |
| Task3 | 84.40 | 85.60 | 93.70 | 89.40 | - | - | - | - | - | - | 88.28 |
| Task4 | 73.80 | 69.90 | 89.30 | 88.00 | 94.40 | - | - | - | - | - | 83.08 |
| Task5 | 49.60 | 47.20 | 78.20 | 82.50 | 86.80 | 93.30 | - | - | - | - | 72.93 |
| Task6 | 30.00 | 22.50 | 67.10 | 74.30 | 80.60 | 85.90 | 96.10 | - | - | - | 65.21 |
| Task7 | 7.50 | 6.80 | 37.50 | 54.30 | 61.20 | 67.30 | 89.90 | 93.40 | - | - | 52.24 |
| Task8 | 2.10 | 0.60 | 6.20 | 24.30 | 32.20 | 49.90 | 82.50 | 88.80 | 96.40 | - | 42.56 |
| Task9 | 0.20 | 0.10 | 0.40 | 8.30 | 10.30 | 28.70 | 72.40 | 72.70 | 93.30 | 95.30 | 38.17 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 38.17 |

Heatmap: [`task_matrix_keep_lora_wp_only.png`](task_matrix_keep_lora_wp_only.png)

### KeepLoRA M-only

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 93.30 | - | - | - | - | - | - | - | - | - | 93.30 |
| Task1 | 91.10 | 89.20 | - | - | - | - | - | - | - | - | 90.15 |
| Task2 | 87.90 | 87.20 | 92.40 | - | - | - | - | - | - | - | 89.17 |
| Task3 | 85.10 | 85.30 | 91.40 | 85.50 | - | - | - | - | - | - | 86.83 |
| Task4 | 81.70 | 82.20 | 90.40 | 82.70 | 84.60 | - | - | - | - | - | 84.32 |
| Task5 | 79.20 | 81.60 | 89.80 | 81.90 | 83.00 | 82.60 | - | - | - | - | 83.02 |
| Task6 | 78.10 | 80.40 | 89.60 | 81.70 | 82.70 | 78.20 | 87.90 | - | - | - | 82.66 |
| Task7 | 76.50 | 79.50 | 89.20 | 80.90 | 79.50 | 75.30 | 85.80 | 73.70 | - | - | 80.05 |
| Task8 | 75.70 | 78.20 | 88.80 | 80.50 | 77.80 | 75.00 | 85.30 | 73.20 | 81.10 | - | 79.51 |
| Task9 | 74.50 | 77.80 | 88.40 | 79.70 | 77.20 | 74.20 | 84.80 | 72.60 | 80.70 | 79.50 | 78.94 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 78.94 |

Heatmap: [`task_matrix_keep_lora_m_only.png`](task_matrix_keep_lora_m_only.png)

### Full KeepLoRA

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 93.00 | - | - | - | - | - | - | - | - | - | 93.00 |
| Task1 | 90.60 | 88.60 | - | - | - | - | - | - | - | - | 89.60 |
| Task2 | 87.80 | 86.00 | 92.10 | - | - | - | - | - | - | - | 88.63 |
| Task3 | 85.00 | 84.30 | 91.10 | 85.10 | - | - | - | - | - | - | 86.38 |
| Task4 | 81.50 | 81.20 | 89.80 | 82.20 | 83.90 | - | - | - | - | - | 83.72 |
| Task5 | 79.00 | 80.60 | 89.00 | 81.30 | 82.50 | 82.70 | - | - | - | - | 82.52 |
| Task6 | 77.70 | 79.20 | 88.70 | 81.10 | 82.20 | 78.70 | 87.40 | - | - | - | 82.14 |
| Task7 | 76.20 | 78.30 | 88.30 | 80.00 | 78.90 | 75.60 | 85.50 | 73.70 | - | - | 79.56 |
| Task8 | 75.30 | 77.20 | 88.00 | 79.70 | 77.40 | 75.30 | 84.80 | 73.00 | 80.50 | - | 79.02 |
| Task9 | 74.30 | 76.70 | 87.50 | 79.20 | 76.70 | 73.90 | 84.40 | 72.30 | 80.20 | 78.70 | 78.39 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 78.39 |

Heatmap: [`task_matrix_keep_lora_full.png`](task_matrix_keep_lora_full.png)

## Protected Space Growth

### Vanilla LoRA (matched control)

Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.

| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 93.30 |
| 1 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 92.10 |
| 2 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 95.40 |
| 3 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 92.60 |
| 4 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 96.00 |
| 5 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 94.70 |
| 6 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 97.40 |
| 7 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 95.80 |
| 8 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 96.30 |
| 9 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 95.00 |

### KeepLoRA Wp-only

Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.

| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 93.00 |
| 1 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 91.20 |
| 2 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 93.80 |
| 3 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 89.40 |
| 4 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 94.40 |
| 5 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 93.30 |
| 6 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 96.10 |
| 7 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 93.40 |
| 8 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 96.40 |
| 9 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 95.30 |

### KeepLoRA M-only

Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.

| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 324.5 | 324.5 | 443.5 | 209 | 42.26 | 93.30 |
| 1 | 0.0 | 414.8 | 414.8 | 353.2 | 81 | 54.01 | 89.20 |
| 2 | 0.0 | 446.3 | 446.3 | 321.7 | 53 | 58.11 | 92.40 |
| 3 | 0.0 | 464.7 | 464.7 | 303.3 | 39 | 60.51 | 85.50 |
| 4 | 0.0 | 473.5 | 473.5 | 294.5 | 34 | 61.65 | 84.60 |
| 5 | 0.0 | 479.1 | 479.1 | 288.9 | 31 | 62.38 | 82.60 |
| 6 | 0.0 | 484.3 | 484.3 | 283.7 | 29 | 63.06 | 87.90 |
| 7 | 0.0 | 487.6 | 487.6 | 280.4 | 27 | 63.49 | 73.70 |
| 8 | 0.0 | 490.2 | 490.2 | 277.8 | 26 | 63.83 | 81.10 |
| 9 | 0.0 | 492.9 | 492.9 | 275.1 | 25 | 64.18 | 79.50 |

### Full KeepLoRA

Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.

| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 248.9 | 211.9 | 460.8 | 307.2 | 118 | 60.00 | 93.00 |
| 1 | 248.9 | 262.6 | 511.6 | 256.4 | 53 | 66.61 | 88.60 |
| 2 | 248.9 | 280.4 | 529.3 | 238.7 | 38 | 68.92 | 92.10 |
| 3 | 248.9 | 290.9 | 539.9 | 228.1 | 29 | 70.29 | 85.10 |
| 4 | 248.9 | 295.8 | 544.7 | 223.3 | 27 | 70.93 | 83.90 |
| 5 | 248.9 | 299.0 | 547.9 | 220.1 | 25 | 71.35 | 82.70 |
| 6 | 248.9 | 302.3 | 551.3 | 216.7 | 23 | 71.78 | 87.40 |
| 7 | 248.9 | 304.3 | 553.2 | 214.8 | 22 | 72.04 | 73.70 |
| 8 | 248.9 | 306.1 | 555.0 | 213.0 | 21 | 72.27 | 80.50 |
| 9 | 248.9 | 307.8 | 556.7 | 211.3 | 20 | 72.49 | 78.70 |

## Interpretation

1. **Wp protection:** Wp-only changes final accuracy by +0.53 points versus the matched vanilla control.
2. **Previous-task M protection:** M-only changes final accuracy by +41.30 points versus vanilla.
3. **Relative contribution:** In this run, the larger standalone final-accuracy contribution comes from M protection (+41.30 points).
4. **Full combination:** Full KeepLoRA changes final accuracy by +40.75 points versus vanilla and -0.55 points versus M-only.
5. **Stability/plasticity:** Full KeepLoRA ends at old/new accuracy 78.36/78.70; vanilla is 31.27/95.00.
6. **Protected growth (KeepLoRA M-only):** final mean protected ratio is 64.18% and its task-wise correlation with new accuracy is r=-0.709 (p=0.0217).
6. **Protected growth (Full KeepLoRA):** final mean protected ratio is 72.49% and its task-wise correlation with new accuracy is r=-0.721 (p=0.0186).

## Integrity Diagnostics

- `vanilla_lora`: max ||A^T Q||=0.000e+00, max protected update ratio=0.000e+00, zero-rank initializations=0/480, optimizer skips=0, peak CUDA=5.55 GiB.
- `keep_lora_wp_only`: max ||A^T Q||=5.694e-07, max protected update ratio=1.596e-07, zero-rank initializations=0/480, optimizer skips=0, peak CUDA=5.55 GiB.
- `keep_lora_m_only`: max ||A^T Q||=9.145e-07, max protected update ratio=2.608e-07, zero-rank initializations=0/480, optimizer skips=0, peak CUDA=5.55 GiB.
- `keep_lora_full`: max ||A^T Q||=9.338e-07, max protected update ratio=2.851e-07, zero-rank initializations=0/480, optimizer skips=0, peak CUDA=5.55 GiB.

All task-matrix consistency checks passed against class-level correct/count values and expanding classifier sizes. Details: [`task_matrix_consistency.json`](task_matrix_consistency.json).

## KeepLoRA Metric Mapping

Original MTIL Transfer evaluates future tasks before learning them. The expanding CIL head has no future-class rows, so Transfer is `N/A`. Here Average means lower-triangle CIL Average Incremental Accuracy and Last means final 100-class single-head accuracy.

## Existing Orthogonal Diagnostic vs KeepLoRA

The earlier repository diagnostic protects class-wise means and centered PCA directions using Q/V LoRA. This experiment separately implements pretrained-weight Wp, residualized task feature M, Q/K/V/O targets, paper Eq. 5-6 initialization, frozen direction/B-only training, and task-end merge. The earlier outputs were not renamed or overwritten.

## Scope

This is a seed-0 short implementation-validation run. Old test images are evaluation-only. No raw replay images, synthetic features, task routing, task-specific inference adapters, or oracle Task-ID are used.
