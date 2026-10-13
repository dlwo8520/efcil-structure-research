# cifar100_10task: four-condition result

All accuracies are single-head class-incremental accuracies. Values are percentage points except drift, update geometry, and memory.

| method | final acc | avg incremental acc | final old acc | final new acc | forgetting | old prototype drift | mean update norm | mean protected ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_lora | 20.49 | 44.81 | 11.90 | 97.80 | 86.62 | 0.47161 | 7.3245e-02 | 7.5088e-01 |
| small_update_lora | 24.97 | 50.37 | 16.79 | 98.60 | 82.11 | 0.55041 | 1.8311e-02 | 7.5189e-01 |
| orthogonal_lora | 26.21 | 50.15 | 18.19 | 98.40 | 80.51 | 0.48124 | 7.4643e-02 | 1.2585e-07 |
| orthogonal_lora_norm_matched | 29.43 | 55.41 | 21.82 | 97.90 | 77.07 | 0.48059 | 1.8311e-02 | 1.2593e-07 |

## Controlled contrasts

- Direction at matched small magnitude: orthogonal norm-matched minus small-update = +4.46 points final accuracy.
- Magnitude with unconstrained direction: small-update minus vanilla = +4.48 points final accuracy.
- Natural combined effect: orthogonal minus vanilla = +5.72 points final accuracy.

## Task trajectory (seen-class accuracy, %)

| task | vanilla | small update | orthogonal | orthogonal, norm matched |
|---:|---:|---:|---:|---:|
| 0 | 98.80 | 99.50 | 98.80 | 99.50 |
| 1 | 75.20 | 72.35 | 81.05 | 79.45 |
| 2 | 45.87 | 55.57 | 65.67 | 56.43 |
| 3 | 39.83 | 52.50 | 44.65 | 54.90 |
| 4 | 39.30 | 48.22 | 37.40 | 54.42 |
| 5 | 38.87 | 49.70 | 38.62 | 55.68 |
| 6 | 40.61 | 43.93 | 44.21 | 57.23 |
| 7 | 25.26 | 27.76 | 33.61 | 35.49 |
| 8 | 23.88 | 29.20 | 31.31 | 31.61 |
| 9 | 20.49 | 24.97 | 26.21 | 29.43 |

## KeepLoRA metrics and CIL mapping

KeepLoRA defines `a_t^(i)` as task-t accuracy after training task i. Its per-task metrics are `Transfer_t = mean_{i<t} a_t^(i)`, `Average_t = mean_{i=1..n} a_t^(i)`, and `Last_t = a_t^(n)`. The official implementation applies these operations directly to the full stage-by-task matrix. See the [KeepLoRA paper, Appendix B.2](https://openreview.net/pdf/5b1c7a77eba775963f7a9e2e29612ab26bd10cd2.pdf) and [official implementation](https://github.com/MaolinLuo/KeepLoRA/blob/main/MTIL/model/utils.py).

KeepLoRA evaluates every MTIL dataset at every stage, including unseen future datasets. This expanding single-head CIL classifier has no rows for future classes, so an equivalent Transfer value—and consequently KeepLoRA's original full-matrix Average—is undefined. Transfer is therefore reported as `N/A`. The `Average` below is explicitly the CIL Average Incremental Accuracy (mean of lower-triangle Seen Avg values), while `Last` is the final 100-class single-head accuracy.

## Task-wise Accuracy Matrix

Each `A[t,j]` evaluates only Task-j test samples, but prediction remains over every class seen through Task t. No Task-ID or task-local candidate restriction is used. Future-task cells are intentionally blank.

### Vanilla LoRA

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 98.80 | - | - | - | - | - | - | - | - | - | 98.80 |
| Task1 | 51.80 | 98.60 | - | - | - | - | - | - | - | - | 75.20 |
| Task2 | 10.60 | 27.80 | 99.20 | - | - | - | - | - | - | - | 45.87 |
| Task3 | 0.20 | 0.80 | 60.50 | 97.80 | - | - | - | - | - | - | 39.83 |
| Task4 | 5.20 | 5.30 | 25.50 | 61.20 | 99.30 | - | - | - | - | - | 39.30 |
| Task5 | 11.90 | 10.10 | 26.40 | 30.20 | 56.30 | 98.30 | - | - | - | - | 38.87 |
| Task6 | 34.40 | 21.10 | 32.90 | 9.50 | 30.50 | 57.00 | 98.90 | - | - | - | 40.61 |
| Task7 | 11.30 | 14.30 | 19.70 | 0.00 | 0.30 | 2.80 | 56.20 | 97.50 | - | - | 25.26 |
| Task8 | 14.10 | 12.20 | 13.00 | 0.00 | 0.00 | 0.00 | 22.30 | 55.00 | 98.30 | - | 23.88 |
| Task9 | 4.80 | 12.10 | 6.40 | 0.00 | 0.00 | 0.00 | 0.50 | 14.50 | 68.80 | 97.80 | 20.49 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 20.49 |

Heatmap: [`task_matrix_vanilla_lora.png`](task_matrix_vanilla_lora.png)

### Small Update LoRA

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 99.50 | - | - | - | - | - | - | - | - | - | 99.50 |
| Task1 | 46.60 | 98.10 | - | - | - | - | - | - | - | - | 72.35 |
| Task2 | 2.10 | 65.00 | 99.60 | - | - | - | - | - | - | - | 55.57 |
| Task3 | 0.00 | 20.60 | 90.70 | 98.70 | - | - | - | - | - | - | 52.50 |
| Task4 | 0.40 | 4.80 | 56.50 | 80.10 | 99.30 | - | - | - | - | - | 48.22 |
| Task5 | 3.90 | 15.10 | 40.50 | 65.30 | 74.30 | 99.10 | - | - | - | - | 49.70 |
| Task6 | 19.10 | 26.90 | 27.40 | 28.90 | 38.10 | 68.00 | 99.10 | - | - | - | 43.93 |
| Task7 | 23.30 | 21.80 | 8.90 | 1.60 | 0.40 | 2.30 | 65.60 | 98.20 | - | - | 27.76 |
| Task8 | 34.50 | 15.10 | 2.60 | 0.00 | 0.00 | 0.10 | 34.50 | 77.50 | 98.50 | - | 29.20 |
| Task9 | 20.30 | 12.50 | 0.10 | 0.00 | 0.00 | 0.00 | 0.10 | 32.50 | 85.60 | 98.60 | 24.97 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 24.97 |

Heatmap: [`task_matrix_small_update_lora.png`](task_matrix_small_update_lora.png)

### Orthogonal LoRA

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 98.80 | - | - | - | - | - | - | - | - | - | 98.80 |
| Task1 | 63.80 | 98.30 | - | - | - | - | - | - | - | - | 81.05 |
| Task2 | 36.30 | 61.20 | 99.50 | - | - | - | - | - | - | - | 65.67 |
| Task3 | 1.00 | 5.90 | 73.10 | 98.60 | - | - | - | - | - | - | 44.65 |
| Task4 | 0.00 | 0.00 | 11.70 | 76.10 | 99.20 | - | - | - | - | - | 37.40 |
| Task5 | 0.00 | 0.10 | 5.00 | 55.20 | 72.70 | 98.70 | - | - | - | - | 38.62 |
| Task6 | 14.90 | 9.10 | 30.60 | 41.20 | 46.60 | 68.10 | 99.00 | - | - | - | 44.21 |
| Task7 | 12.70 | 10.60 | 30.60 | 20.30 | 9.10 | 20.70 | 67.10 | 97.80 | - | - | 33.61 |
| Task8 | 29.00 | 18.50 | 24.40 | 9.00 | 0.80 | 0.30 | 23.90 | 77.50 | 98.40 | - | 31.31 |
| Task9 | 17.20 | 22.70 | 14.60 | 0.70 | 0.00 | 0.00 | 0.00 | 26.80 | 81.70 | 98.40 | 26.21 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 26.21 |

Heatmap: [`task_matrix_orthogonal_lora.png`](task_matrix_orthogonal_lora.png)

### Orthogonal LoRA Norm-Matched

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 99.50 | - | - | - | - | - | - | - | - | - | 99.50 |
| Task1 | 60.90 | 98.00 | - | - | - | - | - | - | - | - | 79.45 |
| Task2 | 2.60 | 67.10 | 99.60 | - | - | - | - | - | - | - | 56.43 |
| Task3 | 0.00 | 27.40 | 93.10 | 99.10 | - | - | - | - | - | - | 54.90 |
| Task4 | 0.00 | 7.20 | 75.90 | 89.70 | 99.30 | - | - | - | - | - | 54.42 |
| Task5 | 0.00 | 7.60 | 63.00 | 82.30 | 81.90 | 99.30 | - | - | - | - | 55.68 |
| Task6 | 6.90 | 31.60 | 68.90 | 62.30 | 58.70 | 73.20 | 99.00 | - | - | - | 57.23 |
| Task7 | 16.10 | 35.10 | 27.50 | 13.90 | 7.60 | 14.70 | 71.10 | 97.90 | - | - | 35.49 |
| Task8 | 25.90 | 24.50 | 10.30 | 1.90 | 0.40 | 1.10 | 36.50 | 85.60 | 98.30 | - | 31.61 |
| Task9 | 25.00 | 14.40 | 0.30 | 0.30 | 0.00 | 0.00 | 2.80 | 59.20 | 94.40 | 97.90 | 29.43 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 29.43 |

Heatmap: [`task_matrix_orthogonal_lora_norm_matched.png`](task_matrix_orthogonal_lora_norm_matched.png)

## Transfer / Average / Last

| Method | Transfer | CIL Average | Last |
|---|---:|---:|---:|
| Vanilla LoRA | N/A | 44.81 | 20.49 |
| Small Update LoRA | N/A | 50.37 | 24.97 |
| Orthogonal LoRA | N/A | 50.15 | 26.21 |
| Orthogonal LoRA Norm-Matched | N/A | 55.41 | 29.43 |

## Final Task-wise Accuracy

| Method | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Average |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla LoRA | 4.80 | 12.10 | 6.40 | 0.00 | 0.00 | 0.00 | 0.50 | 14.50 | 68.80 | 97.80 | 20.49 |
| Small Update LoRA | 20.30 | 12.50 | 0.10 | 0.00 | 0.00 | 0.00 | 0.10 | 32.50 | 85.60 | 98.60 | 24.97 |
| Orthogonal LoRA | 17.20 | 22.70 | 14.60 | 0.70 | 0.00 | 0.00 | 0.00 | 26.80 | 81.70 | 98.40 | 26.21 |
| Orthogonal LoRA Norm-Matched | 25.00 | 14.40 | 0.30 | 0.30 | 0.00 | 0.00 | 2.80 | 59.20 | 94.40 | 97.90 | 29.43 |

## Task-matrix Consistency Checks

All four methods passed: lower-triangle shape and blank future cells; task-wise cells versus sample-count accuracies; row means versus existing Seen Accuracy; final-row mean versus Final Accuracy; mean row average versus Average Incremental Accuracy; and expanding classifier sizes 10, 20, ..., 100. Every task contains the same 1,000 test samples, so task-wise and sample-wise row averages are equal. Full details are in [`task_matrix_consistency.json`](task_matrix_consistency.json).

## Diagnostic questions

1. **Vanilla forgetting — yes.** Final old accuracy is 11.90% and final forgetting is 86.62%, while final new-class accuracy remains 97.80%.
2. **Small magnitude helps, but does not solve forgetting.** A 75% reduction in every measured step norm improves final accuracy by +4.48 points and changes forgetting by -4.51 points.
3. **Natural orthogonal updates help.** They improve final accuracy by +5.72 points and change forgetting by -6.11 points versus vanilla.
4. **Direction matters at equal magnitude.** The two small-norm conditions have the same mean effective update norm (1.8311e-02); orthogonal direction adds +4.46 points final accuracy, +5.03 points old accuracy, and changes forgetting by -5.04 points.
5. **Representation drift is reduced on average across stages, not at the final stage.** Mean old-class cosine drift across tasks 1..9 is 0.3645 for vanilla and 0.2759 for matched orthogonal. At task 9 the corresponding values are 0.4716 and 0.4806; the early/mid-task advantage is no longer visible in this final-only statistic.
6. **Interference tracks forgetting descriptively.** Across vanilla tasks 1..9, mean protected-direction ratio versus forgetting has Pearson r=0.743 (p=0.0217) and Spearman rho=0.667 (p=0.0499). The ratio versus prototype drift has Pearson r=0.968 (p=1.793e-05). This is a nine-point, single-seed association and is not causal evidence.
7. **No large plasticity cost is observed.** Mean new-class accuracy over tasks 1..9 is 98.80% for small-update and 98.71% for its direction-matched orthogonal counterpart.
8. **The protected space grows substantially without clear new-class collapse.** Rank grows to 400/768 (52.08%). For matched orthogonal, rank ratio versus new accuracy has Pearson r=-0.469 (p=0.203); new accuracy stays between 97.90% and 99.60%.

## Integrity diagnostics

- `vanilla_lora`: max ||A Q||=3.976e+00, max protected update ratio=9.280e-01, norm-match saturated=0/1040, optimizer skipped=0/1040, peak CUDA allocated=5.15 GiB.
- `small_update_lora`: max ||A Q||=3.968e+00, max protected update ratio=9.307e-01, norm-match saturated=0/1040, optimizer skipped=0/1040, peak CUDA allocated=5.15 GiB.
- `orthogonal_lora`: max ||A Q||=5.636e-07, max protected update ratio=2.265e-07, norm-match saturated=0/1040, optimizer skipped=0/1040, peak CUDA allocated=5.15 GiB.
- `orthogonal_lora_norm_matched`: max ||A Q||=5.692e-07, max protected update ratio=2.269e-07, norm-match saturated=0/1040, optimizer skipped=0/1040, peak CUDA allocated=5.15 GiB.

The protected component ratio is `||delta_W Q Q^T||_F / ||delta_W||_F`, where columns of `Q` are the accumulated old-class input directions. The residual is `delta_W - delta_W Q Q^T`.

## Scope

This is a controlled seed-0 diagnostic (two epochs per task), not a paper-scale benchmark. The classifier is one expanding cosine head with frozen old weight rows; old test images are evaluation-only; no raw training images, paths, replay samples, task routing, or task-specific inference adapters are retained.
