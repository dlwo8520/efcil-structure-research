# KeepLoRA on Single-Head EFCIL

## 전달용 실험 요약

### 한 문장 요약

원래 multi-domain task-incremental learning을 대상으로 제안된 KeepLoRA를 CIFAR-100 single-head exemplar-free class-incremental learning에 옮겨, PTM principal weight space `Wp`와 이전 task feature space `M`의 보호가 catastrophic forgetting을 줄이는지 네 가지 동일 조건 ablation으로 검증한 실험이다.

### 연구 질문

- 이전 task feature 공간 `M`을 보호하면 이전 class의 망각이 감소하는가?

- pretrained model의 principal weight 공간 `Wp`를 추가로 보호할 가치가 있는가?

- `Wp`와 `M`을 함께 보호할 때 안정성이 좋아지면서도 새 class 학습 능력이 유지되는가?

- task가 누적되어 protected space가 커질 때 residual learning space가 포화되는가?

### 비교한 네 조건

| 조건 | Wp 보호 | 이전 task M 보호 | 의미 |
|---|:---:|:---:|---|
| Vanilla LoRA | X | X | subspace projection이 없는 parameter-matched 대조군 |
| KeepLoRA Wp-only | O | X | PTM의 주요 weight 방향만 보호 |
| KeepLoRA M-only | X | O | 이전 task의 주요 feature 방향만 보호 |
| Full KeepLoRA | O | O | 두 공간을 통합해 보호하는 원 방법에 가장 가까운 조건 |

여기서 Vanilla는 일반적인 two-factor LoRA와 달리 KeepLoRA와 같은 gradient-informed fixed-direction/B-only parameterization을 사용한다. 네 조건의 trainable parameter 수를 같게 두고, 보호 공간의 유무만 비교하기 위한 대조군이다.

### Single-head EFCIL로 변경한 부분

KeepLoRA의 핵심인 `Wp` 구성, residualized `M` 누적, protected-gradient projection, SVD 기반 LoRA 초기화, direction matrix 고정, B-only 학습 및 task-end merge는 유지했다. 원 논문의 domain/task별 MTIL 설정만 다음과 같이 바꿨다.

- CIFAR-100을 고정된 순서의 10개 task로 나누고 task마다 서로 다른 10개 class를 순차 학습했다.

- classifier는 하나의 cosine head를 10개 row씩 확장했다. 이전 row는 고정하고 현재 task row만 학습했다.

- 평가에서는 Task-ID나 task-local mask 없이 현재까지 존재하는 모든 class logit 중 하나를 예측했다.

- old image replay, synthetic replay 및 task-specific adapter routing을 사용하지 않았다.

### 한 task에서 수행한 절차

1. 현재 task의 10개 classifier row를 해당 class의 feature centroid로 초기화한다.

2. 현재 task 첫 batch의 full attention-weight gradient를 계산한다.

3. 조건에 따라 gradient에서 `Wp`, `M` 또는 둘의 unified component를 제거한다.

4. residual gradient를 SVD하여 LoRA rank-16 방향을 정하고, 이 방향 matrix는 고정한 채 나머지 matrix와 새 classifier row만 학습한다.

5. 10 epochs 학습 후 LoRA update를 base ViT에 merge하고 adapter를 초기화한다.

6. 지금까지 학습한 모든 class를 single head로 평가한다.

7. M을 사용하는 조건은 merge된 모델의 현재-task feature에서 기존 protected component를 제거한 뒤 dominant directions를 M에 누적한다. 원본 이미지는 저장하지 않는다.

### 공정한 비교를 위해 고정한 요소

네 조건 모두 동일한 ViT-B/16-IN21K backbone, class order, seed 0, Q/K/V/O 48개 LoRA site, rank 16, alpha 1.0, AdamW, task당 10 epochs, cosine classifier 및 데이터 augmentation을 사용했다. 학습에는 현재 task의 training image만 사용했고 old test image는 평가에만 사용했다.

### 핵심 결과와 결론

- Full KeepLoRA의 final 100-class accuracy는 39.50%로 Vanilla의 35.76%보다 +3.74%p 높았다.

- Full은 final old-class accuracy를 28.79%에서 32.93%로 +4.14%p 높이고, forgetting을 70.31%에서 66.03%로 4.28%p 줄였다.

- 동시에 final new-class accuracy는 Vanilla 98.50%, Full 98.60%로 차이가 +0.10%p에 불과했다. 따라서 이 run에서는 Full의 개선이 뚜렷한 new-class plasticity 희생으로 만들어진 것은 아니다.

- Wp-only와 M-only의 final accuracy 향상은 각각 +2.91%p, +1.76%p였다. Final accuracy 기준으로는 Wp의 독립 기여가 더 컸지만, M-only는 가장 높은 Average Incremental Accuracy 65.01%를 기록해 Vanilla보다 +6.94%p 높았다.

- Full의 마지막 unified protected rank는 48개 site 평균 291.7/768 (37.99%)이고, 가장 좁은 site에도 residual dimension이 368 남았다. rank-16 LoRA가 사용할 공간보다 충분히 커서 이 실험에서는 subspace 포화가 직접적인 병목으로 관찰되지 않았다.

결론적으로, seed-0 결과에서는 `Wp`와 `M` 보호가 모두 single-head CIL의 망각을 완화했고 Full 조합이 가장 높은 최종 정확도를 보였다. 그러나 Full도 old/new accuracy가 32.93%/98.60%로 크게 벌어져 catastrophic forgetting과 classifier recency bias를 해결한 것은 아니다.

### 해석 시 주의사항

- 이 결과는 seed 0 하나의 CIFAR-100 실험이므로 통계적 결론이 아니다. 다음 단계는 최소 3 seeds 반복이다.

- Full은 최종 정확도가 가장 높지만 M-only는 학습 과정 평균 정확도가 가장 높다. 따라서 하나의 지표만으로 절대적 우위를 주장하면 안 된다.

- old classifier row를 고정하는 공통 정책을 사용했기 때문에 representation forgetting과 classifier bias가 함께 반영돼 있다.

- 원 논문의 MTIL을 single-head CIL로 바꾼 adaptation 결과이며, 원 논문의 MTIL 수치를 직접 재현한 실험은 아니다.

## Experimental Setting

| Item | Value |
|---|---|
| Dataset | CIFAR-100 |
| Backbone | `vit_base_patch16_224.augreg_in21k` (ViT-B/16, IN21K) |
| Tasks | 10 tasks × 10 classes |
| LoRA | Q/K/V/O, rank 16, alpha 1.0 |
| Thresholds | epsilon_w=0.85, epsilon_f=0.99 |
| Classifier | One expanding cosine head; old rows frozen, current rows trainable |
| Training | 10 epoch(s)/task, seed 0 |
| Replay / Task ID | None / None |
| Initialization | First-batch full-weight gradient; paper Eq. 5-6; direction frozen, B-only optimization |

`vanilla_lora` is the parameter-matched no-projection control: it retains the same gradient-informed frozen direction/B-only parameterization but uses neither Wp nor M.

## Main Results

| Method | Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |
|---|---:|---:|---:|---:|---:|
| Vanilla LoRA (matched control) | 58.06 | 35.76 | 28.79 | 98.50 | 70.31 |
| KeepLoRA Wp-only | 58.67 | 38.67 | 32.00 | 98.70 | 67.02 |
| KeepLoRA M-only | 65.01 | 37.52 | 30.73 | 98.60 | 68.16 |
| Full KeepLoRA | 61.48 | 39.50 | 32.93 | 98.60 | 66.03 |

## Ablation

| Method | Wp | M | Final Acc | Delta vs Vanilla |
|---|:---:|:---:|---:|---:|
| Vanilla | X | X | 35.76 | +0.00 |
| Wp-only | O | X | 38.67 | +2.91 |
| M-only | X | O | 37.52 | +1.76 |
| Full KeepLoRA | O | O | 39.50 | +3.74 |

## Task-wise Accuracy Matrix

Each A[t,j] uses Task-j samples while prediction remains over every class seen through Task t. Future cells are blank; no Task-ID masking is used.

### Vanilla LoRA (matched control)

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 99.40 | - | - | - | - | - | - | - | - | - | 99.40 |
| Task1 | 77.00 | 98.20 | - | - | - | - | - | - | - | - | 87.60 |
| Task2 | 3.20 | 57.10 | 99.70 | - | - | - | - | - | - | - | 53.33 |
| Task3 | 0.70 | 32.60 | 92.20 | 98.90 | - | - | - | - | - | - | 56.10 |
| Task4 | 1.30 | 19.50 | 74.60 | 83.40 | 99.50 | - | - | - | - | - | 55.66 |
| Task5 | 19.20 | 46.20 | 68.90 | 73.10 | 77.50 | 99.40 | - | - | - | - | 64.05 |
| Task6 | 41.00 | 42.20 | 59.80 | 35.80 | 43.60 | 70.80 | 99.30 | - | - | - | 56.07 |
| Task7 | 44.00 | 41.50 | 26.60 | 4.90 | 0.90 | 18.80 | 74.60 | 98.70 | - | - | 38.75 |
| Task8 | 46.20 | 18.80 | 6.30 | 0.00 | 0.00 | 0.20 | 52.70 | 82.20 | 98.80 | - | 33.91 |
| Task9 | 35.50 | 35.60 | 3.50 | 0.10 | 0.10 | 1.10 | 30.90 | 61.20 | 91.10 | 98.50 | 35.76 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 35.76 |

Heatmap: [`task_matrix_vanilla_lora.png`](task_matrix_vanilla_lora.png)

### KeepLoRA Wp-only

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 99.40 | - | - | - | - | - | - | - | - | - | 99.40 |
| Task1 | 79.30 | 98.30 | - | - | - | - | - | - | - | - | 88.80 |
| Task2 | 7.60 | 60.30 | 99.50 | - | - | - | - | - | - | - | 55.80 |
| Task3 | 1.00 | 35.30 | 92.60 | 98.80 | - | - | - | - | - | - | 56.93 |
| Task4 | 1.70 | 18.00 | 77.00 | 83.90 | 99.40 | - | - | - | - | - | 56.00 |
| Task5 | 9.40 | 34.30 | 68.90 | 73.80 | 78.50 | 99.40 | - | - | - | - | 60.72 |
| Task6 | 33.10 | 39.20 | 61.10 | 36.40 | 44.30 | 71.30 | 99.10 | - | - | - | 54.93 |
| Task7 | 48.10 | 46.60 | 35.00 | 6.40 | 1.30 | 17.00 | 75.80 | 98.60 | - | - | 41.10 |
| Task8 | 47.10 | 22.50 | 7.80 | 0.00 | 0.00 | 0.40 | 50.00 | 82.40 | 98.70 | - | 34.32 |
| Task9 | 40.80 | 46.20 | 14.00 | 1.70 | 0.30 | 4.70 | 28.60 | 60.50 | 91.20 | 98.70 | 38.67 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 38.67 |

Heatmap: [`task_matrix_keep_lora_wp_only.png`](task_matrix_keep_lora_wp_only.png)

### KeepLoRA M-only

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 99.40 | - | - | - | - | - | - | - | - | - | 99.40 |
| Task1 | 85.00 | 98.00 | - | - | - | - | - | - | - | - | 91.50 |
| Task2 | 38.50 | 73.90 | 99.60 | - | - | - | - | - | - | - | 70.67 |
| Task3 | 4.10 | 43.10 | 94.60 | 98.70 | - | - | - | - | - | - | 60.13 |
| Task4 | 0.50 | 17.90 | 80.40 | 88.30 | 99.40 | - | - | - | - | - | 57.30 |
| Task5 | 0.00 | 9.90 | 69.20 | 85.70 | 84.80 | 99.10 | - | - | - | - | 58.12 |
| Task6 | 3.20 | 20.20 | 79.40 | 83.40 | 81.40 | 78.30 | 99.10 | - | - | - | 63.57 |
| Task7 | 26.20 | 64.30 | 87.10 | 69.90 | 59.30 | 44.50 | 76.00 | 98.40 | - | - | 65.71 |
| Task8 | 27.80 | 40.20 | 54.10 | 41.00 | 8.60 | 11.70 | 49.30 | 84.50 | 98.30 | - | 46.17 |
| Task9 | 32.50 | 38.20 | 32.10 | 11.40 | 0.50 | 1.90 | 13.00 | 53.20 | 93.80 | 98.60 | 37.52 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 37.52 |

Heatmap: [`task_matrix_keep_lora_m_only.png`](task_matrix_keep_lora_m_only.png)

### Full KeepLoRA

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 99.40 | - | - | - | - | - | - | - | - | - | 99.40 |
| Task1 | 82.20 | 97.80 | - | - | - | - | - | - | - | - | 90.00 |
| Task2 | 21.10 | 68.60 | 99.60 | - | - | - | - | - | - | - | 63.10 |
| Task3 | 0.50 | 35.50 | 94.40 | 99.00 | - | - | - | - | - | - | 57.35 |
| Task4 | 0.10 | 16.30 | 81.40 | 88.50 | 99.40 | - | - | - | - | - | 57.14 |
| Task5 | 0.50 | 16.50 | 73.60 | 82.70 | 83.70 | 99.40 | - | - | - | - | 59.40 |
| Task6 | 5.90 | 27.90 | 78.00 | 70.70 | 71.00 | 75.00 | 99.20 | - | - | - | 61.10 |
| Task7 | 21.00 | 43.50 | 65.40 | 33.70 | 21.60 | 28.70 | 76.80 | 98.40 | - | - | 48.64 |
| Task8 | 34.70 | 30.60 | 33.80 | 9.30 | 0.30 | 5.60 | 53.20 | 86.30 | 98.50 | - | 39.14 |
| Task9 | 48.40 | 44.60 | 22.90 | 1.90 | 0.00 | 1.00 | 20.40 | 63.40 | 93.80 | 98.60 | 39.50 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 39.50 |

Heatmap: [`task_matrix_keep_lora_full.png`](task_matrix_keep_lora_full.png)

## Protected Space Growth

### Vanilla LoRA (matched control)

Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.

| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 99.40 |
| 1 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 98.20 |
| 2 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 99.70 |
| 3 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 98.90 |
| 4 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 99.50 |
| 5 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 99.40 |
| 6 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 99.30 |
| 7 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 98.70 |
| 8 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 98.80 |
| 9 | 0.0 | 0.0 | 0.0 | 768.0 | 768 | 0.00 | 98.50 |

### KeepLoRA Wp-only

Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.

| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 99.40 |
| 1 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 98.30 |
| 2 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 99.50 |
| 3 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 98.80 |
| 4 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 99.40 |
| 5 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 99.40 |
| 6 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 99.10 |
| 7 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 98.60 |
| 8 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 98.70 |
| 9 | 248.9 | 0.0 | 248.9 | 519.1 | 399 | 32.41 | 98.70 |

### KeepLoRA M-only

Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.

| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 34.8 | 34.8 | 733.2 | 670 | 4.53 | 99.40 |
| 1 | 0.0 | 48.6 | 48.6 | 719.4 | 621 | 6.33 | 98.00 |
| 2 | 0.0 | 57.9 | 57.9 | 710.1 | 592 | 7.54 | 99.60 |
| 3 | 0.0 | 66.8 | 66.8 | 701.2 | 563 | 8.69 | 98.70 |
| 4 | 0.0 | 73.2 | 73.2 | 694.8 | 549 | 9.53 | 99.40 |
| 5 | 0.0 | 77.6 | 77.6 | 690.4 | 537 | 10.11 | 99.10 |
| 6 | 0.0 | 85.6 | 85.6 | 682.4 | 519 | 11.14 | 99.10 |
| 7 | 0.0 | 90.2 | 90.2 | 677.8 | 509 | 11.75 | 98.40 |
| 8 | 0.0 | 93.8 | 93.8 | 674.2 | 503 | 12.22 | 98.30 |
| 9 | 0.0 | 97.0 | 97.0 | 671.0 | 497 | 12.62 | 98.60 |

### Full KeepLoRA

Ranks are means across all 48 Q/K/V/O sites; residual-min is the smallest layer/site residual dimension.

| Task | Wp active rank | M active rank | Unified rank | Residual rank | Residual min | Protected % | New Acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 248.9 | 15.9 | 264.8 | 503.2 | 398 | 34.48 | 99.40 |
| 1 | 248.9 | 21.1 | 270.0 | 498.0 | 397 | 35.16 | 97.80 |
| 2 | 248.9 | 24.5 | 273.4 | 494.6 | 396 | 35.60 | 99.60 |
| 3 | 248.9 | 27.9 | 276.8 | 491.2 | 393 | 36.05 | 99.00 |
| 4 | 248.9 | 30.9 | 279.9 | 488.1 | 387 | 36.44 | 99.40 |
| 5 | 248.9 | 33.0 | 281.9 | 486.1 | 384 | 36.71 | 99.40 |
| 6 | 248.9 | 37.0 | 285.9 | 482.1 | 379 | 37.23 | 99.20 |
| 7 | 248.9 | 39.1 | 288.0 | 480.0 | 374 | 37.50 | 98.40 |
| 8 | 248.9 | 40.9 | 289.9 | 478.1 | 371 | 37.74 | 98.50 |
| 9 | 248.9 | 42.8 | 291.7 | 476.3 | 368 | 37.99 | 98.60 |

## Interpretation

1. **Wp protection:** Wp-only changes final accuracy by +2.91 points versus the matched vanilla control.
2. **Previous-task M protection:** M-only changes final accuracy by +1.76 points versus vanilla.
3. **Relative contribution:** In this run, the larger standalone final-accuracy contribution comes from Wp protection (+2.91 points).
4. **Full combination:** Full KeepLoRA changes final accuracy by +3.74 points versus vanilla and +1.98 points versus M-only.
5. **Stability/plasticity:** Full KeepLoRA ends at old/new accuracy 32.93/98.60; vanilla is 28.79/98.50.
6. **Protected growth (KeepLoRA M-only):** final mean protected ratio is 12.62% and its task-wise correlation with new accuracy is r=-0.316 (p=0.374).
7. **Protected growth (Full KeepLoRA):** final mean protected ratio is 37.99% and its task-wise correlation with new accuracy is r=-0.223 (p=0.536).

## Integrity Diagnostics

- `vanilla_lora`: max ||A^T Q||=0.000e+00, max protected update ratio=0.000e+00, zero-rank initializations=0/480, optimizer skips=0, peak CUDA=8.12 GiB.
- `keep_lora_wp_only`: max ||A^T Q||=5.718e-07, max protected update ratio=1.715e-07, zero-rank initializations=0/480, optimizer skips=0, peak CUDA=8.12 GiB.
- `keep_lora_m_only`: max ||A^T Q||=6.289e-07, max protected update ratio=1.146e-07, zero-rank initializations=0/480, optimizer skips=0, peak CUDA=8.12 GiB.
- `keep_lora_full`: max ||A^T Q||=5.971e-07, max protected update ratio=2.029e-07, zero-rank initializations=0/480, optimizer skips=0, peak CUDA=8.12 GiB.

All task-matrix consistency checks passed against class-level correct/count values and expanding classifier sizes. Details: [`task_matrix_consistency.json`](task_matrix_consistency.json).

## KeepLoRA Metric Mapping

Original MTIL Transfer evaluates future tasks before learning them. The expanding CIL head has no future-class rows, so Transfer is `N/A`. Here Average means lower-triangle CIL Average Incremental Accuracy and Last means final 100-class single-head accuracy.

## Existing Orthogonal Diagnostic vs KeepLoRA

The earlier repository diagnostic protects class-wise means and centered PCA directions using Q/V LoRA. This experiment separately implements pretrained-weight Wp, residualized task feature M, Q/K/V/O targets, paper Eq. 5-6 initialization, frozen direction/B-only training, and task-end merge. The earlier outputs were not renamed or overwritten.

## Scope

This is a seed-0 normal ten-epoch run. Old test images are evaluation-only. No raw replay images, synthetic features, task routing, task-specific inference adapters, or oracle Task-ID are used.

<!-- BEGIN GENERATED PCDS COMPARISON -->

## PCDS Experiment

이 절의 결과는 원 논문의 공식 CIFAR-100 재현값이 아니라, 동일한 CIFAR-100 single-head EFCIL protocol에 PCDS 방법론을 적용한 `PCDS CIFAR-100 adaptation`이다.

### Method

PCDS는 새 task 학습으로 발생한 feature-space migration을 두 구조로 다룬다. APR은 새 class의 학습 전/후 center 이동으로 old mean과 covariance를 보정하고, SCC는 현재 sample과 old prototype 사이의 cosine-relation vector를 직전 frozen model과 일치시킨다. 이후 Eq. (11)의 class-balanced Gaussian synthetic feature로 cosine classifier만 adaptation한다. Old raw train image는 다시 읽거나 저장하지 않는다.

Released CDS 코드의 FFN-output nonlinear bottleneck adapter를 동일 timm ViT에 이식했다. Released covariance helper의 heuristic 대신, 실험 지시서에 명시된 paper Eq. (4)-(8)의 `rho * Delta_k Delta_k^T`를 그대로 사용했다.

### Hyperparameters

| Item | Value | Source |
|---|---:|---|
| Adapter bottleneck | 64 | released CDS config |
| SCC lambda | 2.0 | released CDS config |
| APR gamma | 10.0 | released CDS config |
| APR rho | 10.0 | released CDS config |
| Prototype/APR epsilon | 0.001 / 1e-06 | released implementation |
| Cosine scale tau | 20.0 | released CDS config |
| Optimizer | SGD, lr=0.01, momentum=0.9, wd=0.0005 | released CDS config |
| Scheduler / batch / epochs | cosine / 48 / 10 per task | released config + matched budget |
| Classifier adaptation | 5 epochs, 256 features/class/epoch | released implementation |
| Feature adaptation params | 1,208,064 | runtime model metadata |

CIFAR-100에 맞춘 hyperparameter search는 수행하지 않았다.

### Main Results

| Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |
|---:|---:|---:|---:|---:|
| 94.00 | 89.26 | 88.88 | 92.70 | 6.84 |

### Task-wise Accuracy Matrix

각 `A[t,j]`는 Task t까지 학습한 하나의 expanding head로 Task j sample을 평가한다. Task-local masking이나 Task-ID는 사용하지 않는다.

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 99.30 | - | - | - | - | - | - | - | - | - | 99.30 |
| Task1 | 98.20 | 97.40 | - | - | - | - | - | - | - | - | 97.80 |
| Task2 | 96.40 | 95.80 | 98.50 | - | - | - | - | - | - | - | 96.90 |
| Task3 | 94.80 | 94.60 | 97.40 | 96.30 | - | - | - | - | - | - | 95.78 |
| Task4 | 93.20 | 91.60 | 96.30 | 94.30 | 95.50 | - | - | - | - | - | 94.18 |
| Task5 | 90.70 | 91.40 | 94.90 | 93.80 | 95.40 | 94.20 | - | - | - | - | 93.40 |
| Task6 | 88.80 | 89.30 | 94.00 | 93.60 | 95.20 | 92.70 | 95.50 | - | - | - | 92.73 |
| Task7 | 86.80 | 88.60 | 93.10 | 92.40 | 92.20 | 88.60 | 94.20 | 89.70 | - | - | 90.70 |
| Task8 | 84.20 | 85.40 | 91.90 | 91.10 | 90.30 | 88.40 | 93.30 | 90.30 | 94.50 | - | 89.93 |
| Task9 | 82.20 | 83.30 | 91.50 | 89.90 | 89.50 | 86.70 | 92.60 | 89.60 | 94.60 | 92.70 | 89.26 |
| Final Avg |  |  |  |  |  |  |  |  |  |  | 89.26 |

Heatmap: [`task_matrix_pcds_full.png`](task_matrix_pcds_full.png)

### APR Statistics

| Task | Applied | mean ||delta_j|| | mean ||Delta_k|| | max ||Delta_k|| | Cov trace before | Cov trace after |
|---:|:---:|---:|---:|---:|---:|---:|
| 0 | X | 44.781292 | 0.000000 | 0.000000 | nan | nan |
| 1 | O | 16.757040 | 15.788394 | 16.149239 | 1310.766968 | 3804.078125 |
| 2 | O | 10.221363 | 10.176053 | 10.856746 | 2544.735840 | 3581.617676 |
| 3 | O | 13.107622 | 12.717391 | 13.985129 | 2713.196777 | 4333.605957 |
| 4 | O | 10.782520 | 10.569687 | 11.277246 | 3502.332764 | 4620.583008 |
| 5 | O | 10.476166 | 9.771404 | 10.752290 | 3856.946045 | 4813.306152 |
| 6 | O | 8.594700 | 7.918263 | 8.756651 | 4139.337891 | 4767.100586 |
| 7 | O | 9.820859 | 9.202854 | 9.981487 | 4187.971680 | 5035.485352 |
| 8 | O | 9.979712 | 9.002226 | 9.959454 | 4496.120117 | 5309.147461 |
| 9 | O | 9.329922 | 8.330771 | 9.163215 | 4795.149902 | 5490.278809 |

Task 0에는 old prototype이 없어 APR을 적용하지 않는다.

### SCC Statistics

| Task | SCC | Mean CE | Final CE | Mean SCC | Final SCC | Mean Total | Final Total |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 0 | X | 0.014833 | 0.009057 | 0.000000 | -0.000000 | 0.014833 | 0.009057 |
| 1 | O | 0.020820 | 0.006861 | 0.004046 | 0.002959 | 0.028913 | 0.012778 |
| 2 | O | 0.007068 | 0.001978 | 0.001793 | 0.001037 | 0.010653 | 0.004053 |
| 3 | O | 0.037105 | 0.017769 | 0.005477 | 0.003295 | 0.048058 | 0.024358 |
| 4 | O | 0.016194 | 0.021391 | 0.003525 | 0.002989 | 0.023243 | 0.027369 |
| 5 | O | 0.028393 | 0.012428 | 0.005871 | 0.004844 | 0.040135 | 0.022116 |
| 6 | O | 0.018311 | 0.010737 | 0.003503 | 0.002714 | 0.025317 | 0.016164 |
| 7 | O | 0.045581 | 0.040609 | 0.006431 | 0.005547 | 0.058442 | 0.051703 |
| 8 | O | 0.033787 | 0.013814 | 0.006615 | 0.004779 | 0.047018 | 0.023371 |
| 9 | O | 0.050194 | 0.061068 | 0.007152 | 0.005216 | 0.064499 | 0.071499 |

### Classifier Adaptation Analysis

| Task | CA | Seen before | Seen after | Delta | Old before | Old after | New before | New after |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | X | 99.30 | 99.30 | +0.00%p | N/A | N/A | 99.30 | 99.30 |
| 1 | O | 91.50 | 97.80 | +6.30%p | 85.00 | 98.20 | 98.00 | 97.40 |
| 2 | O | 80.83 | 96.90 | +16.07%p | 71.45 | 96.10 | 99.60 | 98.50 |
| 3 | O | 80.67 | 95.78 | +15.10%p | 74.63 | 95.60 | 98.80 | 96.30 |
| 4 | O | 72.60 | 94.18 | +21.58%p | 65.92 | 93.85 | 99.30 | 95.50 |
| 5 | O | 74.87 | 93.40 | +18.53%p | 70.00 | 93.24 | 99.20 | 94.20 |
| 6 | O | 78.69 | 92.73 | +14.04%p | 75.27 | 92.27 | 99.20 | 95.50 |
| 7 | O | 64.68 | 90.70 | +26.02%p | 59.90 | 90.84 | 98.10 | 89.70 |
| 8 | O | 61.14 | 89.93 | +28.79%p | 56.44 | 89.36 | 98.80 | 94.50 |
| 9 | O | 65.60 | 89.26 | +23.66%p | 61.94 | 88.88 | 98.50 | 92.70 |

Task 1-9에서 CA의 mean Seen-accuracy 변화는 +18.90%p였다. 마지막 task의 old/new 격차는 CA 전 36.56%p에서 CA 후 3.82%p로 변했다.

## Combined Comparison

### Combined CIFAR-100 Single-Head EFCIL Results

| Method | Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |
|---|---:|---:|---:|---:|---:|
| Vanilla LoRA (matched control) | 58.06 | 35.76 | 28.79 | 98.50 | 70.31 |
| KeepLoRA Wp-only | 58.67 | 38.67 | 32.00 | 98.70 | 67.02 |
| KeepLoRA M-only | 65.01 | 37.52 | 30.73 | 98.60 | 68.16 |
| Full KeepLoRA | 61.48 | 39.50 | 32.93 | 98.60 | 66.03 |
| PCDS | 94.00 | 89.26 | 88.88 | 92.70 | 6.84 |

### Old vs New Accuracy and Forgetting

PCDS final accuracy는 Vanilla보다 53.50%p 높았다, Full KeepLoRA보다 49.76%p 높았다. Old accuracy는 각각 60.09%p 높았다, 55.94%p 높았다. Forgetting은 Vanilla 대비 -63.47%p, Full KeepLoRA 대비 -59.19%p 변했다. Average Incremental Accuracy는 Full KeepLoRA 대비 +32.52%p였다.

### Task-wise Comparison

마지막 model의 Task 3-5 accuracy는 다음과 같다.

| Evaluated task | Full KeepLoRA | PCDS | PCDS - Full |
|---:|---:|---:|---:|
| 3 | 1.90 | 89.90 | +88.00%p |
| 4 | 0.00 | 89.50 | +89.50%p |
| 5 | 1.00 | 86.70 | +85.70%p |

세 task 평균 차이는 +87.73%p였다.

### Methodological Differences

| Method | Raw Replay | Synthetic Feature Replay | Task ID | PTM Protection | Previous-task Subspace | APR | SCC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Vanilla | X | X | X | X | X | X | X |
| Wp-only | X | X | X | O | X | X | X |
| M-only | X | X | X | X | O | X | X |
| Full KeepLoRA | X | X | X | O | O | X | X |
| PCDS | X | O (classifier only) | X | X | X | O | O |

PCDS synthetic replay는 저장된 prototype Gaussian에서 매 CA epoch 생성되며 classifier만 최적화한다. Image나 per-sample old feature replay가 아니다.

KeepLoRA의 학습 feature-adaptation parameter는 task당 589,824개이고, PCDS FFN adapter는 1,208,064개다. Classifier row parameter는 task에 따라 추가되므로 이 두 숫자에는 포함하지 않았다.

## Interpretation

### Q1. PCDS가 Vanilla보다 forgetting을 줄이는가?

Average forgetting은 Vanilla 70.31, PCDS 6.84로, PCDS가 Vanilla보다 63.47%p 낮았다. 방향과 크기만 기술하며 seed 0에서 통계적 우월성을 주장하지 않는다.

### Q2. PCDS old accuracy가 KeepLoRA보다 높은가?

PCDS old accuracy 88.88는 Full KeepLoRA 32.93보다 55.94%p 높았다.

### Q3. Synthetic classifier adaptation이 recency bias를 줄였는가?

Final-task old/new gap은 CA 전 36.56%p, CA 후 3.82%p였다. 따라서 gap 변화는 -32.73%p다.

### Q4. Drift prevention과 structure rectification 비교

Full KeepLoRA/PCDS final accuracy는 39.50/89.26이며 PCDS가 49.76%p 높았다. KeepLoRA는 update 방향 자체를 보호하고 PCDS는 SCC로 relation을 제약하면서 APR와 synthetic CA로 drift 결과를 보정한다. CA 및 parameter-count 차이가 있어 이 값만으로 한 핵심 알고리즘의 순수 효과를 분리할 수 없다.

### Q5. KeepLoRA에서 무너진 Task 3-5가 회복되는가?

Final Task 3-5 평균에서 PCDS는 Full KeepLoRA 대비 +87.73%p였다. 위 task-wise 표가 class group별 회복 여부의 직접 근거다.

## Limitations

- PCDS 원 논문은 이 정확한 CIFAR-100 protocol의 결과를 보고하지 않았다. 본 결과는 PCDS methodology의 CIFAR-100 adaptation이다.

- 현재 비교는 seed 0 하나뿐이므로 통계적 결론이 아니다.

- PCDS는 classifier adaptation에 synthetic Gaussian feature를 사용하지만 KeepLoRA run은 synthetic replay를 사용하지 않았다.

- Feature adaptation parameter 수가 KeepLoRA 589,824, PCDS 1,208,064로 다르다.

- Optimizer와 adaptation module도 각 방법의 released setting을 따르므로 final accuracy만으로 모든 방법론적 요인을 분리하지 못한다.

- PCDS는 released CDS FFN adapter를 사용했다. 따라서 논문 본문의 LoRA 표기와 released implementation 사이의 차이를 포함한 implementation adaptation이다.

- 최대 CUDA allocated memory는 3.35 GiB였고 optimizer skip은 0회였다.

<!-- END GENERATED PCDS COMPARISON -->
