# Class-Discriminative Subspace Protection 실험 결과

## 1. 실험 목적

이 실험은 Exemplar-Free Class-Incremental Learning에서 이미 발생한 feature drift를 사후 보정하는 대신, old class를 구별하는 데 중요한 representation subspace가 새로운 task 학습 중 덜 움직이도록 adapter update를 제한할 수 있는지 진단한다.

기존 CDS의 기본 관점은 다음과 같다.

```text
Feature drift
  -> new-class displacement 추정
  -> APR로 old prototype 사후 보정
```

본 실험의 관점은 다음과 같다.

```text
Old class-discriminative subspace 추출
  -> 해당 subspace를 피하도록 adapter update 제한
  -> feature drift 억제
  -> 저장된 old prototype을 보정 없이 사용
```

이 코드는 KeepLoRA와 같은 protected-subspace 철학에서 영감을 받았지만 KeepLoRA 구현은 아니다. CDS의 adapter는 선형 LoRA `BAx`가 아니라 다음과 같은 nonlinear bottleneck adapter이기 때문이다.

```text
up_proj(ReLU(down_proj(x)))
```

따라서 본 실험이 직접 제한하는 대상은 LoRA의 `Delta W`가 아니라 각 adapter의 `down_proj.weight`가 반응하는 입력 방향이다.

## 2. 방법론

### 2.1 Layer-wise adapter input 수집

ViT의 각 MLP adapter `down_proj`에 forward pre-hook을 등록한다. Adapter가 실제로 받는 tensor가 `[B, N, 768]`일 때 CLS token인 `x[:, 0, :]`만 수집한다.

- hook은 task 종료 후 compressed statistics를 계산할 때만 활성화한다.
- tensor는 즉시 `detach()`하고 CPU로 이동한다.
- sample feature 자체는 저장하지 않는다.
- 모든 통계는 해당 class가 처음 등장한 current-task train split에서만 계산한다.
- old-class train image나 old feature sample에는 다시 접근하지 않는다.

### 2.2 Prototype memory

각 layer `l`, class `c`에 대해 current-task feature의 평균을 저장한다.

```math
\mu_c^{(l)} = \frac{1}{N_c}\sum_{i:y_i=c}h_i^{(c,l)}
```

최종 ViT feature에 대해서도 class prototype을 저장하며, old prototype은 이후 task에서 다시 계산하거나 APR로 이동시키지 않는다.

### 2.3 Inter-class discriminative subspace

Layer `l`에서 지금까지 저장된 class prototype을 평균 중심화해 다음 행렬을 구성한다.

```math
P^{(l)} = [\mu_1^{(l)}-\bar{\mu}^{(l)},\ldots,\mu_C^{(l)}-\bar{\mu}^{(l)}]
```

`P^(l)`에 SVD를 적용하고 singular-value energy 95%를 설명하는 최소 rank의 left singular vectors를 `Q_inter^(l)`로 사용한다. 이는 old class 전체 feature가 아니라 class center를 서로 구별하는 방향만 압축해 저장한다.

### 2.4 Generic task-feature subspace

`task_subspace`에서는 label을 사용하지 않고 current-task adapter input 전체의 centered scatter matrix를 streaming 방식으로 계산한다. Energy 95%를 설명하는 basis를 task별로 추출하고 이전 task basis와 orthonormal union한다.

이 조건은 old task가 사용한 feature direction 전반을 보호하는 generic task-feature protection baseline이다.

### 2.5 Intra-class subspace

`inter_intra_subspace`에서는 각 sample의 class residual에 해당하는 within-class scatter를 streaming 방식으로 계산한다.

```math
R^{(l)} = [h_1-\mu_{y_1}^{(l)},\ldots,h_n-\mu_{y_n}^{(l)}]
```

Energy 90%의 dominant basis를 task별로 압축한 뒤 누적하고, 최종 protected basis는 다음 union으로 구성한다.

```math
Q^{(l)} = \operatorname{orth}([Q_{inter}^{(l)}, Q_{intra}^{(l)}])
```

### 2.6 Adapter hard constraint

Layer `l`의 adapter down projection을 `D_l`, protected orthonormal basis를 `Q_l`이라 하면 다음 조건을 적용한다.

```math
D_l Q_l = 0
```

Projection은 explicit `768 x 768` matrix를 만들지 않고 다음 형태로 계산한다.

```math
D_l \leftarrow D_l - (D_lQ_l)Q_l^T
```

제약은 세 시점에 유지된다.

1. Task 시작 전 weight projection
2. Backward 시 gradient projection
3. Optimizer step 후 weight와 SGD momentum 재투영

Protection 조건에서는 `down_proj.bias`와 `up_proj.bias`를 0으로 유지하고 freeze한다. Task 0에는 old class가 없으므로 `Q`가 empty이며 protection 없이 학습한다. CDS의 두 adapter branch 중 현재 task에서 실제 학습되는 branch에만 제약을 적용하고, task 종료 후 기존 lifecycle에 따라 다음 branch로 복사한다.

### 2.7 Prototype classifier

최종 inference는 현재 feature `z`와 모든 seen-class stored prototype 사이 cosine similarity로 수행한다.

```math
s_c(x) = \frac{z^T\mu_c}{\lVert z\rVert\lVert\mu_c\rVert}
```

Prediction 후보는 항상 현재까지 학습한 모든 class이며 oracle Task-ID나 task별 class masking을 사용하지 않는다. 기존 expanding cosine head의 성능도 보조 metric으로 함께 기록한다.

### 2.8 Controlled comparison

| Method | APR | SCC | Synthetic CA | Prototype eval | Protected space |
|---|---:|---:|---:|---:|---|
| Prototype Baseline | X | X | X | O | 없음 |
| Task Subspace | X | X | X | O | Generic task feature |
| Inter Subspace | X | X | X | O | Inter-class prototype geometry |
| Inter + Intra | X | X | X | O | Inter-class + within-class |

모든 조건에서 raw replay, synthetic feature replay, Gaussian replay, prototype rectification, old-sample distillation을 사용하지 않았다. Official CDS 결과는 이번 네 조건의 실행 결과에 포함되지 않는다.

## 3. 실험 설정

| Item | Value |
|---|---|
| Dataset | CIFAR-100 (`cifar224`) |
| Protocol | 10 initial classes + 10 classes/task, 총 10 tasks |
| Backbone | `pretrained_vit_b16_224_in21k_adapter` |
| Feature dimension | 768 |
| Adapter bottleneck | 64 |
| Adapter placement | 모든 ViT block의 MLP output |
| Seed | 1993, 단일 seed |
| Epochs | Task 0: 20, Task 1-9: 10 |
| Batch size | 48 |
| Optimizer | SGD, momentum 0.9 |
| Learning rate | 0.01, cosine schedule |
| Inter/task energy | 0.95 |
| Intra energy | 0.90 |
| Primary classifier | Stored-prototype cosine classifier |
| Physical GPU | GPU 3 (`CUDA_VISIBLE_DEVICES=3`, logical `cuda:0`) |

## 4. 주요 결과

### 4.1 최종 성능

| Method | Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting | Final Top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Prototype Baseline | 93.145 | 89.20 | 88.64 | **94.20** | 6.38 | 98.75 |
| Task Subspace | 93.138 | 89.37 | 88.99 | 92.80 | **4.91** | 98.73 |
| Inter Subspace | 93.172 | 89.35 | 88.87 | 93.70 | 5.89 | **98.83** |
| Inter + Intra | **93.199** | **89.46** | **89.04** | 93.20 | 5.00 | 98.77 |

### 4.2 Prototype baseline 대비 변화

| Method | Final Acc Δ | Old Acc Δ | New Acc Δ | Forgetting Δ |
|---|---:|---:|---:|---:|
| Task Subspace | +0.17%p | +0.35%p | -1.40%p | -1.47%p |
| Inter Subspace | +0.15%p | +0.23%p | -0.50%p | -0.49%p |
| Inter + Intra | **+0.26%p** | **+0.40%p** | -1.00%p | -1.38%p |

Protection 조건은 모두 final/old accuracy를 소폭 높이고 forgetting을 줄였다. 반면 new-class accuracy는 0.5-1.4%p 낮아져 stability와 plasticity 사이의 trade-off가 관찰됐다.

### 4.3 학습 task별 seen-class Top-1

각 column은 해당 task까지 학습한 시점에 모든 seen class를 대상으로 평가한 accuracy이다.

| Method | T0 | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prototype Baseline | 98.10 | 96.90 | 95.97 | 94.48 | 93.20 | 92.25 | 91.71 | 90.50 | 89.14 | 89.20 |
| Task Subspace | 97.90 | 96.60 | 95.93 | 94.42 | 93.24 | 92.42 | 91.67 | 90.29 | 89.54 | 89.37 |
| Inter Subspace | 97.90 | 96.85 | 95.93 | 94.52 | 93.20 | 92.27 | 91.74 | 90.49 | 89.47 | 89.35 |
| Inter + Intra | 97.90 | 96.60 | 95.77 | 94.48 | 93.22 | **92.53** | **91.90** | 90.45 | **89.68** | **89.46** |

### 4.4 최종 모델의 task-wise accuracy

모든 값은 Task 9까지 학습한 하나의 모델이 전체 100-class 후보를 사용해 각 task test subset을 평가한 결과이다.

| Method | Task 0 | Task 1 | Task 2 | Task 3 | Task 4 | Task 5 | Task 6 | Task 7 | Task 8 | Task 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Prototype Baseline | 82.20 | 87.40 | 93.10 | 90.10 | 87.40 | 87.70 | 91.60 | **85.00** | **93.30** | **94.20** |
| Task Subspace | **87.10** | **87.90** | **93.60** | 90.80 | **89.00** | 87.40 | 91.00 | 83.30 | 90.80 | 92.80 |
| Inter Subspace | 84.00 | 87.50 | 92.80 | 89.80 | 87.80 | **88.30** | **91.80** | 84.70 | 93.10 | 93.70 |
| Inter + Intra | 86.50 | 87.80 | 93.40 | **90.90** | 88.70 | 87.40 | 91.50 | 83.70 | 91.50 | 93.20 |

Task/Inter+Intra protection은 오래된 Task 0 accuracy를 baseline 대비 각각 +4.9%p, +4.3%p 개선했다. 동시에 최근 task 일부의 성능은 낮아져 전체 평균 개선 폭은 작았다.

## 5. Representation drift 분석

Prototype cosine drift는 stored old prototype과 현재 모델로 old test sample에서 다시 계산한 prototype 사이의 `1 - cosine`이다. Pairwise geometry drift는 stored/current old prototype cosine matrix의 off-diagonal 평균 절대 차이다. 두 metric 모두 작을수록 geometry가 안정적이다.

| Method | Prototype Cosine Drift | Baseline 대비 | Pairwise Geometry Drift | Baseline 대비 |
|---|---:|---:|---:|---:|
| Prototype Baseline | 0.070491 | - | 0.047917 | - |
| Task Subspace | **0.034317** | **-51.3%** | **0.025408** | **-47.0%** |
| Inter Subspace | 0.063638 | -9.7% | 0.041754 | -12.9% |
| Inter + Intra | 0.040492 | -42.6% | 0.028202 | -41.1% |

Protection strength가 큰 Task/Inter+Intra 조건에서 prototype과 pairwise geometry drift가 크게 감소했다. 이와 함께 forgetting도 6.38에서 각각 4.91, 5.00으로 감소했다. 따라서 이 seed에서는 representation stabilization과 forgetting 감소가 같은 방향으로 나타났다.

Inter-only protection은 훨씬 작은 basis만 사용하면서 drift를 제한적으로 줄였고 final accuracy는 +0.15%p 개선했다. 이는 compact discriminative direction만 보호하는 방식이 plasticity를 상대적으로 덜 제한한다는 가능성을 보여주지만, 단일 seed 결과이므로 확정적인 결론은 아니다.

## 6. Protected rank 및 constraint 진단

### 6.1 Task 9 종료 후 layer-wise protected rank

| Layer | Task Subspace | Inter Subspace | Inter + Intra | Feature dim |
|---:|---:|---:|---:|---:|
| 0 | 40 | 3 | 34 | 768 |
| 1 | 60 | 7 | 57 | 768 |
| 2 | 107 | 11 | 79 | 768 |
| 3 | 328 | 13 | 243 | 768 |
| 4 | 267 | 12 | 191 | 768 |
| 5 | 682 | 22 | 496 | 768 |
| 6 | 768 | 26 | 751 | 768 |
| 7 | 768 | 34 | 768 | 768 |
| 8 | 768 | 46 | 768 | 768 |
| 9 | 768 | 41 | 768 | 768 |
| 10 | 653 | 56 | 550 | 768 |
| 11 | 768 | 68 | 768 | 768 |

Inter-only basis는 최종 시점에도 layer별 rank 3-68로 compact하다. 반면 Task Subspace는 layer 6-9와 11에서, Inter+Intra는 layer 7-9와 11에서 rank 768로 포화됐다.

`Q`가 full rank이면 `D Q = 0`은 사실상 `D ≈ 0`을 요구한다. 따라서 generic task feature와 누적 intra-class basis는 후반 layer에서 보호 범위가 지나치게 커진 것으로 해석해야 한다. Task/Inter+Intra 조건의 new-class accuracy 하락 역시 이러한 over-protection과 일관된 현상이다.

### 6.2 Constraint violation

학습 후 측정한 최대 violation ratio는 다음과 같다.

```math
r_l = \frac{\lVert D_lQ_l\rVert_F}{\lVert D_l\rVert_F+\epsilon}
```

| Method | 전체 task/layer 최대 post-training violation |
|---|---:|
| Task Subspace | 8.75e-6 |
| Inter Subspace | **3.75e-8** |
| Inter + Intra | 1.20e-5 |

모든 protection 조건에서 violation은 작게 유지됐다. Full-rank에 가까운 basis에서는 반복 projection의 수치오차가 상대적으로 커졌지만 최대 약 `1e-5` 수준이었다.

## 7. Prototype classifier와 head 비교

| Method | Prototype Top-1 | Expanding Head Top-1 | Prototype - Head |
|---|---:|---:|---:|
| Prototype Baseline | 89.20 | 89.40 | -0.20%p |
| Task Subspace | 89.37 | 88.74 | +0.63%p |
| Inter Subspace | 89.35 | **89.42** | -0.07%p |
| Inter + Intra | **89.46** | 89.01 | +0.45%p |

Task/Inter+Intra 조건에서는 stored-prototype classifier가 expanding head보다 높았다. Representation은 비교적 안정됐지만 head calibration 또는 old/new class competition이 별도의 문제가 될 수 있음을 시사한다. Inter-only에서는 두 classifier가 거의 동일했다.

## 8. 실행 시간과 메모리

| Method | 누적 training time | 전체 task elapsed 합 | Peak allocated GPU memory |
|---|---:|---:|---:|
| Prototype Baseline | 1.263 h | 1.425 h | 4.66 GiB |
| Task Subspace | 1.261 h | 1.425 h | 4.67 GiB |
| Inter Subspace | 1.261 h | 1.423 h | 4.66 GiB |
| Inter + Intra | 1.263 h | 1.428 h | 4.67 GiB |

네 조건의 runtime과 peak GPU memory는 거의 같았다. Stored checkpoint 크기는 basis memory가 큰 Task/Inter+Intra 조건에서 더 증가한다.

## 9. 해석

1. 모든 protection variant가 prototype baseline보다 final accuracy를 0.15-0.26%p 높이고 forgetting을 줄였다.
2. 가장 낮은 forgetting과 drift는 Task Subspace에서 나타났지만 new-class accuracy가 가장 크게 감소했다. Generic feature protection이 stability는 높이지만 plasticity를 제한하는 결과다.
3. Inter Subspace는 rank 3-68의 작은 basis만 보호하면서 final accuracy +0.15%p, forgetting -0.49%p를 기록했다. 선택적 discriminative direction protection이라는 연구 가설에 가장 잘 부합하는 controlled condition이다.
4. Inter+Intra는 가장 높은 final/average accuracy와 old accuracy를 기록했지만 여러 후반 layer에서 rank가 768로 포화됐다. 따라서 성능 수치와 별개로 현재의 단순 누적 intra basis는 장기 CIL에서 scalable하지 않다.
5. Drift 감소 폭은 Task/Inter+Intra가 크지만 final accuracy 차이는 작다. Representation preservation만으로 classifier calibration과 class competition 문제가 모두 해결되지는 않는다.

## 10. 한계 및 다음 실험

- Seed 1993 한 번의 결과이므로 통계적 유의성을 주장할 수 없다.
- Method 간 final accuracy 차이 0.15-0.26%p는 Task 0 성능 차이 약 0.2%p와 비슷한 크기다. Multi-seed mean/std 검증이 필요하다.
- Official CDS(APR+SCC+synthetic CA)는 이번 네 조건과 함께 실행되지 않았으므로 이 문서에서 직접 성능 비교하지 않는다.
- Task/Intra basis의 rank saturation을 막기 위한 global rank budget, per-layer cap, forgetting factor 또는 incremental SVD가 필요하다.
- Inter-only protection의 energy threshold와 protected layer 선택 ablation이 필요하다.
- 동일 seed에서 `inter_energy`, `intra_energy`, 마지막 4개 layer만 보호하는 설정을 비교하면 stability-plasticity trade-off를 더 명확히 볼 수 있다.

## 11. 원본 결과 파일

- Prototype Baseline: [`results/cdsp/cifar224/prototype_baseline/seed_1993`](results/cdsp/cifar224/prototype_baseline/seed_1993)
- Task Subspace: [`results/cdsp/cifar224/task_subspace/seed_1993`](results/cdsp/cifar224/task_subspace/seed_1993)
- Inter Subspace: [`results/cdsp/cifar224/inter_subspace/seed_1993`](results/cdsp/cifar224/inter_subspace/seed_1993)
- Inter + Intra: [`results/cdsp/cifar224/inter_intra_subspace/seed_1993`](results/cdsp/cifar224/inter_intra_subspace/seed_1993)

각 directory에는 `metrics.json`, `task_metrics.csv`, `task_accuracy_matrix.csv`, `class_accuracy.csv`, `subspace_stats.csv`, `prototype_drift.csv` 및 task별 checkpoint가 저장되어 있다.
