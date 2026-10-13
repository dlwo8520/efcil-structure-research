# PCDS / PCMD Experiments

## Existing PCDS Baseline

`RESULTS_PCMD.md` did not previously exist in this codebase. This baseline section was initialized from the completed PCDS seed-0 output specified in the experiment config; metrics were not hard-coded.

| Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |
|---:|---:|---:|---:|---:|
| 94.00 | 89.26 | 88.88 | 92.70 | 6.84 |

<!-- BEGIN GENERATED PCDS STRUCTURE SGDS -->

## PCDS + SGDS Activation Guidance Experiment

### Method

이 실험은 SGDS 공식 reproduction이 아니다. 기존 PCDS의 APR, SCC, cosine classifier, Gaussian synthetic-feature classifier adaptation을 유지하고, [SGDS 논문](https://arxiv.org/abs/2601.21345)의 Semantic Strategy Formulation, Semantic Exploration, Activation Compaction(Eq. 4-10)만 마지막 ViT block의 FFN-output adapter 입력에 추가했다.

SGDS의 task-specific adapter bank, inference-time retrieval/fusion, entropy selection, SGDS prototype classifier, MOS alignment은 사용하지 않았다. 학습 objective는 기존과 동일한 `CE + 2.0 * SCC`이며 SGDS loss는 추가하지 않았다.

- Eq.4: frozen pretrained backbone으로 누적 저장한 class semantic prototype의 cosine-softmax
- Eq.5-6: old/current probability mass. Eq.6은 수식과 Algorithm에 따라 자기 class를 포함
- Eq.7: semantically weighted old class counter를 이용한 Knowledge Reuse
- Eq.8-9: global counter 기반 New Subspace Allocation과 class별 strategy switch
- Eq.10: epoch 6-10 class-specific counter 기반 Activation Compaction
- Counter: Bernoulli와 Top-K를 모두 지난 실제 non-zero dimension만 sample당 1회 count

논문은 `k=0.6`을 sparsity ratio라고 부르지만 Algorithm 1은 `TopK(floor(kN))`을 유지한다. 본 구현은 수식/알고리즘을 따라 60% retain, 40% zero로 해석했다. ViT token tensor는 sample별 mean absolute magnitude로 dimension score를 만들고 동일 dimension mask를 모든 token에 broadcast했다. Stochastic mask는 training-only이며 evaluation에는 적용하지 않았다.

### Experimental Setting

| Item | Value |
|---|---|
| Dataset | CIFAR-100 |
| Backbone | `vit_base_patch16_224.augreg_in21k` |
| Tasks / seed | 10 × 10 / 0 |
| Epoch / batch | 10 per task / 48 |
| Adapter | FFN-output nonlinear bottleneck 64 |
| APR | gamma=10.0, rho=10.0 |
| SCC | lambda=2.0 |
| Classifier adaptation | 5 epochs, 256 Gaussian features/class |
| SGDS target | final ViT block only (layer 11) |
| SGDS k / beta / gamma | 0.6 / 0.5 / 1.0 |
| Resolved retain / zero target | 60% / 40% |
| Replay / Task ID | no raw replay / none |
| Adapter trainable params | 1,208,064 |
| SGDS trainable params | 0 |

### Main Results

| Method | Avg Incremental Acc | Final Acc | Old Acc | New Acc | Forgetting |
|---|---:|---:|---:|---:|---:|
| PCDS | 94.00 | 89.26 | 88.88 | 92.70 | 6.84 |
| PCDS + SGDS | 94.12 | 89.33 | 88.94 | 92.80 | 6.69 |
| Delta | +0.12%p | +0.07%p | +0.07%p | +0.10%p | -0.16%p |

### Task-wise Accuracy Matrix

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 99.40 | - | - | - | - | - | - | - | - | - | 99.40 |
| Task1 | 98.70 | 97.60 | - | - | - | - | - | - | - | - | 98.15 |
| Task2 | 96.80 | 95.70 | 98.20 | - | - | - | - | - | - | - | 96.90 |
| Task3 | 95.20 | 94.80 | 97.40 | 96.20 | - | - | - | - | - | - | 95.90 |
| Task4 | 93.70 | 91.70 | 96.30 | 94.20 | 95.40 | - | - | - | - | - | 94.26 |
| Task5 | 92.10 | 91.70 | 95.00 | 93.40 | 95.30 | 93.90 | - | - | - | - | 93.57 |
| Task6 | 89.90 | 88.80 | 94.20 | 93.50 | 95.30 | 93.30 | 95.60 | - | - | - | 92.94 |
| Task7 | 88.00 | 87.80 | 93.50 | 91.70 | 92.50 | 89.00 | 93.70 | 89.80 | - | - | 90.75 |
| Task8 | 85.70 | 84.90 | 91.50 | 91.10 | 90.60 | 88.60 | 93.20 | 90.00 | 94.40 | - | 90.00 |
| Task9 | 82.90 | 83.00 | 91.60 | 90.30 | 89.50 | 86.90 | 92.50 | 89.00 | 94.80 | 92.80 | 89.33 |

### SGDS Semantic Strategy Statistics

| Task | New classes | Reuse | New allocation | Reuse ratio |
|---:|---:|---:|---:|---:|
| 0 | 10 | 0 | 10 | 0.0% |
| 1 | 10 | 1 | 9 | 10.0% |
| 2 | 10 | 10 | 0 | 100.0% |
| 3 | 10 | 10 | 0 | 100.0% |
| 4 | 10 | 10 | 0 | 100.0% |
| 5 | 10 | 10 | 0 | 100.0% |
| 6 | 10 | 10 | 0 | 100.0% |
| 7 | 10 | 10 | 0 | 100.0% |
| 8 | 10 | 10 | 0 | 100.0% |
| 9 | 10 | 10 | 0 | 100.0% |

Class-level Eq.4-6 results:

| Task | Class | S_old | S_new | Strategy |
|---:|---:|---:|---:|---|
| 0 | 0 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 1 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 2 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 3 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 4 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 5 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 6 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 7 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 8 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 0 | 9 | 0.000000 | 1.000000 | NEW_SUBSPACE_ALLOCATION |
| 1 | 10 | 0.499919 | 0.500081 | NEW_SUBSPACE_ALLOCATION |
| 1 | 11 | 0.505484 | 0.494516 | KNOWLEDGE_REUSE |
| 1 | 12 | 0.452742 | 0.547259 | NEW_SUBSPACE_ALLOCATION |
| 1 | 13 | 0.460204 | 0.539796 | NEW_SUBSPACE_ALLOCATION |
| 1 | 14 | 0.458071 | 0.541929 | NEW_SUBSPACE_ALLOCATION |
| 1 | 15 | 0.484264 | 0.515736 | NEW_SUBSPACE_ALLOCATION |
| 1 | 16 | 0.465240 | 0.534760 | NEW_SUBSPACE_ALLOCATION |
| 1 | 17 | 0.452958 | 0.547042 | NEW_SUBSPACE_ALLOCATION |
| 1 | 18 | 0.449649 | 0.550351 | NEW_SUBSPACE_ALLOCATION |
| 1 | 19 | 0.453353 | 0.546647 | NEW_SUBSPACE_ALLOCATION |
| 2 | 20 | 0.667315 | 0.332685 | KNOWLEDGE_REUSE |
| 2 | 21 | 0.647111 | 0.352889 | KNOWLEDGE_REUSE |
| 2 | 22 | 0.657611 | 0.342389 | KNOWLEDGE_REUSE |
| 2 | 23 | 0.655434 | 0.344566 | KNOWLEDGE_REUSE |
| 2 | 24 | 0.662539 | 0.337461 | KNOWLEDGE_REUSE |
| 2 | 25 | 0.670379 | 0.329621 | KNOWLEDGE_REUSE |
| 2 | 26 | 0.667059 | 0.332941 | KNOWLEDGE_REUSE |
| 2 | 27 | 0.669439 | 0.330561 | KNOWLEDGE_REUSE |
| 2 | 28 | 0.654507 | 0.345493 | KNOWLEDGE_REUSE |
| 2 | 29 | 0.641162 | 0.358839 | KNOWLEDGE_REUSE |
| 3 | 30 | 0.742608 | 0.257392 | KNOWLEDGE_REUSE |
| 3 | 31 | 0.742899 | 0.257101 | KNOWLEDGE_REUSE |
| 3 | 32 | 0.737328 | 0.262672 | KNOWLEDGE_REUSE |
| 3 | 33 | 0.730898 | 0.269102 | KNOWLEDGE_REUSE |
| 3 | 34 | 0.732836 | 0.267164 | KNOWLEDGE_REUSE |
| 3 | 35 | 0.746163 | 0.253837 | KNOWLEDGE_REUSE |
| 3 | 36 | 0.731190 | 0.268810 | KNOWLEDGE_REUSE |
| 3 | 37 | 0.739632 | 0.260368 | KNOWLEDGE_REUSE |
| 3 | 38 | 0.752169 | 0.247831 | KNOWLEDGE_REUSE |
| 3 | 39 | 0.745804 | 0.254196 | KNOWLEDGE_REUSE |
| 4 | 40 | 0.785980 | 0.214020 | KNOWLEDGE_REUSE |
| 4 | 41 | 0.790769 | 0.209231 | KNOWLEDGE_REUSE |
| 4 | 42 | 0.791746 | 0.208253 | KNOWLEDGE_REUSE |
| 4 | 43 | 0.791324 | 0.208676 | KNOWLEDGE_REUSE |
| 4 | 44 | 0.789291 | 0.210709 | KNOWLEDGE_REUSE |
| 4 | 45 | 0.791816 | 0.208184 | KNOWLEDGE_REUSE |
| 4 | 46 | 0.788220 | 0.211780 | KNOWLEDGE_REUSE |
| 4 | 47 | 0.796839 | 0.203161 | KNOWLEDGE_REUSE |
| 4 | 48 | 0.792487 | 0.207513 | KNOWLEDGE_REUSE |
| 4 | 49 | 0.788813 | 0.211187 | KNOWLEDGE_REUSE |
| 5 | 50 | 0.824420 | 0.175580 | KNOWLEDGE_REUSE |
| 5 | 51 | 0.826075 | 0.173925 | KNOWLEDGE_REUSE |
| 5 | 52 | 0.833501 | 0.166499 | KNOWLEDGE_REUSE |
| 5 | 53 | 0.824623 | 0.175377 | KNOWLEDGE_REUSE |
| 5 | 54 | 0.824429 | 0.175571 | KNOWLEDGE_REUSE |
| 5 | 55 | 0.818775 | 0.181225 | KNOWLEDGE_REUSE |
| 5 | 56 | 0.824524 | 0.175476 | KNOWLEDGE_REUSE |
| 5 | 57 | 0.818707 | 0.181293 | KNOWLEDGE_REUSE |
| 5 | 58 | 0.826782 | 0.173218 | KNOWLEDGE_REUSE |
| 5 | 59 | 0.824839 | 0.175161 | KNOWLEDGE_REUSE |
| 6 | 60 | 0.852757 | 0.147243 | KNOWLEDGE_REUSE |
| 6 | 61 | 0.837276 | 0.162724 | KNOWLEDGE_REUSE |
| 6 | 62 | 0.848069 | 0.151931 | KNOWLEDGE_REUSE |
| 6 | 63 | 0.833810 | 0.166190 | KNOWLEDGE_REUSE |
| 6 | 64 | 0.844104 | 0.155896 | KNOWLEDGE_REUSE |
| 6 | 65 | 0.851156 | 0.148844 | KNOWLEDGE_REUSE |
| 6 | 66 | 0.836868 | 0.163132 | KNOWLEDGE_REUSE |
| 6 | 67 | 0.848646 | 0.151354 | KNOWLEDGE_REUSE |
| 6 | 68 | 0.843617 | 0.156383 | KNOWLEDGE_REUSE |
| 6 | 69 | 0.845233 | 0.154767 | KNOWLEDGE_REUSE |
| 7 | 70 | 0.875613 | 0.124387 | KNOWLEDGE_REUSE |
| 7 | 71 | 0.876959 | 0.123041 | KNOWLEDGE_REUSE |
| 7 | 72 | 0.880443 | 0.119557 | KNOWLEDGE_REUSE |
| 7 | 73 | 0.861035 | 0.138965 | KNOWLEDGE_REUSE |
| 7 | 74 | 0.861718 | 0.138282 | KNOWLEDGE_REUSE |
| 7 | 75 | 0.868201 | 0.131800 | KNOWLEDGE_REUSE |
| 7 | 76 | 0.873991 | 0.126009 | KNOWLEDGE_REUSE |
| 7 | 77 | 0.862119 | 0.137881 | KNOWLEDGE_REUSE |
| 7 | 78 | 0.872936 | 0.127064 | KNOWLEDGE_REUSE |
| 7 | 79 | 0.859810 | 0.140189 | KNOWLEDGE_REUSE |
| 8 | 80 | 0.879822 | 0.120178 | KNOWLEDGE_REUSE |
| 8 | 81 | 0.884445 | 0.115555 | KNOWLEDGE_REUSE |
| 8 | 82 | 0.875117 | 0.124883 | KNOWLEDGE_REUSE |
| 8 | 83 | 0.882539 | 0.117461 | KNOWLEDGE_REUSE |
| 8 | 84 | 0.876837 | 0.123163 | KNOWLEDGE_REUSE |
| 8 | 85 | 0.874395 | 0.125605 | KNOWLEDGE_REUSE |
| 8 | 86 | 0.876835 | 0.123165 | KNOWLEDGE_REUSE |
| 8 | 87 | 0.884769 | 0.115231 | KNOWLEDGE_REUSE |
| 8 | 88 | 0.879575 | 0.120425 | KNOWLEDGE_REUSE |
| 8 | 89 | 0.876055 | 0.123945 | KNOWLEDGE_REUSE |
| 9 | 90 | 0.892406 | 0.107595 | KNOWLEDGE_REUSE |
| 9 | 91 | 0.898947 | 0.101053 | KNOWLEDGE_REUSE |
| 9 | 92 | 0.889601 | 0.110399 | KNOWLEDGE_REUSE |
| 9 | 93 | 0.889479 | 0.110522 | KNOWLEDGE_REUSE |
| 9 | 94 | 0.895461 | 0.104539 | KNOWLEDGE_REUSE |
| 9 | 95 | 0.888244 | 0.111756 | KNOWLEDGE_REUSE |
| 9 | 96 | 0.895271 | 0.104729 | KNOWLEDGE_REUSE |
| 9 | 97 | 0.894379 | 0.105621 | KNOWLEDGE_REUSE |
| 9 | 98 | 0.893769 | 0.106231 | KNOWLEDGE_REUSE |
| 9 | 99 | 0.896113 | 0.103887 | KNOWLEDGE_REUSE |

### Activation Sparsity Statistics

| Task | Layer | Target zero | Actual non-zero | Actual zero | Mean active dims | Global max | Class max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 11 | 40.10 | 59.47 | 40.53 | 456.7 | 30805 | 3295 |
| 1 | 11 | 40.10 | 58.41 | 41.59 | 448.6 | 60807 | 3260 |
| 2 | 11 | 40.10 | 52.42 | 47.58 | 402.5 | 87848 | 2870 |
| 3 | 11 | 40.10 | 54.43 | 45.57 | 418.0 | 116012 | 2959 |
| 4 | 11 | 40.10 | 55.63 | 44.37 | 427.2 | 144721 | 3015 |
| 5 | 11 | 40.10 | 56.38 | 43.62 | 433.0 | 173702 | 3071 |
| 6 | 11 | 40.10 | 56.83 | 43.17 | 436.4 | 202990 | 3071 |
| 7 | 11 | 40.10 | 57.36 | 42.64 | 440.5 | 232598 | 3110 |
| 8 | 11 | 40.10 | 57.58 | 42.42 | 442.2 | 262445 | 3134 |
| 9 | 11 | 40.10 | 57.89 | 42.11 | 444.6 | 292030 | 3118 |

### Representation Effective Rank

`r_0.99`는 current-task class covariance eigen-energy 99%를 설명하는 최소 rank다.

| Task | Mean r_0.99 | Min | Max |
|---:|---:|---:|---:|
| 0 | 313.9 | 257 | 336 |
| 1 | 322.5 | 307 | 332 |
| 2 | 311.9 | 288 | 326 |
| 3 | 310.0 | 289 | 323 |
| 4 | 294.5 | 263 | 327 |
| 5 | 288.2 | 259 | 315 |
| 6 | 287.8 | 261 | 316 |
| 7 | 285.5 | 253 | 304 |
| 8 | 275.6 | 261 | 288 |
| 9 | 267.1 | 255 | 279 |

### Classifier Adaptation Analysis

| Task | Seen before | Seen after | Delta | Old before | Old after | New before | New after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 99.40 | 99.40 | +0.00%p | N/A | N/A | 99.40 | 99.40 |
| 1 | 90.40 | 98.15 | +7.75%p | 82.80 | 98.70 | 98.00 | 97.60 |
| 2 | 80.50 | 96.90 | +16.40%p | 70.95 | 96.25 | 99.60 | 98.20 |
| 3 | 80.10 | 95.90 | +15.80%p | 73.87 | 95.80 | 98.80 | 96.20 |
| 4 | 72.76 | 94.26 | +21.50%p | 66.07 | 93.98 | 99.50 | 95.40 |
| 5 | 74.07 | 93.57 | +19.50%p | 69.02 | 93.50 | 99.30 | 93.90 |
| 6 | 76.14 | 92.94 | +16.80%p | 72.28 | 92.50 | 99.30 | 95.60 |
| 7 | 60.17 | 90.75 | +30.57%p | 54.71 | 90.89 | 98.40 | 89.80 |
| 8 | 52.87 | 90.00 | +37.13%p | 47.14 | 89.45 | 98.70 | 94.40 |
| 9 | 52.55 | 89.33 | +36.78%p | 47.46 | 88.94 | 98.40 | 92.80 |

### Interpretation

1. Final accuracy 변화: +0.07%p. 따라서 seed-0에서 SGDS activation guidance는 PCDS final accuracy를 올렸다.
2. Forgetting 변화: -0.16%p. 음수이면 감소를 뜻한다.
3. Old accuracy 변화: +0.07%p.
4. New-class accuracy 변화: +0.10%p. 음수이면 plasticity 손실이다.
5. 실제 평균 zero ratio는 43.36%이고 non-zero ratio는 56.64%다. 논문의 명칭상 60% sparsity와는 다르지만 Algorithm의 k=0.6 retain 해석과 비교해야 한다.
6. Knowledge Reuse 비율은 전반부 평균 62.0%, 후반부 평균 100.0%로 변했다.
7. Mean r_0.99는 Task0 313.9에서 Task9 267.1로 변했다. 기존 PCDS run에는 동일 rank diagnostic이 없어 SGDS가 baseline 대비 rank를 감소시켰는지는 이 값만으로 확정할 수 없다.
8. Task9 CA는 Seen accuracy를 52.55에서 89.33로 +36.78%p 변화시켰다. 전체 PCDS+SGDS 조합 실험이므로 APR/SCC/CA와 SGDS 각각의 인과 효과를 완전히 분리할 수 없다.

### Integrity Diagnostics

- Peak CUDA allocated: 3.37 GiB
- Optimizer skipped steps: 0
- NaN/Inf steps: 0
- Evaluation masking: disabled; single expanding head over all seen classes
- Counter source: final Bernoulli + Top-K non-zero dimensions only

### Limitations

- Seed 0 한 번의 실험이므로 통계적 결론이 아니다.
- SGDS 공식 reproduction이 아니라 PCDS에 activation mechanism만 추가한 조합 실험이다.
- SGDS 원 논문의 CIFAR-100 protocol은 20×5 classes와 20 epochs/task지만, 본 실험은 PCDS 비교를 위해 10×10과 10 epochs/task를 사용했다.
- SGDS 원 논문의 task-specific adapter retrieval/fusion을 사용하지 않았다.
- 논문의 k 명칭과 Top-K 알고리즘 사이 모호성 때문에 수식상 60% retain으로 구현했다.
- 따라서 이 결과를 SGDS 논문의 reported CIFAR-100 score와 직접 비교할 수 없다.

<!-- END GENERATED PCDS STRUCTURE SGDS -->

<!-- BEGIN GENERATED CARS196 PCDS SGDS -->

## Cars-196: PCDS vs PCDS + SGDS

### Experimental Setting

| Item | Value |
|---|---|
| Dataset | Stanford Cars-196 official train/test split |
| Task split | `[20, 20, 20, 20, 20, 20, 20, 20, 20, 16]` = 20 classes × 9 + 16 classes |
| Training seed / class-order seed | 0 / 1993 |
| Class order | `cds_cars196_seed1993`; SHA-256 `09d54824030fa08e1f3983fbeaf0e93208320db501a0988fe56a137816b84f8e` |
| Backbone | `vit_base_patch16_224.augreg_in21k` |
| Epochs | Task0 70; Task1-9 40 |
| Batch / SGD lr | 32 / 0.005 |
| Adapter | parallel attention K/V nonlinear bottleneck 64 |
| Cosine classifier | scale=20, margin=0.3 |
| APR / SCC | gamma=10, rho=10; lambda=5 |
| CA | 5 epochs, 256 Gaussian features/class |
| SGDS | final block, k=0.6 retain, beta=0.5, gamma=1 |
| Replay / Task ID | none / none |
| GPU | physical GPU 3 only |
| PCDS / SGDS trainable adapter params | 2,416,128 / 2,416,128 |

정확한 196-class order는 각 run의 `class_order.json`에 저장했다:

```text
[59, 134, 162, 9, 42, 72, 118, 99, 109, 79, 43, 100, 16, 144, 27, 10, 172, 26, 191, 80, 189, 6, 87, 153, 97, 70, 107, 0, 88, 178, 35, 4, 2, 170, 171, 141, 51, 166, 145, 188, 17, 167, 101, 116, 23, 157, 180, 129, 68, 85, 32, 54, 103, 104, 19, 113, 106, 175, 36, 11, 159, 169, 8, 98, 3, 28, 57, 147, 138, 187, 56, 49, 44, 125, 62, 151, 165, 94, 55, 92, 12, 122, 78, 66, 193, 15, 135, 168, 184, 91, 137, 186, 52, 126, 40, 112, 65, 142, 64, 136, 45, 77, 89, 154, 90, 71, 190, 74, 30, 119, 96, 84, 67, 50, 183, 150, 69, 21, 18, 111, 108, 58, 174, 131, 148, 110, 194, 1, 173, 83, 81, 60, 13, 163, 14, 156, 63, 117, 5, 22, 143, 121, 38, 41, 82, 127, 114, 20, 31, 53, 37, 195, 192, 130, 152, 176, 86, 76, 24, 34, 181, 149, 33, 128, 182, 155, 146, 139, 120, 140, 102, 47, 25, 158, 123, 46, 164, 61, 7, 115, 75, 133, 160, 105, 132, 179, 124, 48, 73, 93, 39, 95, 29, 177, 185, 161]
```

### Main Results

| Method | Avg Inc. Acc | Final Acc | Old Acc | New Acc | Forgetting | Old-New gap | Peak CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCDS | 59.45 | 33.74 | 28.09 | 96.97 | 67.02 | -68.88%p | 2.98 GiB |
| PCDS + SGDS | 50.91 | 15.48 | 8.09 | 98.18 | 87.50 | -90.09%p | 2.98 GiB |
| Delta | -8.53%p | -18.26%p | -20.00%p | +1.21%p | +20.48%p | -21.21%p | +0.01 GiB |

### Task-wise Results

#### PCDS

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 96.78 | - | - | - | - | - | - | - | - | - | 96.78 |
| Task1 | 81.38 | 96.30 | - | - | - | - | - | - | - | - | 88.84 |
| Task2 | 58.00 | 86.16 | 95.81 | - | - | - | - | - | - | - | 79.82 |
| Task3 | 27.57 | 59.07 | 87.30 | 97.31 | - | - | - | - | - | - | 67.46 |
| Task4 | 6.56 | 32.94 | 63.13 | 90.09 | 96.06 | - | - | - | - | - | 57.31 |
| Task5 | 0.72 | 14.92 | 40.81 | 70.75 | 83.74 | 94.89 | - | - | - | - | 50.81 |
| Task6 | 0.00 | 4.53 | 21.70 | 44.68 | 64.41 | 78.03 | 94.55 | - | - | - | 43.76 |
| Task7 | 0.00 | 1.31 | 7.40 | 24.48 | 44.21 | 66.63 | 81.19 | 92.79 | - | - | 39.58 |
| Task8 | 0.00 | 0.12 | 5.30 | 10.40 | 31.40 | 44.18 | 63.61 | 81.30 | 94.23 | - | 36.39 |
| Task9 | 0.00 | 0.00 | 2.22 | 6.00 | 19.09 | 26.72 | 45.92 | 70.54 | 85.07 | 96.97 | 33.74 |

#### PCDS + SGDS

| trained through | Task0 | Task1 | Task2 | Task3 | Task4 | Task5 | Task6 | Task7 | Task8 | Task9 | Seen Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Task0 | 96.30 | - | - | - | - | - | - | - | - | - | 96.30 |
| Task1 | 82.58 | 96.30 | - | - | - | - | - | - | - | - | 89.44 |
| Task2 | 58.47 | 87.11 | 95.68 | - | - | - | - | - | - | - | 80.26 |
| Task3 | 24.94 | 55.37 | 84.09 | 96.94 | - | - | - | - | - | - | 64.98 |
| Task4 | 6.92 | 25.54 | 52.53 | 85.92 | 95.32 | - | - | - | - | - | 52.82 |
| Task5 | 0.95 | 4.18 | 18.37 | 46.27 | 75.25 | 94.66 | - | - | - | - | 39.90 |
| Task6 | 0.00 | 0.00 | 2.96 | 14.44 | 29.56 | 55.46 | 94.93 | - | - | - | 28.03 |
| Task7 | 0.00 | 0.00 | 0.00 | 1.47 | 8.00 | 22.33 | 52.10 | 95.35 | - | - | 22.27 |
| Task8 | 0.00 | 0.00 | 0.00 | 0.00 | 2.09 | 6.29 | 16.96 | 58.44 | 96.24 | - | 19.67 |
| Task9 | 0.00 | 0.00 | 0.00 | 0.00 | 0.37 | 0.48 | 2.35 | 24.45 | 46.55 | 98.18 | 15.48 |

### SGDS Semantic Strategy

| Task | Old | New | Reuse | Allocation | Reuse ratio | mean S_old | mean S_new |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 20 | 0 | 20 | 0.0% | 0.000000 | 1.000000 |
| 1 | 20 | 20 | 11 | 9 | 55.0% | 0.501886 | 0.498114 |
| 2 | 40 | 20 | 20 | 0 | 100.0% | 0.664522 | 0.335478 |
| 3 | 60 | 20 | 20 | 0 | 100.0% | 0.746524 | 0.253476 |
| 4 | 80 | 20 | 20 | 0 | 100.0% | 0.793989 | 0.206011 |
| 5 | 100 | 20 | 20 | 0 | 100.0% | 0.831948 | 0.168052 |
| 6 | 120 | 20 | 20 | 0 | 100.0% | 0.856070 | 0.143930 |
| 7 | 140 | 20 | 20 | 0 | 100.0% | 0.873867 | 0.126133 |
| 8 | 160 | 20 | 20 | 0 | 100.0% | 0.888093 | 0.111907 |
| 9 | 180 | 16 | 16 | 0 | 100.0% | 0.916885 | 0.083115 |

`corr(|Y_old|, mean S_old) = 0.8165`. 이는 class 수 증가와 old probability mass 증가가 얼마나 함께 움직였는지 나타내며 수식 자체는 변경하지 않았다.

### Fine-Grained Semantic Similarity

| Task | Class | max old cosine | mean old cosine | Strategy | Final class acc |
|---:|---:|---:|---:|---|---:|
| 0 | 0 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 1 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 2 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 3 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 4 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 5 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 6 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 7 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 8 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 9 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 10 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 11 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 12 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 13 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 14 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 15 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 16 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 17 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 18 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 0 | 19 | nan | nan | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 20 | 0.929987 | 0.794391 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 21 | 0.942060 | 0.841389 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 22 | 0.811881 | 0.475457 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 23 | 0.917688 | 0.731047 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 24 | 0.911465 | 0.787603 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 25 | 0.978773 | 0.505212 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 26 | 0.846261 | 0.651107 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 27 | 0.769085 | 0.503884 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 28 | 0.930542 | 0.746941 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 29 | 0.883385 | 0.716626 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 30 | 0.962721 | 0.770081 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 31 | 0.934410 | 0.828494 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 32 | 0.945275 | 0.838262 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 33 | 0.929577 | 0.814856 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 34 | 0.902643 | 0.768328 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 35 | 0.928335 | 0.760122 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 36 | 0.958968 | 0.798613 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 37 | 0.939375 | 0.837442 | KNOWLEDGE_REUSE | 0.00 |
| 1 | 38 | 0.919288 | 0.604410 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 1 | 39 | 0.945928 | 0.678729 | NEW_SUBSPACE_ALLOCATION | 0.00 |
| 2 | 40 | 0.950342 | 0.723479 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 41 | 0.907158 | 0.734137 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 42 | 0.901785 | 0.652529 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 43 | 0.942689 | 0.765669 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 44 | 0.957542 | 0.810871 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 45 | 0.873393 | 0.736808 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 46 | 0.917508 | 0.763740 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 47 | 0.925528 | 0.771875 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 48 | 0.832045 | 0.557191 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 49 | 0.876024 | 0.635872 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 50 | 0.929283 | 0.801991 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 51 | 0.902684 | 0.629300 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 52 | 0.912524 | 0.712231 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 53 | 0.960135 | 0.773331 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 54 | 0.942095 | 0.797157 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 55 | 0.871418 | 0.652139 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 56 | 0.943516 | 0.761603 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 57 | 0.926350 | 0.781395 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 58 | 0.943756 | 0.780026 | KNOWLEDGE_REUSE | 0.00 |
| 2 | 59 | 0.951996 | 0.728618 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 60 | 0.938543 | 0.703593 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 61 | 0.920668 | 0.796735 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 62 | 0.962889 | 0.767011 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 63 | 0.878706 | 0.730497 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 64 | 0.960903 | 0.806950 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 65 | 0.962290 | 0.794264 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 66 | 0.956838 | 0.760683 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 67 | 0.951851 | 0.716290 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 68 | 0.952873 | 0.793588 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 69 | 0.964714 | 0.799428 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 70 | 0.887557 | 0.689368 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 71 | 0.936279 | 0.781922 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 72 | 0.909819 | 0.692917 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 73 | 0.916261 | 0.751746 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 74 | 0.951896 | 0.789579 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 75 | 0.834209 | 0.667711 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 76 | 0.946930 | 0.521916 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 77 | 0.949060 | 0.721802 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 78 | 0.925994 | 0.725010 | KNOWLEDGE_REUSE | 0.00 |
| 3 | 79 | 0.871489 | 0.722344 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 80 | 0.949173 | 0.805745 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 81 | 0.858696 | 0.705297 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 82 | 0.955294 | 0.803994 | KNOWLEDGE_REUSE | 4.17 |
| 4 | 83 | 0.968195 | 0.805113 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 84 | 0.913886 | 0.731421 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 85 | 0.960987 | 0.755119 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 86 | 0.933801 | 0.757262 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 87 | 0.924658 | 0.601420 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 88 | 0.943869 | 0.796058 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 89 | 0.921372 | 0.794769 | KNOWLEDGE_REUSE | 2.50 |
| 4 | 90 | 0.954854 | 0.806979 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 91 | 0.964887 | 0.809467 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 92 | 0.960893 | 0.708171 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 93 | 0.949840 | 0.685018 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 94 | 0.948243 | 0.786012 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 95 | 0.971962 | 0.630030 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 96 | 0.929795 | 0.790001 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 97 | 0.955474 | 0.652727 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 98 | 0.945768 | 0.675715 | KNOWLEDGE_REUSE | 0.00 |
| 4 | 99 | 0.958560 | 0.806369 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 100 | 0.939615 | 0.742841 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 101 | 0.945853 | 0.739371 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 102 | 0.953892 | 0.673660 | KNOWLEDGE_REUSE | 2.44 |
| 5 | 103 | 0.955906 | 0.755381 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 104 | 0.968633 | 0.648395 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 105 | 0.954198 | 0.806525 | KNOWLEDGE_REUSE | 2.22 |
| 5 | 106 | 0.920522 | 0.723567 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 107 | 0.962782 | 0.590654 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 108 | 0.958119 | 0.750717 | KNOWLEDGE_REUSE | 4.55 |
| 5 | 109 | 0.962606 | 0.717005 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 110 | 0.944826 | 0.800771 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 111 | 0.950454 | 0.645911 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 112 | 0.952634 | 0.737812 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 113 | 0.961420 | 0.815719 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 114 | 0.952584 | 0.789899 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 115 | 0.950946 | 0.717879 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 116 | 0.969487 | 0.609504 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 117 | 0.961204 | 0.787799 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 118 | 0.958407 | 0.787684 | KNOWLEDGE_REUSE | 0.00 |
| 5 | 119 | 0.925841 | 0.682447 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 120 | 0.970186 | 0.706061 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 121 | 0.956257 | 0.726000 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 122 | 0.918819 | 0.707974 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 123 | 0.943784 | 0.801739 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 124 | 0.966097 | 0.740474 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 125 | 0.974972 | 0.649550 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 126 | 0.952011 | 0.758224 | KNOWLEDGE_REUSE | 4.65 |
| 6 | 127 | 0.950973 | 0.785755 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 128 | 0.921443 | 0.699474 | KNOWLEDGE_REUSE | 7.32 |
| 6 | 129 | 0.918897 | 0.758065 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 130 | 0.933314 | 0.736060 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 131 | 0.976529 | 0.785019 | KNOWLEDGE_REUSE | 16.28 |
| 6 | 132 | 0.967606 | 0.795438 | KNOWLEDGE_REUSE | 2.38 |
| 6 | 133 | 0.939681 | 0.782537 | KNOWLEDGE_REUSE | 2.33 |
| 6 | 134 | 0.956340 | 0.781731 | KNOWLEDGE_REUSE | 2.33 |
| 6 | 135 | 0.923715 | 0.685782 | KNOWLEDGE_REUSE | 11.11 |
| 6 | 136 | 0.969468 | 0.538876 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 137 | 0.968760 | 0.722077 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 138 | 0.907182 | 0.736376 | KNOWLEDGE_REUSE | 0.00 |
| 6 | 139 | 0.972305 | 0.814945 | KNOWLEDGE_REUSE | 0.00 |
| 7 | 140 | 0.942479 | 0.745319 | KNOWLEDGE_REUSE | 21.74 |
| 7 | 141 | 0.976409 | 0.669424 | KNOWLEDGE_REUSE | 47.50 |
| 7 | 142 | 0.948428 | 0.712245 | KNOWLEDGE_REUSE | 5.56 |
| 7 | 143 | 0.952192 | 0.778182 | KNOWLEDGE_REUSE | 85.29 |
| 7 | 144 | 0.947378 | 0.796493 | KNOWLEDGE_REUSE | 22.50 |
| 7 | 145 | 0.952884 | 0.800569 | KNOWLEDGE_REUSE | 7.69 |
| 7 | 146 | 0.970497 | 0.788144 | KNOWLEDGE_REUSE | 42.22 |
| 7 | 147 | 0.961624 | 0.715300 | KNOWLEDGE_REUSE | 45.24 |
| 7 | 148 | 0.953651 | 0.760326 | KNOWLEDGE_REUSE | 17.07 |
| 7 | 149 | 0.977767 | 0.636370 | KNOWLEDGE_REUSE | 0.00 |
| 7 | 150 | 0.957319 | 0.725038 | KNOWLEDGE_REUSE | 7.50 |
| 7 | 151 | 0.891145 | 0.602016 | KNOWLEDGE_REUSE | 20.00 |
| 7 | 152 | 0.936500 | 0.769732 | KNOWLEDGE_REUSE | 19.51 |
| 7 | 153 | 0.941561 | 0.760565 | KNOWLEDGE_REUSE | 19.05 |
| 7 | 154 | 0.913621 | 0.645610 | KNOWLEDGE_REUSE | 45.45 |
| 7 | 155 | 0.947055 | 0.760441 | KNOWLEDGE_REUSE | 2.27 |
| 7 | 156 | 0.974527 | 0.631720 | KNOWLEDGE_REUSE | 9.09 |
| 7 | 157 | 0.948609 | 0.798523 | KNOWLEDGE_REUSE | 10.00 |
| 7 | 158 | 0.955105 | 0.763349 | KNOWLEDGE_REUSE | 20.51 |
| 7 | 159 | 0.955965 | 0.798858 | KNOWLEDGE_REUSE | 46.34 |
| 8 | 160 | 0.967601 | 0.804333 | KNOWLEDGE_REUSE | 73.91 |
| 8 | 161 | 0.921963 | 0.708944 | KNOWLEDGE_REUSE | 86.11 |
| 8 | 162 | 0.956978 | 0.782671 | KNOWLEDGE_REUSE | 29.55 |
| 8 | 163 | 0.972312 | 0.798180 | KNOWLEDGE_REUSE | 84.21 |
| 8 | 164 | 0.943602 | 0.765640 | KNOWLEDGE_REUSE | 16.67 |
| 8 | 165 | 0.892408 | 0.725687 | KNOWLEDGE_REUSE | 58.97 |
| 8 | 166 | 0.977818 | 0.689242 | KNOWLEDGE_REUSE | 95.45 |
| 8 | 167 | 0.953039 | 0.792442 | KNOWLEDGE_REUSE | 50.00 |
| 8 | 168 | 0.949958 | 0.775598 | KNOWLEDGE_REUSE | 72.73 |
| 8 | 169 | 0.958141 | 0.782752 | KNOWLEDGE_REUSE | 52.94 |
| 8 | 170 | 0.973479 | 0.655639 | KNOWLEDGE_REUSE | 92.31 |
| 8 | 171 | 0.962238 | 0.750582 | KNOWLEDGE_REUSE | 71.43 |
| 8 | 172 | 0.948695 | 0.755901 | KNOWLEDGE_REUSE | 0.00 |
| 8 | 173 | 0.955018 | 0.722984 | KNOWLEDGE_REUSE | 30.56 |
| 8 | 174 | 0.939115 | 0.619460 | KNOWLEDGE_REUSE | 5.13 |
| 8 | 175 | 0.955934 | 0.779905 | KNOWLEDGE_REUSE | 20.00 |
| 8 | 176 | 0.958808 | 0.801449 | KNOWLEDGE_REUSE | 4.55 |
| 8 | 177 | 0.960565 | 0.671622 | KNOWLEDGE_REUSE | 27.03 |
| 8 | 178 | 0.963104 | 0.712796 | KNOWLEDGE_REUSE | 35.56 |
| 8 | 179 | 0.940069 | 0.525697 | KNOWLEDGE_REUSE | 10.81 |
| 9 | 180 | 0.972771 | 0.738214 | KNOWLEDGE_REUSE | 97.67 |
| 9 | 181 | 0.966978 | 0.802001 | KNOWLEDGE_REUSE | 96.97 |
| 9 | 182 | 0.940386 | 0.737432 | KNOWLEDGE_REUSE | 100.00 |
| 9 | 183 | 0.963480 | 0.602674 | KNOWLEDGE_REUSE | 100.00 |
| 9 | 184 | 0.965331 | 0.791317 | KNOWLEDGE_REUSE | 95.24 |
| 9 | 185 | 0.946877 | 0.701430 | KNOWLEDGE_REUSE | 97.62 |
| 9 | 186 | 0.964574 | 0.597055 | KNOWLEDGE_REUSE | 100.00 |
| 9 | 187 | 0.963663 | 0.793964 | KNOWLEDGE_REUSE | 100.00 |
| 9 | 188 | 0.978877 | 0.602871 | KNOWLEDGE_REUSE | 100.00 |
| 9 | 189 | 0.967001 | 0.779236 | KNOWLEDGE_REUSE | 95.35 |
| 9 | 190 | 0.930417 | 0.754722 | KNOWLEDGE_REUSE | 100.00 |
| 9 | 191 | 0.963530 | 0.797348 | KNOWLEDGE_REUSE | 97.56 |
| 9 | 192 | 0.938610 | 0.768148 | KNOWLEDGE_REUSE | 100.00 |
| 9 | 193 | 0.951017 | 0.794773 | KNOWLEDGE_REUSE | 97.56 |
| 9 | 194 | 0.968087 | 0.701207 | KNOWLEDGE_REUSE | 94.74 |
| 9 | 195 | 0.954205 | 0.794557 | KNOWLEDGE_REUSE | 97.78 |

- corr(max old cosine, Knowledge Reuse indicator): 0.3356
- corr(max old cosine, final class accuracy): 0.2558
- mean max cosine, Reuse / Allocation: 0.9430 / 0.8942

### Effective Rank

| Task | PCDS rank | PCDS+SGDS rank | Delta |
|---:|---:|---:|---:|
| 0 | 39.0 | 39.0 | +0.0 |
| 1 | 38.6 | 38.5 | -0.1 |
| 2 | 37.2 | 37.2 | +0.0 |
| 3 | 36.8 | 36.8 | +0.0 |
| 4 | 36.2 | 36.2 | +0.0 |
| 5 | 36.6 | 36.7 | +0.1 |
| 6 | 34.6 | 34.5 | -0.1 |
| 7 | 34.5 | 34.4 | -0.1 |
| 8 | 33.1 | 33.2 | +0.1 |
| 9 | 33.7 | 33.4 | -0.2 |

### Classifier Adaptation

| Method | Task | Seen before/after | Old before/after | New before/after |
|---|---:|---:|---:|---:|
| PCDS | 0 | 96.78 / 96.78 | N/A / N/A | 96.78 / 96.78 |
| PCDS | 1 | 72.85 / 88.84 | 48.21 / 81.38 | 97.49 / 96.30 |
| PCDS | 2 | 56.57 / 79.82 | 37.35 / 72.08 | 96.30 / 95.81 |
| PCDS | 3 | 36.68 / 67.46 | 16.73 / 57.66 | 97.43 / 97.31 |
| PCDS | 4 | 29.49 / 57.31 | 12.89 / 47.79 | 97.04 / 96.06 |
| PCDS | 5 | 21.80 / 50.81 | 6.34 / 41.79 | 97.39 / 94.89 |
| PCDS | 6 | 15.56 / 43.76 | 2.20 / 35.48 | 97.52 / 94.55 |
| PCDS | 7 | 13.32 / 39.58 | 1.49 / 32.03 | 96.70 / 92.79 |
| PCDS | 8 | 11.99 / 36.39 | 1.70 / 29.39 | 96.99 / 94.23 |
| PCDS | 9 | 8.42 / 33.74 | 0.37 / 28.09 | 98.48 / 96.97 |
| PCDS+SGDS | 0 | 96.30 / 96.30 | N/A / N/A | 96.30 / 96.30 |
| PCDS+SGDS | 1 | 70.05 / 89.44 | 42.48 / 82.58 | 97.61 / 96.30 |
| PCDS+SGDS | 2 | 48.65 / 80.26 | 25.66 / 72.79 | 96.18 / 95.68 |
| PCDS+SGDS | 3 | 27.39 / 64.98 | 4.34 / 54.48 | 97.55 / 96.94 |
| PCDS+SGDS | 4 | 21.31 / 52.82 | 2.72 / 42.37 | 96.92 / 95.32 |
| PCDS+SGDS | 5 | 16.42 / 39.90 | 0.07 / 28.69 | 96.32 / 94.66 |
| PCDS+SGDS | 6 | 13.56 / 28.03 | 0.00 / 17.12 | 96.78 / 94.93 |
| PCDS+SGDS | 7 | 12.03 / 22.27 | 0.00 / 11.90 | 96.82 / 95.35 |
| PCDS+SGDS | 8 | 10.57 / 19.67 | 0.06 / 10.40 | 97.37 / 96.24 |
| PCDS+SGDS | 9 | 8.16 / 15.48 | 0.00 / 8.09 | 99.39 / 98.18 |

### CIFAR-100 Comparison

| Dataset | PCDS Final | PCDS+SGDS Final | Delta Final | PCDS forgetting | SGDS forgetting |
|---|---:|---:|---:|---:|---:|
| CIFAR-100 | 89.26 | 89.33 | +0.07%p | 6.84 | 6.69 |
| Cars-196 | 33.74 | 15.48 | -18.26%p | 67.02 | 87.50 |

### Interpretation

1. Cars-196 Final Acc 변화는 -18.26%p이므로 SGDS는 PCDS 성능을 낮췄다.
2. CIFAR-100의 +0.07%p와 비교하면 Cars 개선폭은 더 크지 않다 (-18.26%p).
3. Old Acc 변화는 -20.00%p이다.
4. Forgetting 변화는 +20.48%p이다. 음수이면 감소다.
5. New Acc 변화는 +1.21%p이다. 음수이면 plasticity 감소다.
6. 평균 effective-rank delta는 -0.04다. 음수이면 SGDS rank가 감소했다.
7. 0%와 100% 사이의 mixed-strategy task는 Cars 1개, CIFAR 1개다.
8. 마지막 task Reuse ratio는 100.0%다.
9. max cosine과 Reuse의 상관은 0.3356다.
10. max cosine과 final class accuracy 상관은 0.2558다. 이 단일 seed 상관만으로 fine-grained geometry가 인과 원인이라고 주장할 수 없다.

### Activation Sparsity

| Task | Requested zero | Actual zero | Actual non-zero |
|---:|---:|---:|---:|
| 0 | 40.10 | 40.54 | 59.46 |
| 1 | 40.10 | 47.39 | 52.61 |
| 2 | 40.10 | 48.25 | 51.75 |
| 3 | 40.10 | 46.25 | 53.75 |
| 4 | 40.10 | 45.19 | 54.81 |
| 5 | 40.10 | 44.28 | 55.72 |
| 6 | 40.10 | 43.80 | 56.20 |
| 7 | 40.10 | 43.38 | 56.62 |
| 8 | 40.10 | 43.02 | 56.98 |
| 9 | 40.10 | 42.42 | 57.58 |

### Integrity and Limitations

- PCDS / SGDS optimizer skipped steps: 0 / 0
- PCDS / SGDS NaN/Inf steps: 0 / 0
- Seed 0 단일 screening run이므로 statistical significance를 주장하지 않는다.
- 논문 보고값 80.23±0.68은 참고값이며 이번 baseline으로 하드코딩하지 않았다.
- PCDS+SGDS는 SGDS 공식 reproduction이 아니라 Eq. 4-10 activation guidance 조합이다.
- SGDS의 task-specific adapter retrieval/fusion은 사용하지 않았다.

<!-- END GENERATED CARS196 PCDS SGDS -->
