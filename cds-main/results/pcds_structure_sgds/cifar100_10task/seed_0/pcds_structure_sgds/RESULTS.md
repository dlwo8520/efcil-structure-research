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
