# Representation / Class-Relation Diagnostic Final Report

> IA5/CAR5 T5 · seed 1993 · frozen Structure checkpoints · oracle diagnostic only

## Executive summary

이번 진단에서 가장 중요한 결과는 세 가지다.

1. **기존 Linear probe의 큰 성능 저하는 representation 한계가 아니라 regularization 설정의 영향이었다.** Cosine head를 linear weight로 정확히 변환했을 때 logits가 일치했고, 약한 L2로 다시 학습한 Linear는 IA5에서 Structure와 0.59 pp 차이, CAR5에서는 Structure보다 1.36 pp 높았다.
2. **실제 오류는 일부 class pair에 집중되지 않았다.** Global top-20 방향 pair가 차지한 오류는 IA5 8.8%, CAR5 12.8%에 불과했다. 따라서 소수 pair만 선택적으로 보정하는 방법의 전체 성능 개선 상한은 낮다.
3. **T4→T5 incremental update가 주요 관계를 새로 악화시켰다는 근거는 없었다.** 동일 class set·동일 sample·동일 probe 조건에서 T5 성능은 두 protocol 모두 소폭 향상됐다. 현재 결과는 task-induced pair degradation보다 **broad distributed error와 held-out generalization/coverage 문제**를 더 강하게 지지한다.

## 통합 첫 페이지 결론

| 질문 | IA5 | CAR5 |
|---|---|---|
| Q0. 이전 낮은 Linear 결과가 설정/regularization artifact였는가? | **YES** — validated linear test=65.83%, Structure=66.43% (delta=-0.59 pp) | **YES** — validated linear test=82.55%, Structure=81.20% (delta=1.36 pp) |
| Q1. 오류가 소수 class pair에 집중되는가? | **NO** — global top-20 pair error share=8.8% (510 total errors) | **NO** — global top-20 pair error share=12.8% (1512 total errors) |
| Q2. 주요 오류의 old/new 유형은 무엇인가? | **YES — old-old** — largest pair type share=59.8% | **YES — old-old** — largest pair type share=64.9% |
| Q3. Train에서 분리되지만 held-out에서 무너지는가? | **YES** — focused-pair median negative margin train=0.0%, test=50.0%, delta=50.0% | **YES** — focused-pair median negative margin train=2.9%, test=25.0%, delta=22.4% |
| Q4. Cosine 오류를 k-NN이 의미 있게 회복하는가? | **NO** — k-NN−Structure=-8.62 pp; B1=10/510 cosine errors | **NO** — k-NN−Structure=-14.84 pp; B1=0/1512 cosine errors |
| Q5. 낮은 neighborhood purity가 오류와 연결되는가? | **YES** — purity correct=0.638, wrong=0.127, gap=0.511 | **YES** — purity correct=0.479, wrong=0.149, gap=0.330 |
| Q6. T5 pair가 T4 representation보다 악화됐는가? | **NO** — confusion↑ + q05↓ + purity↓: 1/40 focused pairs | **NO** — confusion↑ + q05↓ + purity↓: 0/23 focused pairs |
| Q7. 동일 diagnostic probe에서도 악화가 유지되는가? | **INCONCLUSIVE** — focused probe pair errors T4=47, T5=47; overall micro=64.91→65.70, macro=53.69→54.55 | **NO** — focused probe pair errors T4=192, T5=189; overall micro=80.38→80.97, macro=80.36→80.98 |
| Q8. 현재 evidence가 가장 지지하는 설명은 무엇인가? | **YES — broad distributed error** — broad distributed error | **YES — broad distributed error** — broad distributed error |

## 1. 실행 범위와 무결성

- 대상: IA5 T5, CAR5 T5, seed 1993
- T4는 동일 sample/class-set transition 진단에만 사용
- T5 frozen feature cache 재사용, T4 frozen feature만 추가 추출
- backbone, adapter, APR, SCC 및 main classifier는 갱신하지 않음
- IA10, CAR10, T2, T3는 실행하지 않음
- 실제 old sample 사용 결과는 method 성능이 아닌 oracle diagnostic으로만 해석
- 총 diagnostic 실행 시간: 152.3초

| Protocol | Train | Diagnostic train | Validation | Test | Checkpoint unchanged | Elapsed |
|---|---:|---:|---:|---:|---|---:|
| IA5 | 5981 | 4865 | 1116 | 1519 | True | 54.9s |
| CAR5 | 8144 | 6598 | 1546 | 8041 | True | 97.4s |

## 2. Stage 0 — Linear probe validity

Cosine classifier의 각 weight를 `v_k = 20 × w_k / ||w_k||`로 변환했다. 두 protocol 모두 전체 test set에서 prediction mismatch가 0이었으며, 기존 Structure accuracy와 재구성 accuracy도 정확히 일치했다.

| Protocol | Equivalence | Max logit diff | Mismatch | Historical/reconstructed Structure |
|---|---|---:|---:|---:|
| IA5 | PASS | 3.815e-06 | 0 | 66.43% / 66.43% |
| CAR5 | PASS | 3.815e-06 | 0 | 81.20% / 81.20% |

### Regularization sanity

Test는 L2 선택에 사용하지 않았고 validation macro, validation micro, 더 강한 L2 순으로 선택했다.

| Protocol | L2 | Train micro | Val micro | Val macro | Test micro | Test macro | Selected |
|---|---:|---:|---:|---:|---:|---:|---|
| IA5 | 5.0e-04 | 75.60 | 62.99 | 53.05 | 59.12 | 50.10 | False |
| IA5 | 5.0e-05 | 92.42 | 66.67 | 53.17 | 65.83 | 54.52 | True |
| IA5 | 5.0e-06 | 99.96 | 66.04 | 51.15 | 65.57 | 53.96 | False |
| CAR5 | 5.0e-04 | 77.30 | 65.59 | 66.12 | 63.65 | 64.12 | False |
| CAR5 | 5.0e-05 | 96.20 | 80.60 | 80.63 | 79.39 | 79.42 | False |
| CAR5 | 5.0e-06 | 99.88 | 82.99 | 82.99 | 82.55 | 82.47 | True |

IA5에서는 `5e-4`의 test micro 59.12%가 `5e-5`에서 65.83%로 회복되어 Structure 66.43%에 근접했다. CAR5에서는 63.65%가 `5e-6`에서 82.55%로 회복되어 Structure 81.20%를 넘었다. 따라서 과거 Linear의 -6~-18 pp 차이를 근거로 classifier 원인을 배제한 판단은 수정해야 한다.

## 3. Directional class-pair confusion

| Protocol | Total errors | Top-20 pair share | old-old | old-new | new-old | new-new |
|---|---:|---:|---:|---:|---:|---:|
| IA5 | 510 | 8.8% | 59.8% | 21.0% | 16.7% | 2.5% |
| CAR5 | 1512 | 12.8% | 64.9% | 16.4% | 15.3% | 3.4% |

두 protocol 모두 old-old 오류가 가장 많지만, 이는 하나의 특정 old-old pair가 지배한다는 뜻이 아니다. IA5의 가장 큰 방향 pair도 3건, CAR5의 가장 큰 pair도 16건으로 전체 오류의 1.1% 이하였다.

### Global contribution 상위 pair

| Protocol | Pair | Type | Errors | All-error share | Conditional confusion | Train neg | Test neg |
|---|---|---|---:|---:|---:|---:|---:|
| IA5 | 8→99 | old-old | 3 | 0.6% | 18.8% | 0.0% | 18.8% |
| IA5 | 39→33 | old-old | 3 | 0.6% | 15.8% | 3.1% | 15.8% |
| IA5 | 45→160 | old-new | 3 | 0.6% | 18.8% | 2.4% | 18.8% |
| IA5 | 151→2 | old-old | 3 | 0.6% | 23.1% | 2.2% | 23.1% |
| IA5 | 169→25 | new-old | 3 | 0.6% | 23.1% | 2.6% | 30.8% |
| IA5 | 2→151 | old-old | 2 | 0.4% | 12.5% | 0.0% | 12.5% |
| IA5 | 12→93 | old-old | 2 | 0.4% | 9.1% | 3.3% | 9.1% |
| IA5 | 12→109 | old-old | 2 | 0.4% | 9.1% | 1.6% | 18.2% |
| IA5 | 15→134 | old-old | 2 | 0.4% | 11.8% | 3.0% | 17.6% |
| IA5 | 37→168 | old-new | 2 | 0.4% | 40.0% | 0.0% | 40.0% |
| CAR5 | 22→76 | old-old | 16 | 1.1% | 41.0% | 0.0% | 43.6% |
| CAR5 | 6→136 | old-old | 14 | 0.9% | 20.6% | 10.9% | 25.0% |
| CAR5 | 25→136 | old-old | 14 | 0.9% | 40.0% | 17.9% | 45.7% |
| CAR5 | 4→143 | old-old | 13 | 0.9% | 28.3% | 18.9% | 41.3% |
| CAR5 | 117→80 | old-old | 12 | 0.8% | 28.6% | 5.7% | 38.1% |
| CAR5 | 85→12 | old-old | 10 | 0.7% | 23.3% | 2.8% | 25.6% |
| CAR5 | 144→129 | old-old | 10 | 0.7% | 25.0% | 21.2% | 25.0% |
| CAR5 | 149→188 | old-new | 10 | 0.7% | 25.0% | 9.4% | 27.5% |
| CAR5 | 4→10 | old-old | 9 | 0.6% | 19.6% | 5.4% | 23.9% |
| CAR5 | 12→85 | old-old | 9 | 0.6% | 22.5% | 0.0% | 22.5% |

## 4. Train–held-out pair margin

Focused pair의 median negative-margin fraction은 IA5에서 0.0%→50.0%, CAR5에서 2.9%→25.0%로 증가했다. 즉 training feature에서는 분리되던 관계가 held-out에서 무너지는 경향은 분명하다. 다만 오류가 소수 pair에 집중되지 않았으므로 이는 특정 relation 하나의 문제가 아니라 더 넓은 generalization/coverage 현상으로 보는 편이 타당하다.

- [IA5 margin plot](IA5_T5/pair_margin/train_validation_test_margin.png)
- [CAR5 margin plot](CAR5_T5/pair_margin/train_validation_test_margin.png)

## 5. k-NN 및 neighborhood purity

| Protocol | Selected k | Structure micro/macro | Linear micro/macro | k-NN micro/macro | k-NN Δ | B1 | B2 | B3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IA5 | 5 | 66.43/55.22 | 65.83/54.52 | 57.80/44.90 | -8.62 pp | 10 | 168 | 24 |
| CAR5 | 9 | 81.20/81.25 | 82.55/82.47 | 66.36/66.54 | -14.84 pp | 0 | 567 | 14 |

k-NN은 IA5에서 Structure보다 8.62 pp, CAR5에서 14.84 pp 낮았다. 반면 cosine 정답 sample과 오답 sample의 purity 차이는 IA5 0.511, CAR5 0.330이었다. 낮은 purity가 오류와 연결되기는 하지만, local neighborhood를 그대로 사용하는 classifier가 현재 오류를 해결하지는 못한다. 특히 CAR5에서는 high-purity B1이 0건이고 low-purity shared-wrong B2가 567건이었다.

- [IA5 neighborhood purity](IA5_T5/neighborhood_purity/purity_distribution.png)
- [CAR5 neighborhood purity](CAR5_T5/neighborhood_purity/purity_distribution.png)
- [IA5 cosine × k-NN matrix](IA5_T5/knn_vs_cosine/correctness_matrix.png)
- [CAR5 cosine × k-NN matrix](CAR5_T5/knn_vs_cosine/correctness_matrix.png)

## 6. T4→T5 동일 조건 비교

T4와 T5 representation에 동일한 T5 class set, train/validation/test sample, probe 절차와 k-NN 규칙을 적용했다. 후보 class 수가 다른 original T4/T5 head accuracy는 직접 비교하지 않았다.

| Protocol | Probe micro T4→T5 | Probe macro T4→T5 | Focused probe pair errors T4→T5 | Simultaneous degradation pairs |
|---|---:|---:|---:|---:|
| IA5 | 64.91→65.70 | 53.69→54.55 | 47→47 | 1/40 |
| CAR5 | 80.38→80.97 | 80.36→80.98 | 192→189 | 0/23 |

IA5는 overall probe micro/macro가 각각 +0.79/+0.86 pp 향상됐고 focused pair 오류는 47건으로 동일했다. CAR5도 +0.59/+0.62 pp 향상되고 focused pair 오류가 192→189로 감소했다. Confusion 증가·margin q05 감소·purity 감소가 동시에 발생한 pair는 IA5 1/40, CAR5 0/23이었다. 따라서 T5 update가 주요 relation을 전반적으로 훼손했다는 가설은 지지되지 않는다.

- [IA5 transition plots](IA5_T5/task_transition/)
- [CAR5 transition plots](CAR5_T5/task_transition/)

## 7. Cross-protocol decision

두 protocol의 자동 판정은 모두 **broad distributed error**를 우선 지지한다.

권장 해석 순서는 다음과 같다.

1. Linear classifier family 자체는 충분한 표현력을 가진다. 향후 classifier 비교에서는 regularization과 validation selection을 먼저 공정하게 맞춰야 한다.
2. 선택적 소수-pair separation은 현재 전체 오류 구조와 맞지 않는다. Pair-specific method를 추가하기 전에 오류 coverage 상한을 계산해야 한다.
3. 단순 k-NN/local geometry 활용은 우선순위가 낮다. Neighborhood purity가 낮은 held-out sample이 많아 reference coverage/generalization 진단이 더 직접적이다.
4. T4→T5 relation preservation failure는 현재 seed/protocol에서 주원인이 아니다. 새로운 preservation loss를 추가할 근거로 사용하지 않는다.
5. 다음 검증은 독립 seed 또는 별도 held-out에서 진행해야 한다. 현재 focused pair는 test confusion으로 발견했으므로 hypothesis-generation 용도다.

## 8. 한계와 해석 범위

- Validation image는 diagnostic head training에서는 제외됐지만 과거 CIL backbone/adapter 학습에는 사용됐으므로 representation에 완전히 unseen인 표본은 아니다.
- Focused pair는 같은 test set에서 선정·기술했으므로 새로운 method의 유효성 검정으로 사용할 수 없다.
- T4/T5 original heads는 class 후보 수가 달라 비교하지 않았으므로 original-head-only alignment 변화는 직접 판정하지 않았다.
- IA5/CAR5 seed 1993만 분석했으며 10-task 결과로 일반화하지 않는다.

## 9. 상세 보고서와 원시 산출물

### IA5

- [상세 보고서](IA5_T5/IA5_representation_relation_report.md)
- [Directional confusion](IA5_T5/directional_pair_confusion.csv)
- [Pair margins](IA5_T5/pair_margin_distribution.csv)
- [k-NN sample diagnostic](IA5_T5/knn_neighborhood_diagnostic.csv)
- [T4→T5 transition](IA5_T5/pair_task_transition_analysis.csv)
- 자동 우선 판정: **broad distributed error**

### CAR5

- [상세 보고서](CAR5_T5/CAR5_representation_relation_report.md)
- [Directional confusion](CAR5_T5/directional_pair_confusion.csv)
- [Pair margins](CAR5_T5/pair_margin_distribution.csv)
- [k-NN sample diagnostic](CAR5_T5/knn_neighborhood_diagnostic.csv.gz)
- [T4→T5 transition](CAR5_T5/pair_task_transition_analysis.csv)
- 자동 우선 판정: **broad distributed error**

