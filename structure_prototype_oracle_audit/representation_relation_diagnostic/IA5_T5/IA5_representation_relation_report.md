# IA5 T5 Representation / Class-Relation Diagnostic

> 상태: **COMPLETE** · seed 1993 · diagnostic only · checkpoint unchanged: **True**

## 첫 페이지 결론 — YES / NO / INCONCLUSIVE

| 질문 | 판정 | 직접 근거 |
|---|---|---|
| Q0. 이전 낮은 Linear 결과가 학습 설정/regularization artifact였는가? | **YES** | validated linear test=65.83%, Structure=66.43% (delta=-0.59 pp) |
| Q1. 오류는 소수 class pair에 집중되는가? | **NO** | global top-20 pair error share=8.8% (510 total errors) |
| Q2. 주요 error pair는 어느 old/new 유형인가? | **YES — old-old** | largest pair type share=59.8% |
| Q3. 주요 pair가 train에서는 분리되고 held-out에서 무너지는가? | **YES** | focused-pair median negative margin train=0.0%, test=50.0%, delta=50.0% |
| Q4. Cosine 오류를 k-NN이 의미 있게 회복하는가? | **NO** | k-NN−Structure=-8.62 pp; B1=10/510 cosine errors |
| Q5. 낮은 neighborhood purity가 confusion과 연결되는가? | **YES** | purity correct=0.638, wrong=0.127, gap=0.511 |
| Q6. T5 주요 pair가 T4 representation보다 악화됐는가? | **NO** | confusion↑ + q05↓ + purity↓: 1/40 focused pairs |
| Q7. 동일 diagnostic probe에서도 T4→T5 악화가 유지되는가? | **INCONCLUSIVE** | focused probe pair errors T4=47, T5=47; overall micro=64.91→65.70, macro=53.69→54.55 |
| Q8. 현재 evidence가 가장 지지하는 설명은 무엇인가? | **YES — broad distributed error** | broad distributed error |

## 1. Linear validity

| Equivalence | Max logit diff | Mean logit diff | Prediction mismatch |
|---|---:|---:|---:|
| PASS | 3.815e-06 | 2.260e-07 | 0 |

Historical Structure accuracy regression: **PASS** (historical=66.43%, reconstructed=66.43%).

| L2 | Train micro | Val micro | Val macro | Test micro | Test macro | Grad norm | Selected |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5.0e-04 | 75.60 | 62.99 | 53.05 | 59.12 | 50.10 | 1.054e-05 | False |
| 5.0e-05 | 92.42 | 66.67 | 53.17 | 65.83 | 54.52 | 5.142e-06 | True |
| 5.0e-06 | 99.96 | 66.04 | 51.15 | 65.57 | 53.96 | 2.154e-06 | False |

## 2. Directional error concentration

Structure test 오류는 총 **510개**이며 global top-20 방향 pair가 **8.8%**를 차지한다.

| Pair type | Error count | Share |
|---|---:|---:|
| old-old | 305 | 59.8% |
| old-new | 107 | 21.0% |
| new-old | 85 | 16.7% |
| new-new | 13 | 2.5% |

### Focused pair

| True→Pred | Type | Errors | All-error share | Conditional | Train neg | Test neg | Train q05 | Test q05 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8→99 | old-old | 3 | 0.6% | 18.8% | 0.0% | 18.8% | 1.277 | -1.397 |
| 39→33 | old-old | 3 | 0.6% | 15.8% | 3.1% | 15.8% | 1.661 | -2.597 |
| 45→160 | old-new | 3 | 0.6% | 18.8% | 2.4% | 18.8% | 0.862 | -0.978 |
| 151→2 | old-old | 3 | 0.6% | 23.1% | 2.2% | 23.1% | 0.797 | -0.967 |
| 169→25 | new-old | 3 | 0.6% | 23.1% | 2.6% | 30.8% | 1.811 | -2.918 |
| 2→151 | old-old | 2 | 0.4% | 12.5% | 0.0% | 12.5% | 0.420 | -0.489 |
| 12→93 | old-old | 2 | 0.4% | 9.1% | 3.3% | 9.1% | 1.123 | -0.326 |
| 12→109 | old-old | 2 | 0.4% | 9.1% | 1.6% | 18.2% | 1.220 | -0.464 |
| 15→134 | old-old | 2 | 0.4% | 11.8% | 3.0% | 17.6% | 0.744 | -0.613 |
| 37→168 | old-new | 2 | 0.4% | 40.0% | 0.0% | 40.0% | 2.164 | -1.534 |
| 58→27 | old-old | 2 | 0.4% | 33.3% | 0.0% | 50.0% | 6.412 | -3.285 |
| 59→50 | old-old | 2 | 0.4% | 22.2% | 0.0% | 22.2% | 4.977 | -0.707 |
| 60→15 | old-old | 2 | 0.4% | 13.3% | 0.0% | 13.3% | 0.547 | -0.149 |
| 64→177 | old-new | 2 | 0.4% | 28.6% | 0.0% | 42.9% | 2.778 | -3.200 |
| 66→136 | old-old | 2 | 0.4% | 11.1% | 3.0% | 11.1% | 1.636 | -0.579 |
| 69→22 | old-old | 2 | 0.4% | 25.0% | 0.0% | 37.5% | 0.723 | -1.790 |
| 70→4 | old-old | 2 | 0.4% | 20.0% | 3.6% | 20.0% | 2.035 | -0.394 |
| 75→133 | old-old | 2 | 0.4% | 10.5% | 0.0% | 15.8% | 2.566 | -0.393 |
| 80→92 | old-old | 2 | 0.4% | 22.2% | 0.0% | 44.4% | 4.715 | -3.743 |
| 81→109 | old-old | 2 | 0.4% | 11.8% | 0.0% | 11.8% | 3.367 | -2.973 |
| 125→199 | old-new | 2 | 0.4% | 66.7% | 0.0% | 66.7% | 3.521 | -8.244 |
| 1→95 | old-old | 1 | 0.2% | 100.0% | 0.0% | 100.0% | 5.642 | -3.868 |
| 4→85 | old-old | 1 | 0.2% | 50.0% | 0.0% | 50.0% | 4.575 | -0.265 |
| 4→188 | old-new | 1 | 0.2% | 50.0% | 0.0% | 50.0% | 2.525 | -0.950 |
| 16→165 | old-new | 1 | 0.2% | 50.0% | 0.0% | 50.0% | 3.604 | -1.917 |

## 3. k-NN / neighborhood diagnostic

Validation에서 선택된 k는 **5**이다. 후보는 `[1, 3, 5, 9]`이며 test는 선택에 사용하지 않았다.

| Method | Micro | Macro | Old | New |
|---|---:|---:|---:|---:|
| Structure cosine | 66.43 | 55.22 | 66.99 | 63.84 |
| Validated Linear | 65.83 | 54.52 | 66.43 | 63.10 |
| k-NN (k=5) | 57.80 | 44.90 | 58.97 | 52.40 |

Sample case 집계: B1(head miss/local support) **10**, B2(shared wrong/low purity) **168**, B3(train-high/held-out-low purity) **24**.

### Focused-pair k-NN recovery

| Pair | Structure errors | k-NN corrected | Correction rate | Same wrong rate |
|---|---:|---:|---:|---:|
| 8→99 | 3 | 0 | 0.0% | 33.3% |
| 39→33 | 3 | 2 | 66.7% | 33.3% |
| 45→160 | 3 | 2 | 66.7% | 33.3% |
| 151→2 | 3 | 0 | 0.0% | 33.3% |
| 169→25 | 3 | 0 | 0.0% | 66.7% |
| 2→151 | 2 | 0 | 0.0% | 0.0% |
| 12→93 | 2 | 0 | 0.0% | 50.0% |
| 12→109 | 2 | 0 | 0.0% | 100.0% |
| 15→134 | 2 | 0 | 0.0% | 100.0% |
| 37→168 | 2 | 0 | 0.0% | 50.0% |
| 58→27 | 2 | 0 | 0.0% | 100.0% |
| 59→50 | 2 | 1 | 50.0% | 0.0% |
| 60→15 | 2 | 0 | 0.0% | 100.0% |
| 64→177 | 2 | 0 | 0.0% | 50.0% |
| 66→136 | 2 | 1 | 50.0% | 50.0% |
| 69→22 | 2 | 0 | 0.0% | 0.0% |
| 70→4 | 2 | 0 | 0.0% | 100.0% |
| 75→133 | 2 | 0 | 0.0% | 50.0% |
| 80→92 | 2 | 0 | 0.0% | 0.0% |
| 81→109 | 2 | 0 | 0.0% | 100.0% |
| 125→199 | 2 | 0 | 0.0% | 50.0% |
| 1→95 | 1 | 0 | 0.0% | 0.0% |
| 4→85 | 1 | 0 | 0.0% | 0.0% |
| 4→188 | 1 | 0 | 0.0% | 0.0% |
| 16→165 | 1 | 0 | 0.0% | 0.0% |

## 4. T4→T5 동일 sample / 동일 class-set transition

T4/T5 original head는 후보 class 수가 달라 직접 비교하지 않았다. 따라서 C2의 'original head에서만 악화, retrained probe에서 회복'은 직접 확정하지 않고, 동일 절차 diagnostic probe와 k-NN 변화만 보고한다.

| Pair | Probe confusion T4→T5 | Margin q05 Δ | Purity Δ | k-NN pair error Δ | Interpretation |
|---|---:|---:|---:|---:|---|
| 8→99 | 2→2 | 0.105 | 0.025 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 39→33 | 2→2 | -0.068 | 0.000 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 45→160 | 2→2 | -0.274 | 0.037 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 151→2 | 2→1 | -0.036 | -0.015 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 169→25 | 3→3 | 0.240 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 2→151 | 1→1 | -0.039 | -0.013 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 12→93 | 2→2 | 0.347 | -0.018 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 12→109 | 2→1 | -0.270 | -0.018 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 15→134 | 1→1 | -0.056 | -0.012 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 37→168 | 1→1 | -0.139 | -0.040 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 58→27 | 2→2 | 0.442 | 0.000 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 59→50 | 1→1 | 0.067 | -0.022 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 60→15 | 2→2 | -0.008 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 64→177 | 1→1 | -0.173 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 66→136 | 1→1 | 0.063 | 0.033 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 69→22 | 2→2 | 0.161 | -0.025 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 70→4 | 1→1 | -0.002 | 0.040 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 75→133 | 1→2 | -0.003 | -0.011 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 80→92 | 2→2 | -0.184 | -0.022 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 81→109 | 2→2 | 0.273 | 0.012 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 125→199 | 1→2 | -0.385 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 1→95 | 0→0 | 0.227 | 0.000 | 0 | MIXED_OR_IMPROVED |
| 4→85 | 1→1 | 0.178 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 4→188 | 0→0 | 0.041 | 0.000 | 0 | MIXED_OR_IMPROVED |
| 16→165 | 1→1 | 0.194 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |

## 5. Decision tree 결과

현재 자동 판정이 우선 지지하는 설명은 **broad distributed error**이다.

이 결과는 old actual sample을 사용한 oracle diagnostic이며 EF-CIL 성능으로 해석하지 않는다. 또한 test confusion에서 focused pair를 찾았으므로 pair 결과는 hypothesis generation용이다. 후속 method 검증에는 다른 seed, 별도 held-out 또는 다음 transition이 필요하다.

판정 기준: top-20 오류집중 YES≥50%, NO≤25%; held-out collapse YES는 median negative-fraction 증가≥20%p이면서 train≤20%; k-NN recovery YES는 +1pp 또는 cosine error 중 B1≥5%; purity 연결 YES는 correct−wrong purity≥0.10; T4→T5 악화 YES는 focused pair의 절반 이상에서 confusion↑, q05↓, purity↓가 동시에 성립할 때다.

## 6. 재현 정보

- T4 checkpoint: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA5/Structure_AUDIT/seed_1993/task_checkpoints/task_04.pt`
- T5 checkpoint: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA5/Structure_AUDIT/seed_1993/task_checkpoints/task_05.pt`
- diagnostic train/validation/test: 4865 / 1116 / 1519
- elapsed: 54.9 s
- backbone/adapter update: 없음
- T2/T3 및 IA10/CAR10: 실행하지 않음
