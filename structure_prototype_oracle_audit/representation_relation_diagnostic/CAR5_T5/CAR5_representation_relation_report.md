# CAR5 T5 Representation / Class-Relation Diagnostic

> 상태: **COMPLETE** · seed 1993 · diagnostic only · checkpoint unchanged: **True**

## 첫 페이지 결론 — YES / NO / INCONCLUSIVE

| 질문 | 판정 | 직접 근거 |
|---|---|---|
| Q0. 이전 낮은 Linear 결과가 학습 설정/regularization artifact였는가? | **YES** | validated linear test=82.55%, Structure=81.20% (delta=1.36 pp) |
| Q1. 오류는 소수 class pair에 집중되는가? | **NO** | global top-20 pair error share=12.8% (1512 total errors) |
| Q2. 주요 error pair는 어느 old/new 유형인가? | **YES — old-old** | largest pair type share=64.9% |
| Q3. 주요 pair가 train에서는 분리되고 held-out에서 무너지는가? | **YES** | focused-pair median negative margin train=2.9%, test=25.0%, delta=22.4% |
| Q4. Cosine 오류를 k-NN이 의미 있게 회복하는가? | **NO** | k-NN−Structure=-14.84 pp; B1=0/1512 cosine errors |
| Q5. 낮은 neighborhood purity가 confusion과 연결되는가? | **YES** | purity correct=0.479, wrong=0.149, gap=0.330 |
| Q6. T5 주요 pair가 T4 representation보다 악화됐는가? | **NO** | confusion↑ + q05↓ + purity↓: 0/23 focused pairs |
| Q7. 동일 diagnostic probe에서도 T4→T5 악화가 유지되는가? | **NO** | focused probe pair errors T4=192, T5=189; overall micro=80.38→80.97, macro=80.36→80.98 |
| Q8. 현재 evidence가 가장 지지하는 설명은 무엇인가? | **YES — broad distributed error** | broad distributed error |

## 1. Linear validity

| Equivalence | Max logit diff | Mean logit diff | Prediction mismatch |
|---|---:|---:|---:|
| PASS | 3.815e-06 | 2.142e-07 | 0 |

Historical Structure accuracy regression: **PASS** (historical=81.20%, reconstructed=81.20%).

| L2 | Train micro | Val micro | Val macro | Test micro | Test macro | Grad norm | Selected |
|---:|---:|---:|---:|---:|---:|---:|---|
| 5.0e-04 | 77.30 | 65.59 | 66.12 | 63.65 | 64.12 | 2.601e-06 | False |
| 5.0e-05 | 96.20 | 80.60 | 80.63 | 79.39 | 79.42 | 9.461e-06 | False |
| 5.0e-06 | 99.88 | 82.99 | 82.99 | 82.55 | 82.47 | 2.945e-06 | True |

## 2. Directional error concentration

Structure test 오류는 총 **1512개**이며 global top-20 방향 pair가 **12.8%**를 차지한다.

| Pair type | Error count | Share |
|---|---:|---:|
| old-old | 981 | 64.9% |
| old-new | 248 | 16.4% |
| new-old | 231 | 15.3% |
| new-new | 52 | 3.4% |

### Focused pair

| True→Pred | Type | Errors | All-error share | Conditional | Train neg | Test neg | Train q05 | Test q05 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 22→76 | old-old | 16 | 1.1% | 41.0% | 0.0% | 43.6% | 0.457 | -1.293 |
| 6→136 | old-old | 14 | 0.9% | 20.6% | 10.9% | 25.0% | -0.574 | -1.663 |
| 25→136 | old-old | 14 | 0.9% | 40.0% | 17.9% | 45.7% | -0.826 | -1.591 |
| 4→143 | old-old | 13 | 0.9% | 28.3% | 18.9% | 41.3% | -1.433 | -1.387 |
| 117→80 | old-old | 12 | 0.8% | 28.6% | 5.7% | 38.1% | 0.108 | -2.556 |
| 85→12 | old-old | 10 | 0.7% | 23.3% | 2.8% | 25.6% | 0.909 | -2.186 |
| 144→129 | old-old | 10 | 0.7% | 25.0% | 21.2% | 25.0% | -0.412 | -2.183 |
| 149→188 | old-new | 10 | 0.7% | 25.0% | 9.4% | 27.5% | -0.643 | -2.116 |
| 4→10 | old-old | 9 | 0.6% | 19.6% | 5.4% | 23.9% | 0.026 | -2.992 |
| 12→85 | old-old | 9 | 0.6% | 22.5% | 0.0% | 22.5% | 0.942 | -1.222 |
| 132→118 | old-old | 9 | 0.6% | 21.4% | 8.6% | 38.1% | -0.471 | -1.746 |
| 188→149 | new-old | 9 | 0.6% | 20.9% | 2.8% | 30.2% | 0.062 | -1.948 |
| 52→170 | old-new | 8 | 0.5% | 19.0% | 2.9% | 21.4% | 0.544 | -1.413 |
| 177→98 | new-old | 8 | 0.5% | 21.6% | 0.0% | 21.6% | 1.362 | -0.772 |
| 25→6 | old-old | 7 | 0.5% | 20.0% | 10.7% | 28.6% | -0.519 | -0.839 |
| 49→156 | old-old | 7 | 0.5% | 16.7% | 2.9% | 23.8% | 0.954 | -1.327 |
| 77→180 | old-new | 7 | 0.5% | 15.6% | 2.7% | 15.6% | 1.376 | -1.384 |
| 108→30 | old-old | 7 | 0.5% | 15.9% | 5.6% | 20.5% | 0.413 | -2.238 |
| 114→46 | old-old | 7 | 0.5% | 17.5% | 0.0% | 22.5% | 1.429 | -3.386 |
| 143→4 | old-old | 7 | 0.5% | 20.6% | 3.6% | 26.5% | 0.526 | -1.822 |
| 190→94 | new-old | 7 | 0.5% | 17.9% | 0.0% | 17.9% | 0.755 | -0.306 |
| 94→10 | old-old | 6 | 0.4% | 17.1% | 0.0% | 25.7% | 1.097 | -0.902 |
| 136→25 | old-old | 6 | 0.4% | 20.7% | 0.0% | 24.1% | 0.153 | -1.100 |

## 3. k-NN / neighborhood diagnostic

Validation에서 선택된 k는 **9**이다. 후보는 `[1, 3, 5, 9]`이며 test는 선택에 사용하지 않았다.

| Method | Micro | Macro | Old | New |
|---|---:|---:|---:|---:|
| Structure cosine | 81.20 | 81.25 | 80.99 | 82.05 |
| Validated Linear | 82.55 | 82.47 | 82.50 | 82.75 |
| k-NN (k=9) | 66.36 | 66.54 | 65.72 | 68.99 |

Sample case 집계: B1(head miss/local support) **0**, B2(shared wrong/low purity) **567**, B3(train-high/held-out-low purity) **14**.

### Focused-pair k-NN recovery

| Pair | Structure errors | k-NN corrected | Correction rate | Same wrong rate |
|---|---:|---:|---:|---:|
| 22→76 | 16 | 4 | 25.0% | 75.0% |
| 6→136 | 14 | 2 | 14.3% | 85.7% |
| 25→136 | 14 | 3 | 21.4% | 64.3% |
| 4→143 | 13 | 3 | 23.1% | 30.8% |
| 117→80 | 12 | 4 | 33.3% | 33.3% |
| 85→12 | 10 | 1 | 10.0% | 70.0% |
| 144→129 | 10 | 0 | 0.0% | 90.0% |
| 149→188 | 10 | 1 | 10.0% | 80.0% |
| 4→10 | 9 | 3 | 33.3% | 66.7% |
| 12→85 | 9 | 0 | 0.0% | 88.9% |
| 132→118 | 9 | 0 | 0.0% | 88.9% |
| 188→149 | 9 | 4 | 44.4% | 44.4% |
| 52→170 | 8 | 0 | 0.0% | 100.0% |
| 177→98 | 8 | 4 | 50.0% | 37.5% |
| 25→6 | 7 | 2 | 28.6% | 57.1% |
| 49→156 | 7 | 0 | 0.0% | 85.7% |
| 77→180 | 7 | 3 | 42.9% | 14.3% |
| 108→30 | 7 | 0 | 0.0% | 100.0% |
| 114→46 | 7 | 0 | 0.0% | 42.9% |
| 143→4 | 7 | 2 | 28.6% | 42.9% |
| 190→94 | 7 | 4 | 57.1% | 42.9% |
| 94→10 | 6 | 2 | 33.3% | 33.3% |
| 136→25 | 6 | 2 | 33.3% | 33.3% |

## 4. T4→T5 동일 sample / 동일 class-set transition

T4/T5 original head는 후보 class 수가 달라 직접 비교하지 않았다. 따라서 C2의 'original head에서만 악화, retrained probe에서 회복'은 직접 확정하지 않고, 동일 절차 diagnostic probe와 k-NN 변화만 보고한다.

| Pair | Probe confusion T4→T5 | Margin q05 Δ | Purity Δ | k-NN pair error Δ | Interpretation |
|---|---:|---:|---:|---:|---|
| 22→76 | 15→13 | 0.058 | 0.000 | -1 | MIXED_OR_IMPROVED |
| 6→136 | 10→11 | 0.043 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 25→136 | 10→10 | 0.049 | 0.003 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 4→143 | 6→5 | -0.042 | 0.000 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 117→80 | 11→12 | -0.074 | 0.005 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 85→12 | 9→10 | 0.045 | 0.021 | 2 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 144→129 | 11→11 | 0.007 | 0.019 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 149→188 | 10→11 | -0.002 | 0.017 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 4→10 | 7→6 | 0.047 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 12→85 | 7→8 | -0.134 | 0.011 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 132→118 | 8→8 | -0.132 | 0.013 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 188→149 | 6→6 | 0.014 | 0.000 | 0 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 52→170 | 6→6 | -0.031 | -0.003 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 177→98 | 10→10 | 0.120 | 0.009 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 25→6 | 8→8 | -0.071 | 0.003 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 49→156 | 7→7 | 0.027 | 0.032 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 77→180 | 6→6 | -0.030 | 0.025 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 108→30 | 9→9 | 0.089 | 0.003 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 114→46 | 5→5 | -0.233 | 0.025 | -1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 143→4 | 12→10 | -0.056 | 0.023 | 0 | MIXED_OR_IMPROVED |
| 190→94 | 7→5 | 0.066 | 0.014 | -1 | MIXED_OR_IMPROVED |
| 94→10 | 3→3 | -0.041 | 0.013 | 1 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |
| 136→25 | 9→9 | 0.118 | 0.019 | -4 | C3_PRE_EXISTING_HARD_RELATION_CANDIDATE |

## 5. Decision tree 결과

현재 자동 판정이 우선 지지하는 설명은 **broad distributed error**이다.

이 결과는 old actual sample을 사용한 oracle diagnostic이며 EF-CIL 성능으로 해석하지 않는다. 또한 test confusion에서 focused pair를 찾았으므로 pair 결과는 hypothesis generation용이다. 후속 method 검증에는 다른 seed, 별도 held-out 또는 다음 transition이 필요하다.

판정 기준: top-20 오류집중 YES≥50%, NO≤25%; held-out collapse YES는 median negative-fraction 증가≥20%p이면서 train≤20%; k-NN recovery YES는 +1pp 또는 cosine error 중 B1≥5%; purity 연결 YES는 correct−wrong purity≥0.10; T4→T5 악화 YES는 focused pair의 절반 이상에서 confusion↑, q05↓, purity↓가 동시에 성립할 때다.

## 6. 재현 정보

- T4 checkpoint: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR5/Structure_AUDIT/seed_1993/task_checkpoints/task_04.pt`
- T5 checkpoint: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR5/Structure_AUDIT/seed_1993/task_checkpoints/task_05.pt`
- diagnostic train/validation/test: 6598 / 1546 / 8041
- elapsed: 97.4 s
- backbone/adapter update: 없음
- T2/T3 및 IA10/CAR10: 실행하지 않음
