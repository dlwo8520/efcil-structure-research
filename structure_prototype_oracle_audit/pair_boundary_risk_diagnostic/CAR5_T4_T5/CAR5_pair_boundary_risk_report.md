# CAR5 T4→T5 Pair Boundary-Risk Diagnostic

> 상태: COMPLETE · seed 1993 · METHOD READY: **NO** · checkpoint unchanged

## First-page answers

| 질문 | 판정 | 근거 |
|---|---|---|
| Q1. 동일 sample에서 newly-degraded directional pair가 충분히 존재하는가? | **NO** | eligible=27/38220 (0.1%); new errors=47 < recovered=73; fixed-head accuracy Δ=0.323 pp; boundary violations Δ=-318 |
| Q2. Confusion 증가는 correct→wrong과 wrong-destination 변경 중 무엇인가? | **DESCRIPTIVE** | correct→wrong-j=47, wrong-C→wrong-B=68, recovered=73 |
| Q3. delta_B와 same-sample margin 악화가 일치하는가? | **INCONCLUSIVE** | Spearman(delta_B, -delta_q05)=0.079, sign agreement=69.3% |
| Q4. Oracle gamma가 actual risky pair를 ranking하는가? | **NO** | new-risk B/C: Oracle beats random 4/4, prototype 0/4; correlations remain small |
| Q5. Stored gamma가 Oracle risk ranking을 재현하는가? | **YES** | Spearman(gamma_stored, gamma_oracle)=0.988 |
| Q6. Stored gamma가 prototype similarity보다 나은가? | **NO** | Stored gamma beats prototype on 0/4 B/C recall/mass comparisons; consistent superiority requires 4/4 |
| Q7. Mean/covariance로 held-out boundary risk를 설명할 수 있는가? | **NO** | New-risk ranking is not consistently superior; Oracle target Spearman: A_B_T5=0.371, B_POSITIVE_DELTA_B=0.093, C_NEW_ACTUAL_ERROR=0.049 |
| Q8. 현재 evidence가 가장 지지하는 case는 무엇인가? | **D — incremental-specific degradation 자체가 거의 없음** | D — incremental-specific degradation 자체가 거의 없음 |
| Q9. METHOD READY인가? | **NO** | R1/Q4/Q5/Q6 must all be YES; relative ranking evidence, not a single absolute cutoff |

## Same-sample transition

| Transition | Count |
|---|---:|
| correct_to_correct | 6456 |
| correct_to_wrong_B | 47 |
| wrong_B_to_correct | 73 |
| wrong_B_to_wrong_B | 1397 |
| wrong_C_to_wrong_B | 68 |

Fixed-head accuracy는 T4 **80.87%**에서 T5 **81.20%**로 변해 Δ=+0.32 pp였다. 전체 ordered sample-pair boundary violation은 7072→6754 (Δ=-318)였다. Positive degradation mass 273보다 improvement mass 591가 컸다.

T4/T5 모두 frozen T5 Structure cosine head를 공통 readout으로 사용했다. 따라서 이 표는 historical T4 head 성능 비교가 아니라 current boundary에 대한 representation transition이다.
Prototype-similarity baseline은 stored mean cosine이며 방향성이 없으므로 `i→j`와 `j→i`에 동일한 값이 들어간다.

## Newly-degraded pairs

| Rank | Pair | Type | n | ΔC | ΔB | New errors | Δmargin q05 | Oracle γ | Stored γ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 133→176 | old-new | 43 | 0.070 | 0.047 | 2 | -0.181 | 2.272 | 2.274 |
| 2 | 177→98 | new-old | 37 | 0.054 | 0.054 | 2 | -0.161 | 2.673 | 2.673 |
| 3 | 65→162 | old-new | 42 | 0.048 | 0.048 | 2 | -0.189 | 2.414 | 2.724 |
| 4 | 45→85 | old-old | 29 | 0.034 | 0.034 | 1 | -0.021 | 4.637 | 5.310 |
| 5 | 7→193 | old-new | 33 | 0.030 | 0.030 | 1 | -0.300 | 3.495 | 3.830 |
| 6 | 3→178 | old-new | 33 | 0.030 | 0.030 | 1 | -0.164 | 1.882 | 2.152 |
| 7 | 169→132 | new-old | 34 | 0.029 | 0.029 | 1 | -0.189 | 3.708 | 3.708 |
| 8 | 48→141 | old-old | 38 | 0.026 | 0.026 | 1 | -0.116 | 3.793 | 4.140 |
| 9 | 48→107 | old-old | 38 | 0.026 | 0.026 | 1 | -0.012 | 1.473 | 1.616 |
| 10 | 174→141 | new-old | 39 | 0.026 | 0.026 | 1 | -0.119 | 2.225 | 2.225 |
| 11 | 165→190 | new-new | 39 | 0.026 | 0.026 | 1 | -0.067 | 4.392 | 4.392 |
| 12 | 31→69 | old-old | 40 | 0.025 | 0.025 | 1 | -0.268 | 2.527 | 2.360 |
| 13 | 30→159 | old-new | 41 | 0.024 | 0.049 | 1 | -0.310 | 2.990 | 3.112 |
| 14 | 103→168 | old-new | 42 | 0.024 | 0.024 | 1 | -0.192 | 3.724 | 4.027 |
| 15 | 50→159 | old-new | 42 | 0.024 | 0.024 | 1 | -0.076 | 3.464 | 3.557 |
| 16 | 110→159 | old-new | 42 | 0.024 | 0.024 | 1 | -0.065 | 3.108 | 3.234 |
| 17 | 40→85 | old-old | 42 | 0.024 | 0.024 | 1 | -0.062 | 2.486 | 2.627 |
| 18 | 16→70 | old-old | 43 | 0.023 | 0.023 | 1 | -0.005 | 2.898 | 2.819 |
| 19 | 27→186 | old-new | 44 | 0.023 | 0.023 | 1 | -0.069 | 1.903 | 2.698 |
| 20 | 28→189 | old-new | 44 | 0.023 | 0.023 | 1 | -0.121 | 2.697 | 3.172 |
| 21 | 66→129 | old-old | 44 | 0.023 | 0.023 | 1 | -0.055 | 3.387 | 3.310 |
| 22 | 138→154 | old-old | 44 | 0.023 | 0.023 | 1 | -0.050 | 3.178 | 3.167 |
| 23 | 77→180 | old-new | 45 | 0.022 | 0.022 | 1 | -0.033 | 2.580 | 2.904 |
| 24 | 44→54 | old-old | 45 | 0.022 | 0.022 | 1 | -0.042 | 2.038 | 1.962 |
| 25 | 54→59 | old-old | 46 | 0.022 | 0.022 | 1 | -0.047 | 4.212 | 4.361 |
| 26 | 140→108 | old-old | 46 | 0.022 | 0.022 | 1 | -0.036 | 2.575 | 2.571 |
| 27 | 78→119 | old-old | 46 | 0.022 | 0.022 | 1 | -0.003 | 2.673 | 2.807 |

## Ranking comparison

아래 값은 K={10,20,50,100}의 평균이다. Gamma는 작을수록 위험하여 ranking score로는 `-gamma`를 사용했다.

| Target | Ranking | Mean Recall@K | Mean captured mass@K | Spearman |
|---|---|---:|---:|---:|
| A_B_T5 | RANDOM_EXPECTED | 0.001 | 0.001 | -0.005 |
| A_B_T5 | PROTOTYPE_SIMILARITY | 0.265 | 0.029 | 0.355 |
| A_B_T5 | ORACLE_GAMMA | 0.540 | 0.047 | 0.371 |
| A_B_T5 | STORED_GAMMA | 0.467 | 0.045 | 0.371 |
| B_POSITIVE_DELTA_B | RANDOM_EXPECTED | 0.001 | 0.001 | -0.008 |
| B_POSITIVE_DELTA_B | PROTOTYPE_SIMILARITY | 0.073 | 0.034 | 0.092 |
| B_POSITIVE_DELTA_B | ORACLE_GAMMA | 0.065 | 0.033 | 0.093 |
| B_POSITIVE_DELTA_B | STORED_GAMMA | 0.053 | 0.026 | 0.091 |
| C_NEW_ACTUAL_ERROR | RANDOM_EXPECTED | 0.001 | 0.001 | -0.006 |
| C_NEW_ACTUAL_ERROR | PROTOTYPE_SIMILARITY | 0.148 | 0.165 | 0.050 |
| C_NEW_ACTUAL_ERROR | ORACLE_GAMMA | 0.090 | 0.138 | 0.049 |
| C_NEW_ACTUAL_ERROR | STORED_GAMMA | 0.075 | 0.112 | 0.048 |

Static current risk(Target A)와 incremental new risk(Target B/C)는 분리해서 해석한다. Oracle mean Recall@K는 A=0.540, B=0.065, C=0.090였다. Target A가 높더라도 B/C가 낮으면 incremental-specific selection 근거가 아니다.

## Interpretation boundary

`q_hat=Phi(-gamma)`는 Gaussian projection의 보조값이며 multiclass에서 competitor로 최종 예측될 확률이 아니다. 결과는 ranking association 진단이고 causal factor 또는 method 성능이 아니다. Test에서 관찰한 target은 Stage D selection에 사용할 수 없다.

## Artifacts

- `same_sample_prediction_transition.csv`
- `pair_confusion_boundary_metrics.csv`
- `same_sample_pair_margin_transition.csv`
- `stored_oracle_boundary_risk.csv`
- `risk_recall_at_k.csv`, `captured_degradation_mass.csv`, `rank_correlation_summary.csv`
- `optional_region_support_diagnostic.csv` (설명용 only)
- `method_readiness_notes.md`
