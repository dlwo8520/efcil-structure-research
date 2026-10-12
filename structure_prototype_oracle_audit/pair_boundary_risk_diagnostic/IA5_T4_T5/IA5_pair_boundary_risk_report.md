# IA5 T4→T5 Pair Boundary-Risk Diagnostic

> 상태: COMPLETE · seed 1993 · METHOD READY: **NO** · checkpoint unchanged

## First-page answers

| 질문 | 판정 | 근거 |
|---|---|---|
| Q1. 동일 sample에서 newly-degraded directional pair가 충분히 존재하는가? | **NO** | eligible=9/39800 (0.0%); new errors=14 < recovered=26; fixed-head accuracy Δ=0.790 pp; boundary violations Δ=-151 |
| Q2. Confusion 증가는 correct→wrong과 wrong-destination 변경 중 무엇인가? | **DESCRIPTIVE** | correct→wrong-j=14, wrong-C→wrong-B=32, recovered=26 |
| Q3. delta_B와 same-sample margin 악화가 일치하는가? | **INCONCLUSIVE** | Spearman(delta_B, -delta_q05)=0.115, sign agreement=56.1% |
| Q4. Oracle gamma가 actual risky pair를 ranking하는가? | **NO** | new-risk B/C: Oracle beats random 3/4, prototype 3/4; correlations remain small |
| Q5. Stored gamma가 Oracle risk ranking을 재현하는가? | **YES** | Spearman(gamma_stored, gamma_oracle)=0.980 |
| Q6. Stored gamma가 prototype similarity보다 나은가? | **NO** | Stored gamma beats prototype on 3/4 B/C recall/mass comparisons; consistent superiority requires 4/4 |
| Q7. Mean/covariance로 held-out boundary risk를 설명할 수 있는가? | **NO** | New-risk ranking is not consistently superior; Oracle target Spearman: A_B_T5=0.080, B_POSITIVE_DELTA_B=0.018, C_NEW_ACTUAL_ERROR=0.026 |
| Q8. 현재 evidence가 가장 지지하는 case는 무엇인가? | **D — incremental-specific degradation 자체가 거의 없음** | D — incremental-specific degradation 자체가 거의 없음 |
| Q9. METHOD READY인가? | **NO** | R1/Q4/Q5/Q6 must all be YES; relative ranking evidence, not a single absolute cutoff |

## Same-sample transition

| Transition | Count |
|---|---:|
| correct_to_correct | 983 |
| correct_to_wrong_B | 14 |
| wrong_B_to_correct | 26 |
| wrong_B_to_wrong_B | 464 |
| wrong_C_to_wrong_B | 32 |

Fixed-head accuracy는 T4 **65.64%**에서 T5 **66.43%**로 변해 Δ=+0.79 pp였다. 전체 ordered sample-pair boundary violation은 5986→5835 (Δ=-151)였다. Positive degradation mass 271보다 improvement mass 422가 컸다.

T4/T5 모두 frozen T5 Structure cosine head를 공통 readout으로 사용했다. 따라서 이 표는 historical T4 head 성능 비교가 아니라 current boundary에 대한 representation transition이다.
Prototype-similarity baseline은 stored mean cosine이며 방향성이 없으므로 `i→j`와 `j→i`에 동일한 값이 들어간다.

## Newly-degraded pairs

| Rank | Pair | Type | n | ΔC | ΔB | New errors | Δmargin q05 | Oracle γ | Stored γ |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 103→96 | old-old | 5 | 0.200 | 0.200 | 1 | -0.315 | 6.552 | 6.372 |
| 2 | 122→28 | old-old | 7 | 0.143 | 0.143 | 1 | -0.079 | 3.522 | 3.461 |
| 3 | 59→50 | old-old | 9 | 0.111 | 0.111 | 1 | -0.005 | 3.553 | 3.470 |
| 4 | 51→160 | old-new | 11 | 0.091 | 0.091 | 1 | -0.642 | 2.724 | 3.405 |
| 5 | 151→2 | old-old | 13 | 0.077 | 0.077 | 1 | -0.151 | 2.077 | 2.202 |
| 6 | 17→181 | old-new | 15 | 0.067 | 0.067 | 1 | -0.570 | 2.744 | 3.250 |
| 7 | 8→174 | old-new | 16 | 0.062 | 0.062 | 1 | -0.448 | 2.060 | 2.100 |
| 8 | 119→33 | old-old | 17 | 0.059 | 0.059 | 1 | -0.362 | 2.231 | 2.362 |
| 9 | 63→109 | old-old | 23 | 0.043 | 0.043 | 1 | -0.053 | 2.597 | 2.888 |

## Ranking comparison

아래 값은 K={10,20,50,100}의 평균이다. Gamma는 작을수록 위험하여 ranking score로는 `-gamma`를 사용했다.

| Target | Ranking | Mean Recall@K | Mean captured mass@K | Spearman |
|---|---|---:|---:|---:|
| A_B_T5 | RANDOM_EXPECTED | 0.001 | 0.001 | 0.004 |
| A_B_T5 | PROTOTYPE_SIMILARITY | 0.000 | 0.007 | 0.288 |
| A_B_T5 | ORACLE_GAMMA | 0.000 | 0.006 | 0.080 |
| A_B_T5 | STORED_GAMMA | 0.000 | 0.007 | 0.076 |
| B_POSITIVE_DELTA_B | RANDOM_EXPECTED | 0.001 | 0.001 | 0.005 |
| B_POSITIVE_DELTA_B | PROTOTYPE_SIMILARITY | 0.000 | 0.006 | 0.072 |
| B_POSITIVE_DELTA_B | ORACLE_GAMMA | 0.000 | 0.008 | 0.018 |
| B_POSITIVE_DELTA_B | STORED_GAMMA | 0.000 | 0.006 | 0.015 |
| C_NEW_ACTUAL_ERROR | RANDOM_EXPECTED | 0.001 | 0.002 | 0.001 |
| C_NEW_ACTUAL_ERROR | PROTOTYPE_SIMILARITY | 0.000 | 0.000 | 0.026 |
| C_NEW_ACTUAL_ERROR | ORACLE_GAMMA | 0.010 | 0.054 | 0.026 |
| C_NEW_ACTUAL_ERROR | STORED_GAMMA | 0.010 | 0.054 | 0.026 |

Static current risk(Target A)와 incremental new risk(Target B/C)는 분리해서 해석한다. Oracle mean Recall@K는 A=0.000, B=0.000, C=0.010였다. Target A가 높더라도 B/C가 낮으면 incremental-specific selection 근거가 아니다.

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
