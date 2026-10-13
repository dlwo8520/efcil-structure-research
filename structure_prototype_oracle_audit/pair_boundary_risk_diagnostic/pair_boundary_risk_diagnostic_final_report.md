# Pair Boundary-Risk Diagnostic Final Report

> IA5/CAR5 · T4→T5 · seed 1993 · frozen diagnostic only

## Executive conclusion

**최종 결론은 METHOD READY = NO이다.** 일부 directional pair의 악화는 관찰됐지만 전체 pair와 sample 수에 비해 희소했고, 두 protocol 모두 새 오류보다 복구가 많았으며 fixed-head accuracy와 전체 boundary violation은 오히려 개선됐다. Stored gamma는 Oracle gamma 자체는 매우 잘 재현했지만, 핵심 target인 newly-degraded boundary와 newly-created actual error의 Top-K ranking에서는 prototype similarity보다 일관되게 우수하지 않았다.

따라서 현재 결과는 **D. incremental-specific degradation 자체가 전체 성능의 주된 문제는 아님**을 가장 강하게 지지한다. Mean/covariance boundary risk는 CAR5의 현재 static boundary risk를 설명하는 데는 일부 유효했지만, 다음 단계의 risk-weighted preservation method를 시작할 충분한 cross-protocol 근거는 아니다.

## First-page answers

| Question | IA5 | CAR5 |
|---|---|---|
| Q1 | **NO** — eligible=9/39800 (0.0%); new errors=14 < recovered=26; fixed-head accuracy Δ=0.790 pp; boundary violations Δ=-151 | **NO** — eligible=27/38220 (0.1%); new errors=47 < recovered=73; fixed-head accuracy Δ=0.323 pp; boundary violations Δ=-318 |
| Q2 | **DESCRIPTIVE** — correct→wrong-j=14, wrong-C→wrong-B=32, recovered=26 | **DESCRIPTIVE** — correct→wrong-j=47, wrong-C→wrong-B=68, recovered=73 |
| Q3 | **INCONCLUSIVE** — Spearman(delta_B, -delta_q05)=0.115, sign agreement=56.1% | **INCONCLUSIVE** — Spearman(delta_B, -delta_q05)=0.079, sign agreement=69.3% |
| Q4 | **NO** — new-risk B/C: Oracle beats random 3/4, prototype 3/4; correlations remain small | **NO** — new-risk B/C: Oracle beats random 4/4, prototype 0/4; correlations remain small |
| Q5 | **YES** — Spearman(gamma_stored, gamma_oracle)=0.980 | **YES** — Spearman(gamma_stored, gamma_oracle)=0.988 |
| Q6 | **NO** — Stored gamma beats prototype on 3/4 B/C recall/mass comparisons; consistent superiority requires 4/4 | **NO** — Stored gamma beats prototype on 0/4 B/C recall/mass comparisons; consistent superiority requires 4/4 |
| Q7 | **NO** — New-risk ranking is not consistently superior; Oracle target Spearman: A_B_T5=0.080, B_POSITIVE_DELTA_B=0.018, C_NEW_ACTUAL_ERROR=0.026 | **NO** — New-risk ranking is not consistently superior; Oracle target Spearman: A_B_T5=0.371, B_POSITIVE_DELTA_B=0.093, C_NEW_ACTUAL_ERROR=0.049 |
| Q8 | **D — incremental-specific degradation 자체가 거의 없음** — D — incremental-specific degradation 자체가 거의 없음 | **D — incremental-specific degradation 자체가 거의 없음** — D — incremental-specific degradation 자체가 거의 없음 |
| Q9 | **NO** — R1/Q4/Q5/Q6 must all be YES; relative ranking evidence, not a single absolute cutoff | **NO** — R1/Q4/Q5/Q6 must all be YES; relative ranking evidence, not a single absolute cutoff |

## 1. 실행 범위와 score 정의

- IA5/CAR5, seed 1993, T4→T5만 분석
- T4/T5 frozen feature cache와 T5 Structure checkpoint의 stored mean/covariance 사용
- backbone, adapter, APR, SCC, main classifier 업데이트 없음
- T4 original head에는 T5 신규 class가 없으므로 frozen T5 Structure cosine head를 T4/T5 feature 모두에 적용한 fixed common readout 사용
- 따라서 prediction transition은 historical T4 head 성능이 아니라 current T5 boundary에 대한 representation 변화
- `a_ij = normalize(w_i)-normalize(w_j)` sign equivalence는 두 protocol 모두 PASS
- Gamma는 ranking diagnostic이며 actual multiclass error probability가 아님

| Protocol | Samples | Classes | Ordered pairs | Newly-degraded eligible | Checkpoint unchanged | Runtime |
|---|---:|---:|---:|---:|---|---:|
| IA5 | 1519 | 200 | 39800 | 9 (0.0%) | True | 40.6s |
| CAR5 | 8041 | 196 | 38220 | 27 (0.1%) | True | 40.6s |

## 2. Same-sample prediction transition

| Protocol | T4 fixed-head acc | T5 fixed-head acc | Δ | Correct→Wrong | Wrong→Correct | Wrong destination change |
|---|---:|---:|---:|---:|---:|---:|
| IA5 | 65.64% | 66.43% | +0.79 pp | 14 | 26 | 32 |
| CAR5 | 80.87% | 81.20% | +0.32 pp | 47 | 73 | 68 |

IA5는 새 오류 14개보다 복구 26개가 많아 +0.79 pp 개선됐고, CAR5도 새 오류 47개보다 복구 73개가 많아 +0.32 pp 개선됐다. 즉 newly-degraded sample은 실제로 존재하지만 전체 transition의 순효과는 악화가 아니다.

## 3. Pairwise boundary transition

| Protocol | B-violation T4 | B-violation T5 | Net Δ | Positive degradation mass | Improvement mass | ΔB>0 pairs | ΔB<0 pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| IA5 | 5986 | 5835 | -151 | 271 | 422 | 268 | 415 |
| CAR5 | 7072 | 6754 | -318 | 273 | 591 | 260 | 558 |

IA5는 boundary violation이 151건, CAR5는 318건 순감소했다. 악화 mass만 보면 IA5 271, CAR5 273건이지만 반대 방향 improvement mass가 각각 422, 591건으로 더 크다. Positive mass만 보고 relation degradation이 지배적이라고 결론 내리면 안 된다.

## 4. Newly-degraded pair의 규모

Eligibility 조건은 `n>=5`, `ΔC>0`, `ΔB>0`, q05 margin 악화, correct→wrong-j 1개 이상을 동시에 만족하는 것이다.

| Protocol | Eligible pairs | Pair share | New errors | Largest new-error count/pair | Main pair types |
|---|---:|---:|---:|---:|---|
| IA5 | 9 | 0.023% | 14 | 1 | old-new=3, old-old=6 |
| CAR5 | 27 | 0.071% | 47 | 2 | new-new=1, new-old=3, old-new=11, old-old=12 |

IA5 eligible pair는 전체의 0.023%, CAR5는 0.071%였다. IA5 각 pair는 새 오류 1개씩이었고 CAR5도 상위 3개만 2개, 나머지는 1개였다. 이는 candidate relation의 존재를 보여주지만 selective method가 해결할 충분한 error mass를 입증하지는 않는다.

## 5. ΔB와 same-sample margin

| Protocol | Spearman(ΔB, −Δq05) | Sign agreement | 판정 |
|---|---:|---:|---|
| IA5 | 0.115 | 56.1% | INCONCLUSIVE |
| CAR5 | 0.079 | 69.3% | INCONCLUSIVE |

두 correlation은 양수지만 0.115/0.079로 작다. ΔB와 q05 margin이 완전히 무관하다고 할 수는 없지만, 같은 현상을 강하게 일관되게 측정한다고 보기도 어렵다.

## 6. Risk ranking — static target과 incremental target 분리

표의 값은 K={10,20,50,100} 평균이다.

| Protocol | Target | Ranking | Mean Recall@K | Mean captured mass@K | Spearman |
|---|---|---|---:|---:|---:|
| IA5 | A_B_T5 | PROTOTYPE_SIMILARITY | 0.000 | 0.007 | 0.288 |
| IA5 | A_B_T5 | ORACLE_GAMMA | 0.000 | 0.006 | 0.080 |
| IA5 | A_B_T5 | STORED_GAMMA | 0.000 | 0.007 | 0.076 |
| IA5 | B_POSITIVE_DELTA_B | PROTOTYPE_SIMILARITY | 0.000 | 0.006 | 0.072 |
| IA5 | B_POSITIVE_DELTA_B | ORACLE_GAMMA | 0.000 | 0.008 | 0.018 |
| IA5 | B_POSITIVE_DELTA_B | STORED_GAMMA | 0.000 | 0.006 | 0.015 |
| IA5 | C_NEW_ACTUAL_ERROR | PROTOTYPE_SIMILARITY | 0.000 | 0.000 | 0.026 |
| IA5 | C_NEW_ACTUAL_ERROR | ORACLE_GAMMA | 0.010 | 0.054 | 0.026 |
| IA5 | C_NEW_ACTUAL_ERROR | STORED_GAMMA | 0.010 | 0.054 | 0.026 |
| CAR5 | A_B_T5 | PROTOTYPE_SIMILARITY | 0.265 | 0.029 | 0.355 |
| CAR5 | A_B_T5 | ORACLE_GAMMA | 0.540 | 0.047 | 0.371 |
| CAR5 | A_B_T5 | STORED_GAMMA | 0.467 | 0.045 | 0.371 |
| CAR5 | B_POSITIVE_DELTA_B | PROTOTYPE_SIMILARITY | 0.073 | 0.034 | 0.092 |
| CAR5 | B_POSITIVE_DELTA_B | ORACLE_GAMMA | 0.065 | 0.033 | 0.093 |
| CAR5 | B_POSITIVE_DELTA_B | STORED_GAMMA | 0.053 | 0.026 | 0.091 |
| CAR5 | C_NEW_ACTUAL_ERROR | PROTOTYPE_SIMILARITY | 0.148 | 0.165 | 0.050 |
| CAR5 | C_NEW_ACTUAL_ERROR | ORACLE_GAMMA | 0.090 | 0.138 | 0.049 |
| CAR5 | C_NEW_ACTUAL_ERROR | STORED_GAMMA | 0.075 | 0.112 | 0.048 |

### Target A — 현재 T5 pairwise boundary risk

CAR5에서는 Oracle gamma가 mean Recall@K 0.540, Stored gamma가 0.467로 static current risk ranking에 의미 있는 신호가 있었다. 반면 IA5는 Oracle/Stored/Prototype 모두 Recall@K가 0이었고 prototype correlation(0.288)이 gamma보다 높았다.

### Target B/C — 새 boundary 악화와 새 actual error

핵심 incremental target에서는 결과가 약하다. IA5 Target B의 Oracle/Stored Recall@K는 모두 0이며, Target C도 mean 0.010이다. CAR5도 Target B Oracle 0.065/Stored 0.053, Target C Oracle 0.090/Stored 0.075에 그쳤고, prototype baseline이 각각 0.073/0.148로 같거나 더 높았다. 따라서 static risk 식별 가능성을 newly-degraded pair 식별 능력으로 확대 해석할 수 없다.

## 7. Stored vs Oracle gamma

| Protocol | Gamma rank Spearman | 의미 |
|---|---:|---|
| IA5 | 0.980 | Stored statistics가 Oracle gamma 순서 자체는 매우 잘 재현 |
| CAR5 | 0.988 | Stored statistics가 Oracle gamma 순서 자체는 매우 잘 재현 |

IA5 0.980, CAR5 0.988은 stored statistics 품질이 낮아서 ranking이 실패한 것이 아님을 보여준다. Oracle gamma 자체도 newly-degraded target과의 상관이 낮기 때문에 주된 한계는 stored estimation보다는 **mean/covariance Gaussian boundary score와 실제 incremental error target 사이의 약한 연결**이다.

## 8. METHOD READINESS

| Gate | IA5 | CAR5 | Cross-protocol |
|---|---|---|---|
| R1. 충분한 net degradation | NO | NO | FAIL |
| R2. Oracle new-risk ranking | NO | NO | FAIL |
| R3. Stored > prototype | NO | NO | FAIL |
| Final | **NO** | **NO** | **METHOD READY = NO** |

따라서 현재 결과만으로 risk-weighted relation preservation 또는 selective pair intervention을 구현·성능 실험할 단계는 아니다. 먼저 다른 seed/transition에서 newly-degraded mass가 반복되는지 확인하거나, mean/covariance 외 tail/local/coverage 정보가 new-error ranking을 개선하는지 별도 diagnostic으로 검증해야 한다.

## 9. Scientific limitations

- Fixed T5 head를 공통 readout으로 썼으므로 historical T4 classifier의 실제 배포 성능 비교가 아니다.
- Test label로 degradation target을 정의했으므로 이 pair를 Stage D training selection에 그대로 사용할 수 없다.
- `q=Phi(-gamma)`는 projected Gaussian 보조값이지 multiclass error probability가 아니다.
- Gamma correlation은 association이며 causal evidence가 아니다.
- IA5/CAR5 seed 1993 한 개만 분석했으므로 10-task 또는 다른 seed로 일반화할 수 없다.
- Optional Mahalanobis region 표는 selected pair 설명용이며 readiness 판정에 사용하지 않았다.

## 10. Detailed reports and artifacts

- [IA5 report](IA5_T4_T5/IA5_pair_boundary_risk_report.md) — METHOD READY NO
- [CAR5 report](CAR5_T4_T5/CAR5_pair_boundary_risk_report.md) — METHOD READY NO

- [IA5 raw risk table](IA5_T4_T5/stored_oracle_boundary_risk.csv.gz)
- [CAR5 raw risk table](CAR5_T4_T5/stored_oracle_boundary_risk.csv.gz)
- [IA5 Recall@K plots](IA5_T4_T5/recall_at_k/)
- [CAR5 Recall@K plots](CAR5_T4_T5/recall_at_k/)

모든 수치는 association/ranking diagnostic이다. Gamma를 actual error probability로, Gaussian을 실제 분포의 정확한 모델로, correlation을 causal evidence로 해석하지 않는다.
