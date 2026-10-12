# Structure prototype oracle audit — CAR10

> 상태: **COMPLETE** · mode: **SCIENTIFIC** · CA regression: **PASS** · audited tasks: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` · primary oracle: `oracle_train_statistics`

이 보고서는 frozen checkpoint의 post-hoc 진단이다. old raw sample은 oracle mean/covariance 계산에만 사용했고, A–D optimizer에는 synthetic feature와 diagnostic head만 들어갔다. 본 학습의 backbone/adapter/checkpoint는 수정하지 않았다.

## 핵심 수치

| 항목 | 값 |
|---|---:|
| old-class Structure↔oracle cosine 평균 | 0.9899 |
| old-class prototype relative L2 평균 | 0.1271 |
| APR error reduction 평균 | 2.6343 |
| one-step drift error 평균 | 1.9473 |
| cumulative prototype error 평균 | 5.2441 |
| covariance Frobenius relative error 평균 | 0.3909 |
| covariance trace ratio 평균 | 1.1223 |
| A→B Oracle Mean gain (pp) | 0.5348 |
| A→C Oracle Covariance gain (pp) | 0.0746 |
| A→D Full Oracle gain (pp) | 1.1068 |
| final new-only−single-head new (pp) | 14.8909 |

APR class 집계: `improved`=889, `worsened`=5.

## Final-task oracle classifier replacement

진단 task: `10` · initialization: `exact_pre_ca_sidecar`

| condition | seen acc | old acc | new acc | old-only | new-only | old→new | new→old | Δ vs A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL_STRUCTURE | 78.9578 | 78.6836 | 81.5148 | 79.9504 | 96.4056 | 2.5062 | 17.5866 | INCONCLUSIVE |
| A_STRUCTURE | 78.9578 | 78.6836 | 81.5148 | 79.9504 | 96.4056 | 2.5062 | 17.5866 | 0.0000 |
| B_ORACLE_MEAN | 79.4926 | 79.3996 | 80.3594 | 80.5701 | 96.2773 | 2.2721 | 18.7420 | 0.5348 |
| C_ORACLE_COVARIANCE | 79.0325 | 78.7937 | 81.2580 | 80.1019 | 96.2773 | 2.4924 | 17.8434 | 0.0746 |
| D_FULL_ORACLE | 80.0647 | 80.0468 | 80.2311 | 81.0796 | 96.4056 | 2.1344 | 18.8703 | 1.1068 |

## Task progression

| Task | Structure | Oracle Mean | Oracle Cov | Full Oracle |
|---:|---:|---:|---:|---:|
| 2 | 85.9189 | 86.2172 | 85.9785 | 86.3365 |
| 3 | 85.0422 | 85.4443 | 85.1628 | 85.4041 |
| 4 | 83.6864 | 83.9286 | 84.0194 | 84.5036 |
| 5 | 83.6735 | 84.1108 | 84.1108 | 84.4995 |
| 6 | 82.5534 | 83.2190 | 82.7148 | 83.5821 |
| 7 | 82.0674 | 82.5214 | 82.1722 | 82.8357 |
| 8 | 80.7196 | 81.1193 | 80.6273 | 81.5652 |
| 9 | 79.2895 | 80.2671 | 79.7990 | 80.8317 |
| 10 | 78.9578 | 79.4926 | 79.0325 | 80.0647 |

## Case A–F 자동 판정

임계값: mean relative L2 ≤0.20(작음), ≥0.40(큼), head gain ≥1 pp, new-only gap ≥5 pp, new-only <50%(낮음). 이는 재현 가능한 진단 규칙이지 유의성 검정이 아니다.

Case A: mean tracking은 주요 병목일 가능성이 낮음; Case E: old-new score competition/classifier bias가 유력.

## 추가 Q1–Q10 답변

1. **Q1 APR는 pre-APR보다 가까워지는가?** 평균 L2 reduction은 `2.6343`, improved 비율은 `0.9944`다.
2. **Q2 일부 class는 악화되는가?** worsened는 `5/894` class-task다.
3. **Q3 class age와 관계있는가?** age↔APR reduction Spearman `-0.8195`, age↔cumulative relative error Pearson `0.8469`다.
4. **Q4 Oracle Mean 회복량:** B−A seen accuracy는 `0.5348` pp다.
5. **Q5 Oracle Covariance 회복량:** C−A는 `0.0746` pp다.
6. **Q6 Full Oracle 회복량:** D−A는 `1.1068` pp다.
7. **Q7 L2와 boundary 중 어느 쪽이 더 관련되는가?** forgetting Spearman은 L2 `0.4177`, boundary `0.3315`이고 old→new confusion은 L2 `0.0075`, boundary `0.0179`다. 절대값이 큰 지표가 이 run에서 더 연관되지만 인과관계는 아니다.
8. **Q8 representation 손상인가 competition인가?** single-head new `81.5148`, new-only `96.4056`, gap `14.8909` pp, sample-level new→old 비율 `0.1759`다. margin↔new→old Spearman `-0.6558`, margin↔strongest-old prototype error `0.0834`, margin↔boundary error `0.0543`다.
9. **Q9 covariance가 과도하게 팽창하는가?** trace ratio `1.1223`, Structure Δtrace `53.2524`, oracle Δtrace `-20.8114`, Structure 증가 비율 `1.0000`, Structure↑이면서 oracle≤0인 비율 `0.5526`다. 연속 task snapshot이 없으면 Δtrace는 INCONCLUSIVE다.
10. **Q10 현재 병목:** 자동 근거상 **old-new score competition**에 가장 가깝다. 증거가 충돌하거나 partial/single-seed이면 확정 결론은 보류한다.

## 기존 audit 및 Gaussian 보조 결과

- APR/true one-step drift cosine `0.9414`; legacy drift CSV cosine `0.9414`.
- diagonal covariance relative L2 `0.1789`, diagonal Mahalanobis D² Pearson `0.9950`.
- nearest-old ranking 변경률 `0.3740`; 기존 oracle nearest-prototype final gain `14.1649` pp.
- mean |skew| `0.3067`, mean |excess kurtosis| `0.5757`, D²/dimension `0.9754`.

## 해석 제한

- t-SNE 좌표는 task별 독립 좌표다. task 간 이동 해석에는 shared PCA만 사용한다.
- final checkpoint 하나만 있으면 one-step/age/trace-delta는 INCONCLUSIVE다.
- correlation은 인과성을 뜻하지 않으며 single-seed 결과는 반복 검증이 필요하다.
- `shared_post_ca_legacy_init`은 통계 조건 간 공정 비교지만 원래 CA 재현은 아니다.
- class/sample/head override를 쓴 sanity output은 과학적 결과가 아니다.
