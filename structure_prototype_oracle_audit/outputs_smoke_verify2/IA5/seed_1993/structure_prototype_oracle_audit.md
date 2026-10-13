# Structure prototype oracle audit — IA5

> 상태: **PARTIAL** · audited tasks: `[5]` · primary oracle: `oracle_train_statistics`

이 보고서는 frozen checkpoint의 post-hoc 진단이다. old raw sample은 oracle mean/covariance 계산에만 사용했고, A–D optimizer에는 synthetic feature와 diagnostic head만 들어갔다. 본 학습의 backbone/adapter/checkpoint는 수정하지 않았다.

## 핵심 수치

| 항목 | 값 |
|---|---:|
| old-class Structure↔oracle cosine 평균 | 0.6167 |
| old-class prototype relative L2 평균 | 0.7884 |
| APR error reduction 평균 | INCONCLUSIVE |
| one-step drift error 평균 | INCONCLUSIVE |
| cumulative prototype error 평균 | INCONCLUSIVE |
| covariance Frobenius relative error 평균 | 0.9838 |
| covariance trace ratio 평균 | INCONCLUSIVE |
| A→B Oracle Mean gain (pp) | INCONCLUSIVE |
| A→C Oracle Covariance gain (pp) | INCONCLUSIVE |
| A→D Full Oracle gain (pp) | INCONCLUSIVE |
| final new-only−single-head new (pp) | INCONCLUSIVE |

APR class 집계: available row 없음.

## Final-task oracle classifier replacement

진단 task: `not run` · initialization: `not-run`

| condition | seen acc | old acc | new acc | old-only | new-only | Δ vs A |
|---|---:|---:|---:|---:|---:|---:|
| A_STRUCTURE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| B_ORACLE_MEAN | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| C_ORACLE_COVARIANCE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |
| D_FULL_ORACLE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE | INCONCLUSIVE |

## Case A–F 자동 판정

임계값: mean relative L2 ≤0.20(작음), ≥0.40(큼), head gain ≥1 pp, new-only gap ≥5 pp, new-only <50%(낮음). 이는 재현 가능한 진단 규칙이지 유의성 검정이 아니다.

임계값을 만족하는 단일 Case가 없어 INCONCLUSIVE.

## 추가 Q1–Q10 답변

1. **Q1 APR는 pre-APR보다 가까워지는가?** 평균 L2 reduction은 `INCONCLUSIVE`, improved 비율은 `0.0000`다.
2. **Q2 일부 class는 악화되는가?** worsened는 `0/0` class-task다.
3. **Q3 class age와 관계있는가?** age↔APR reduction Spearman `INCONCLUSIVE`, age↔cumulative relative error Pearson `INCONCLUSIVE`다.
4. **Q4 Oracle Mean 회복량:** B−A seen accuracy는 `INCONCLUSIVE` pp다.
5. **Q5 Oracle Covariance 회복량:** C−A는 `INCONCLUSIVE` pp다.
6. **Q6 Full Oracle 회복량:** D−A는 `INCONCLUSIVE` pp다.
7. **Q7 L2와 boundary 중 어느 쪽이 더 관련되는가?** forgetting Spearman은 L2 `INCONCLUSIVE`, boundary `INCONCLUSIVE`이고 old→new confusion은 L2 `INCONCLUSIVE`, boundary `INCONCLUSIVE`다. 절대값이 큰 지표가 이 run에서 더 연관되지만 인과관계는 아니다.
8. **Q8 representation 손상인가 competition인가?** single-head new `INCONCLUSIVE`, new-only `INCONCLUSIVE`, gap `INCONCLUSIVE` pp, sample-level new→old 비율 `INCONCLUSIVE`다.
9. **Q9 covariance가 과도하게 팽창하는가?** trace ratio `INCONCLUSIVE`, Structure Δtrace `INCONCLUSIVE`, oracle Δtrace `INCONCLUSIVE`, Structure 증가 비율 `INCONCLUSIVE`다. 연속 task snapshot이 없으면 Δtrace는 INCONCLUSIVE다.
10. **Q10 현재 병목:** 자동 근거상 **inconclusive**에 가장 가깝다. 증거가 충돌하거나 partial/single-seed이면 확정 결론은 보류한다.

## 기존 audit 및 Gaussian 보조 결과

- APR/true one-step drift cosine `INCONCLUSIVE`; legacy drift CSV cosine `INCONCLUSIVE`.
- diagonal covariance relative L2 `0.9245`, diagonal Mahalanobis D² Pearson `INCONCLUSIVE`.
- nearest-old ranking 변경률 `INCONCLUSIVE`; 기존 oracle nearest-prototype final gain `33.3333` pp.
- mean |skew| `INCONCLUSIVE`, mean |excess kurtosis| `INCONCLUSIVE`, D²/dimension `INCONCLUSIVE`.

## 해석 제한

- t-SNE 좌표는 task별 독립 좌표다. task 간 이동 해석에는 shared PCA만 사용한다.
- final checkpoint 하나만 있으면 one-step/age/trace-delta는 INCONCLUSIVE다.
- correlation은 인과성을 뜻하지 않으며 single-seed 결과는 반복 검증이 필요하다.
- `shared_post_ca_legacy_init`은 통계 조건 간 공정 비교지만 원래 CA 재현은 아니다.
- class/sample/head override를 쓴 sanity output은 과학적 결과가 아니다.
