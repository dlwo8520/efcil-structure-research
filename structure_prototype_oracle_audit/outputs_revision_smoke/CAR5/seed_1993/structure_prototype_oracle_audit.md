# Structure prototype oracle audit — CAR5

> 상태: **PARTIAL** · mode: **SANITY ONLY** · audited tasks: `[5]` · primary oracle: `oracle_train_statistics`

이 보고서는 frozen checkpoint의 post-hoc 진단이다. old raw sample은 oracle mean/covariance 계산에만 사용했고, A–D optimizer에는 synthetic feature와 diagnostic head만 들어갔다. 본 학습의 backbone/adapter/checkpoint는 수정하지 않았다. Legacy checkpoint에는 pre-CA sidecar가 없어 본 CA의 exact replay가 아니며, 동일 post-CA 시작점에서 네 statistics 조건의 상대 효과만 비교한다.

## 핵심 수치

| 항목 | 값 |
|---|---:|
| old-class Structure↔oracle cosine 평균 | 0.7527 |
| old-class prototype relative L2 평균 | 0.6679 |
| APR error reduction 평균 | 0.2465 |
| one-step drift error 평균 | INCONCLUSIVE |
| cumulative prototype error 평균 | 39.1482 |
| covariance Frobenius relative error 평균 | 1.0115 |
| covariance trace ratio 평균 | 1.5709 |
| A→B Oracle Mean gain (pp) | 0.0000 |
| A→C Oracle Covariance gain (pp) | 0.0000 |
| A→D Full Oracle gain (pp) | 0.0000 |
| final new-only−single-head new (pp) | 0.0000 |

APR class 집계: `improved`=1, `worsened`=1.

## Final-task oracle classifier replacement

진단 task: `5` · initialization: `shared_post_ca_legacy_init`

| condition | seen acc | old acc | new acc | old-only | new-only | Δ vs A |
|---|---:|---:|---:|---:|---:|---:|
| A_STRUCTURE | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| B_ORACLE_MEAN | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| C_ORACLE_COVARIANCE | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| D_FULL_ORACLE | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |

## Case A–F 자동 판정

임계값: mean relative L2 ≤0.20(작음), ≥0.40(큼), head gain ≥1 pp, new-only gap ≥5 pp, new-only <50%(낮음). 이는 재현 가능한 진단 규칙이지 유의성 검정이 아니다.

SANITY ONLY — 축소/override 실행에서는 Case A–F를 과학적으로 판정하지 않음.

## 추가 Q1–Q10 답변

1. **Q1 APR는 pre-APR보다 가까워지는가?** 평균 L2 reduction은 `0.2465`, improved 비율은 `0.5000`다.
2. **Q2 일부 class는 악화되는가?** worsened는 `1/2` class-task다.
3. **Q3 class age와 관계있는가?** age↔APR reduction Spearman `INCONCLUSIVE`, age↔cumulative relative error Pearson `INCONCLUSIVE`다.
4. **Q4 Oracle Mean 회복량:** B−A seen accuracy는 `0.0000` pp다.
5. **Q5 Oracle Covariance 회복량:** C−A는 `0.0000` pp다.
6. **Q6 Full Oracle 회복량:** D−A는 `0.0000` pp다.
7. **Q7 L2와 boundary 중 어느 쪽이 더 관련되는가?** forgetting Spearman은 L2 `INCONCLUSIVE`, boundary `INCONCLUSIVE`이고 old→new confusion은 L2 `INCONCLUSIVE`, boundary `INCONCLUSIVE`다. 절대값이 큰 지표가 이 run에서 더 연관되지만 인과관계는 아니다.
8. **Q8 representation 손상인가 competition인가?** single-head new `100.0000`, new-only `100.0000`, gap `0.0000` pp, sample-level new→old 비율 `0.0000`다. margin↔new→old Spearman `INCONCLUSIVE`, margin↔strongest-old prototype error `-0.2582`, margin↔boundary error `0.2582`다.
9. **Q9 covariance가 과도하게 팽창하는가?** trace ratio `1.5709`, Structure Δtrace `INCONCLUSIVE`, oracle Δtrace `INCONCLUSIVE`, Structure 증가 비율 `INCONCLUSIVE`, Structure↑이면서 oracle≤0인 비율 `INCONCLUSIVE`다. 연속 task snapshot이 없으면 Δtrace는 INCONCLUSIVE다.
10. **Q10 현재 병목:** 자동 근거상 **inconclusive (sanity override)**에 가장 가깝다. 증거가 충돌하거나 partial/single-seed이면 확정 결론은 보류한다.

## 기존 audit 및 Gaussian 보조 결과

- APR/true one-step drift cosine `INCONCLUSIVE`; legacy drift CSV cosine `INCONCLUSIVE`.
- diagonal covariance relative L2 `0.8850`, diagonal Mahalanobis D² Pearson `INCONCLUSIVE`.
- nearest-old ranking 변경률 `0.2500`; 기존 oracle nearest-prototype final gain `0.0000` pp.
- mean |skew| `INCONCLUSIVE`, mean |excess kurtosis| `INCONCLUSIVE`, D²/dimension `0.4901`.

## 해석 제한

- t-SNE 좌표는 task별 독립 좌표다. task 간 이동 해석에는 shared PCA만 사용한다.
- final checkpoint 하나만 있으면 one-step/age/trace-delta는 INCONCLUSIVE다.
- correlation은 인과성을 뜻하지 않으며 single-seed 결과는 반복 검증이 필요하다.
- `shared_post_ca_legacy_init`은 통계 조건 간 공정 비교지만 원래 CA 재현은 아니다.
- class/sample/head override를 쓴 sanity output은 과학적 결과가 아니다.
