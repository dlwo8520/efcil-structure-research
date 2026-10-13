# Structure prototype oracle audit — IA5

> 상태: **COMPLETE** · mode: **SCIENTIFIC** · audited tasks: `[1, 2, 3, 4, 5]` · primary oracle: `oracle_train_statistics`

이 보고서는 frozen checkpoint의 post-hoc 진단이다. old raw sample은 oracle mean/covariance 계산에만 사용했고, A–D optimizer에는 synthetic feature와 diagnostic head만 들어갔다. 본 학습의 backbone/adapter/checkpoint는 수정하지 않았다.

## 핵심 수치

| 항목 | 값 |
|---|---:|
| old-class Structure↔oracle cosine 평균 | 0.9948 |
| old-class prototype relative L2 평균 | 0.0987 |
| APR error reduction 평균 | 2.8392 |
| one-step drift error 평균 | 1.6265 |
| cumulative prototype error 평균 | 2.7006 |
| covariance Frobenius relative error 평균 | 0.3030 |
| covariance trace ratio 평균 | 1.1297 |
| A→B Oracle Mean gain (pp) | -0.3292 |
| A→C Oracle Covariance gain (pp) | -0.3950 |
| A→D Full Oracle gain (pp) | -0.5267 |
| final new-only−single-head new (pp) | 16.9742 |

APR class 집계: `improved`=400.

## Final-task oracle classifier replacement

진단 task: `5` · initialization: `exact_pre_ca_sidecar`

| condition | seen acc | old acc | new acc | old-only | new-only | Δ vs A |
|---|---:|---:|---:|---:|---:|---:|
| A_STRUCTURE | 66.5569 | 67.0673 | 64.2066 | 69.2308 | 80.8118 | 0.0000 |
| B_ORACLE_MEAN | 66.2278 | 67.2276 | 61.6236 | 69.1506 | 80.8118 | -0.3292 |
| C_ORACLE_COVARIANCE | 66.1619 | 66.8269 | 63.0996 | 69.0705 | 80.8118 | -0.3950 |
| D_FULL_ORACLE | 66.0303 | 66.9071 | 61.9926 | 68.9103 | 81.1808 | -0.5267 |

## Case A–F 자동 판정

임계값: mean relative L2 ≤0.20(작음), ≥0.40(큼), head gain ≥1 pp, new-only gap ≥5 pp, new-only <50%(낮음). 이는 재현 가능한 진단 규칙이지 유의성 검정이 아니다.

Case A: mean tracking은 주요 병목일 가능성이 낮음; Case D: moment 보정보다 Gaussian 가정·CA·representation overlap을 더 의심; Case E: old-new score competition/classifier bias가 유력.

## 추가 Q1–Q10 답변

1. **Q1 APR는 pre-APR보다 가까워지는가?** 평균 L2 reduction은 `2.8392`, improved 비율은 `1.0000`다.
2. **Q2 일부 class는 악화되는가?** worsened는 `0/400` class-task다.
3. **Q3 class age와 관계있는가?** age↔APR reduction Spearman `-0.7891`, age↔cumulative relative error Pearson `0.7915`다.
4. **Q4 Oracle Mean 회복량:** B−A seen accuracy는 `-0.3292` pp다.
5. **Q5 Oracle Covariance 회복량:** C−A는 `-0.3950` pp다.
6. **Q6 Full Oracle 회복량:** D−A는 `-0.5267` pp다.
7. **Q7 L2와 boundary 중 어느 쪽이 더 관련되는가?** forgetting Spearman은 L2 `0.1949`, boundary `0.1436`이고 old→new confusion은 L2 `-0.0224`, boundary `-0.0070`다. 절대값이 큰 지표가 이 run에서 더 연관되지만 인과관계는 아니다.
8. **Q8 representation 손상인가 competition인가?** single-head new `63.8376`, new-only `80.8118`, gap `16.9742` pp, sample-level new→old 비율 `0.3137`다. margin↔new→old Spearman `-0.7625`, margin↔strongest-old prototype error `-0.0254`, margin↔boundary error `-0.0489`다.
9. **Q9 covariance가 과도하게 팽창하는가?** trace ratio `1.1297`, Structure Δtrace `23.8075`, oracle Δtrace `-58.3876`, Structure 증가 비율 `1.0000`, Structure↑이면서 oracle≤0인 비율 `0.9100`다. 연속 task snapshot이 없으면 Δtrace는 INCONCLUSIVE다.
10. **Q10 현재 병목:** 자동 근거상 **old-new score competition**에 가장 가깝다. 증거가 충돌하거나 partial/single-seed이면 확정 결론은 보류한다.

## 기존 audit 및 Gaussian 보조 결과

- APR/true one-step drift cosine `0.9456`; legacy drift CSV cosine `0.9456`.
- diagonal covariance relative L2 `0.1478`, diagonal Mahalanobis D² Pearson `0.9954`.
- nearest-old ranking 변경률 `0.4961`; 기존 oracle nearest-prototype final gain `30.2172` pp.
- mean |skew| `0.3454`, mean |excess kurtosis| `0.8336`, D²/dimension `0.9320`.

## 해석 제한

- t-SNE 좌표는 task별 독립 좌표다. task 간 이동 해석에는 shared PCA만 사용한다.
- final checkpoint 하나만 있으면 one-step/age/trace-delta는 INCONCLUSIVE다.
- correlation은 인과성을 뜻하지 않으며 single-seed 결과는 반복 검증이 필요하다.
- `shared_post_ca_legacy_init`은 통계 조건 간 공정 비교지만 원래 CA 재현은 아니다.
- class/sample/head override를 쓴 sanity output은 과학적 결과가 아니다.
