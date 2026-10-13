# Structure prototype oracle audit — IA10

> 상태: **COMPLETE** · mode: **SCIENTIFIC** · CA regression: **PASS** · audited tasks: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]` · primary oracle: `oracle_train_statistics`

이 보고서는 frozen checkpoint의 post-hoc 진단이다. old raw sample은 oracle mean/covariance 계산에만 사용했고, A–D optimizer에는 synthetic feature와 diagnostic head만 들어갔다. 본 학습의 backbone/adapter/checkpoint는 수정하지 않았다.

## 핵심 수치

| 항목 | 값 |
|---|---:|
| old-class Structure↔oracle cosine 평균 | 0.9931 |
| old-class prototype relative L2 평균 | 0.1086 |
| APR error reduction 평균 | 2.5054 |
| one-step drift error 평균 | 1.4484 |
| cumulative prototype error 평균 | 3.3595 |
| covariance Frobenius relative error 평균 | 0.3741 |
| covariance trace ratio 평균 | 1.1587 |
| A→B Oracle Mean gain (pp) | -0.0658 |
| A→C Oracle Covariance gain (pp) | 0.1317 |
| A→D Full Oracle gain (pp) | -0.0658 |
| final new-only−single-head new (pp) | 17.7419 |

APR class 집계: `improved`=900.

## Final-task oracle classifier replacement

진단 task: `10` · initialization: `exact_pre_ca_sidecar`

| condition | seen acc | old acc | new acc | old-only | new-only | old→new | new→old | Δ vs A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ORIGINAL_STRUCTURE | 64.6478 | 64.4444 | 66.9355 | 65.3763 | 84.6774 | 3.5125 | 32.2581 | INCONCLUSIVE |
| A_STRUCTURE | 64.6478 | 64.4444 | 66.9355 | 65.3763 | 84.6774 | 3.5125 | 32.2581 | 0.0000 |
| B_ORACLE_MEAN | 64.5820 | 64.5878 | 64.5161 | 65.5914 | 84.6774 | 3.2975 | 34.6774 | -0.0658 |
| C_ORACLE_COVARIANCE | 64.7795 | 64.5161 | 67.7419 | 65.5197 | 84.6774 | 3.5842 | 31.4516 | 0.1317 |
| D_FULL_ORACLE | 64.5820 | 64.5161 | 65.3226 | 65.5197 | 84.6774 | 3.1541 | 33.8710 | -0.0658 |

## Task progression

| Task | Structure | Oracle Mean | Oracle Cov | Full Oracle |
|---:|---:|---:|---:|---:|
| 2 | 83.3333 | 82.7778 | 83.0556 | 82.2222 |
| 3 | 77.5210 | 77.5210 | 77.7311 | 77.3109 |
| 4 | 78.2235 | 78.2235 | 78.0802 | 77.6504 |
| 5 | 74.1974 | 73.9596 | 73.9596 | 73.4839 |
| 6 | 72.6154 | 72.4103 | 72.4103 | 72.3077 |
| 7 | 69.4023 | 70.0268 | 69.4023 | 69.6699 |
| 8 | 66.9872 | 67.2276 | 67.3077 | 67.3878 |
| 9 | 65.0896 | 65.3763 | 65.0896 | 65.0179 |
| 10 | 64.6478 | 64.5820 | 64.7795 | 64.5820 |

## Case A–F 자동 판정

임계값: mean relative L2 ≤0.20(작음), ≥0.40(큼), head gain ≥1 pp, new-only gap ≥5 pp, new-only <50%(낮음). 이는 재현 가능한 진단 규칙이지 유의성 검정이 아니다.

Case A: mean tracking은 주요 병목일 가능성이 낮음; Case D: moment 보정보다 Gaussian 가정·CA·representation overlap을 더 의심; Case E: old-new score competition/classifier bias가 유력.

## 추가 Q1–Q10 답변

1. **Q1 APR는 pre-APR보다 가까워지는가?** 평균 L2 reduction은 `2.5054`, improved 비율은 `1.0000`다.
2. **Q2 일부 class는 악화되는가?** worsened는 `0/900` class-task다.
3. **Q3 class age와 관계있는가?** age↔APR reduction Spearman `-0.8486`, age↔cumulative relative error Pearson `0.7936`다.
4. **Q4 Oracle Mean 회복량:** B−A seen accuracy는 `-0.0658` pp다.
5. **Q5 Oracle Covariance 회복량:** C−A는 `0.1317` pp다.
6. **Q6 Full Oracle 회복량:** D−A는 `-0.0658` pp다.
7. **Q7 L2와 boundary 중 어느 쪽이 더 관련되는가?** forgetting Spearman은 L2 `0.2543`, boundary `0.2016`이고 old→new confusion은 L2 `-0.0149`, boundary `0.0658`다. 절대값이 큰 지표가 이 run에서 더 연관되지만 인과관계는 아니다.
8. **Q8 representation 손상인가 competition인가?** single-head new `66.9355`, new-only `84.6774`, gap `17.7419` pp, sample-level new→old 비율 `0.3226`다. margin↔new→old Spearman `-0.8025`, margin↔strongest-old prototype error `-0.0203`, margin↔boundary error `-0.0172`다.
9. **Q9 covariance가 과도하게 팽창하는가?** trace ratio `1.1587`, Structure Δtrace `28.5066`, oracle Δtrace `-29.3748`, Structure 증가 비율 `1.0000`, Structure↑이면서 oracle≤0인 비율 `0.7022`다. 연속 task snapshot이 없으면 Δtrace는 INCONCLUSIVE다.
10. **Q10 현재 병목:** 자동 근거상 **old-new score competition**에 가장 가깝다. 증거가 충돌하거나 partial/single-seed이면 확정 결론은 보류한다.

## 기존 audit 및 Gaussian 보조 결과

- APR/true one-step drift cosine `0.9553`; legacy drift CSV cosine `0.9553`.
- diagonal covariance relative L2 `0.1816`, diagonal Mahalanobis D² Pearson `0.9930`.
- nearest-old ranking 변경률 `0.7738`; 기존 oracle nearest-prototype final gain `30.9414` pp.
- mean |skew| `0.3432`, mean |excess kurtosis| `0.8322`, D²/dimension `0.9319`.

## 해석 제한

- t-SNE 좌표는 task별 독립 좌표다. task 간 이동 해석에는 shared PCA만 사용한다.
- final checkpoint 하나만 있으면 one-step/age/trace-delta는 INCONCLUSIVE다.
- correlation은 인과성을 뜻하지 않으며 single-seed 결과는 반복 검증이 필요하다.
- `shared_post_ca_legacy_init`은 통계 조건 간 공정 비교지만 원래 CA 재현은 아니다.
- class/sample/head override를 쓴 sanity output은 과학적 결과가 아니다.

## NEXT_DIAGNOSTIC_RECOMMENDATION

Full Oracle D−A가 1 pp 미만이므로 다음 단계에서는 actual old training features를 사용한 diagnostic balanced classifier를 우선 검토한다. 그것도 개선이 작을 때 actual replay representation reference로 넘어간다. 이번 queue에서는 두 후속 실험을 실행하지 않았다.
