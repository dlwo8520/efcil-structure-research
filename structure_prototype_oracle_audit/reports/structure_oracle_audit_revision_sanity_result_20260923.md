# Structure Prototype Oracle Audit 개정판 — GPU 1 Sanity 결과

작성일: 2026-09-23  
실행 환경: `cds`, physical GPU 1  
상태: **구현 및 축소 동작 검증 완료 / 정식 실험 미실행**

## 1. 목적

기존 Structure prototype audit에 다음 진단을 추가한 뒤, 실제 IA5와 CAR5 final checkpoint에서 전체 경로가 정상 작동하는지 확인했다.

- pre-APR / APR / oracle prototype 비교
- one-step drift error와 cumulative prototype error 분리
- Structure, Oracle Mean, Oracle Covariance, Full Oracle의 classifier adaptation A–D 비교
- single-head / new-only / old-only accuracy와 old-new error decomposition
- boundary-direction prototype error와 sample margin
- covariance trace, effective rank, Gaussian approximation 보조 진단
- pre-APR → APR → oracle 시각화
- Case A–F 및 Q1–Q10 자동 보고서 생성

이번 실행은 **과학적 결론을 얻기 위한 실험이 아니라 코드 동작 검증용 sanity run**이다.

## 2. Sanity 실행 범위

| 항목 | 설정 |
|---|---|
| Protocol | IA5 T5, CAR5 T5 |
| Checkpoint | 기존 완료된 legacy final checkpoint |
| Audited classes | protocol당 old 2개 + current/new 2개 |
| Oracle samples | class당 최대 2개 |
| A–D adaptation | 1 epoch |
| Synthetic samples | class당 2개 |
| Mahalanobis mode | diagonal |
| Visualization | IA5에서 t-SNE/shared PCA 회귀 검증 |

정식 설정인 모든 class·sample, CA 5 epochs, class당 synthetic sample 256개를 사용하지 않았으므로 아래 정확도와 correlation은 연구 결과로 해석하지 않는다.

## 3. 실행 및 안전성 검증 결과

| 검증 항목 | 결과 |
|---|---|
| Python unit/leakage test | 6/6 통과 |
| IA5 sanity elapsed time | 5.30초 |
| CAR5 sanity elapsed time | 7.79초 |
| IA5/CAR5 A–D 완료 | 성공 |
| 요청 CSV 생성 | 성공 |
| 자동 보고서 생성 | 성공 |
| pre-APR/APR/oracle figure 생성 | 성공 |
| 과거 NPZ 시각화 호환성 | 성공 |
| checkpoint fingerprint 전후 비교 | 동일 |
| backbone/adapter optimizer 포함 | 없음 |
| diagnostic 입력 checksum 변경 | 없음 |

old raw sample은 frozen model의 oracle mean/covariance 계산에만 사용했다. A–D adaptation optimizer에는 diagnostic cosine head만 포함되며, backbone과 adapter는 업데이트되지 않았다. 생성된 diagnostic head와 oracle statistics는 본 학습 checkpoint나 다음 task에 전달되지 않는다.

## 4. 핵심 sanity 수치

| 항목 | IA5 | CAR5 |
|---|---:|---:|
| old Structure↔oracle cosine | 0.6167 | 0.7527 |
| prototype relative L2 | 0.7884 | 0.6679 |
| APR L2 error reduction | 0.4457 | 0.2465 |
| APR improved | 2/2 | 1/2 |
| APR worsened | 0/2 | 1/2 |
| cumulative prototype L2 error | 29.9991 | 39.1482 |
| covariance Frobenius relative error | 0.9838 | 1.0115 |
| covariance trace ratio | 1.3887 | 1.5709 |
| nearest-old ranking changed rate | 0.6667 | 0.2500 |
| oracle nearest-prototype gain | +16.6667 pp | +0.0000 pp |

표본이 class당 최대 2개뿐이어서 covariance, Gaussian shape, Mahalanobis correlation 수치는 매우 불안정하다. 특히 one-step drift, class-age correlation, task별 covariance trace 변화는 T1–T4 snapshot이 없어 `INCONCLUSIVE`다.

## 5. Oracle classifier replacement A–D

### IA5

| Condition | Seen | Old | New | Old-only | New-only | Δ vs A |
|---|---:|---:|---:|---:|---:|---:|
| A — Structure | 66.6667 | 66.6667 | 66.6667 | 66.6667 | 100.0000 | 0.0000 |
| B — Oracle Mean | 66.6667 | 66.6667 | 66.6667 | 66.6667 | 100.0000 | 0.0000 |
| C — Oracle Covariance | 66.6667 | 66.6667 | 66.6667 | 66.6667 | 100.0000 | 0.0000 |
| D — Full Oracle | 66.6667 | 66.6667 | 66.6667 | 66.6667 | 100.0000 | 0.0000 |

### CAR5

| Condition | Seen | Old | New | Old-only | New-only | Δ vs A |
|---|---:|---:|---:|---:|---:|---:|
| A — Structure | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| B — Oracle Mean | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| C — Oracle Covariance | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |
| D — Full Oracle | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 100.0000 | 0.0000 |

두 protocol 모두 축소 A–D 정확도 차이가 0 pp였지만, 4개 class와 class당 최대 2개 sample만 사용했기 때문에 “oracle statistics가 효과가 없다”는 결론을 내릴 수 없다.

또한 사용한 기존 checkpoint에는 CA 직전 classifier/RNG sidecar가 없다. 따라서 네 조건은 모두 동일한 **post-CA checkpoint head**에서 시작했고 `shared_post_ca_legacy_init`로 기록됐다. 조건 간 statistics 교체는 공정하지만 원래 Structure CA의 exact replay는 아니다. 새 audit-ready 학습부터는 `pre_ca_classifier/task_XX.pt`를 저장하므로 exact pre-CA 비교가 가능하다.

## 6. New/old competition 진단

| 항목 | IA5 | CAR5 |
|---|---:|---:|
| single-head new accuracy | 100.0000 | 100.0000 |
| new-only accuracy | 100.0000 | 100.0000 |
| new-only − single-head | 0.0000 pp | 0.0000 pp |
| new→old sample rate | 0.0000 | 0.0000 |

sanity subset에서는 old-new competition 신호가 나타나지 않았다. 그러나 test sample 수가 IA5 6개, CAR5 8개뿐이므로 representation/competition 병목 판정은 보류한다.

## 7. 결론

이번 실행으로 확인된 것은 다음과 같다.

1. 기존 Structure 학습 경로를 변경하지 않고 확장 audit가 작동한다.
2. pre-APR/APR/oracle 및 one-step/cumulative 필드가 정상 생성된다.
3. A–D head는 동일 초기화·seed·optimizer 설정으로 실행된다.
4. old/new-only, confusion, margin, boundary, covariance, Gaussian CSV가 생성된다.
5. checkpoint는 read-only로 유지되고 oracle 정보가 본 학습에 누출되지 않는다.
6. IA5와 task partition이 불균등한 CAR5 모두 실행된다.

현재 병목은 **inconclusive**다. prototype tracking, covariance estimation, Gaussian approximation, classifier adaptation, score competition, representation learning 중 어느 하나를 선택하려면 audit-ready T1–T5 snapshot과 전체 class/sample을 사용하는 정식 IA5/CAR5 실행이 필요하다.

## 8. 생성 파일

- [IA5 상세 보고서](../outputs_revision_smoke/IA5/seed_1993/structure_prototype_oracle_audit.md)
- [CAR5 상세 보고서](../outputs_revision_smoke/CAR5/seed_1993/structure_prototype_oracle_audit.md)
- [통합 보고서](../outputs_revision_smoke/structure_prototype_oracle_audit.md)
- [Covariance 코드 감사](../structure_covariance_code_audit.md)
- [Revision 계획](../structure_oracle_audit_revision_plan.md)
- [IA5 A–D CSV](../outputs_revision_smoke/IA5/seed_1993/oracle_classifier_replacement.csv)
- [CAR5 A–D CSV](../outputs_revision_smoke/CAR5/seed_1993/oracle_classifier_replacement.csv)
- [IA5 APR focused PCA](../outputs_revision_smoke/IA5/seed_1993/pca/task_05_APR_improved_shared_pca.png)

## 9. 다음 정식 단계

먼저 GPU 1에서 audit-ready IA5와 CAR5의 T1–T5 checkpoint를 생성하고 final-task A–D를 정식 설정으로 실행한다. 이 결과에서 유망한 원인과 방법을 좁힌 뒤 IA10/CAR10으로 확장한다.

정식 결과가 생성되면 이 sanity 보고서의 수치를 덮어쓰지 않고 별도 scientific report로 작성해야 한다.
