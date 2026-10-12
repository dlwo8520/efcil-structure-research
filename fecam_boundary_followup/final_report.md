# FeCAM + Boundary-Aware Sampling 후속 실험

## 출처와 실험상 변경점

- [FeCAM 논문](https://arxiv.org/abs/2309.14062)과 [공식 코드](https://github.com/dipamgoswami/FeCAM)의 class-wise covariance shrinkage (`alpha1=alpha2=10`), covariance correlation normalization, normalized mean/query의 Mahalanobis distance를 사용했다. Tukey transform은 껐다.
- Structure 저장 covariance는 원 feature 공간의 moment이므로 공식 pretrained-ViT 예제의 *sample별 L2 정규화 후 covariance*와 같지 않다. F2와 F3는 동일한 원공간 moment 정의로 비교하고, score 단계에서 정규화한다. 따라서 정확한 pretrained-ViT FeCAM 재현이 아닌 Structure 통계 이식 실험이다.
- CEOS는 [공식 코드](https://github.com/HXuSz11/ACB_CEOS_CVPR2026_Findings)의 `expansive_oversampling`에서 nearest enemy, `lambda ~ Uniform(0.7,1)`, L2 정규화를 이식했다. 공식 current minibatch 대신 고정된 전체 current-task actual train feature pool에서 enemy를 찾는다. ACB/focal/class weighting은 사용하지 않는다.
- 모든 조건은 seed 1993, Structure final T10 frozen feature/checkpoint를 사용한다. Oracle old training feature는 진단 전용이다.

## FeCAM 결과 (%)

| Protocol | 조건 | Seen | Old | New | Old-only | New-only | Macro | Old→New | New→Old |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IA10 | F0 | 64.71 | 64.52 | 66.94 | 65.45 | 84.68 | 53.59 | 0.04 | 0.32 |
| IA10 | F1 | 60.76 | 61.08 | 57.26 | 61.72 | 82.26 | 48.74 | 0.03 | 0.42 |
| IA10 | F2 | 59.91 | 60.14 | 57.26 | 61.15 | 83.06 | 45.05 | 0.04 | 0.43 |
| IA10 | F3 | 60.37 | 60.50 | 58.87 | 61.65 | 83.06 | 45.65 | 0.04 | 0.41 |
| CAR10 | F0 | 78.96 | 78.68 | 81.51 | 79.95 | 96.41 | 79.06 | 0.03 | 0.18 |
| CAR10 | F1 | 62.67 | 62.41 | 65.08 | 64.00 | 90.37 | 62.87 | 0.05 | 0.32 |
| CAR10 | F2 | 74.59 | 74.13 | 78.95 | 75.78 | 94.22 | 74.66 | 0.04 | 0.19 |
| CAR10 | F3 | 77.13 | 77.14 | 77.02 | 78.39 | 94.22 | 77.05 | 0.03 | 0.21 |

F0=기존 Structure CA, F1=공식 pretrained-ViT NCM 정의(샘플별 정규화 후 평균), F2=Stored-FeCAM, F3=Oracle-FeCAM(진단 전용). 전이율은 0–1 범위다. Old-only/New-only는 해당 그룹의 class만 후보로 제한한 정확도이다.

## Boundary margin audit

| Protocol | Source | Macro q05 gap | Macro q10 gap | q05 positive class fraction | q10 positive class fraction |
|---|---|---:|---:|---:|---:|
| IA10 | BB_oracle_gaussian | 0.73 | 0.65 | 0.79 | 0.82 |
| IA10 | BC_stored_gaussian | 0.95 | 0.85 | 0.82 | 0.86 |
| CAR10 | BB_oracle_gaussian | 0.28 | 0.23 | 0.76 | 0.75 |
| CAR10 | BC_stored_gaussian | 0.40 | 0.30 | 0.79 | 0.78 |

양의 gap은 Gaussian이 실제 old feature보다 해당 lower-tail margin에서 쉬움을 뜻한다. 상세 class·draw 수치는 `boundary_audit/*/per_class_margins.csv`에 있다.

## CEOS-only matched CA

| Protocol | 조건 | Seen | Old | New | Old-only | New-only | Macro | Old→New | New→Old |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IA10 | C0_matched_stored_gaussian | 64.38 | 64.23 | 66.13 | 65.16 | 83.87 | 53.86 | 0.04 | 0.33 |
| IA10 | C1_ceos_only | 63.73 | 63.87 | 62.10 | 64.73 | 83.06 | 53.26 | 0.03 | 0.37 |
| CAR10 | C0_matched_stored_gaussian | 78.56 | 78.39 | 80.10 | 79.72 | 96.92 | 78.63 | 0.03 | 0.19 |
| CAR10 | C1_ceos_only | 78.09 | 78.49 | 74.33 | 79.28 | 96.79 | 78.17 | 0.02 | 0.25 |

C0는 앞선 matched stored-Gaussian CA, C1은 동일한 stored Gaussian old 샘플에 CEOS만 적용한 CA이다. 두 조건은 256/class, 같은 new bank·pre-CA 초기화·permutation·optimizer·step 수를 쓴다.

| Protocol | C0 q05 | C1 q05 | C0 q10 | C1 q10 | C0 negative | C1 negative | C0 train loss | C1 train loss | C0 score reproduction max diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IA10 | 0.574 | 0.342 | 1.150 | 0.915 | 0.023 | 0.032 | 0.412 | 0.434 | 0 |
| CAR10 | 0.147 | -0.075 | 0.624 | 0.405 | 0.040 | 0.056 | 0.605 | 0.660 | 0 |
고정 Structure head에서 CEOS bank의 q05/q10이 낮아졌으나, C1의 final test accuracy는 개선되지 않았다. C0 재학습 점수가 저장된 matched C0와 정확히 일치하는지 위 마지막 열로 검증했다.

## 판정

- Stored-FeCAM − Structure Seen: IA10 -4.81 pp, CAR10 -4.37 pp. Old/New trade-off와 macro를 함께 확인해야 한다.
- Stored-FeCAM − NCM Seen: IA10 -0.86 pp, CAR10 +11.93 pp. NCM은 샘플별 정규화 후 mean, F2는 저장 원공간 moment이므로 이 차이를 covariance 효과만으로 분해할 수 없다.
- Oracle-FeCAM − Stored-FeCAM Seen: IA10 +0.46 pp, CAR10 +2.54 pp. CAR10에서 저장 통계 품질의 영향 가능성이 더 크지만, Oracle도 기존 Structure F0를 넘지 못한다.
- FECAM PROTOTYPE READY = NO (탐색적·단일 seed 기준)
- CEOS − C0 Seen: IA10 -0.66 pp, CAR10 -0.47 pp.
- CEOS Old/New 변화: IA10 -0.36/-4.03 pp, CAR10 +0.10/-5.78 pp. CAR10의 작은 Old 이득은 큰 New 손실로 상쇄된다.
- 두 protocol 모두 Gaussian의 boundary-hard sample 부족 신호는 있었지만, CEOS로 margin을 낮춰도 downstream CA 정확도는 오르지 않았다. 이 실험에서는 boundary gap이 주된 성능 병목이라는 인과 주장을 지지하지 않는다.
- CEOS PROTOTYPE READY = NO (탐색적·단일 seed 기준)
- FULL METHOD READY = NO. 독립 seed 및 task transition 검증 전 일반화 주장 불가.

## 제한 및 다음 단계

FeCAM은 공식 pretrained-ViT의 정규화-feature covariance와 다른 Structure 저장 원공간 covariance를 사용한다. CEOS는 공식 minibatch enemy가 아닌 전체 current-task train pool을 사용한다. 두 차이 모두 결과 해석을 제한한다. 현재 설정은 두 protocol 공통 개선이 없어 FeCAM/CEOS 축을 우선 보류한다. 후속 재검증 시에는 정규화 moment를 class 최초 등장 때 저장하는 설계와 official minibatch CEOS를 별도 비교해야 한다.

세부 class 결과와 covariance status는 각 하위 폴더의 CSV/JSON에, 공식 소스 대조는 `source_audit.md`에 있다.
