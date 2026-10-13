# Coverage Source + Pretrained-vs-Final Generalization Diagnostic

## 1. 실행 범위와 통제

- Protocol: `['IA10', 'CAR10']`, seed `1993`, final task only.
- 기존 Structure checkpoint와 final-feature cache를 읽기 전용으로 사용했다. Backbone/CIL training은 재실행하지 않았다.
- B/C Gaussian은 class별 A의 unique reference 수와 동일하게 맞추고 `3`개 독립 draw를 사용했다.
- Gaussian은 raw feature space에서 생성한 뒤 support 계산 직전에 L2 normalize했다.
- New-class bank는 actual-current로 고정했으며 old/new query 결과를 구분했다.
- Pretrained/final cosine probe는 같은 train/validation index, optimizer, LR grid, epoch permutation과 budget을 사용했다. Test label은 선택에 사용하지 않았다.
- Negative classifier-margin fraction은 별도 representation evidence로 재사용하지 않았다.

## 2. 핵심 결과

### 2.1 Coverage source

| Protocol | A margin | A correct | A wrong | B-A micro | B-A macro | C-B micro | C-B macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| IA10 | 0.0225 | 0.0761 | -0.0758 | -0.0070 ± 0.0006 | -0.0020 | -0.0047 ± 0.0011 | -0.0037 |
| CAR10 | 0.0333 | 0.0548 | -0.0474 | -0.0104 ± 0.0004 | -0.0104 | -0.0120 ± 0.0004 | -0.0118 |

`B-A`와 `C-B`의 ± 값은 Gaussian draw 간 표준편차다. Micro와 class-macro를 함께 제시하며, query-level paired delta는 각 protocol CSV에 보존했다.

### 2.2 Pretrained vs Structure final

| Protocol | P0 train | P1 train | P0 test | P1 test | Δ test | P0 macro | P1 macro | Δ support |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IA10 | 89.70 | 90.79 | 50.43 | 61.82 | +11.39pp | 40.74 | 50.21 | +0.0342 |
| CAR10 | 92.16 | 99.38 | 56.59 | 80.81 | +24.23pp | 56.54 | 80.70 | +0.0430 |

![Pretrained와 Structure final cosine-probe test accuracy](cross_protocol/plots/pretrained_vs_final_probe_accuracy.png)

## 3. 필수 질문 Q1–Q9

### Q1. Actual-current bank에서도 wrong held-out query support가 낮은가?

IA10 correct/wrong margin은 `0.0761/-0.0758`; CAR10 correct/wrong margin은 `0.0548/-0.0474`이다. 이 결과는 실제 current training reference와 cosine metric 기준의 coverage를 말하며 representation overfitting 자체를 확정하지 않는다.

### Q2. Oracle Gaussian은 actual 대비 support margin을 감소시키는가?

IA10 B-A `-0.0070` (class-macro `-0.0020`, paired 95% CI `[-0.0088, -0.0052]`); CAR10 B-A `-0.0104` (class-macro `-0.0104`, paired 95% CI `[-0.0111, -0.0097]`). 감소의 크기·방향이 두 protocol과 class-macro에서 일치하는지를 기준으로 판단해야 한다.

### Q3. Structure Gaussian은 Oracle Gaussian보다 추가 손실을 만드는가?

IA10 C-B `-0.0047` (class-macro `-0.0037`, paired 95% CI `[-0.0063, -0.0031]`); CAR10 C-B `-0.0120` (class-macro `-0.0118`, paired 95% CI `[-0.0127, -0.0113]`). 이 차이는 local support effect이며 기존 Oracle classifier-adaptation 결과와 별도로 downstream 병목을 뜻하지 않는다.

### Q4. Micro뿐 아니라 class-macro에서도 유지되는가?

위 표의 micro/macro 부호와 크기를 함께 기준으로 삼았다. 둘 중 하나에서만 나타난 차이는 일부 class 또는 sample 수 불균형의 영향으로 간주한다.

### Q5. Low-reference-count class 때문만인가?

각 protocol의 `reference_count_support.png`와 `paired_support_delta.csv`에 `01-04`, `05-09`, `10-19`, `20-49`, `50+` bin을 기록했다. A/B/C bank size는 모든 class에서 동일하고 A에는 replacement가 없다.

### Q6. Structure final의 train fit, held-out 성능, support는 좋아졌는가?

IA10: final-pretrained train `+1.09pp`, test `+11.39pp`, support `+0.0342`; CAR10: final-pretrained train `+7.21pp`, test `+24.23pp`, support `+0.0430`. 절대 train/test accuracy도 2.2 표에 함께 제시했다.

### Q7. Continual adaptation이 held-out generalization을 훼손했다는 evidence가 있는가?

강한 evidence는 두 protocol에서 `final train↑`, `final test↓`, `final support↓`가 동시에 나타나고 paired support CI가 0 아래일 때만 인정한다. 최종 상태는 decision table에 반영했다.

### Q8. Support metric과 cosine-probe accuracy 변화가 같은 방향인가?

IA10: probe Δ `+11.39pp`, support Δ `+0.0342`; CAR10: probe Δ `+24.23pp`, support Δ `+0.0430`. 부호가 충돌하면 local cosine support를 representation quality의 충분한 proxy로 사용하지 않는다.

### Q9. 현재 evidence로 method를 만들 준비가 되었는가?

**METHOD READY = NO**

Rule E: protocol 공통의 뚜렷한 source/representation 차이가 확립되지 않았다.

## 4. Hypothesis decision table

| Candidate explanation | Evidence | Counter-evidence | Status |
|---|---|---|---|
| Actual training coverage limitation | IA10: correct/wrong A=0.0761/-0.0758; CAR10: correct/wrong A=0.0548/-0.0474 | Low actual-bank support does not identify whether the representation or finite samples caused it. | **SUPPORTED** |
| Representation generalization degradation | IA10: train/test/support delta=+1.09pp/+11.39pp/+0.0342; CAR10: train/test/support delta=+7.21pp/+24.23pp/+0.0430 | Single seed; pretrained and final features may improve different aspects of geometry. | **NOT SUPPORTED** |
| Gaussianization information loss | IA10: B-A micro/macro=-0.0070/-0.0020; CAR10: B-A micro/macro=-0.0104/-0.0104 | A geometry effect alone is not downstream error recovery. | **NOT SUPPORTED** |
| Stored moment approximation loss | IA10: C-B micro/macro=-0.0047/-0.0037; CAR10: C-B micro/macro=-0.0120/-0.0118 | Must be reconciled with the small/inconsistent Oracle classifier-adaptation gains. | **NOT SUPPORTED** |
| Single-head competition | A/B/C support and representation-specific cosine probes expose geometry/decision disagreement. | This diagnostic does not isolate calibration or candidate-count causally. | **INCONCLUSIVE** |
| Local cosine metric limitation | Support-margin and probe-accuracy directions do not show a material sign conflict. | Agreement in direction still does not make support a sufficient quality metric. | **INCONCLUSIVE** |

## 5. Protocol별 시각화

### IA10

![IA10 A/B/C support margin](IA10/plots/abc_support_margin_distribution.png)

![IA10 A/B/C own support](IA10/plots/abc_own_support_distribution.png)

![IA10 A/B/C rival support](IA10/plots/abc_rival_support_distribution.png)

![IA10 paired B minus A](IA10/plots/paired_B_minus_A.png)

![IA10 paired C minus B](IA10/plots/paired_C_minus_B.png)

![IA10 correct wrong support](IA10/plots/correct_wrong_support.png)

![IA10 reference count support](IA10/plots/reference_count_support.png)

![IA10 class age support](IA10/plots/class_age_support.png)

![IA10 pretrained final scatter](IA10/plots/pretrained_vs_final_support_scatter.png)

![IA10 final minus pretrained support](IA10/plots/final_minus_pretrained_support.png)

![IA10 representation correct wrong support](IA10/plots/representation_correct_wrong_support.png)

### CAR10

![CAR10 A/B/C support margin](CAR10/plots/abc_support_margin_distribution.png)

![CAR10 A/B/C own support](CAR10/plots/abc_own_support_distribution.png)

![CAR10 A/B/C rival support](CAR10/plots/abc_rival_support_distribution.png)

![CAR10 paired B minus A](CAR10/plots/paired_B_minus_A.png)

![CAR10 paired C minus B](CAR10/plots/paired_C_minus_B.png)

![CAR10 correct wrong support](CAR10/plots/correct_wrong_support.png)

![CAR10 reference count support](CAR10/plots/reference_count_support.png)

![CAR10 class age support](CAR10/plots/class_age_support.png)

![CAR10 pretrained final scatter](CAR10/plots/pretrained_vs_final_support_scatter.png)

![CAR10 final minus pretrained support](CAR10/plots/final_minus_pretrained_support.png)

![CAR10 representation correct wrong support](CAR10/plots/representation_correct_wrong_support.png)

## 6. 산출물

- [`coverage_generalization_audit.md`](coverage_generalization_audit.md)
- [`cross_protocol/summary_table.csv`](cross_protocol/summary_table.csv)
- [`cross_protocol/decision_table.csv`](cross_protocol/decision_table.csv)
- [`run_manifest.json`](run_manifest.json)

각 protocol 폴더에는 요청된 bank 통계, query-level A/B/C, paired delta, pretrained/final support, probe 결과와 plot이 저장되어 있다.

## 7. 한계

1. Seed 1993 단일 seed다.
2. Actual old feature는 diagnostic-only이며 replay-free method 성능으로 해석할 수 없다.
3. Historical train split은 backbone이 과거에 본 sample이므로 representation-unseen validation이 아니다.
4. Gaussian support 차이만으로 multimodality나 multi-center method의 필요성을 결론내릴 수 없다.
5. Candidate-count stress test는 must-have 두 실험보다 우선순위가 낮아 이번 기본 실행에서 제외했다.
