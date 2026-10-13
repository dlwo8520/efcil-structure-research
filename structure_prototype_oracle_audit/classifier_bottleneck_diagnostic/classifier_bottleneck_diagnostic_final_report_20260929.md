# Structure Classifier Bottleneck Diagnostic 최종 보고서

> 작성일: 2026-09-29  
> 실행 장치: GPU 1  
> 대상: IA5, CAR5 / seed 1993 / T2–T5  
> 상태: **COMPLETE**  
> 기존 Structure checkpoint 변경: **없음**

## 1. 최종 결론

이번 실험은 Structure를 처음부터 다시 학습한 실험이 아니다. 기존 task checkpoint와 scientific audit 결과를 재사용하고, frozen backbone에서 feature를 추출한 뒤 classifier adaptation 병목만 진단했다.

핵심 결론은 다음과 같다.

1. Actual old feature를 직접 사용한 CA는 fixed Gaussian CA보다 모든 transition에서 높았지만, 개선 폭은 `+0.3283~+1.0417 pp`로 작았다.
2. Actual feature 사용이 final-task New→Old confusion을 줄이지는 못했다.
3. Oracle-NCM은 Full Oracle Gaussian CA보다 모든 transition에서 낮았다.
4. 단일 old/new bias 조절로 Old 정확도를 유지하면서 New 정확도를 크게 회복하지 못했다.
5. 따라서 Gaussian replacement가 일부 손실을 만드는 것은 확인됐지만, 현재 classifier 병목의 주원인이라는 강한 근거는 아니다.
6. 남는 후보는 CA objective, classifier form, optimization 및 feature-space class overlap이지만 이번 실험만으로 하나를 원인으로 확정하지 않는다.

## 2. Q1–Q8 판정

| 질문 | 통합 판정 | IA5 | CAR5 |
|---|---|---|---|
| Q1. Oracle NCM은 Full Oracle Gaussian CA보다 높은가? | **NO** | NO | NO |
| Q2. Actual old feature CA는 matched Gaussian CA보다 높은가? | **YES** | YES | YES |
| Q3. Gaussian→Actual이 New→Old confusion을 줄이는가? | **INCONCLUSIVE** | INCONCLUSIVE | INCONCLUSIVE |
| Q4. Actual CA에도 New-only gap이 크게 남는가? | **INCONCLUSIVE** | YES | NO |
| Q5. 단일 bias로 Old를 크게 잃지 않고 New를 회복할 수 있는가? | **NO** | NO | NO |
| Q6. Structure/D/R이 서로 다른 bias frontier를 만드는가? | **YES** | YES | YES |
| Q7. CA 이후 weight가 class mean 방향에서 멀어지는가? | **NO** | NO | NO |
| Q8. G/R 차이를 optimization/fitting 차이로 설명할 수 있는가? | **INCONCLUSIVE** | INCONCLUSIVE | INCONCLUSIVE |

## 3. 실험 구성

### 3.1 재사용한 결과

- task별 Structure checkpoint
- frozen backbone 및 adapter
- pre-CA classifier sidecar와 RNG state
- Structure/APR mean 및 covariance
- 기존 scientific audit의 Structure CA, Oracle Mean CA, Full Oracle Gaussian CA 결과

Full Oracle Gaussian CA의 accuracy는 기존 결과를 재사용했다. Bias 및 margin 분석에 필요한 D head만 저장된 pre-CA RNG로 재구성했으며, 재구성된 Seen accuracy는 기존 결과와 모든 task에서 정확히 일치했다.

### 3.2 새로 계산한 조건

- APR-NCM
- Oracle-NCM
- G-fixed: 고정된 matched-count oracle Gaussian old bank
- R-fixed: 실제 old training feature bank
- Structure CA, Full Oracle Gaussian CA, R-fixed에 대한 bias sweep
- score, margin, calibration, weight–mean alignment 및 optimization trace

APR-NCM과 Oracle-NCM은 current/new prototype을 동일하게 유지하고 old prototype source만 변경했다.

G-fixed와 R-fixed는 다음 조건을 동일하게 유지했다.

- pre-CA classifier initialization
- classifier architecture
- optimizer, learning rate, weight decay
- epoch와 batch size
- shuffle seed
- current/new fixed Structure Gaussian bank
- evaluation sample과 global label mapping

달라지는 변수는 old feature source뿐이다.

### 3.3 Feature 및 classifier 정의

- feature: frozen network의 raw `sfeatures`
- classifier: bias가 없는 cosine classifier
- score: `20 × cosine(L2-normalized feature, L2-normalized weight)`
- L2 normalization: classifier 내부에서 수행
- classifier adaptation epoch: 5
- configured synthetic count: class당 256
- feature dimension: 768

모든 실제 old class의 train sample 수가 256보다 작았다.

- IA5: class당 2–86개
- CAR5: class당 24–68개

따라서 이번 데이터에서는 Matched Count와 Full Actual bank가 정확히 같으며, `R_FIXED`가 두 조건을 동시에 나타낸다.

## 4. Final-task 절대 정확도

### 4.1 IA5 T5

| Condition | Seen | Old | New | Old-only | New-only | New→Old |
|---|---:|---:|---:|---:|---:|---:|
| Structure CA | 66.4253 | 66.9872 | 63.8376 | 69.1506 | 80.8118 | 31.3653 |
| Oracle Mean CA | 66.0303 | 66.9872 | 61.6236 | 68.9904 | 80.8118 | 33.5793 |
| Full Oracle Gaussian CA | 66.0961 | 66.9071 | 62.3616 | 68.9103 | 81.5498 | 33.2103 |
| Fixed Oracle Gaussian CA | 62.8703 | 60.3365 | 74.5387 | 69.0705 | 80.8118 | 11.4391 |
| Actual Feature CA | 63.1995 | 60.6571 | 74.9077 | 69.3109 | 80.4428 | 11.4391 |
| APR-NCM | 61.6195 | 62.0192 | 59.7786 | 64.8237 | 75.6458 | 33.5793 |
| Oracle-NCM | 61.5537 | 62.6603 | 56.4576 | 65.1442 | 75.6458 | 38.0074 |

IA5 T5에서 R-fixed−G-fixed Seen은 `+0.3292 pp`였다. Actual CA의 New-only−single-head New gap은 `5.5351 pp`였다.

### 4.2 CAR5 T5

| Condition | Seen | Old | New | Old-only | New-only | New→Old |
|---|---:|---:|---:|---:|---:|---:|
| Structure CA | 81.1964 | 80.9870 | 82.0545 | 83.1219 | 92.5808 | 14.6481 |
| Oracle Mean CA | 81.5073 | 81.4356 | 81.8009 | 83.4004 | 92.5808 | 14.7749 |
| Full Oracle Gaussian CA | 81.8679 | 81.8379 | 81.9911 | 83.7407 | 92.7077 | 14.5212 |
| Fixed Oracle Gaussian CA | 75.9980 | 72.4319 | 90.6151 | 82.7197 | 92.3906 | 2.7901 |
| Actual Feature CA | 76.8934 | 73.5149 | 90.7419 | 83.0600 | 92.3906 | 2.8535 |
| APR-NCM | 63.5991 | 63.5056 | 63.9822 | 66.5223 | 81.2936 | 27.5206 |
| Oracle-NCM | 65.3401 | 65.5167 | 64.6164 | 68.3478 | 81.2936 | 26.6963 |

CAR5 T5에서 R-fixed−G-fixed Seen은 `+0.8954 pp`였다. Actual CA의 New-only−single-head New gap은 `1.6487 pp`였다.

## 5. 전체 transition 결과

| Protocol | Task | R−G Seen | Oracle-NCM−Full Oracle Gaussian CA | ΔNew→Old |
|---|---:|---:|---:|---:|
| IA5 | T2 | +0.4298 | -3.8682 | +0.5917 |
| IA5 | T3 | +0.6154 | -4.4103 | +0.3610 |
| IA5 | T4 | +1.0417 | -3.6859 | +1.4652 |
| IA5 | T5 | +0.3292 | -4.5425 | +0.0000 |
| CAR5 | T2 | +0.3369 | -13.8132 | +0.0000 |
| CAR5 | T3 | +0.3283 | -15.4115 | +0.0622 |
| CAR5 | T4 | +0.8199 | -15.6714 | +0.0000 |
| CAR5 | T5 | +0.8954 | -16.5278 | +0.0634 |

`ΔNew→Old`는 R-fixed−G-fixed percentage point다. 음수일 때 Actual 교체가 New→Old confusion을 줄였다는 의미인데, 이번 결과에서는 감소가 관찰되지 않았다.

## 6. NCM prediction-change 결과

Final task에서 APR-NCM과 Oracle-NCM 예측이 달라진 sample은 다음과 같다.

| Protocol | 변경 sample | APR 오답→Oracle 정답 | APR 정답→Oracle 오답 |
|---|---:|---:|---:|
| IA5 | 81 | 23 | 24 |
| CAR5 | 417 | 192 | 52 |

CAR5에서는 oracle prototype이 APR-NCM보다 개선을 만들었지만, Oracle-NCM의 절대 정확도는 Full Oracle Gaussian CA보다 16.5278 pp 낮았다. 따라서 작은 prototype 변화가 일부 boundary sample을 바꾸는 현상과 NCM이 CA보다 강하다는 주장은 구분해야 한다.

## 7. Bias 및 calibration 해석

- IA5: Old 손실을 1 pp 이하로 제한했을 때 R-fixed New는 74.9077에서 최대 75.2768로만 증가했다.
- CAR5: 같은 조건에서 R-fixed New는 90.7419에서 증가하지 않았다.
- 전체 class에 동일 temperature를 적용하면 argmax prediction은 변하지 않는다.
- bias sweep은 held-out test에서 계산한 **ORACLE/TEST-SWEEP DIAGNOSTIC**이다.
- 별도 validation에서 beta를 선택하지 않았으므로 sweep상의 최적값을 정식 method 성능으로 해석하면 안 된다.

## 8. 실행 시간 검증

### 8.1 Task별 처리시간

| Protocol | Task | Train feature | Test feature | 처리시간 |
|---|---:|---:|---:|---:|
| IA5 | T2 | 2,766 | 698 | 39.72초 |
| IA5 | T3 | 3,813 | 975 | 48.12초 |
| IA5 | T4 | 4,883 | 1,248 | 57.83초 |
| IA5 | T5 | 5,981 | 1,519 | 69.67초 |
| CAR5 | T2 | 3,300 | 3,265 | 57.27초 |
| CAR5 | T3 | 4,932 | 4,873 | 79.31초 |
| CAR5 | T4 | 6,545 | 6,464 | 103.96초 |
| CAR5 | T5 | 8,144 | 8,041 | 125.69초 |

task manifest의 처리시간 합계는 약 `581.6초`, 즉 `9분 42초`다. 먼저 T5를 실행한 뒤 T2–T4로 확장했으며, 확장 실행에서 완료된 T5는 marker와 cache를 이용해 중복 계산하지 않았다.

### 8.2 빠르게 끝난 이유

이번 진단에는 다음 작업이 포함되지 않았다.

- Structure backbone 재학습
- adapter 재학습
- CIL task sequence 재학습
- IA10/CAR10 실행
- 새로운 feature loss 또는 method sweep

GPU에서 수행한 주요 작업은 각 checkpoint의 frozen forward와 작은 cosine classifier head 학습이다. 최대 head 크기는 약 `200 × 768`이며 G-fixed/R-fixed 각각 5 epoch만 학습하므로 전체 Structure 학습보다 훨씬 빠르다.

## 9. 실행 및 산출물 검증

검증 결과:

- 단위 테스트 9개 통과
- synthetic fixed-bank smoke test 통과
- IA5/CAR5 모두 T2–T5 coverage 확인
- protocol별 필수 CSV 11개 생성
- protocol별 plot 16개 생성
- G/R current-new bank 동일성 확인
- Matched Count와 Full Actual 동일성 확인
- D 재구성 accuracy와 기존 scientific audit accuracy의 정확한 일치 확인
- 실행 전후 checkpoint 크기 및 edge SHA-256 fingerprint 동일
- main checkpoint write 없음

CSV 총 data row 수:

- IA5: 45,506행
- CAR5: 191,759행

## 10. Leakage 방지

- 실제 old feature는 diagnostic memory와 diagnostic output에서만 사용했다.
- actual old feature는 main CIL training, backbone/adapter update, 다음 task training에 전달하지 않았다.
- 정식 Structure checkpoint에 diagnostic head나 actual feature를 저장하지 않았다.
- classifier optimizer에는 diagnostic head parameter만 포함했다.
- epoch별 test metric은 기록만 했으며 epoch 선택에 사용하지 않았다.

## 11. 후속 실험 권고

IA5/CAR5 모든 transition에서 R−G와 Oracle-NCM−D의 방향이 일관됐다. 따라서 IA10/CAR10은 작은 G/R 차이가 긴 task sequence에서 누적되는지를 확인하는 후속 검증으로 수행할 근거가 있다.

다만 현재 결과만으로도 다음은 비교적 명확하다.

- Actual old feature 교체만으로 큰 성능 회복은 발생하지 않았다.
- Oracle-NCM이 CA보다 우수하지 않았다.
- 단순 scalar bias가 주요 해결책은 아니었다.

그러므로 IA10/CAR10은 현재 결론을 내리기 위한 필수 조건이라기보다 sequence-length robustness 검증으로 취급하는 것이 적절하다.

Optional G/R Linear는 실행하지 않았다. Main G/R 비교가 모든 transition에서 작고 같은 방향이어서 old source의 주효과 판단이 모호하지 않았기 때문이다.

## 12. 관련 파일

- 통합 자동 보고서: `classifier_bottleneck_diagnostic_report.md`
- IA5 상세 보고서: `IA5/seed_1993/IA5_classifier_bottleneck_report.md`
- CAR5 상세 보고서: `CAR5/seed_1993/CAR5_classifier_bottleneck_report.md`
- IA5 실행 manifest: `IA5/seed_1993/run_manifest.json`
- CAR5 실행 manifest: `CAR5/seed_1993/run_manifest.json`
- 계획 문서: `../classifier_bottleneck_diagnostic_plan.md`
- 실행 코드: `../run_classifier_diagnostic.py`
- 진단 파이프라인: `../classifier_diagnostic/pipeline.py`

