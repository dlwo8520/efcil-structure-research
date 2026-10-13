# CAR5 Classifier Bottleneck Diagnostic

> 상태: **COMPLETE** · seed 1993 · completed tasks `[2, 3, 4, 5]` · main checkpoint modified: **NO**

## 맨 앞 결론 — YES / NO / INCONCLUSIVE

| 질문 | 답 | 직접 근거 |
|---|---|---|
| Q1. Oracle NCM > Full Oracle Gaussian CA? | **NO** | Oracle NCM − Full Oracle Gaussian CA Seen = -16.5278 pp |
| Q2. Actual old feature CA > matched Gaussian CA? | **YES** | R-fixed − G-fixed Seen = 0.8954 pp |
| Q3. Gaussian→Actual이 New→Old를 줄이는가? | **INCONCLUSIVE** | R-fixed − G-fixed New→Old = 0.0634 pp |
| Q4. Actual CA에도 New-only gap이 크게 남는가? | **NO** | R-fixed New-only − New = 1.6487 pp |
| Q5. 단일 bias로 Old를 크게 잃지 않고 New를 회복하는가? | **NO** | test-sweep diagnostic에서 Old 손실≤1 pp 조건의 New: 90.7419 → 90.7419 |
| Q6. A/D/R이 서로 다른 bias frontier를 만드는가? | **YES** | 공통 Old 구간에서 bias trade-off curve 최대 New 차이 = 65.0602 pp |
| Q7. CA 후 weight가 class mean에서 멀어지는가? | **NO** | old mean alignment pre=0.7478, Structure=0.7486, D=0.7504, R=0.7332 |
| Q8. G/R 차이가 optimization/fitting 차이일 가능성이 있는가? | **INCONCLUSIVE** | final fixed-bank train accuracy R−G = -0.3993 pp; 이 진단만으로 optimization 인과성은 확정하지 않음 |

## Final-task 절대 정확도

| Condition | Old source | Seen | Old | New | Old-only | New-only | New→Old |
|---|---|---:|---:|---:|---:|---:|---:|
| Structure CA | Structure Gaussian regenerated per epoch | 81.1964 | 80.9870 | 82.0545 | 83.1219 | 92.5808 | 14.6481 |
| Oracle Mean CA | oracle mean + Structure covariance | 81.5073 | 81.4356 | 81.8009 | 83.4004 | 92.5808 | 14.7749 |
| Full Oracle Gaussian CA | oracle Gaussian regenerated per epoch | 81.8679 | 81.8379 | 81.9911 | 83.7407 | 92.7077 | 14.5212 |
| Fixed Oracle Gaussian CA | fixed matched-count oracle Gaussian | 75.9980 | 72.4319 | 90.6151 | 82.7197 | 92.3906 | 2.7901 |
| Actual Feature CA | fixed matched/full actual old features | 76.8934 | 73.5149 | 90.7419 | 83.0600 | 92.3906 | 2.8535 |
| APR-NCM | APR mean | 63.5991 | 63.5056 | 63.9822 | 66.5223 | 81.2936 | 27.5206 |
| Oracle-NCM | oracle train mean | 65.3401 | 65.5167 | 64.6164 | 68.3478 | 81.2936 | 26.6963 |

## G-fixed 대비 R-fixed delta

| Metric | R − G |
|---|---:|
| Seen | 0.8954 |
| Old | 1.0829 |
| New | 0.1268 |
| Old-only | 0.3403 |
| New-only | 0.0000 |
| New→Old | 0.0634 |
| Old→New | -1.1293 |
| NLL | -0.0136 |
| ECE | 0.0106 |
| New-vs-Old margin | -0.0270 |
| Old-vs-New margin | 0.0355 |
| old weight-mean alignment | 0.0001 |

## Transition progression

| Task | Structure | Full Oracle G | APR-NCM | Oracle-NCM | G-fixed | R-fixed | R−G Seen | ΔNew→Old (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 86.6156 | 86.8913 | 72.4349 | 73.0781 | 84.1960 | 84.5329 | 0.3369 | 0.0000 |
| 3 | 85.1837 | 85.4299 | 69.0129 | 70.0185 | 80.0328 | 80.3612 | 0.3283 | 0.0622 |
| 4 | 82.6887 | 82.7351 | 65.8106 | 67.0637 | 77.7537 | 78.5736 | 0.8199 | 0.0000 |
| 5 | 81.1964 | 81.8679 | 63.5991 | 65.3401 | 75.9980 | 76.8934 | 0.8954 | 0.0634 |

R−G Seen은 전 transition에서 `0.3283~0.8954` pp였고, Oracle-NCM−Full Oracle Gaussian CA는 `-16.5278~-13.8132` pp였다.

## NCM prediction-change diagnostic

Final task에서 APR-NCM과 Oracle-NCM의 예측이 달라진 sample은 **417개**였다. 그중 APR 오답→Oracle 정답은 192개, APR 정답→Oracle 오답은 52개였다. sample별 score, margin, prototype shift는 `ncm_prediction_change_analysis.csv`에 저장했다.

## Bias frontier operating-point diagnostic

기준은 R-fixed beta=0의 Old=73.5149, New=90.7419이며, 아래 값은 test sweep을 이용한 진단값일 뿐 선택된 method 성능이 아니다.

| Condition | max New (Old ≥ R-old−1pp) | max Old (New ≥ R-new−1pp) |
|---|---:|---:|
| Structure CA | 91.2492 | 76.6089 |
| Full Oracle Gaussian CA | 91.3760 | 76.6553 |
| Actual Feature CA | 90.7419 | 74.5050 |

## Matched Count / Full Actual 확인

모든 transition에서 actual old class count가 class당 24~68개로 configured cap보다 작아 **Matched Count와 Full Actual이 정확히 같은 bank**였다. 따라서 별도 Full Actual 행은 중복이며 `R_FIXED`가 두 버전을 동시에 나타낸다.

## 해석

Actual-vs-Gaussian 차이는 존재하지만 단일 seed에서 결정적이라고 보기 어렵다. task progression과 paired margin/optimization trace를 함께 해석해야 한다.

Bias sweep은 held-out test에서 그린 **ORACLE/TEST-SWEEP DIAGNOSTIC**이다. 별도 validation으로 beta를 선택하지 않았으므로 최적 beta accuracy를 정식 method result로 제시하지 않는다. 전체 class에 동일 temperature를 적용하면 argmax는 변하지 않는다.

## 구현 및 안전성

- raw frozen `sfeatures`를 사용하고 L2 normalization은 cosine head 내부에서 수행했다.
- G/R current-new source는 동일한 fixed Structure Gaussian bank다.
- R old bank는 actual old training feature이며 main CIL training이나 다음 task에 전달하지 않았다.
- 기존 D accuracy는 재사용했고 bias/logit용 head만 saved RNG로 재구성하여 기존 accuracy와 일치시켰다.
- classifier는 bias가 없고 weight norm은 cosine score에 직접 들어가지 않는다.
- optimizer는 diagnostic head만 포함하고 backbone/adapter/checkpoint는 read-only다.

## 산출물

요청된 11개 CSV와 bias/score/margin/alignment plot은 이 보고서와 같은 protocol output에 저장했다.
