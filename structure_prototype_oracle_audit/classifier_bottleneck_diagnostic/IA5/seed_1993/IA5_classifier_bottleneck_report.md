# IA5 Classifier Bottleneck Diagnostic

> 상태: **COMPLETE** · seed 1993 · completed tasks `[2, 3, 4, 5]` · main checkpoint modified: **NO**

## 맨 앞 결론 — YES / NO / INCONCLUSIVE

| 질문 | 답 | 직접 근거 |
|---|---|---|
| Q1. Oracle NCM > Full Oracle Gaussian CA? | **NO** | Oracle NCM − Full Oracle Gaussian CA Seen = -4.5425 pp |
| Q2. Actual old feature CA > matched Gaussian CA? | **YES** | R-fixed − G-fixed Seen = 0.3292 pp |
| Q3. Gaussian→Actual이 New→Old를 줄이는가? | **INCONCLUSIVE** | R-fixed − G-fixed New→Old = 0.0000 pp |
| Q4. Actual CA에도 New-only gap이 크게 남는가? | **YES** | R-fixed New-only − New = 5.5351 pp |
| Q5. 단일 bias로 Old를 크게 잃지 않고 New를 회복하는가? | **NO** | test-sweep diagnostic에서 Old 손실≤1 pp 조건의 New: 74.9077 → 75.2768 |
| Q6. A/D/R이 서로 다른 bias frontier를 만드는가? | **YES** | 공통 Old 구간에서 bias trade-off curve 최대 New 차이 = 34.3173 pp |
| Q7. CA 후 weight가 class mean에서 멀어지는가? | **NO** | old mean alignment pre=0.7303, Structure=0.7338, D=0.7338, R=0.6998 |
| Q8. G/R 차이가 optimization/fitting 차이일 가능성이 있는가? | **INCONCLUSIVE** | final fixed-bank train accuracy R−G = -1.9176 pp; 이 진단만으로 optimization 인과성은 확정하지 않음 |

## Final-task 절대 정확도

| Condition | Old source | Seen | Old | New | Old-only | New-only | New→Old |
|---|---|---:|---:|---:|---:|---:|---:|
| Structure CA | Structure Gaussian regenerated per epoch | 66.4253 | 66.9872 | 63.8376 | 69.1506 | 80.8118 | 31.3653 |
| Oracle Mean CA | oracle mean + Structure covariance | 66.0303 | 66.9872 | 61.6236 | 68.9904 | 80.8118 | 33.5793 |
| Full Oracle Gaussian CA | oracle Gaussian regenerated per epoch | 66.0961 | 66.9071 | 62.3616 | 68.9103 | 81.5498 | 33.2103 |
| Fixed Oracle Gaussian CA | fixed matched-count oracle Gaussian | 62.8703 | 60.3365 | 74.5387 | 69.0705 | 80.8118 | 11.4391 |
| Actual Feature CA | fixed matched/full actual old features | 63.1995 | 60.6571 | 74.9077 | 69.3109 | 80.4428 | 11.4391 |
| APR-NCM | APR mean | 61.6195 | 62.0192 | 59.7786 | 64.8237 | 75.6458 | 33.5793 |
| Oracle-NCM | oracle train mean | 61.5537 | 62.6603 | 56.4576 | 65.1442 | 75.6458 | 38.0074 |

## G-fixed 대비 R-fixed delta

| Metric | R − G |
|---|---:|
| Seen | 0.3292 |
| Old | 0.3205 |
| New | 0.3690 |
| Old-only | 0.2404 |
| New-only | -0.3690 |
| New→Old | 0.0000 |
| Old→New | -0.5609 |
| NLL | -0.0113 |
| ECE | 0.0019 |
| New-vs-Old margin | -0.0629 |
| Old-vs-New margin | 0.0334 |
| old weight-mean alignment | -0.0011 |

## Transition progression

| Task | Structure | Full Oracle G | APR-NCM | Oracle-NCM | G-fixed | R-fixed | R−G Seen | ΔNew→Old (pp) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 79.3696 | 79.0831 | 75.2149 | 75.2149 | 76.2178 | 76.6476 | 0.4298 | 0.5917 |
| 3 | 74.1538 | 74.4615 | 70.4615 | 70.0513 | 71.3846 | 72.0000 | 0.6154 | 0.3610 |
| 4 | 69.1506 | 68.9904 | 64.9840 | 65.3045 | 66.9872 | 68.0288 | 1.0417 | 1.4652 |
| 5 | 66.4253 | 66.0961 | 61.6195 | 61.5537 | 62.8703 | 63.1995 | 0.3292 | 0.0000 |

R−G Seen은 전 transition에서 `0.3292~1.0417` pp였고, Oracle-NCM−Full Oracle Gaussian CA는 `-4.5425~-3.6859` pp였다.

## NCM prediction-change diagnostic

Final task에서 APR-NCM과 Oracle-NCM의 예측이 달라진 sample은 **81개**였다. 그중 APR 오답→Oracle 정답은 23개, APR 정답→Oracle 오답은 24개였다. sample별 score, margin, prototype shift는 `ncm_prediction_change_analysis.csv`에 저장했다.

## Bias frontier operating-point diagnostic

기준은 R-fixed beta=0의 Old=60.6571, New=74.9077이며, 아래 값은 test sweep을 이용한 진단값일 뿐 선택된 method 성능이 아니다.

| Condition | max New (Old ≥ R-old−1pp) | max Old (New ≥ R-new−1pp) |
|---|---:|---:|
| Structure CA | 74.9077 | 60.8173 |
| Full Oracle Gaussian CA | 75.6458 | 61.6987 |
| Actual Feature CA | 75.2768 | 62.1795 |

## Matched Count / Full Actual 확인

모든 transition에서 actual old class count가 class당 2~86개로 configured cap보다 작아 **Matched Count와 Full Actual이 정확히 같은 bank**였다. 따라서 별도 Full Actual 행은 중복이며 `R_FIXED`가 두 버전을 동시에 나타낸다.

## 해석

Actual feature source로 교체해도 fixed Gaussian 대비 큰 회복이 없다. 이는 old source 교체만으로 부족하다는 뜻이며 representation 원인을 직접 증명하지 않는다. CA objective, optimization, classifier form과 score overlap이 남는다.
R-fixed에서도 new-only gap이 크게 남아 actual old feature 사용만으로 old-new competition이 해소되지 않는다.

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
