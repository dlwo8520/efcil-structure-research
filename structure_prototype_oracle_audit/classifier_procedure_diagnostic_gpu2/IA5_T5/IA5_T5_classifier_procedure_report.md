# IA5 T5 Classifier Procedure Diagnostic

> 상태: **COMPLETE** · seed 1993 · T5 only · checkpoint unchanged: **True**

## 첫 페이지 결론 — YES / NO / INCONCLUSIVE

| 질문 | 판정 | 직접 근거 |
|---|---|---|
| Q1. 기존 G/R-fixed의 Old/New exposure가 불균형했는가? | **YES** | old/class/epoch mean=30.5188, new/class/epoch=256.0000; steps=300 vs 256-paired=1000 |
| Q2. G-fixed→G256에서 성능이 회복되는가? | **YES** | G256−G-fixed Seen=3.8183 pp |
| Q3. R-fixed→R256에서 Old/New balance가 회복되는가? | **YES** | R256−R-fixed: Seen=3.4233, Old=6.8910, New=-12.5461; |Old−New|=14.2507→5.1865 |
| Q4. 동일 256 exposure에서 R256이 G256보다 높은가? | **INCONCLUSIVE** | R256−G256 Seen=-0.0658 pp |
| Q5. 현재 5-epoch cosine classifier는 충분히 수렴했는가? | **NO** | 5ep→100ep trajectory train CE reduction=24.3973%, epoch100 grad=0.4045; validation-best epoch=1 |
| Q6. Converged cosine이 5-epoch cosine보다 높은가? | **NO** | Converged−5ep test Seen=-0.9217 pp |
| Q7. Linear no-bias가 converged cosine보다 높은가? | **NO** | Linear no-bias−Converged Seen=-6.4516 pp |
| Q8. Bias를 허용한 linear가 추가 개선되는가? | **YES** | Linear with-bias−no-bias Seen=0.7900 pp |
| Q9. 현재 evidence가 지시하는 다음 우선순위는 무엇인가? | **YES — sampling/weighting** | sampling/weighting |

## Exposure table

| Condition | Old/class draws | New/class draws | Total draws | Steps | Fixed/Dynamic |
|---|---:|---:|---:|---:|---|
| FULL_ORACLE_GAUSSIAN_CA | 256–256 / epoch | 256–256 / epoch | 256,000 | 1,000 | dynamic_regenerated_each_epoch |
| G_FIXED | 2–86 / epoch | 256–256 / epoch | 75,615 | 300 | fixed_repeated |
| R_FIXED | 2–86 / epoch | 256–256 / epoch | 75,615 | 300 | fixed_repeated |
| G256_FIXED | 256–256 / epoch | 256–256 / epoch | 256,000 | 1,000 | fixed_repeated |
| R256_FIXED | 256–256 / epoch | 256–256 / epoch | 256,000 | 1,000 | fixed_repeated |

## G/R procedure accuracy

| Condition | Seen | Old | New | Group balance | Old-only | New-only | New→Old | Old→New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_ORACLE_GAUSSIAN_CA | 66.0961 | 66.9071 | 62.3616 | 64.6343 | 68.9103 | 81.5498 | 33.2103 | 7.6122 |
| G_FIXED | 62.8703 | 60.3365 | 74.5387 | 67.4376 | 69.0705 | 80.8118 | 11.4391 | 22.3558 |
| R_FIXED | 63.1995 | 60.6571 | 74.9077 | 67.7824 | 69.3109 | 80.4428 | 11.4391 | 21.7949 |
| G256_FIXED | 66.6886 | 67.6282 | 62.3616 | 64.9949 | 69.5513 | 81.5498 | 32.8413 | 7.6923 |
| R256_FIXED | 66.6228 | 67.5481 | 62.3616 | 64.9549 | 69.7115 | 81.1808 | 33.2103 | 7.6923 |

## Actual-feature classifier probe

| Classifier | Seen | Old | New | Old-only | New-only | New→Old | Old→New | Train | Validation | Train CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STRUCTURE_CA | 66.4253 | 66.9872 | 63.8376 | 69.1506 | 80.8118 | 31.3653 | 8.5737 | N/A | N/A | N/A |
| COSINE_5EP | 66.4911 | 67.0673 | 63.8376 | 69.4712 | 80.4428 | 28.4133 | 8.0128 | 97.8633 | 83.3333 | 0.3029 |
| COSINE_CONVERGED | 65.5695 | 66.2660 | 62.3616 | 68.6699 | 81.9188 | 32.1033 | 8.4936 | 96.0469 | 87.0968 | 0.3645 |
| LINEAR_NO_BIAS | 59.1178 | 60.6571 | 52.0295 | 62.4199 | 72.3247 | 42.0664 | 7.1314 | 75.5807 | 62.9032 | 3.1832 |
| LINEAR_WITH_BIAS | 59.9078 | 61.3782 | 53.1365 | 63.0609 | 72.3247 | 41.3284 | 7.1314 | 75.7862 | 63.3513 | 3.1778 |

## Convergence 및 validation

- validation sample: 1,116; validation class: 192; validation 미포함 class: 8
- COSINE_CONVERGED 실행 epoch: 100, 선택 epoch: 1
- stop reason: `max_epoch`
- validation은 convergence 선택에만 사용했고 test는 epoch/regularization 선택에 사용하지 않았다.
- Linear L2는 0.0005로 사전 고정했으며 test 기반 sweep을 하지 않았다.

## 해석

동일 exposure/step 조건에서 G256과 R256의 차이는 작다. Gaussian-vs-actual source보다 sampling/weighting 또는 classifier 측 요인이 우선이다.
추가 학습은 train loss를 낮췄지만 held-out Seen 개선으로 이어지지 않았다. 단순 epoch 확대보다 generalization 제약을 우선 고려해야 한다.
Bias는 linear 내부에서 소폭 개선을 만들었지만 linear 자체가 cosine보다 낮다. 따라서 intercept의 국소 효과는 있으나 classifier form이 현재 주병목이라는 증거는 아니다.

## Leakage 및 안전성

- frozen T5 feature cache와 기존 checkpoint만 재사용했다.
- backbone, adapter, APR, SCC 및 main checkpoint를 갱신하지 않았다.
- optimizer에는 diagnostic classifier parameter만 포함했다.
- actual feature는 diagnostic-only이며 next task로 전달하지 않았다.
- 실행 전후 checkpoint edge fingerprint와 size가 동일했다.
- T2–T4 및 IA10/CAR10은 실행하지 않았다.
