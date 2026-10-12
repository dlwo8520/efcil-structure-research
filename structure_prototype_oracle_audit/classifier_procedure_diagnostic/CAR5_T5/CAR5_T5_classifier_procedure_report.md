# CAR5 T5 Classifier Procedure Diagnostic

> 상태: **COMPLETE** · seed 1993 · T5 only · checkpoint unchanged: **True**

## 첫 페이지 결론 — YES / NO / INCONCLUSIVE

| 질문 | 판정 | 직접 근거 |
|---|---|---|
| Q1. 기존 G/R-fixed의 Old/New exposure가 불균형했는가? | **YES** | old/class/epoch mean=41.6879, new/class/epoch=256.0000; steps=325 vs 256-paired=980 |
| Q2. G-fixed→G256에서 성능이 회복되는가? | **YES** | G256−G-fixed Seen=5.7207 pp |
| Q3. R-fixed→R256에서 Old/New balance가 회복되는가? | **YES** | R256−R-fixed: Seen=5.3725, Old=8.6943, New=-8.2435; |Old−New|=17.2271→0.2893 |
| Q4. 동일 256 exposure에서 R256이 G256보다 높은가? | **YES** | R256−G256 Seen=0.5472 pp |
| Q5. 현재 5-epoch cosine classifier는 충분히 수렴했는가? | **INCONCLUSIVE** | train CE relative reduction=-13.5063%, validation accuracy delta=1.8758 pp |
| Q6. Converged cosine이 5-epoch cosine보다 높은가? | **NO** | Converged−5ep test Seen=-0.1865 pp |
| Q7. Linear no-bias가 converged cosine보다 높은가? | **NO** | Linear no-bias−Converged Seen=-17.9331 pp |
| Q8. Bias를 허용한 linear가 추가 개선되는가? | **YES** | Linear with-bias−no-bias Seen=0.7710 pp |
| Q9. 현재 evidence가 지시하는 다음 우선순위는 무엇인가? | **YES — sampling/weighting** | sampling/weighting |

## Exposure table

| Condition | Old/class draws | New/class draws | Total draws | Steps | Fixed/Dynamic |
|---|---:|---:|---:|---:|---|
| FULL_ORACLE_GAUSSIAN_CA | 256–256 / epoch | 256–256 / epoch | 250,880 | 980 | dynamic_regenerated_each_epoch |
| G_FIXED | 24–68 / epoch | 256–256 / epoch | 82,645 | 325 | fixed_repeated |
| R_FIXED | 24–68 / epoch | 256–256 / epoch | 82,645 | 325 | fixed_repeated |
| G256_FIXED | 256–256 / epoch | 256–256 / epoch | 250,880 | 980 | fixed_repeated |
| R256_FIXED | 256–256 / epoch | 256–256 / epoch | 250,880 | 980 | fixed_repeated |

## G/R procedure accuracy

| Condition | Seen | Old | New | Group balance | Old-only | New-only | New→Old | Old→New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FULL_ORACLE_GAUSSIAN_CA | 81.8679 | 81.8379 | 81.9911 | 81.9145 | 83.7407 | 92.7077 | 14.5212 | 3.5736 |
| G_FIXED | 75.9980 | 72.4319 | 90.6151 | 81.5235 | 82.7197 | 92.3906 | 2.7901 | 17.5897 |
| R_FIXED | 76.8934 | 73.5149 | 90.7419 | 82.1284 | 83.0600 | 92.3906 | 2.8535 | 16.4604 |
| G256_FIXED | 81.7187 | 81.6522 | 81.9911 | 81.8217 | 83.4158 | 92.5174 | 14.6481 | 3.6046 |
| R256_FIXED | 82.2659 | 82.2092 | 82.4984 | 82.3538 | 84.0192 | 92.7077 | 14.2676 | 3.4499 |

## Actual-feature classifier probe

| Classifier | Seen | Old | New | Old-only | New-only | New→Old | Old→New | Train | Validation | Train CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STRUCTURE_CA | 81.1964 | 80.9870 | 82.0545 | 83.1219 | 92.5808 | 14.6481 | 3.8366 | N/A | N/A | N/A |
| COSINE_5EP | 81.6192 | 81.4975 | 82.1179 | 83.5860 | 92.7711 | 14.7749 | 3.8212 | 98.2681 | 90.6856 | 0.4070 |
| COSINE_CONVERGED | 81.4327 | 81.2655 | 82.1179 | 83.6170 | 92.8979 | 14.1408 | 4.3626 | 96.5761 | 92.5614 | 0.4619 |
| LINEAR_NO_BIAS | 63.4996 | 63.4127 | 63.8554 | 66.4449 | 81.3570 | 27.8377 | 7.9517 | 77.0537 | 65.5239 | 3.2964 |
| LINEAR_WITH_BIAS | 64.2706 | 64.2791 | 64.2359 | 67.2184 | 81.6741 | 27.7743 | 7.6578 | 77.5386 | 65.5239 | 3.2912 |

## Convergence 및 validation

- validation sample: 1,546; validation class: 196; validation 미포함 class: 0
- COSINE_CONVERGED 실행 epoch: 100, 선택 epoch: 1
- stop reason: `max_epoch`
- validation은 convergence 선택에만 사용했고 test는 epoch/regularization 선택에 사용하지 않았다.
- Linear L2는 0.0005로 사전 고정했으며 test 기반 sweep을 하지 않았다.

## 해석

동일 exposure/step 조건에서도 R256이 G256보다 높다. Actual feature가 보존하는 추가 정보가 유효할 가능성이 있으나 이를 multimodality 하나로 단정하지 않는다.
Linear probe 개선은 cosine normalization/fixed scale 또는 class-specific intercept를 포함한 classifier form 제약 가능성을 지지한다.

## Leakage 및 안전성

- frozen T5 feature cache와 기존 checkpoint만 재사용했다.
- backbone, adapter, APR, SCC 및 main checkpoint를 갱신하지 않았다.
- optimizer에는 diagnostic classifier parameter만 포함했다.
- actual feature는 diagnostic-only이며 next task로 전달하지 않았다.
- 실행 전후 checkpoint edge fingerprint와 size가 동일했다.
- T2–T4 및 IA10/CAR10은 실행하지 않았다.
