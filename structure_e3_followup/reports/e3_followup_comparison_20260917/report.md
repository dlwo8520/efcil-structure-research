# ImageNet-A 5-task: Structure E3 후속 실험 비교 보고서

작성 기준: 동일 source snapshot에서 완료된 pilot seed 1993, 모든 비교 n=1. 정확도 단위는 %, 차이와 forgetting은 percentage point(pp)다. 저장된 기록만 재집계했으며 추가 학습이나 checkpoint 재평가는 수행하지 않았다.

## 1. 결과 요약

**이 pilot에서 Inc-Acc와 Last-Acc를 함께 가장 높인 방법은 AM-Drift**다. B0 대비 Inc-Acc +0.17 pp, Last-Acc +0.26 pp이며 Forgetting도 -0.87 pp 낮다. 가장 낮은 forgetting은 AM-Variance의 6.79 pp(B0 대비 -1.05 pp)다.

그러나 old/new trade-off는 남는다. AM-Drift의 최종 old accuracy는 B0보다 +0.56 pp 높지만 new accuracy는 -1.11 pp 낮다. FCS PIC와 CEOS는 반대로 new accuracy를 높였지만 old 보존과 forgetting을 크게 악화했다. **n=1이므로 이 순위는 관측된 pilot 결과이지 평균 성능이나 통계적 유의성 결론이 아니다.**

## 2. 실험 목록과 비교 가능성

13개 run은 모두 ImageNet-A 200 classes, 5 tasks×40 classes, seed 1993, single-head CIL이고 5/5 tasks를 완료했다. train 5,981장, test 1,519장, class order와 split content SHA-256이 완전히 일치한다. 모든 run의 source snapshot manifest SHA-256과 base working-tree digest도 동일하며, feature 학습 및 classifier retraining의 epoch·processed samples·optimizer steps·LR을 대조했다. CEOS도 class당 총 replay 256개를 유지한다.

| 방법 | 실제 변경 | 상태/task | seed/n | sep λ | CEOS 비율 | adaptive |
| --- | --- | --- | --- | --- | --- | --- |
| Structure | Structure baseline | complete / 5 | 1993 / 1 | 0.0 | 0.0 | none |
| E3 λ=.1 | diagonal separation, λ=.1 | complete / 5 | 1993 / 1 | 0.1 | 0.0 | none |
| E3 λ=.3 | diagonal separation, λ=.3 | complete / 5 | 1993 / 1 | 0.3 | 0.0 | none |
| E3 λ=.5 | diagonal separation, λ=.5 | complete / 5 | 1993 / 1 | 0.5 | 0.0 | none |
| E3 λ=1.0 | diagonal separation, λ=1.0 | complete / 5 | 1993 / 1 | 1.0 | 0.0 | none |
| FCS PIC | adapted point-prototype contrastive | complete / 5 | 1993 / 1 | 0.0 | 0.0 | none |
| E3-Norm | normalized diagonal approximation | complete / 5 | 1993 / 1 | 0.3 | 0.0 | none |
| E3 + AHN | λ=.3 + adapted angular hard negative | complete / 5 | 1993 / 1 | 0.3 | 0.0 | none |
| CEOS50 | 50% Gaussian + 50% adapted CEOS | complete / 5 | 1993 / 1 | 0.0 | 0.5 | none |
| CEOS100 | 100% adapted CEOS for old classes | complete / 5 | 1993 / 1 | 0.0 | 1.0 | none |
| AM-Variance | λ=.3 + variance-aware tau | complete / 5 | 1993 / 1 | 0.3 | 0.0 | variance |
| AM-Age | λ=.3 + age-aware tau | complete / 5 | 1993 / 1 | 0.3 | 0.0 | age |
| AM-Drift | λ=.3 + drift-aware tau | complete / 5 | 1993 / 1 | 0.3 | 0.0 | drift |

모든 방법은 Multi-Center Off이고 K-means를 사용하지 않는다. 따라서 이전 E0–E7 보고서의 K-means 분할 분석은 이번 비교에 **미적용**이다. `B1_E3`는 L10과 동일 설정이라 중복 실행하지 않았다. 계획된 seed 1996·1997은 아직 없어 표본 SD와 paired Δ 분산은 N/A다.

## 3. 전체 성능

| 방법 | n | Inc | ΔInc | Last | ΔLast | F↓ | ΔF↓ | Old | ΔOld | New | ΔNew |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Structure | 1 | 74.60 | 0.00 | 66.43 | 0.00 | 7.84 | 0.00 | 66.99 | 0.00 | 63.84 | 0.00 |
| E3 λ=.1 | 1 | 74.69 | +0.09 | 66.36 | -0.07 | 7.22 | -0.62 | 67.31 | +0.32 | 61.99 | -1.85 |
| E3 λ=.3 | 1 | 74.55 | -0.05 | 66.56 | +0.13 | 7.04 | -0.80 | 67.39 | +0.40 | 62.73 | -1.11 |
| E3 λ=.5 | 1 | 74.55 | -0.04 | 66.36 | -0.07 | 7.08 | -0.76 | 67.31 | +0.32 | 61.99 | -1.85 |
| E3 λ=1.0 | 1 | 74.53 | -0.07 | 66.49 | +0.07 | 7.53 | -0.31 | 67.39 | +0.40 | 62.36 | -1.48 |
| FCS PIC | 1 | 74.17 | -0.43 | 65.11 | -1.32 | 12.16 | +4.32 | 64.74 | -2.24 | 66.79 | +2.95 |
| E3-Norm | 1 | 74.58 | -0.02 | 66.56 | +0.13 | 7.61 | -0.23 | 67.23 | +0.24 | 63.47 | -0.37 |
| E3 + AHN | 1 | 74.44 | -0.15 | 66.23 | -0.20 | 7.69 | -0.16 | 66.91 | -0.08 | 63.10 | -0.74 |
| CEOS50 | 1 | 74.04 | -0.56 | 66.16 | -0.26 | 11.56 | +3.71 | 65.54 | -1.44 | 69.00 | +5.17 |
| CEOS100 | 1 | 71.32 | -3.28 | 61.29 | -5.13 | 21.24 | +13.39 | 58.25 | -8.73 | 75.28 | +11.44 |
| AM-Variance | 1 | 74.41 | -0.19 | 66.56 | +0.13 | 6.79 | -1.05 | 67.55 | +0.56 | 61.99 | -1.85 |
| AM-Age | 1 | 74.42 | -0.18 | 66.49 | +0.07 | 7.00 | -0.85 | 67.47 | +0.48 | 61.99 | -1.85 |
| AM-Drift | 1 | 74.77 | +0.17 | 66.69 | +0.26 | 6.97 | -0.87 | 67.55 | +0.56 | 62.73 | -1.11 |

ΔInc/ΔLast/ΔOld/ΔNew는 양수가 관측 개선이고, ΔF는 음수가 개선이다. Last-Acc의 0.06583 pp는 최종 test set에서 정답 1개 차이에 해당하므로 작은 차이를 과해석하지 않는다. AM-Drift가 세 headline 지표를 동시에 개선했지만 최종 new는 감소했다.

![performance_deltas](figures/performance_deltas.png)

AM-Drift만 Inc-Acc와 Last-Acc가 모두 B0보다 높다. L01은 Inc만, L03·E3-Norm·AM-Variance·AM-Age는 Last만 높다. 초록/빨강은 관측 부호이며 유의성을 뜻하지 않는다.  
[SVG 원본](figures/performance_deltas.svg)

![accuracy_by_task](figures/accuracy_by_task.png)

T1은 모든 방법이 동일하다. CEOS100은 T2부터 하락이 누적되고, AM-Drift는 T3–T5에서 B0보다 일관되게 높다. 선 간 작은 간격은 seed 변동이 없어 안정성을 판단할 수 없다.  
[SVG 원본](figures/accuracy_by_task.svg)

![accuracy_delta_by_task](figures/accuracy_delta_by_task.png)

0선 위가 같은 seed B0 대비 개선이다. FCS PIC는 T2–T3에서 소폭 높다가 T4–T5에 하락하고, CEOS 계열은 새 task 향상과 별개로 seen 전체 정확도가 낮다.  
[SVG 원본](figures/accuracy_delta_by_task.svg)

## 4. 정확도 행렬 heatmap

![B0_absolute](figures/B0_absolute.png)

행은 학습 완료 task, 열은 평가 task다. 모든 값은 해당 시점의 모든 seen-class logits를 사용하는 single-head 정확도이며 미래 task 셀은 비워 두었다.  
[SVG 원본](figures/B0_absolute.svg)

![heatmap_A_overview](figures/heatmap_A_overview.png)

파란 대각선은 각 새 task를 막 학습한 절대 정확도이고 아래 삼각형은 B0 paired Δ다. CEOS·PIC의 대각선 개선과 과거 task 저하가 동시에 보여 plasticity/stability 교환을 확인할 수 있다.  
[SVG 원본](figures/heatmap_A_overview.svg)

![heatmap_B_overview](figures/heatmap_B_overview.png)

대각선을 포함한 모든 유효 셀이 B0 paired Δ다. 모든 패널은 같은 대칭 색 범위를 사용하며 초록은 관측 개선, 빨강은 관측 저하다. 색은 통계적 유의성을 뜻하지 않는다.  
[SVG 원본](figures/heatmap_B_overview.svg)

### E3 λ=.1

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_L01.png) | ![](figures/heatmap_B_L01.png) |

전체 유효 셀 최대 변화는 T5→T1의 +1.67 pp, 최소 변화는 T5→T5의 -1.85 pp다. [A SVG](figures/heatmap_A_L01.svg) · [B SVG](figures/heatmap_B_L01.svg)

### E3 λ=.3

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_L03.png) | ![](figures/heatmap_B_L03.png) |

전체 유효 셀 최대 변화는 T5→T2의 +1.78 pp, 최소 변화는 T2→T2의 -1.78 pp다. [A SVG](figures/heatmap_A_L03.svg) · [B SVG](figures/heatmap_B_L03.svg)

### E3 λ=.5

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_L05.png) | ![](figures/heatmap_B_L05.png) |

전체 유효 셀 최대 변화는 T4→T1의 +2.78 pp, 최소 변화는 T5→T5의 -1.85 pp다. [A SVG](figures/heatmap_A_L05.svg) · [B SVG](figures/heatmap_B_L05.svg)

### E3 λ=1.0

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_L10.png) | ![](figures/heatmap_B_L10.png) |

전체 유효 셀 최대 변화는 T5→T1의 +2.22 pp, 최소 변화는 T2→T2의 -2.37 pp다. [A SVG](figures/heatmap_A_L10.svg) · [B SVG](figures/heatmap_B_L10.svg)

### FCS PIC

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_FCS_PIC.png) | ![](figures/heatmap_B_FCS_PIC.png) |

전체 유효 셀 최대 변화는 T4→T4의 +4.03 pp, 최소 변화는 T5→T1의 -6.11 pp다. [A SVG](figures/heatmap_A_FCS_PIC.svg) · [B SVG](figures/heatmap_B_FCS_PIC.svg)

### E3-Norm

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_E3_NORM.png) | ![](figures/heatmap_B_E3_NORM.png) |

전체 유효 셀 최대 변화는 T5→T1의 +0.56 pp, 최소 변화는 T3→T2의 -0.89 pp다. [A SVG](figures/heatmap_A_E3_NORM.svg) · [B SVG](figures/heatmap_B_E3_NORM.svg)

### E3 + AHN

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_E3_AHN.png) | ![](figures/heatmap_B_E3_AHN.png) |

전체 유효 셀 최대 변화는 T5→T1의 +1.11 pp, 최소 변화는 T5→T4의 -1.47 pp다. [A SVG](figures/heatmap_A_E3_AHN.svg) · [B SVG](figures/heatmap_B_E3_AHN.svg)

### CEOS50

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_CEOS50.png) | ![](figures/heatmap_B_CEOS50.png) |

전체 유효 셀 최대 변화는 T5→T5의 +5.17 pp, 최소 변화는 T2→T1의 -4.17 pp다. [A SVG](figures/heatmap_A_CEOS50.svg) · [B SVG](figures/heatmap_B_CEOS50.svg)

### CEOS100

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_CEOS100.png) | ![](figures/heatmap_B_CEOS100.png) |

전체 유효 셀 최대 변화는 T5→T5의 +11.44 pp, 최소 변화는 T5→T1의 -14.17 pp다. [A SVG](figures/heatmap_A_CEOS100.svg) · [B SVG](figures/heatmap_B_CEOS100.svg)

### AM-Variance

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_AM_VARIANCE.png) | ![](figures/heatmap_B_AM_VARIANCE.png) |

전체 유효 셀 최대 변화는 T5→T2의 +1.78 pp, 최소 변화는 T5→T5의 -1.85 pp다. [A SVG](figures/heatmap_A_AM_VARIANCE.svg) · [B SVG](figures/heatmap_B_AM_VARIANCE.svg)

### AM-Age

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_AM_AGE.png) | ![](figures/heatmap_B_AM_AGE.png) |

전체 유효 셀 최대 변화는 T5→T1의 +1.67 pp, 최소 변화는 T2→T2의 -2.07 pp다. [A SVG](figures/heatmap_A_AM_AGE.svg) · [B SVG](figures/heatmap_B_AM_AGE.svg)

### AM-Drift

| 혼합 heatmap | 전체 paired Δ heatmap |
|---|---|
| ![](figures/heatmap_A_AM_DRIFT.png) | ![](figures/heatmap_B_AM_DRIFT.png) |

전체 유효 셀 최대 변화는 T4→T1의 +1.39 pp, 최소 변화는 T2→T2의 -1.18 pp다. [A SVG](figures/heatmap_A_AM_DRIFT.svg) · [B SVG](figures/heatmap_B_AM_DRIFT.svg)

## 5. Old/new와 forgetting

![old_new_accuracy](figures/old_new_accuracy.png)

PIC와 CEOS는 current-task accuracy를 높이는 대신 old accuracy를 낮춘다. diagonal E3 및 adaptive 계열은 대체로 old를 소폭 높이고 final new를 낮춘다. 이는 방법별 주효과 방향이 서로 반대임을 보여 주지만 원인을 확정하지는 않는다.  
[SVG 원본](figures/old_new_accuracy.svg)

![final_task_accuracy](figures/final_task_accuracy.png)

최종 모델의 평가 task별 정확도다. 오른쪽 다섯 task Δ의 단순 평균은 표본 수가 서로 달라 ΔLast-Acc와 같지 않으며, Last는 correct/total로 가중 집계했다.  
[SVG 원본](figures/final_task_accuracy.svg)

![forgetting_by_task](figures/forgetting_by_task.png)

AM-Variance의 평균 forgetting이 가장 낮고 AM-Drift·AM-Age·L03가 뒤를 잇는다. CEOS100은 과거 task 전체에서 forgetting이 크게 증가한다. forgetting은 낮을수록 좋고 음수 값을 임의로 0으로 자르지 않았다.  
[SVG 원본](figures/forgetting_by_task.svg)

## 6. Lambda sweep

| λ | ΔInc | ΔLast | ΔF↓ | ΔOld | ΔNew | T5 norm | T5 intrusion |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | +0.09 | -0.07 | -0.62 | +0.32 | -1.85 | 57.01 | 0.802 |
| 0.3 | -0.05 | +0.13 | -0.80 | +0.40 | -1.11 | 72.49 | 0.627 |
| 0.5 | -0.04 | -0.07 | -0.76 | +0.32 | -1.85 | 79.20 | 0.449 |
| 1.0 | -0.07 | +0.07 | -0.31 | +0.40 | -1.48 | 83.56 | 0.353 |

![lambda_inc_acc](figures/lambda_inc_acc.png)

Inc-Acc는 λ=.1에서만 B0보다 높고 이후 비단조적이다.  
[SVG 원본](figures/lambda_inc_acc.svg)

![lambda_last_acc](figures/lambda_last_acc.png)

Last-Acc는 λ=.3과 1.0에서 소폭 높지만 정답 1–2개 규모다.  
[SVG 원본](figures/lambda_last_acc.svg)

![lambda_old_new_delta](figures/lambda_old_new_delta.png)

모든 λ에서 final old는 증가하고 new는 감소해 E3 방향의 trade-off가 반복된다.  
[SVG 원본](figures/lambda_old_new_delta.svg)

![lambda_feature_norm](figures/lambda_feature_norm.png)

λ가 커질수록 T5 feature norm이 57.01→83.56으로 단조 증가한다.  
[SVG 원본](figures/lambda_feature_norm.svg)

![lambda_intrusion](figures/lambda_intrusion.png)

λ가 커질수록 T5 intrusion이 0.802→0.353으로 감소한다. norm 증가와 동시에 일어나 인과 분리가 필요하다.  
[SVG 원본](figures/lambda_intrusion.svg)

낮은 λ가 전체 trade-off를 자동으로 해결하지는 않았다. L01은 Inc가 +0.09 pp지만 final new가 −1.85 pp이며, L03은 Last와 forgetting이 더 좋지만 Inc는 −0.05 pp다. 따라서 이 seed에서 λ=.3을 후속 비교의 기준으로 사용한 것은 합리적이나 ‘best’로 확정할 근거는 부족하다.

## 7. PIC · E3-Norm · AHN과 geometry/gradient

![nearest_old_cosine](figures/nearest_old_cosine.png)

선은 epoch 마지막 batch 집계가 아니라 저장된 epoch 전체 요약의 mean이고 음영은 p10–p90이다. PIC는 nearest-old cosine을 낮추지 않았음에도 new accuracy를 높였고 old accuracy는 낮아졌다.  
[SVG 원본](figures/nearest_old_cosine.svg)

![cosine_gap](figures/cosine_gap.png)

correct-current prototype과 nearest-old prototype의 cosine gap은 task가 진행될수록 모든 방법에서 줄어든다. PIC의 gap은 비교 방법보다 크지만 이것만으로 old 보존 성능을 설명하지 못한다.  
[SVG 원본](figures/cosine_gap.svg)

![rho_distribution](figures/rho_distribution.png)

E3-Norm은 모든 task에서 rho가 1보다 훨씬 작고 intrusion이 1.0이어서 근사 normalized covariance와 birth tau가 제대로 보정되지 않았다. 성능은 B0에 가깝지만 이 ablation을 잘 보정된 normalized E3로 해석하면 안 된다.  
[SVG 원본](figures/rho_distribution.svg)

![intrusion_by_task](figures/intrusion_by_task.png)

raw E3는 λ가 커질수록 intrusion이 줄지만 feature norm은 증가한다. AM-Age는 커진 tau 때문에 T5 intrusion이 1.0에 도달하고, AM-Drift는 tau를 완화해 낮은 intrusion과 비교적 작은 norm을 보인다.  
[SVG 원본](figures/intrusion_by_task.svg)

![gradient_norms](figures/gradient_norms.png)

실선은 해당 old-new component, 점선은 classification gradient norm이다. 서로 목적함수 스케일이 다르므로 norm 크기만으로 우열을 판단하지 않는다. 기록은 configurable interval batch의 평균이다.  
[SVG 원본](figures/gradient_norms.svg)

![gradient_cosine](figures/gradient_cosine.png)

diagonal separation은 classification과 평균적으로 약한 음의 cosine을 보여 충돌 신호가 있다. PIC와 AHN은 양의 cosine이지만 최종 성능 개선으로 이어지지 않아 gradient 정렬만으로 충분하지 않다.  
[SVG 원본](figures/gradient_cosine.svg)

FCS PIC는 ΔInc -0.43 pp, ΔLast -1.32 pp, ΔF +4.32 pp다. final new는 +2.95 pp지만 old는 -2.24 pp다. E3+AHN은 L03보다 Inc/Last가 낮고 T5 norm도 71.79로 높아 현재 설정에서는 norm 안정화 가설을 지지하지 않는다. E3-Norm의 T5 norm은 46.43으로 B0 45.67에 가깝지만 boundary calibration 실패 신호가 있다.

## 8. CEOS boundary replay

| 방법 | ΔInc | ΔLast | ΔF↓ | ΔOld | ΔNew | old 예측률 | 평균 logit gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CEOS50 | -0.56 | -0.26 | +3.71 | -1.44 | +5.17 | 1.000 | 8.89 |
| CEOS100 | -3.28 | -5.13 | +13.39 | -8.73 | +11.44 | 1.000 | 7.83 |

![ceos_logit_gap](figures/ceos_logit_gap.png)

CEOS sample은 생성 시점 classifier에서 source old로 100% 예측되고 old−enemy logit gap도 약 8 수준이다. 즉 기록상 decision boundary에 가까운 hard sample이라기보다 old 쪽에 강하게 남아 있다.  
[SVG 원본](figures/ceos_logit_gap.svg)

![ceos_boundary_proximity](figures/ceos_boundary_proximity.png)

gap이 모두 양수여서 이 실행에서는 |gap|과 raw gap의 평균이 같다. CEOS100의 gap이 CEOS50보다 약간 작지만 여전히 0에서 멀고, old test 보존은 오히려 크게 악화했다.  
[SVG 원본](figures/ceos_boundary_proximity.svg)

CEOS50/100 모두 class당 총 replay 256개와 5 CA epochs를 유지했으며 dominance constraint는 기록상 100%다. 그럼에도 CEOS100은 final new +11.44 pp, old −8.73 pp, forgetting +13.39 pp로 심한 plasticity 편향을 보였다. 현재 adapted CEOS는 ‘boundary-aware old 보존’ 가설을 지지하지 않으며, endpoint normalization·interpolation과 Structure classifier retraining의 결합을 다시 설계해야 한다.

## 9. Adaptive threshold

| 방법 | T5 scale 평균[min,max] | ΔInc | ΔLast | ΔF↓ | ΔOld | ΔNew |
| --- | --- | --- | --- | --- | --- | --- |
| AM-Variance | 0.997 [0.849, 1.139] | -0.19 | +0.13 | -1.05 | +0.56 | -1.85 |
| AM-Age | 1.625 [1.250, 2.000] | -0.18 | +0.07 | -0.85 | +0.48 | -1.85 |
| AM-Drift | 0.802 [0.789, 0.811] | +0.17 | +0.26 | -0.87 | +0.56 | -1.11 |

![adaptive_tau_distribution](figures/adaptive_tau_distribution.png)

AM-Age는 시간이 지날수록 tau를 크게 늘리고, AM-Drift는 약 0.8배로 완화한다. Variance rule은 평균 scale이 거의 1이라 주로 class 간 재배분 역할을 한다.  
[SVG 원본](figures/adaptive_tau_distribution.svg)

![adaptive_tau_intrusion](figures/adaptive_tau_intrusion.png)

T5 class별 tau와 intrusion의 기술적 산점도다. 같은 방법 안에서도 class별 분산이 크며, 이 상관을 tau의 인과 효과나 class 정확도 개선으로 해석하지 않는다.  
[SVG 원본](figures/adaptive_tau_intrusion.svg)

![adaptive_drift_margin](figures/adaptive_drift_margin.png)

AM-Drift는 APR shift가 큰 class의 tau scale을 낮추도록 설계되었고 기록도 그 식을 반영한다. 색은 class intrusion이며, 실제 class accuracy와의 인과 연결은 추가 seed와 class-level paired 분석이 필요하다.  
[SVG 원본](figures/adaptive_drift_margin.svg)

AM-Drift는 이 pilot에서 유일하게 Inc/Last/Forgetting을 동시에 개선하고 feature norm도 L03보다 낮췄다. AM-Variance는 headline 정확도는 낮지만 forgetting이 가장 낮다. AM-Age는 tau 증가와 intrusion 증가가 동반되어 현재 방향·강도는 지지되지 않는다. 세 규칙은 AMGC 재현이 아니라 E3용 독립 extension이다.

## 10. Loss · 수치 안정성 · 자원

![loss_task5](figures/loss_task5.png)

T5 feature stage의 raw/weighted 항을 구분했다. PIC/AHN/E3-Norm은 목적함수 자체가 다르므로 total loss 절대값을 성능 순위로 사용하지 않는다. 모든 epoch 값은 batch-size weighted 평균이다.  
[SVG 원본](figures/loss_task5.svg)

| 방법 | old-new 항 | T5 norm | T5 p95 norm | T5 gap | T5 intrusion | grad cos(cls,old-new) | NaN/비유한 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Structure | none | 45.67 | 55.89 | 0.044 | N/A | N/A | 0/0 |
| E3 λ=.1 | separation | 57.01 | 69.22 | 0.041 | 0.802 | -0.044 | 0/0 |
| E3 λ=.3 | separation | 72.49 | 83.37 | 0.039 | 0.627 | -0.074 | 0/0 |
| E3 λ=.5 | separation | 79.20 | 86.51 | 0.040 | 0.449 | -0.078 | 0/0 |
| E3 λ=1.0 | separation | 83.56 | 88.64 | 0.041 | 0.353 | -0.064 | 0/0 |
| FCS PIC | pic | 42.52 | 51.64 | 0.062 | N/A | 0.655 | 0/0 |
| E3-Norm | separation | 46.43 | 56.74 | 0.045 | 1.000 | 0.045 | 0/0 |
| E3 + AHN | separation+ahn | 71.79 | 82.97 | 0.042 | 0.620 | 0.515 | 0/0 |
| CEOS50 | none | 45.52 | 55.26 | 0.045 | N/A | N/A | 0/0 |
| CEOS100 | none | 45.56 | 55.05 | 0.044 | N/A | N/A | 0/0 |
| AM-Variance | separation | 74.29 | 84.84 | 0.039 | 0.721 | -0.076 | 0/0 |
| AM-Age | separation | 76.77 | 86.10 | 0.040 | 1.000 | -0.083 | 0/0 |
| AM-Drift | separation | 63.30 | 74.79 | 0.041 | 0.363 | -0.058 | 0/0 |

기록된 NaN/Inf flag와 검사한 비유한 수치는 모두 0이다. E3-Norm intrusion 1.0은 수치 폭주가 아니라 normalized covariance/tau calibration 문제이며, CEOS의 성능 저하는 replay sample 수 증가 때문이 아니다. gradient cosine은 설정된 interval batch에서만 측정했으므로 전체 학습의 정확한 적분값은 아니다.

![resource_costs](figures/resource_costs.png)

run wall time은 약 49–55분이고 peak CUDA reserved는 약 10.1 GiB로 비슷하다. Separation 계열은 old region covariance를 추가 저장해 final statistics가 약 451 MiB에서 902 MiB로 증가한다. wall time에는 common geometry/cache 진단이 포함되어 이전 보고서 E0 시간과 직접 비교하지 않는다.  
[SVG 원본](figures/resource_costs.svg)

| 방법 | wall | train 분 | GPU alloc GiB | GPU reserved GiB | statistics MiB | cache MiB |
| --- | --- | --- | --- | --- | --- | --- |
| Structure | 0:49:54 | 48.4 | 5.85 | 10.09 | 451.2 | 0.00 |
| E3 λ=.1 | 0:53:42 | 52.4 | 5.85 | 10.11 | 901.8 | 0.94 |
| E3 λ=.3 | 0:53:02 | 51.8 | 5.85 | 10.11 | 901.8 | 0.94 |
| E3 λ=.5 | 0:52:40 | 51.4 | 5.85 | 10.11 | 901.8 | 0.94 |
| E3 λ=1.0 | 0:54:12 | 52.9 | 5.85 | 10.11 | 901.8 | 0.94 |
| FCS PIC | 0:49:45 | 48.5 | 5.85 | 10.11 | 451.2 | 0.00 |
| E3-Norm | 0:52:06 | 50.8 | 5.85 | 10.11 | 901.8 | 0.94 |
| E3 + AHN | 0:54:41 | 53.1 | 5.85 | 10.11 | 901.8 | 0.94 |
| CEOS50 | 0:48:37 | 47.4 | 5.85 | 10.08 | 451.2 | 0.00 |
| CEOS100 | 0:48:33 | 47.3 | 5.85 | 10.11 | 451.2 | 0.00 |
| AM-Variance | 0:53:06 | 51.8 | 5.85 | 10.11 | 901.8 | 0.94 |
| AM-Age | 0:52:29 | 51.2 | 5.85 | 10.11 | 901.8 | 0.94 |
| AM-Drift | 0:52:36 | 51.3 | 5.85 | 10.11 | 901.8 | 0.94 |

## 11. 연구 질문별 답변

### RQ1

**질문:** E3 제한은 아이디어 자체인가, loss 강도인가?  
**관측 결론:** lambda 효과가 비단조적이며 L01만 Inc-Acc가 소폭 양수. 모든 lambda가 final new accuracy를 낮춤.  
**불확실성:** seed 1개; 강도와 raw feature norm 반응이 함께 바뀜.

### RQ2

**질문:** PIC보다 covariance-aware E3가 유리한가?  
**관측 결론:** 이 seed에서는 L03가 PIC보다 Inc/Last/Forgetting이 모두 높음. PIC는 new를 높이고 old를 낮춤.  
**불확실성:** 구현은 FCS 전체가 아닌 adapted PIC component.

### RQ3

**질문:** E3가 feature norm을 키워 constraint를 회피하는가?  
**관측 결론:** lambda 증가와 함께 norm 증가, intrusion 감소가 선명하게 동반됨.  
**불확실성:** 동반 관찰이며 인과 증명은 아님.

### RQ4

**질문:** AHN이 geometry를 안정화하는가?  
**관측 결론:** 현재 AHN은 L03/B0보다 성능이 낮고 norm inflation도 유지되어 지지되지 않음.  
**불확실성:** lambda_ahn 및 margin sweep 없음.

### RQ5

**질문:** CEOS가 plasticity를 더 잘 보존하는가?  
**관측 결론:** new accuracy는 증가했으나 old/forgetting 손실이 더 커져 전체 성능은 저하.  
**불확실성:** Structure용 adapted CEOS이며 sample이 기록상 boundary에 가깝지 않음.

### RQ6

**질문:** adaptive tau가 trade-off를 개선하는가?  
**관측 결론:** AM-Drift만 Inc/Last/Forgetting을 동시에 관측 개선; variance는 forgetting 최저.  
**불확실성:** seed 1개이며 rule coefficient sweep 없음.

## 12. 한계와 다음 실행

가장 큰 한계는 seed 1993 하나뿐이라는 점이다. 현재 차이 상당수는 최종 정답 1–몇 개 규모여서 seed 1996·1997의 같은-seed paired Δ 없이 일반화할 수 없다. 다음 3-seed 우선순위는 AM-Drift, L03, E3-Norm, B0이며, CEOS100은 현 설정 그대로 확대하기보다 boundary proximity가 실제로 작아지도록 sampler를 수정한 뒤 별도 ablation하는 편이 타당하다. AHN은 λ_ahn/margin sweep 전에는 기각이 아니라 ‘현재 설정에서 미지지’로 남긴다.

권장 순서: (1) B0/L03/AM-Drift/E3-Norm을 seed 1996·1997로 확인, (2) 공통 seed mean±sample SD 및 paired Δ SD 계산, (3) E3-Norm tau를 unit-sphere covariance에 맞게 다시 보정, (4) CEOS logit gap이 0 근처가 되도록 boundary sampler를 재검토, (5) 필요할 때 PIC/AHN coefficient sweep을 수행한다.

## 13. 부록: 정의·원자료·재생성

`a(t,i)`는 task t 학습과 classifier retraining을 완료한 모델을 task i test samples에 평가한 single-head 정확도다. `A(t)=100×Σ correct/Σ samples`, `Inc-Acc=mean_t A(t)`, `Last-Acc=A(T)`, `F_i=max_{u=i..T−1}a(u,i)−a(T,i)`, `Forgetting=mean_{i<T}F_i`다. paired Δ는 같은 seed·같은 셀에서 method−B0로 계산했고 forgetting은 낮을수록 좋다.

| 자료 | 파일 |
| --- | --- |
| seed별 핵심 성능·자원 | [metrics_per_seed.csv](tables/metrics_per_seed.csv) |
| task별 A/old/new | [task_metrics_recomputed.csv](tables/task_metrics_recomputed.csv) |
| 정확도 행렬 cell | [cells_per_seed_and_sd.csv](tables/cells_per_seed_and_sd.csv) |
| task별 forgetting | [forgetting_per_task.csv](tables/forgetting_per_task.csv) |
| class별 최종 paired Δ | [class_final_deltas.csv](tables/class_final_deltas.csv) |
| lambda sweep | [lambda_sweep.csv](tables/lambda_sweep.csv) |
| gradient·수치 진단 | [numerical_diagnostics.csv](tables/numerical_diagnostics.csv) |
| CEOS 진단 | [ceos_diagnostics.csv](tables/ceos_diagnostics.csv) |
| adaptive 요약 | [adaptive_summary.csv](tables/adaptive_summary.csv) |
| 연구 질문 요약 | [research_questions.csv](tables/research_questions.csv) |
| 검산 | [validation_checks.csv](tables/validation_checks.csv) |
| 출처·해시·정의 | [metadata.json](metadata.json) |

각 방법의 절대/차이 행렬은 `tables/<METHOD>_absolute_matrix.csv`와 `tables/<METHOD>_delta_B0_matrix.csv`에 있다. raw task/class/epoch/gradient/adaptive/CEOS CSV도 `tables/raw_*`로 합쳤다. 대용량 CEOS per-sample CSV는 원본 run에 보존하고 보고서에는 저장된 epoch quantile summary만 사용했다.

재생성 명령:

```bash
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
python ${RESEARCH_WORKSPACE}/structure_e3_followup/reports/e3_followup_comparison_20260917/build_report.py \
  --project-root ${RESEARCH_WORKSPACE}/structure_e3_followup \
  --output-dir ${RESEARCH_WORKSPACE}/structure_e3_followup/reports/e3_followup_comparison_20260917
```

재생성에는 numpy와 matplotlib만 필요하며 GPU를 사용하지 않는다.
