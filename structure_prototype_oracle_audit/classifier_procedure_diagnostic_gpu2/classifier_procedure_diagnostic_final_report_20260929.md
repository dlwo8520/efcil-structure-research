# Structure Classifier Procedure Diagnostic 최종 보고서

> 작성일: 2026-09-29  
> 실행 GPU: 2  
> 대상: IA5 T5, CAR5 T5 / seed 1993  
> 상태: **COMPLETE**  
> T2–T4, IA10/CAR10: **실행하지 않음**  
> 기존 Structure checkpoint 변경: **없음**

## 1. 첫 페이지 결론

| 질문 | IA5 | CAR5 | 핵심 근거 |
|---|---|---|---|
| Q1. 기존 G/R-fixed의 Old/New exposure가 불균형했는가? | **YES** | **YES** | Old/class/epoch 평균 30.52/41.69 vs New 256; step 300/325 vs vanilla 1,000/980 |
| Q2. G-fixed→G256에서 성능이 회복되는가? | **YES** | **YES** | Seen +3.8183/+5.7207 pp |
| Q3. R-fixed→R256에서 Old/New balance가 회복되는가? | **YES** | **YES** | `|Old−New|` 14.25→5.19 / 17.23→0.29 pp |
| Q4. 동일 256 exposure에서 R256이 G256보다 높은가? | **INCONCLUSIVE** | **YES, 작음** | Seen −0.0658/+0.5472 pp |
| Q5. 현재 5-epoch cosine은 optimization 관점에서 수렴했는가? | **NO** | **NO** | 5→100 epoch train CE 추가 감소 24.40%/19.33%; epoch 100 gradient도 작지 않음 |
| Q6. 추가 cosine fitting이 held-out 성능을 높이는가? | **NO** | **NO** | validation-selected extended cosine의 Seen −0.9217/−0.1865 pp |
| Q7. Linear no-bias가 extended cosine보다 높은가? | **NO** | **NO** | Seen −6.4516/−17.9331 pp |
| Q8. Linear에 bias를 허용하면 추가 개선되는가? | **YES, 작음** | **YES, 작음** | Linear 내부 Seen +0.7900/+0.7710 pp; 여전히 cosine보다 크게 낮음 |
| Q9. 현재 evidence 기준 다음 우선순위는? | **sampling/weighting** | **sampling/weighting** | 256 exposure 복원 효과가 source·추가 fitting·linear form 효과보다 큼 |

## 2. 핵심 해석

가장 큰 효과는 Gaussian을 actual feature로 바꾸는 데서가 아니라 class exposure와 optimizer-step budget을 원래 CA 수준으로 복원할 때 발생했다.

- IA5: G-fixed→G256 `+3.82 pp`, R-fixed→R256 `+3.42 pp`
- CAR5: G-fixed→G256 `+5.72 pp`, R-fixed→R256 `+5.37 pp`
- 동일 exposure에서 R256−G256은 IA5 `−0.07 pp`, CAR5 `+0.55 pp`

따라서 이전 G_FIXED/R_FIXED의 낮은 Seen 성능과 New 쪽 이동은 old class가 new class보다 훨씬 적게 노출된 sampling/weighting 차이에 크게 영향을 받았다. Exposure를 동일하게 맞춘 뒤에는 Gaussian과 actual source의 차이가 작았다.

다만 exposure 256 복원은 모든 그룹 성능을 동시에 높이는 방식이 아니었다. Old는 크게 회복하고 New는 내려가면서 원래 Structure/Full Oracle CA의 operating point로 돌아갔다. 특히 IA5에서는 group-balance 평균이 오히려 감소했다. 그러므로 Seen 회복을 순수한 분류정보 증가로 해석하면 안 된다.

## 3. 원본 CA sampling audit

원본 함수는 `${RESEARCH_WORKSPACE}/structure_multicenter_separation/models/base.py`의 `_stage2_compact_classifier()`다.

- 모든 seen class에서 매 epoch Gaussian 256개 생성
- epoch마다 Gaussian 전체 regenerate
- 전체 sample을 `torch.randperm`으로 섞은 뒤 256개 단위 batch
- DataLoader/sampler 없음
- IA5 T5: epoch당 200 step, 5 epoch 총 1,000 step
- CAR5 T5: epoch당 196 step, 5 epoch 총 980 step
- scheduler: `CosineAnnealingLR(T_max=5)`, epoch마다 step
- classifier: bias 없는 scale-20 cosine classifier

### Exposure 비교

#### IA5 T5

| Condition | Old/class/epoch | New/class/epoch | Total draws | Steps | Lifecycle |
|---|---:|---:|---:|---:|---|
| Full Oracle Gaussian CA | 256 | 256 | 256,000 | 1,000 | Dynamic |
| G_FIXED | 2–86 | 256 | 75,615 | 300 | Fixed |
| R_FIXED | 2–86 | 256 | 75,615 | 300 | Fixed |
| G256_FIXED | 256 | 256 | 256,000 | 1,000 | Fixed |
| R256_FIXED | 256 | 256 | 256,000 | 1,000 | Fixed |

#### CAR5 T5

| Condition | Old/class/epoch | New/class/epoch | Total draws | Steps | Lifecycle |
|---|---:|---:|---:|---:|---|
| Full Oracle Gaussian CA | 256 | 256 | 250,880 | 980 | Dynamic |
| G_FIXED | 24–68 | 256 | 82,645 | 325 | Fixed |
| R_FIXED | 24–68 | 256 | 82,645 | 325 | Fixed |
| G256_FIXED | 256 | 256 | 250,880 | 980 | Fixed |
| R256_FIXED | 256 | 256 | 250,880 | 980 | Fixed |

## 4. G256/R256 paired control

G256_FIXED와 R256_FIXED에서 동일하게 유지한 조건:

- exact pre-CA classifier initialization
- class당 256 exposure
- current/new fixed Structure Gaussian bank
- epoch별 label sequence
- batch boundary와 label order
- SGD, learning rate, momentum, weight decay
- total optimizer step
- epoch-based LR schedule
- evaluation split

유일한 의도된 차이는 old feature source다.

- G256 old: oracle mean/covariance Gaussian 256개/class
- R256 old: actual old feature에서 replacement sampling 256개/class

검증 결과:

| Protocol | Total draws | Steps | Label sequence hash | G/R new bank |
|---|---:|---:|---|---|
| IA5 | 256,000 | 1,000 | `c32679d7939394f15...` | checksum 동일 |
| CAR5 | 250,880 | 980 | `a8327a513ffb71364...` | checksum 동일 |

## 5. G/R procedure accuracy

### IA5 T5

| Condition | Seen | Old | New | Group balance | Old-only | New-only | New→Old | Old→New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Oracle Gaussian CA | 66.0961 | 66.9071 | 62.3616 | 64.6343 | 68.9103 | 81.5498 | 33.2103 | 7.6122 |
| G_FIXED | 62.8703 | 60.3365 | 74.5387 | 67.4376 | 69.0705 | 80.8118 | 11.4391 | 22.3558 |
| R_FIXED | 63.1995 | 60.6571 | 74.9077 | 67.7824 | 69.3109 | 80.4428 | 11.4391 | 21.7949 |
| G256_FIXED | 66.6886 | 67.6282 | 62.3616 | 64.9949 | 69.5513 | 81.5498 | 32.8413 | 7.6923 |
| R256_FIXED | 66.6228 | 67.5481 | 62.3616 | 64.9549 | 69.7115 | 81.1808 | 33.2103 | 7.6923 |

IA5에서 G256과 R256은 사실상 동일하다. R256−G256 Seen은 `−0.0658 pp`다.

### CAR5 T5

| Condition | Seen | Old | New | Group balance | Old-only | New-only | New→Old | Old→New |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Oracle Gaussian CA | 81.8679 | 81.8379 | 81.9911 | 81.9145 | 83.7407 | 92.7077 | 14.5212 | 3.5736 |
| G_FIXED | 75.9980 | 72.4319 | 90.6151 | 81.5235 | 82.7197 | 92.3906 | 2.7901 | 17.5897 |
| R_FIXED | 76.8934 | 73.5149 | 90.7419 | 82.1284 | 83.0600 | 92.3906 | 2.8535 | 16.4604 |
| G256_FIXED | 81.7187 | 81.6522 | 81.9911 | 81.8217 | 83.4158 | 92.5174 | 14.6481 | 3.6046 |
| R256_FIXED | 82.2659 | 82.2092 | 82.4984 | 82.3538 | 84.0192 | 92.7077 | 14.2676 | 3.4499 |

CAR5에서는 R256이 G256보다 `+0.5472 pp` 높다. 차이는 존재하지만 exposure 복원 효과 약 `+5.4~+5.7 pp`보다 훨씬 작다.

## 6. Exposure 복원에 따른 Old/New 이동

| Protocol | 비교 | ΔSeen | ΔOld | ΔNew | `|Old−New|` 변화 | ΔGroup balance |
|---|---|---:|---:|---:|---:|---:|
| IA5 | G256−G_FIXED | +3.8183 | +7.2917 | −12.1771 | 14.20→5.27 | −2.4427 |
| IA5 | R256−R_FIXED | +3.4233 | +6.8910 | −12.5461 | 14.25→5.19 | −2.8275 |
| CAR5 | G256−G_FIXED | +5.7207 | +9.2203 | −8.6240 | 18.18→0.34 | +0.2982 |
| CAR5 | R256−R_FIXED | +5.3725 | +8.6943 | −8.2435 | 17.23→0.29 | +0.2254 |

Seen accuracy에서 old sample의 비중이 크기 때문에 IA5에서도 Seen은 상승하지만, equal-weight group balance는 감소한다. 이는 exposure 복원이 Old/New trade-off의 operating point를 바꾼다는 뜻이며, 두 그룹 모두의 성능을 동시에 개선했다는 뜻은 아니다.

## 7. Actual-feature classifier probe

Old와 current/new actual training feature를 모두 사용하는 diagnostic oracle probe다. 모든 feature는 L2-normalize했고 test는 최종 평가에만 사용했다.

### IA5 T5

| Classifier | Seen | Old | New | Train | Validation | Train CE | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Structure CA | 66.4253 | 66.9872 | 63.8376 | N/A | N/A | N/A | 1.4133 | 0.0703 |
| COSINE_5EP | 66.4911 | 67.0673 | 63.8376 | 97.8633 | 83.3333 | 0.3029 | 1.4501 | 0.0955 |
| COSINE_CONVERGED* | 65.5695 | 66.2660 | 62.3616 | 96.0469 | 87.0968 | 0.3645 | 1.4317 | 0.0771 |
| LINEAR_NO_BIAS | 59.1178 | 60.6571 | 52.0295 | 75.5807 | 62.9032 | 3.1832 | 3.4713 | 0.5419 |
| LINEAR_WITH_BIAS | 59.9078 | 61.3782 | 53.1365 | 75.7862 | 63.3513 | 3.1778 | 3.4665 | 0.5497 |

### CAR5 T5

| Classifier | Seen | Old | New | Train | Validation | Train CE | NLL | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Structure CA | 81.1964 | 80.9870 | 82.0545 | N/A | N/A | N/A | 0.9597 | 0.2607 |
| COSINE_5EP | 81.6192 | 81.4975 | 82.1179 | 98.2681 | 90.6856 | 0.4070 | 0.9640 | 0.2716 |
| COSINE_CONVERGED* | 81.4327 | 81.2655 | 82.1179 | 96.5761 | 92.5614 | 0.4619 | 0.9656 | 0.2691 |
| LINEAR_NO_BIAS | 63.4996 | 63.4127 | 63.8554 | 77.0537 | 65.5239 | 3.2964 | 3.4722 | 0.5935 |
| LINEAR_WITH_BIAS | 64.2706 | 64.2791 | 64.2359 | 77.5386 | 65.5239 | 3.2912 | 3.4676 | 0.6010 |

`COSINE_CONVERGED*`는 max 100 epoch trajectory에서 validation CE가 가장 낮았던 epoch의 classifier다. 두 protocol 모두 선택 epoch는 1이었다. Strict optimization convergence에는 도달하지 않았으므로 명칭을 “수학적으로 수렴한 해”로 해석하면 안 된다.

## 8. Cosine fitting 분석

### IA5

| 지점 | Train CE | Train accuracy | Validation accuracy | Gradient norm |
|---|---:|---:|---:|---:|
| 5-epoch run, epoch 1 | 0.4208 | 93.8047 | 87.0968 | 0.4155 |
| 5-epoch run, epoch 5 | 0.3042 | 97.8164 | 83.3333 | 0.3093 |
| extended run, epoch 100 | 0.2300 | 99.4941 | 67.2043 | 0.4045 |

### CAR5

| 지점 | Train CE | Train accuracy | Validation accuracy | Gradient norm |
|---|---:|---:|---:|---:|
| 5-epoch run, epoch 1 | 0.5042 | 95.1770 | 92.5614 | 0.4955 |
| 5-epoch run, epoch 5 | 0.4085 | 98.2741 | 90.6856 | 0.3869 |
| extended run, epoch 100 | 0.3295 | 99.5436 | 82.7296 | 0.5396 |

결론은 두 부분으로 나뉜다.

1. Optimization 기준: 5 epoch에서 train loss가 완전히 수렴하지 않았다.
2. Generalization 기준: 더 오래 fitting할수록 validation이 악화됐고 validation 최적 epoch는 1이었다. 따라서 단순 epoch 증가는 해결책이 아니다.

즉 현재 문제는 “5 epoch라서 학습을 덜 했다”기보다, 작은 actual dataset을 더 강하게 fitting할 때 발생하는 generalization/regularization 문제에 가깝다.

## 9. Linear probe 해석

사전 고정한 L2 `0.0005`와 class-balanced CE에서 두 linear probe는 cosine보다 낮았다.

- Linear no-bias는 extended cosine 대비 IA5 `−6.45 pp`, CAR5 `−17.93 pp`
- Bias 추가는 linear 내부에서 IA5 `+0.79 pp`, CAR5 `+0.77 pp`
- 하지만 bias linear도 cosine보다 크게 낮고 ECE가 높다.

따라서 class-specific intercept의 작은 국소 효과는 확인되지만 classifier form이 현재 주병목이라는 근거는 아니다. 또한 이 결과는 사전 정의한 L2 한 설정에 대한 diagnostic이며 모든 linear regularization 설정이 열등하다는 뜻은 아니다.

## 10. Margin 결과

Opposite-group margin 평균:

| Protocol | Classifier | Old-vs-New | New-vs-Old |
|---|---|---:|---:|
| IA5 | Structure CA | 3.1125 | 1.1511 |
| IA5 | COSINE_5EP | 2.9451 | 1.0245 |
| IA5 | COSINE_CONVERGED | 2.9009 | 1.0750 |
| IA5 | LINEAR_NO_BIAS | 0.8490 | 0.1439 |
| IA5 | LINEAR_WITH_BIAS | 0.8503 | 0.1628 |
| CAR5 | Structure CA | 2.8063 | 1.6903 |
| CAR5 | COSINE_5EP | 2.7436 | 1.6534 |
| CAR5 | COSINE_CONVERGED | 2.6170 | 1.6363 |
| CAR5 | LINEAR_NO_BIAS | 0.5403 | 0.2132 |
| CAR5 | LINEAR_WITH_BIAS | 0.5506 | 0.2149 |

Linear probe는 opposite-group margin도 크게 작아 classifier 제약 완화로 더 나은 separation을 얻지 못했다.

## 11. Validation 및 제한사항

- IA5 validation: 1,116 sample, 192/200 class
- IA5에서 sample 수 5 미만인 8개 class는 validation에서 제외하고 train에 유지
- CAR5 validation: 1,546 sample, 196/196 class
- test는 epoch 또는 regularization 선택에 사용하지 않음
- Linear L2는 0.0005로 사전 고정; hyperparameter sweep 없음
- Extended cosine은 100 epoch safety cap에서도 gradient 기반 수렴에 도달하지 않음
- 단일 seed 결과이므로 작은 CAR5 R256−G256 차이는 추가 seed 없이는 강한 source 효과로 확정할 수 없음

## 12. Leakage 및 checkpoint 안전성

- frozen T5 feature cache만 사용
- backbone/adapter update 없음
- APR/SCC update 없음
- actual feature를 next task나 main CIL training에 전달하지 않음
- diagnostic classifier만 update
- main Structure checkpoint write 없음
- 실행 전후 checkpoint size와 edge SHA-256 fingerprint 동일

| Protocol | Checkpoint size | Edge SHA-256 |
|---|---:|---|
| IA5 | 1,414,369,793 bytes | `b30bfb61f594bf6b16b4f33edf3ef686b98728c0f4402f34c978684b224f1771` |
| CAR5 | 1,393,004,545 bytes | `c0eaebc19b16b6d80f16b5d2fa23e5a4b5543cb6c79d279a43ca8ad0ed8fadfe` |

## 13. 실행 및 재현성 검증

- GPU 2 IA5: 50.2초
- GPU 2 CAR5: 48.2초
- 필수 CSV: protocol별 7개 생성
- 추가 optimization trace CSV 생성
- plot: protocol별 4종 생성
- unit test: 13개 통과
- cosine/L-BFGS synthetic smoke test 통과
- paired label checksum 일치
- paired current/new bank checksum 일치
- GPU 1과 GPU 2의 `g256_r256_accuracy.csv` SHA-256가 protocol별로 동일
- GPU 1과 GPU 2의 `classifier_probe_accuracy.csv` SHA-256가 protocol별로 동일

실행이 빠른 이유는 기존 frozen feature cache를 재사용하고 작은 classifier만 학습했기 때문이다. Backbone forward 및 CIL 재학습은 포함하지 않았다.

## 14. T2–T4 확장 권고

**HOLD**를 권고한다.

IA5와 CAR5 모두 가장 큰 변화는 exposure를 256/class로 복원할 때 발생했고, 동일 exposure에서 G/R source 차이는 작았다. T5에서 두 protocol의 우선 병목 판정이 sampling/weighting으로 일치하므로 현재 원인 분리를 위해 T2–T4를 자동 확장할 필요는 낮다.

다음 단계가 필요하다면 우선순위는 다음과 같다.

1. 기존 CA의 class-balanced exposure/step을 유지한다.
2. 단순 epoch 증가 대신 validation-aware regularization/generalization을 검토한다.
3. CAR5의 작은 R256 이득이 재현되는지 추가 seed로 확인한다.
4. Linear classifier form을 다시 볼 경우 L2 scaling을 사전 명시한 소규모 진단으로 제한한다.

이는 새로운 method 개발 권고가 아니라 다음 원인 진단의 우선순위다.

## 15. 산출물

- Sampling audit: `ca_sampling_audit.md`
- IA5 상세 보고서: `IA5_T5/IA5_T5_classifier_procedure_report.md`
- CAR5 상세 보고서: `CAR5_T5/CAR5_T5_classifier_procedure_report.md`
- 자동 통합 보고서: `classifier_procedure_diagnostic_final_report.md`
- Paired manifest: `{IA5_T5,CAR5_T5}/paired_sampling_manifest.json`
- 실행 manifest: `{IA5_T5,CAR5_T5}/run_manifest.json`
- Accuracy: `{IA5_T5,CAR5_T5}/g256_r256_accuracy.csv`
- Classifier probe: `{IA5_T5,CAR5_T5}/classifier_probe_accuracy.csv`
- Convergence trace: `{IA5_T5,CAR5_T5}/cosine_convergence_trace.csv`
- Margin: `{IA5_T5,CAR5_T5}/classifier_margin_comparison.csv`
- Calibration: `{IA5_T5,CAR5_T5}/classifier_calibration.csv`

