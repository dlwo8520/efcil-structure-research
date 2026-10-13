# Structure Prototype Oracle Audit 최종 실험 보고서

작성일: 2026-09-28  
실험 실행일: 2026-09-23~2026-09-24  
대상: vanilla Structure, seed 1993  
Protocols: IA5, CAR5, IA10, CAR10  
상태: **모든 protocol COMPLETE / 모든 Condition-A regression PASS / 실행 실패 0건**

## 1. Executive summary

이번 실험은 Structure/APR가 보존하는 old-class mean과 covariance의 정확도를 측정하고, 그 오차를 실제 old training feature에서 구한 oracle moment로 제거했을 때 classifier adaptation 성능이 얼마나 회복되는지 확인했다.

가장 중요한 결론은 다음과 같다.

1. **Structure의 old-class mean 추정은 이미 상당히 정확했다.** old-class Structure–oracle cosine은 `0.9899~0.9948`, relative L2는 `0.0987~0.1271`이었다. APR는 거의 모든 class-task에서 pre-APR prototype보다 oracle mean에 가까워졌다.
2. **Covariance에는 체계적인 오차와 팽창 경향이 있었다.** Frobenius relative error는 `0.2884~0.3909`였고, Structure covariance trace가 증가하는 동안 oracle trace가 유지 또는 감소한 class-task가 `55.3~91.0%`였다.
3. **그러나 exact mean/covariance가 대부분의 성능을 회복시키지는 못했다.** final task Full Oracle의 Seen gain은 IA5 `-0.3292%p`, CAR5 `+0.6716%p`, IA10 `-0.0658%p`, CAR10 `+1.1068%p`였다. Covariance만 oracle로 교체한 효과는 모든 protocol에서 final `+0.1317%p` 이하였다.
4. **Cars에서는 oracle moment 효과가 작지만 일관됐고, ImageNet-A에서는 일관되지 않았다.** Full Oracle은 CAR5의 모든 transition과 CAR10의 모든 transition에서 양수였지만, IA5에서는 4개 중 1개, IA10에서는 9개 중 2개 transition에서만 양수였다.
5. **가장 강한 공통 신호는 old-new score competition이었다.** final new-only accuracy는 `80.81~96.41%`로 높았지만 single-head New accuracy보다 `10.53~17.74%p` 높았다. 실제 new→old 오류율도 `14.65~32.26%`였다. 즉 current-class 내부 구분 능력은 상당 부분 살아 있지만, 모든 seen class를 함께 예측할 때 old class가 new sample을 빼앗는 현상이 크다.
6. 따라서 **mean/covariance 추정 오차는 존재하지만 보편적인 1차 병목으로 보기는 어렵다.** 현재 우선순위는 actual old feature를 사용한 frozen-representation balanced classifier 진단으로 Gaussian approximation과 classifier adaptation 자체를 분리하는 것이다.

이 결론은 seed 1993 한 번의 결과에 근거한다. 통계적 일반화를 위해서는 최종 후보 진단을 multi-seed로 반복해야 한다.

## 2. 연구 질문

이번 audit은 다음 질문을 검증하도록 구성했다.

1. Structure/APR가 old-class feature distribution의 mean과 covariance를 얼마나 정확히 추정하는가?
2. old mean만 정확히 알 때, covariance만 정확히 알 때, 둘 다 정확히 알 때 성능이 얼마나 회복되는가?
3. New 성능 저하는 current representation 자체의 손상인가, old-new single-head competition인가?
4. 정확한 1·2차 moment로도 회복되지 않는다면 다음 병목은 Gaussian approximation, classifier adaptation, representation overlap 중 어디인가?

## 3. 실험 설계

### 3.1 Protocol과 학습 조건

| Protocol | Dataset | Task partition | Classes | Seed |
|---|---|---|---:|---:|
| IA5 | ImageNet-A | 40×5 | 200 | 1993 |
| CAR5 | Cars196 | 40, 39, 39, 39, 39 | 196 | 1993 |
| IA10 | ImageNet-A | 20×10 | 200 | 1993 |
| CAR10 | Cars196 | 20×6, 19×4 | 196 | 1993 |

모든 protocol은 vanilla Structure의 동일한 설정을 사용했다.

- initial training 30 epochs, incremental training 15 epochs
- classifier adaptation 5 epochs
- synthetic samples 256/class
- SGD, learning rate `0.01`, weight decay `5e-4`
- cosine scale `20`
- APR 활성, SCC 활성, multi-center 비활성
- training oracle statistics에는 가능한 모든 old training sample 사용
- evaluation은 분리된 test split 사용
- backbone/adapter는 audit 동안 frozen, `model.eval()`과 `torch.no_grad()` 적용

final task에서 사용된 전체 seen sample 수는 IA가 train `5,981` / test `1,519`, CAR가 train `8,144` / test `8,041`이었다. sample cap은 사용하지 않았다.

### 3.2 A–D classifier adaptation

각 T2 이후 task에서 동일한 saved pre-CA classifier, RNG, optimizer, epoch 수, synthetic sample 수로 다음 네 조건을 독립 실행했다. 교체 대상은 old class statistics뿐이며 current/new class statistics는 공통으로 유지했다.

| Condition | Old mean | Old covariance |
|---|---|---|
| A — Structure | APR mean | Structure covariance |
| B — Oracle Mean | Oracle train mean | Structure covariance |
| C — Oracle Covariance | APR mean | Oracle train covariance |
| D — Full Oracle | Oracle train mean | Oracle train covariance |

여기서 D는 “실제 old feature replay”가 아니다. 정확한 mean/covariance로부터 Gaussian synthetic feature를 생성한 classifier adaptation이다. 따라서 D가 개선되지 않는 경우에는 moment 오차의 가치가 작다는 결론은 가능하지만, 실제 feature distribution 전체가 가치 없다는 결론은 내릴 수 없다.

### 3.3 Scientific validity 검사

- IA5 사전검증 8개 항목: 모두 PASS
- Condition A regression: 전체 26개 transition PASS
- A와 original post-CA classifier weight relative L2 최대값: `0.0`
- A와 original prediction mismatch: `0`
- checkpoint fingerprint 전후: 동일
- diagnostic optimizer: classifier head parameter만 포함
- 기존 sanity 결과: 정식 결과에서 제외

따라서 아래 A–D 차이는 diagnostic 구현 차이로 생긴 수치가 아니라, 교체한 statistics의 차이로 해석할 수 있다.

## 4. Vanilla Structure 기준 성능

| Protocol | Inc-Acc | Final Seen | Forgetting |
|---|---:|---:|---:|
| IA5 | 74.5977 | 66.4253 | 7.8444 |
| CAR5 | 84.9292 | 81.1964 | 4.4314 |
| IA10 | 73.8303 | 64.6478 | 9.1034 |
| CAR10 | 83.2004 | 78.9578 | 5.1198 |

10-task는 대응하는 5-task보다 final accuracy가 IA에서 `1.7775%p`, CAR에서 `2.2385%p` 낮았고 forgetting도 더 높았다. 더 긴 task sequence에서 누적 간섭이 커지는 방향과 일치하지만, 본 실험은 단일 seed이므로 차이의 분산은 평가하지 못했다.

## 5. Final-task Oracle classifier adaptation 결과

### 5.1 전체 A–D 결과

| Protocol | Condition | Seen | Old | New | Old-only | New-only |
|---|---|---:|---:|---:|---:|---:|
| IA5 | A — Structure | 66.4253 | 66.9872 | 63.8376 | 69.1506 | 80.8118 |
| IA5 | B — Oracle Mean | 66.0303 | 66.9872 | 61.6236 | 68.9904 | 80.8118 |
| IA5 | C — Oracle Cov | 66.3594 | 66.9071 | 63.8376 | 69.0705 | 81.5498 |
| IA5 | D — Full Oracle | 66.0961 | 66.9071 | 62.3616 | 68.9103 | 81.5498 |
| CAR5 | A — Structure | 81.1964 | 80.9870 | 82.0545 | 83.1219 | 92.5808 |
| CAR5 | B — Oracle Mean | 81.5073 | 81.4356 | 81.8009 | 83.4004 | 92.5808 |
| CAR5 | C — Oracle Cov | 81.2212 | 80.9715 | 82.2448 | 83.0910 | 92.6443 |
| CAR5 | D — Full Oracle | 81.8679 | 81.8379 | 81.9911 | 83.7407 | 92.7077 |
| IA10 | A — Structure | 64.6478 | 64.4444 | 66.9355 | 65.3763 | 84.6774 |
| IA10 | B — Oracle Mean | 64.5820 | 64.5878 | 64.5161 | 65.5914 | 84.6774 |
| IA10 | C — Oracle Cov | 64.7795 | 64.5161 | 67.7419 | 65.5197 | 84.6774 |
| IA10 | D — Full Oracle | 64.5820 | 64.5161 | 65.3226 | 65.5197 | 84.6774 |
| CAR10 | A — Structure | 78.9578 | 78.6836 | 81.5148 | 79.9504 | 96.4056 |
| CAR10 | B — Oracle Mean | 79.4926 | 79.3996 | 80.3594 | 80.5701 | 96.2773 |
| CAR10 | C — Oracle Cov | 79.0325 | 78.7937 | 81.2580 | 80.1019 | 96.2773 |
| CAR10 | D — Full Oracle | 80.0647 | 80.0468 | 80.2311 | 81.0796 | 96.4056 |

### 5.2 A 대비 final 변화

| Protocol | ΔB Seen | ΔC Seen | ΔD Seen | ΔD Old | ΔD New |
|---|---:|---:|---:|---:|---:|
| IA5 | -0.3950 | -0.0658 | -0.3292 | -0.0801 | -1.4760 |
| CAR5 | +0.3109 | +0.0249 | +0.6716 | +0.8509 | -0.0634 |
| IA10 | -0.0658 | +0.1317 | -0.0658 | +0.0717 | -1.6129 |
| CAR10 | +0.5348 | +0.0746 | **+1.1068** | **+1.3632** | -1.2837 |

핵심 관찰은 다음과 같다.

- **Oracle Covariance 단독 효과는 매우 작다.** final Seen 변화의 절대값이 네 protocol 모두 `0.132%p` 이하다. covariance 추정 오차는 측정되지만 현재 Gaussian CA 성능의 주된 제한으로 보이지 않는다.
- **Oracle Mean 효과는 dataset-dependent다.** Cars에서는 양수지만 IA에서는 0 또는 음수다.
- **CAR10의 +1.1068%p는 old 개선과 new 하락의 결합이다.** Old는 `+1.3632%p` 회복했지만 New는 `-1.2837%p` 하락했다. old class가 177개, new class가 19개인 final evaluation에서 old 개선의 비중이 더 커 전체 Seen이 상승했다.
- CAR5도 Old `+0.8509%p`, New `-0.0634%p`로 비슷하지만 더 약한 trade-off를 보였다.
- IA5와 IA10에서는 Full Oracle이 New 성능을 각각 `-1.4760%p`, `-1.6129%p` 낮춰 전체 이득이 사라졌다.

즉 exact old moments는 일부 old-class retention에는 도움이 될 수 있지만, 이를 그대로 사용하는 것이 old-new 균형을 자동으로 해결하지는 않는다.

## 6. Task progression

아래 표는 transition별 Full Oracle `D−A Seen` 변화다.

| Task | IA5 | CAR5 | IA10 | CAR10 |
|---:|---:|---:|---:|---:|
| T2 | -0.2865 | +0.2757 | -1.1111 | +0.4177 |
| T3 | +0.3077 | +0.2463 | -0.2101 | +0.3619 |
| T4 | -0.1603 | +0.0464 | -0.5731 | +0.8172 |
| T5 | -0.3292 | +0.6716 | -0.7134 | +0.8260 |
| T6 | — | — | -0.3077 | +1.0286 |
| T7 | — | — | +0.2676 | +0.7683 |
| T8 | — | — | +0.4006 | +0.8456 |
| T9 | — | — | -0.0717 | +1.5423 |
| T10 | — | — | -0.0658 | +1.1068 |

| Protocol | Mean ΔB | Mean ΔC | Mean ΔD | Positive D transitions |
|---|---:|---:|---:|---:|
| IA5 | -0.1335 | -0.0066 | -0.1171 | 1/4 |
| CAR5 | +0.1662 | +0.0292 | +0.3100 | 4/4 |
| IA10 | +0.0097 | -0.0224 | -0.2650 | 2/9 |
| CAR10 | +0.4902 | +0.1898 | +0.8572 | 9/9 |

Cars에서는 Full Oracle 효과가 모든 transition에서 양수이고 CAR10 후반부에서 커졌다. 반대로 ImageNet-A에서는 effect sign이 불안정하며 평균적으로 0 이하이다. 따라서 moment replacement의 가치는 task 수만으로 설명되기보다 dataset/feature geometry와 강하게 상호작용한다.

## 7. Prototype 및 APR audit

| Protocol | Structure↔Oracle cosine | Prototype relative L2 | APR improved | Age↔cumulative error |
|---|---:|---:|---:|---:|
| IA5 | 0.9948 | 0.0987 | 400/400 | 0.7915 |
| CAR5 | 0.9932 | 0.1064 | 392/394 | 0.8193 |
| IA10 | 0.9931 | 0.1086 | 900/900 | 0.7936 |
| CAR10 | 0.9899 | 0.1271 | 889/894 | 0.8469 |

APR는 대부분의 class-task에서 oracle mean 방향으로 prototype을 이동시켰다. 따라서 APR가 임의 방향으로 prototype을 악화시키는 것이 전체 성능 저하의 주원인이라는 가설은 지지되지 않는다.

다만 class age와 cumulative prototype error 사이에는 `0.79~0.85`의 높은 양의 상관이 있었다. 오래된 class일수록 오차가 누적되는 현상은 분명하다. 그럼에도 Oracle Mean replacement의 CA gain이 IA에서 거의 없고 Cars에서도 작았으므로, 현재 크기의 mean error가 single-head decision boundary를 지배한다고 보기는 어렵다.

Prototype L2와 forgetting의 Spearman 상관은 IA5 `0.1949`, CAR5 `0.2731`, IA10 `0.2543`, CAR10 `0.4177`이었다. 연관은 CAR10에서 가장 강했지만 전체적으로 약하거나 중간 수준이며 인과관계를 뜻하지 않는다.

## 8. Covariance audit

| Protocol | Frobenius relative error | Trace ratio | Structure trace 증가율 | Structure↑ / Oracle≤0 |
|---|---:|---:|---:|---:|
| IA5 | 0.3030 | 1.1297 | 100.0% | 91.0% |
| CAR5 | 0.2884 | 1.0588 | 100.0% | 80.7% |
| IA10 | 0.3741 | 1.1587 | 100.0% | 70.2% |
| CAR10 | 0.3909 | 1.1223 | 100.0% | 55.3% |

Structure covariance는 task가 진행될수록 지속적으로 팽창하는 반면 oracle covariance는 자주 감소했다. 이 현상은 코드나 표본 오차 수준을 넘어선 체계적인 mismatch 후보다.

하지만 exact covariance만 사용한 Condition C의 평균 Seen gain은 IA5 `-0.0066%p`, CAR5 `+0.0292%p`, IA10 `-0.0224%p`, CAR10 `+0.1898%p`로 작았다. 따라서 covariance mismatch는 존재하지만, 현재 classifier adaptation의 최종 정확도 관점에서는 직접적인 1차 병목으로 보이지 않는다.

Cars에서 D가 B와 C 각각의 효과보다 더 큰 경우가 반복되어 mean–covariance interaction 가능성은 남아 있다. 다만 단일 seed와 작은 effect size 때문에 후속 반복 전에는 확정하지 않는다.

## 9. Representation과 old-new competition

| Protocol | Single-head New | New-only | Gap | New→Old | Old→New |
|---|---:|---:|---:|---:|---:|
| IA5 | 63.8376 | 80.8118 | 16.9742 | 31.3653% | 8.5737% |
| CAR5 | 82.0545 | 92.5808 | 10.5263 | 14.6481% | 3.8366% |
| IA10 | 66.9355 | 84.6774 | 17.7419 | 32.2581% | 3.5125% |
| CAR10 | 81.5148 | 96.4056 | 14.8909 | 17.5866% | 2.5062% |

네 protocol 모두 new-only accuracy가 single-head New보다 크게 높았다. New class들끼리만 비교하면 구분 능력이 상당히 유지되지만, old class까지 후보에 포함하면 New accuracy가 크게 하락한다.

또한 old→new보다 new→old 오류가 훨씬 높아 경쟁이 비대칭적이다. new-vs-old margin과 new→old 오류의 Spearman 상관은 IA5 `-0.7625`, CAR5 `-0.5995`, IA10 `-0.8025`, CAR10 `-0.6558`이었다. margin이 작아질수록 new sample이 old class로 넘어가는 관계가 강하다.

이 결과는 current representation이 완전히 붕괴했다기보다, single-head score calibration 또는 classifier competition이 더 직접적인 병목이라는 해석을 지지한다. 다만 new-only accuracy도 100%는 아니므로 representation 문제가 전혀 없다는 뜻은 아니다.

## 10. Gaussian approximation과 classifier adaptation

Oracle nearest-prototype classifier는 final task에서 Structure prototype 대비 IA5 `+30.22%p`, CAR5 `+12.96%p`, IA10 `+30.94%p`, CAR10 `+14.16%p` 높았다. 반면 같은 oracle mean을 Gaussian classifier adaptation에 넣은 Condition B의 이득은 훨씬 작았다.

두 결과는 서로 다른 classifier를 평가하므로 직접적인 모순은 아니다. 오히려 다음 가능성을 제시한다.

1. oracle mean 자체는 class 구분 정보를 갖지만 현재 synthetic Gaussian CA가 그 정보를 classifier weight로 충분히 전달하지 못한다.
2. mean/covariance만으로 표현되지 않는 multimodality, skew, tail 또는 local structure가 중요할 수 있다.
3. old statistics 개선이 old score를 높이는 동시에 new score 경쟁을 악화시켜 전체 이득을 상쇄할 수 있다.

관측된 mean absolute skew는 `0.300~0.345`, mean absolute excess kurtosis는 `0.572~0.834`였다. Gaussian deviation이 존재하지만, 이 수치만으로 성능 병목의 인과성을 확정할 수 없다.

## 11. 연구 질문에 대한 최종 답변

### Q1. Structure/APR는 실제 old distribution을 정확히 추정하는가?

- Mean: 비교적 정확하다. cosine 약 `0.99`, relative L2 약 `0.10~0.13`이다.
- APR: 거의 항상 pre-APR보다 oracle mean에 가까워진다.
- Covariance: mean보다 오차가 크며 trace가 과도하게 증가하는 경향이 있다.
- 장기 sequence: class age가 증가할수록 cumulative prototype error가 커진다.

### Q2. Mean/covariance 오차를 제거하면 성능이 회복되는가?

- ImageNet-A: 실질적인 회복이 없다.
- Cars: 작지만 일관된 회복이 있다. CAR10 Full Oracle final gain은 `+1.1068%p`이다.
- Covariance 단독 교체 효과는 전반적으로 매우 작다.
- 따라서 moment estimation error는 dataset-dependent한 보조 병목이지만 보편적인 주 병목은 아니다.

### Q3. New 손실은 representation 문제인가 competition 문제인가?

높은 new-only accuracy, 큰 new-only gap, 높은 new→old 오류율을 함께 보면 **old-new score competition이 더 강한 공통 신호**다. Representation 손상 가능성은 남지만 우선순위는 competition/classifier calibration 쪽이다.

### Q4. Exact moments로도 회복되지 않는다면 다음 병목은 무엇인가?

현재 증거만으로는 Gaussian approximation과 classifier adaptation을 분리할 수 없다. 다음 실험은 frozen representation에서 actual old training features를 직접 사용한 balanced classifier여야 한다.

- actual feature classifier가 크게 개선되면: Gaussian moment approximation 또는 synthetic sampling이 병목
- actual feature classifier도 개선이 작으면: classifier objective/optimization 또는 representation overlap이 병목
- actual replay representation reference에서만 개선되면: feature representation drift/overlap이 병목

## 12. 권장 후속 실험

우선순위는 다음과 같다.

1. **Actual-old-feature balanced classifier diagnostic**  
   동일 pre-CA head와 frozen feature extractor를 사용하고, old training feature와 current feature를 균형 있게 넣어 학습한다. 본 학습 checkpoint에는 반영하지 않는다.
2. **Gaussian vs actual-feature paired comparison**  
   동일 sample 수, optimizer, batch order, epoch, seed에서 D Full Oracle Gaussian과 actual-feature classifier를 비교한다.
3. **Old-new score calibration diagnostic**  
   representation과 classifier weight는 고정하고 task-wise temperature/bias 조정만으로 New gap이 얼마나 줄어드는지 확인한다.
4. **Multi-seed 재현**  
   위 진단으로 병목 후보를 하나로 줄인 후 최소 3 seeds에서 IA5/CAR5를 반복하고, 유망한 경우 IA10/CAR10으로 확장한다.

현재 단계에서는 새로운 prototype method, RP sweep, Multi-View RP 또는 추가 loss를 늘리는 것보다 위 세 진단으로 classifier/Gaussian/representation 병목을 먼저 분리하는 것이 타당하다.

## 13. 한계

- 모든 결과는 seed 1993 한 번의 실행이며 confidence interval이나 유의성 검정이 없다.
- Oracle statistics는 training split에서 계산했지만 실제 feature replay가 아니라 1·2차 moment 기반 Gaussian replay에 사용됐다.
- A–D는 frozen feature extractor에서 classifier adaptation만 반복한 진단이며 전체 CIL model 재학습 결과가 아니다.
- condition 간 paired RNG를 사용했지만 dataset별 sample 수와 class partition이 달라 protocol 간 절대 수치를 직접 동일 조건처럼 비교할 수 없다.
- t-SNE는 task별 독립 좌표이므로 task 간 이동 해석에는 사용하지 않았다.
- correlation은 병목의 인과성을 증명하지 않는다.

## 14. 결과 파일

- [Cross-protocol 자동 보고서](structure_oracle_cross_protocol_report.md)
- [IA5 상세 보고서](IA5/seed_1993/IA5_structure_oracle_report.md)
- [CAR5 상세 보고서](CAR5/seed_1993/CAR5_structure_oracle_report.md)
- [IA10 상세 보고서](IA10/seed_1993/IA10_structure_oracle_report.md)
- [CAR10 상세 보고서](CAR10/seed_1993/CAR10_structure_oracle_report.md)
- [Queue 실행 상태](holiday_oracle_audit_status.md)
- Protocol별 원본 CSV: `scientific_audit/<PROTOCOL>/seed_1993/`

## 15. 최종 결론

Structure/APR의 mean tracking은 예상보다 정확했고 covariance mismatch는 분명했지만, exact old mean/covariance를 넣는 것만으로는 대부분의 성능이 회복되지 않았다. Cars, 특히 CAR10에서는 old retention이 개선됐지만 New 성능 일부를 희생했고, ImageNet-A에서는 전체 이득이 없었다.

네 protocol에서 공통으로 가장 강한 현상은 높은 new-only accuracy와 큰 new→old 오류였다. 따라서 현재의 가장 유력한 병목은 **old distribution moment 추정 자체보다 old-new classifier competition 및 synthetic classifier adaptation의 한계**다. 다음 단계는 새로운 방법론을 추가하는 것이 아니라 actual old features를 사용하는 balanced diagnostic classifier로 Gaussian approximation과 classifier adaptation을 분리하는 것이다.
