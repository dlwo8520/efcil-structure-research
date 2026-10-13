# Classifier Bottleneck Diagnostic 구현 전 계획

작성일: 2026-09-29  
대상: IA5, CAR5, seed 1993  
목적: 새 method 개발이 아니라 Structure statistics가 classifier로 변환되는 과정에서 정보와 old-new 균형이 손실되는 지점을 분리한다.

## 1. 현재 Structure CA의 실제 feature source

원본 경로는 `structure_multicenter_separation/models/base.py::_stage2_compact_classifier`이다.

- old와 current/new를 포함한 **모든 seen class**에 대해 Gaussian synthetic feature를 생성한다.
- class `c`의 입력은 `MultivariateNormal(class_means[c] * mean_scale, class_covs[c])`에서 sampling한다.
- class당 epoch별 `replay_samples_per_class=256`개를 새로 생성한다.
- `mean_scale = 0.9 + ((birth_task_index + 1) / (current_task_index + 1)) * 0.1`이다. 실제 원본 구현은 class ID를 현재 `task_size`로 나눈 값을 birth-task 근사로 사용한다.
- 생성된 raw feature를 `model.ca_forward()`에 전달한다.
- classifier는 `SimpleContinualLinear`이며 bias가 없는 task별 cosine head다.
- 실제 score는 `scale * cosine(feature, weight)`, scale은 20이다.

따라서 기존 D Full Oracle Gaussian CA도 old/new 모두 Gaussian이다. D에서 old만 oracle mean/covariance로 교체되고 current/new Gaussian source는 A와 동일하다.

## 2. Normalization 경로

Feature extraction은 frozen network의 `output["sfeatures"]`를 저장한다. CA 입력 직전에는 별도 명시적 normalization이 없지만, `SimpleContinualLinear.forward()` 안에서 다음을 수행한다.

```text
F.linear(F.normalize(feature), F.normalize(weight))
```

따라서 R-CA에는 frozen model이 출력한 raw `sfeatures`를 넣고 head 내부 normalization을 그대로 사용한다. 미리 normalize한 feature와 raw feature를 섞지 않는다. Weight norm과 raw feature norm은 diagnostic으로 저장하지만 cosine prediction score에는 직접 들어가지 않는다고 명시한다.

## 3. 기존 scientific output 재사용 가능성

재사용 가능:

- `training_outputs/<PROTOCOL>/Structure_AUDIT/seed_1993/task_checkpoints/task_XX.pt`
- `pre_ca_classifier/task_XX.pt`
- Structure/APR mean, Structure covariance, task partition, config, RNG
- `scientific_audit/<PROTOCOL>/seed_1993/oracle_classifier_replacement.csv`의 A/B/C/D accuracy
- A post-CA weight는 task checkpoint에서 직접 재사용
- 기존 D accuracy는 새 main 결과로 중복 계산하지 않고 reference로 재사용

직접 재사용 불가:

- scientific `features/task_XX.npz`는 시각화용 class당 최대 50개 train feature만 저장하므로 R-CA bank로 사용할 수 없다.
- 기존 D의 final diagnostic weight/logit은 저장되지 않았다. D bias sweep/alignment를 위해 동일 saved pre-CA RNG로 D head를 **재구성**하고, 재구성 accuracy가 기존 CSV와 일치하는지 검증한 뒤 weight/logit artifact로만 사용한다.
- 기존 prototype classifier audit은 Structure mean과 oracle mean을 모든 seen class에 각각 적용했다. 요청한 APR-NCM vs Oracle-NCM은 current/new prototype을 동일하게 고정해야 하므로 새로 계산한다.

## 4. NCM 정의와 기존 +30pp 검증

새 NCM main diagnostic은 full test feature에 cosine NCM을 적용한다.

- APR-NCM old: checkpoint APR mean
- Oracle-NCM old: current frozen model의 old train feature mean
- 두 NCM의 current/new prototype: 동일한 current/new train feature mean
- 동일 frozen checkpoint, test feature, normalization, seen class set, global label mapping 사용

기존 `prototype_classifier_audit.csv` 비교와 새 old-only replacement 비교를 별도 표기한다. 특히 기존 비교가 current/new mean까지 교체했음을 보고서에 명시한다.

## 5. Actual feature 삽입 위치

원본 training 코드는 수정하지 않는다. 새 project 내부 diagnostic trainer에서 다음 순서로 구현한다.

1. `oracle_audit.runtime.load_frozen_learner`로 task checkpoint를 read-only 로드
2. `extract_split(..., source="train")`으로 모든 seen training `sfeatures` 추출
3. old class feature bank를 class별로 구성
4. current/new는 두 조건 모두 동일한 fixed Gaussian bank 사용
5. pre-CA classifier를 task별 Parameter layout으로 복원
6. diagnostic head만 SGD/CosineAnnealingLR로 학습
7. main checkpoint와 feature extractor는 checksum 및 requires-grad guard로 보호

## 6. G-fixed / R-fixed paired 설계

Main comparison은 fixed bank다.

- G-fixed old: oracle train mean/covariance Gaussian bank
- R-fixed old: actual old train `sfeatures` bank
- New: 두 조건 모두 동일 fixed Gaussian bank
- 동일 pre-CA head, optimizer, LR, weight decay, epochs, scale, class order, bank shuffle seed, update budget
- 모든 bank는 한 번 생성한 뒤 모든 epoch에서 반복 사용

Matched Count:

```text
n_k = min(actual old count, 256)
```

G/R old bank에서 class별 같은 `n_k`를 사용한다. Actual subsampling은 seed 기반 deterministic selection이다. 현재 IA/CAR train class 수는 모두 256 미만이므로 Full Actual과 Matched Count의 actual bank가 같을 가능성이 높다. 실제 count audit 후 같으면 중복 학습하지 않고 `matched_equals_full_actual=true`로 기록한다.

기존 CA의 class-age `mean_scale`은 G-fixed 생성에 유지한다. R-fixed는 actual feature를 직접 사용하는 정의를 우선하며 synthetic mean 이동을 적용하지 않는다. 이 차이는 actual-vs-Gaussian source 정의에 포함되지만, 보고서에서 잠재적 confound로 명시한다. 필요하면 결과 해석용 보조 control로 actual residual을 scaled oracle centroid에 재중심화하는 조건을 추가할 수 있으나 main result로 섞지 않는다.

## 7. Bias 적용 위치

Head 재학습 없이 scaled cosine logits에 적용한다.

```text
new-class logits += beta
old-class logits unchanged
```

- 기본 범위 `[-5, 5]`, 101 points
- A, reconstructed D, G-fixed, R-fixed에 동일 적용
- test sweep은 diagnostic trade-off curve로만 사용
- test에서 선택한 best beta를 정식 성능처럼 제시하지 않음
- 별도 validation으로 선택하지 않는 한 `ORACLE/TEST-SWEEP DIAGNOSTIC` 표기
- 동일 global temperature는 argmax를 바꾸지 않는다는 점을 보고서에 명시

## 8. 수집 metric과 파일

새 output root:

```text
classifier_bottleneck_diagnostic/
  IA5/seed_1993/
  CAR5/seed_1993/
```

Protocol별 생성:

- `ncm_absolute_comparison.csv`
- `ncm_prediction_change_analysis.csv`
- `gaussian_vs_actual_classifier_accuracy.csv`
- `ca_margin_shift.csv`
- `score_distribution_summary.csv`
- `weight_mean_alignment.csv`
- `classifier_weight_diagnostics.csv`
- `ca_optimization_trace.csv`
- `calibration_metrics.csv`
- `bias_sweep.csv`
- `old_new_group_margin.csv`
- `run_manifest.json`
- `IA5_classifier_bottleneck_report.md` 또는 `CAR5_classifier_bottleneck_report.md`

Plots:

- `bias_tradeoff/`
- `score_distributions/`
- `margin_distributions/`
- `weight_alignment/`

Metric 범위:

- Seen/Old/New/Old-only/New-only와 old/new confusion decomposition
- per-task, per-class accuracy
- pre/post internal 및 opposite-group margin
- score mean/std/median/q05/q25/q50/q75/q95
- weight–mean cosine, weight norm, bias 존재 여부
- epoch loss/accuracy, gradient norm, weight update norm
- NLL, multiclass Brier, ECE
- old/new group margin과 group AUROC
- R-fixed − G-fixed delta
- bias trade-off와 approximate frontier dominance

## 9. Leakage 방지

- old raw sample은 diagnostic feature extraction 및 diagnostic head에만 사용
- backbone/adapter는 eval, no-grad, requires-grad false
- optimizer parameter ID가 diagnostic head parameter ID와 정확히 일치하는지 검사
- checkpoint edge fingerprint를 실행 전후 비교
- diagnostic weight를 main learner/checkpoint에 load하지 않음
- output은 기존 `scientific_audit/`와 분리
- IA10/CAR10 자동 실행 금지
- test bias sweep은 curve 분석만 수행하고 deployable best-beta 결과로 표현하지 않음

## 10. 실행 순서

1. IA5/CAR5 checkpoint, sidecar, A/B/C/D CSV, class order 검증
2. final task T5에서 full train/test feature 추출
3. 요청 정의의 APR-NCM/Oracle-NCM 계산 및 prediction-change 분석
4. A와 기존 D reference 검증, A/D bias sweep
5. fixed bank G/R paired head 학습
6. score/margin/alignment/optimization/calibration/group-margin 수집
7. G-fixed/R-fixed bias curve 비교
8. final-task 결과가 유의미한 진단 신호를 주는지 판정
9. 신호가 있으면 동일 코드로 T2~T5 확장
10. protocol별 및 IA5/CAR5 통합 report 생성
11. optional G/R Linear는 결과가 모호할 때만 별도 승인 없이 실행하지 않고 recommendation으로 남김
12. IA10/CAR10은 실행하지 않고 report에 확장 여부만 권고

## 11. 예상 해석 경계

- R > G는 Gaussian replacement가 현재 head 조건에서 제한 요인이라는 증거지만 multimodality를 단독 원인으로 확정하지 않는다.
- R ≈ G는 actual source 교체만으로 부족하다는 뜻이며 representation 원인을 확정하지 않는다.
- bias sweep에서 좋은 작동점이 있어도 test-oracle diagnostic이며 정식 EF-CIL 결과가 아니다.
- new-only gap은 competition 신호지만 calibration 단독 원인을 확정하지 않는다.
