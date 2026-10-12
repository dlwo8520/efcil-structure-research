# Classifier Procedure Diagnostic 구현 계획

> 대상: IA5 T5, CAR5 T5 / seed 1993  
> 범위: diagnostic-only classifier experiment  
> 제외: backbone/adapter/CIL 재학습, T2–T4 자동 확장, IA10/CAR10, 신규 method

## 1. 현재 Structure CA sampler의 실제 동작

원본 구현은 `${RESEARCH_WORKSPACE}/structure_multicenter_separation/models/base.py`의
`BaseLearner._stage2_compact_classifier()`이다.

- line 57–71: classifier parameter만 SGD에 넣고 `CosineAnnealingLR(T_max=ca_epochs)`를 생성한다.
- line 80: epoch loop를 시작한다.
- line 89: `replay_samples_per_class=256`을 읽는다.
- line 91–116: 모든 seen class를 순회하며 class당 256개 Gaussian sample을 생성한다.
- line 111–114: vanilla single-center에서는 `MultivariateNormal(mean * mean_scale, covariance)`에서 매 epoch 새로 sample한다.
- line 118–124: class별 sample을 모두 이어 붙인 뒤 `torch.randperm`으로 전체 sample을 한 번 섞는다.
- line 126–164: 256개씩 contiguous slice하여 update하며, seen class 수만큼 optimizer step을 수행한다.
- line 166: scheduler는 optimizer step마다가 아니라 epoch 종료 시 한 번 step한다.

별도 DataLoader나 sampler 객체는 없다. Gaussian sampling은 iid sampling이며 replacement 개념에 해당하는 finite bank는 없다. 매 epoch bank 전체가 regenerate된다. `torch.randperm`에 의해 batch는 class-balanced batch가 아니라 global shuffle 뒤의 mixed batch다.

Classifier는 `${RESEARCH_WORKSPACE}/structure_multicenter_separation/network/classifier.py`의
`SimpleContinualLinear.forward()`에서 feature와 weight를 모두 L2-normalize하며 bias는 없다. 실제 score는 config의 scale 20을 곱한 cosine score다.

### T5 vanilla CA exposure

| Protocol | Seen classes | Old/New classes | Per-class/epoch | Draws/epoch | Steps/epoch | Epoch | Total draws | Total steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IA5 | 200 | 160/40 | 256 | 51,200 | 200 | 5 | 256,000 | 1,000 |
| CAR5 | 196 | 157/39 | 256 | 50,176 | 196 | 5 | 250,880 | 980 |

각 class는 총 1,280회 노출된다. IA5 old/new 총 노출은 각각 204,800/51,200이며 CAR5는 200,960/49,920이다.

### 기존 G_FIXED/R_FIXED exposure

기존 `classifier_diagnostic.pipeline._actual_old_bank()`와 `_train_fixed_head()`는 old class에 actual count를 cap 256까지만 사용하고 current/new class에는 256개 fixed Structure Gaussian을 사용한다. 두 protocol 모두 actual count가 256 미만이므로 old exposure가 class별로 다르다.

| Protocol | Old unique/class | Old draws/epoch | New draws/epoch | Total/epoch | Steps/epoch | Total steps |
|---|---:|---:|---:|---:|---:|---:|
| IA5 | 2–86 | 4,883 | 10,240 | 15,123 | 60 | 300 |
| CAR5 | 24–68 | 6,545 | 9,984 | 16,529 | 65 | 325 |

따라서 기존 G_FIXED/R_FIXED는 서로 paired였지만 vanilla/Full Oracle CA와는 class exposure 및 optimizer step이 다르다.

## 2. Class별 draw count 계산

각 condition에 대해 다음을 별도로 기록한다.

- unique bank size/class
- exposure/class/epoch
- exposure/class/전체 학습
- old/new total exposure
- min/max/mean exposure per class
- batches/epoch 및 total optimizer steps
- fixed/dynamic bank 여부

결과는 `ca_sampling_audit.csv`와 `class_exposure_summary.csv`에 저장한다.

## 3. G256_FIXED 구현

T5 frozen feature에서 계산한 oracle old mean/covariance를 사용한다.

- old: class별 oracle Gaussian 256개를 실험 시작 시 한 번 생성
- new/current: 기존 Structure mean/covariance Gaussian 256개를 한 번 생성
- mean scaling: 원본 CA의 `class_id // task_size_argument` 및 `mean_scale`을 그대로 적용
- bank는 5 epoch 동안 고정
- IA5: epoch당 51,200 sample, 200 step, 총 1,000 step
- CAR5: epoch당 50,176 sample, 196 step, 총 980 step

기존 Full Oracle Gaussian CA가 dynamic reference이므로 별도 G256_DYNAMIC은 중복 실행하지 않는다.

## 4. R256_FIXED replacement sampling

각 old class의 실제 training feature를 current T5 frozen extractor로 얻는다. 각 class bank에서 seed가 고정된 replacement sampling으로 정확히 256개를 생성한다.

- old: actual feature에서 replacement draw 256개/class
- new/current: G256_FIXED와 byte-identical한 fixed Structure Gaussian bank
- 동일 feature가 반복될 수 있으며 새로운 정보를 생성하는 것이 아니다.
- unique sample 수와 exposure 수를 분리 기록한다.

## 5. Paired label sequence

class별 256개 bank를 class-id 순으로 조립한다. epoch별 global permutation을 미리 한 번 생성하고 G256/R256에 같은 permutation을 적용한다.

`paired_sampling_manifest.json`에 다음을 저장한다.

- seed와 epoch별 permutation seed
- epoch별 label sequence hash와 전체 hash
- class draw counts
- total draws
- batches/epoch와 total optimizer steps
- G/R label checksum
- G/R new-bank checksum 및 동일성 assertion

## 6. Optimizer step 및 LR schedule 일치

G256/R256 모두 exact pre-CA sidecar weight에서 시작한다.

- SGD, lr 0.01, momentum 0.9, weight decay 0.0005
- batch size 256
- 5 epoch
- cosine scale 20
- `CosineAnnealingLR(T_max=5)`를 epoch당 한 번 step
- 동일 epoch permutation과 batch boundary
- classifier parameter만 optimizer에 포함

따라서 feature source를 제외한 label order, class exposure, optimizer step, LR trajectory를 동일하게 만든다.

## 7. Cosine convergence 판정

Actual old와 current/new training feature를 모두 사용하되 train/validation을 먼저 고정 분리한다. Training feature는 class별 256 exposure가 되도록 replacement-balanced fixed bank를 만든다.

`COSINE_5EP`:

- exact pre-CA initialization
- scale 20 cosine head
- SGD 및 Structure와 같은 기본 optimizer 설정
- 5 epoch

`COSINE_CONVERGED`:

- 같은 initialization, score, loss 및 balanced bank
- max epoch 100
- test를 보지 않고 validation CE를 기록
- minimum 15 epoch
- validation CE best state 저장
- validation CE가 12 epoch 개선되지 않고 최근 relative train-loss 개선이 1e-4 미만이면 convergence stop
- gradient norm도 기록하며 max epoch을 safety cap으로 사용

선택 epoch와 stop reason을 manifest/report에 기록한다.

## 8. Linear probe 구현

입력은 `u = z / ||z||`로 고정한다.

- `LINEAR_NO_BIAS`: `w^T u`, bias 없음
- `LINEAR_WITH_BIAS`: `w^T u + b`
- class-balanced CE는 unique training sample에 inverse-frequency weight를 적용
- fixed L2 coefficient 0.0005 사용; test로 선택하지 않음
- GPU full-batch L-BFGS, max iteration 100, solver tolerance로 수렴 판정
- weight는 pre-CA classifier 방향으로 초기화하고 bias는 0으로 초기화

L-BFGS closure별 objective, train accuracy, validation accuracy, gradient norm을 저장한다.

## 9. Validation strategy

각 class에서 sample 수가 5개 이상이면 deterministic stratified 20%를 validation으로 분리하되 최소 2개는 train에 남긴다. 5개 미만 class는 validation으로 분리하지 않고 모두 train에 유지한다.

- validation 미포함 class 수와 class coverage를 report한다.
- validation은 convergence/early stop에만 사용한다.
- regularization은 사전 고정하며 validation/test sweep을 하지 않는다.
- 기존 held-out test는 최종 평가에만 사용한다.

## 10. Leakage 방지

- backbone/adapter는 eval, frozen, `torch.no_grad()`로 feature만 추출한다.
- actual feature는 diagnostic output/cache에서만 사용한다.
- APR/SCC/main Structure state를 갱신하지 않는다.
- next task 학습에 전달하지 않는다.
- diagnostic classifier parameter만 optimizer가 보유한다.
- 실행 전후 main checkpoint size 및 edge SHA-256 fingerprint를 비교한다.
- test metric은 epoch 선택이나 hyperparameter 선택에 사용하지 않는다.

## 11. 기존 checkpoint/result 재사용 위치

- checkpoint: `training_outputs/{IA5,CAR5}/Structure_AUDIT/seed_1993/task_checkpoints/task_05.pt`
- exact pre-CA state: 같은 run의 `pre_ca_classifier/task_05.pt`
- scientific A/B/D: `scientific_audit/{protocol}/seed_1993/oracle_classifier_replacement.csv`
- 기존 G_FIXED/R_FIXED와 T5 feature cache: `classifier_bottleneck_diagnostic/{protocol}/seed_1993/`

기존 결과와 cache는 read-only로 사용한다.

## 12. 출력과 실행 순서

출력 root는 `classifier_procedure_diagnostic/`이다.

1. 원본 CA code audit 및 `ca_sampling_audit.md` 생성
2. IA5 T5 feature/cache/checkpoint validation
3. paired G256/R256 bank 및 manifest 생성
4. G256/R256 5-epoch paired training
5. actual-feature train/validation split
6. COSINE_5EP 및 COSINE_CONVERGED
7. LINEAR_NO_BIAS 및 LINEAR_WITH_BIAS
8. metric/plot/protocol report 생성
9. CAR5 T5에 동일 절차 적용
10. 통합 report 및 artifact/checkpoint 검증

T2–T4는 자동 실행하지 않는다.

## 13. 예상 runtime

Frozen T5 feature cache는 기존 diagnostic에서 재사용 가능하다. G256/R256은 protocol당 약 2,000 classifier optimizer step, cosine probe는 5 epoch와 최대 100 epoch, linear probe는 L-BFGS 최대 100 iteration이다.

예상 runtime은 GPU 상태와 convergence epoch에 따라 protocol당 약 10–30분, 전체 약 20–60분이다. Max-epoch에 도달해도 safety cap을 넘지 않는다.

