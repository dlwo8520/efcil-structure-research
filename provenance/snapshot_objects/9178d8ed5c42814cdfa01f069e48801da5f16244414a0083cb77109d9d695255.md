# Structure multi-center / old-new separation 분석

## 기준과 출처

- 공식 CDS 저장소 `https://github.com/lambor9973/cds`의 `main` 브랜치, 커밋
  `a0db2168663982a4cad24981231721a3518ddf29`를 새 폴더에 clone한 뒤 수정했다.
- Structure 논문은 CVPR 2026 원문과 공개 코드를 함께 확인했다.
  논문의 APR 식 (4)--(8), SCC 식 (9), 전체 loss 식 (12)를 코드와 대조했다.
- FCS는 공식 저장소 `https://github.com/zhoujiahuan1991/CVPR2024-FCS`의 커밋
  `41a4175b039bf44e06ed9e3665191285fe730606`과 CVPR 2024 원문을 확인했다.
- MPFR은 DOI `10.3390/s26030979` 및 PMC 원문을 확인했다. 논문은 K-means
  multi-center와 centroid별 isotropic diffusion을 정의하지만 코드 링크는 제공하지
  않는다. 따라서 이 프로젝트의 구현은 MPFR 공식 코드 이식이 아니라 원문 기반
  참고 구현이다.

cluster별 full covariance, class-level APR의 local 적용, birth quantile, epoch APR
cache, normalized Mahalanobis hinge는 이 프로젝트의 제안 사양이다. FCS 또는 MPFR의
원래 방법이나 검증된 novelty로 부르지 않는다.

## 공개 CDS 실제 pipeline

| 단계 | 실제 위치와 호출 | shape / 공간 | gradient |
|---|---|---|---|
| 실행·config | `main.py -> trainer.CDS_train -> trainer._train` | JSON dict | 없음 |
| 분할·class order | `DataManager._setup_data`, `_map_new_class_index` | 원 class ID를 seed permutation의 index로 remap | 없음 |
| task 학습 | `Learner.incremental_train -> _train -> _init_train` | image `[B,3,224,224]` | student adapter와 classifier |
| dual adapter | `VisionTransformer.forward` | `tfeatures`, `sfeatures`: `[B,768]`; 둘 다 raw feature | teacher detach, student gradient |
| classification | `AngularPenaltySMLoss` | 새 head logits `[B,40]` | 있음 |
| SCC | `LobachevskyRelativeGeomLoss` | new sample과 old class mean의 cosine 관계 `[B,C_old]` | old mean/teacher detach, student gradient |
| task-start current stats | `_conpute_now_task_old_class_mean` | current classes의 pre-training mean `[40,768]`, covariance `[40,768,768]` | eval/no-grad |
| task-end stats | `_compute_class_mean` | class mean NumPy, covariance Torch; raw student feature, covariance에 `1e-3 I` | eval/no-grad |
| APR | `LobachevskyPrototypeCorrector.correct_prototypes` | old mean/cov를 current feature 공간으로 추정 | 없음 |
| replay·stage 2 | `BaseLearner._stage2_compact_classifier` | class당 256 Gaussian feature, single-head cosine classifier | classifier만 |
| 평가 | `BaseLearner._eval_cnn`, `_evaluate` | 지금까지의 모든 head를 concat하여 single-head top-k | 없음 |
| resume | 프로젝트의 `save_experiment_checkpoint`, `load_experiment_checkpoint` | model/statistics/optimizer/scheduler/RNG/source snapshot | 없음 |

공개 코드의 호출 순서는 task-start current-class 통계 추출, stage-1 학습, task-end 새
통계 계산, APR, Gaussian replay stage-2이다. 따라서 task-end APR 결과를 앞선 stage-1
학습에 사용했다고 해석하면 안 된다.

SCC는 이미 new feature와 old prototype의 상대 각 관계를 보존한다. 이번 separation은
그 관계 보존과 달리, old Gaussian region 내부로 들어온 new feature에 직접 penalty를
준다. 두 gradient가 항상 일치한다고 보장할 수 없으므로 둘을 동시에 켠 E2--E7에서
각 loss와 separation gradient norm을 기록한다.

## 논문과 공개 코드에서 확인한 차이

- Structure 논문 본문은 Query/Value LoRA를 설명하지만 공개 저장소는 두 벌의 Adapter를
  번갈아 teacher/student로 사용하고 ImageNet-A JSON은 `adapter_placement="mlp"`이다.
  이 프로젝트는 재현 가능한 공개 코드 경로를 baseline으로 삼는다.
- 논문은 ImageNet-A 5/10/20-task 결과를 보고하지만 공개
  `adapter_imageneta.json`은 `20+20`, 즉 200 classes에서 10 tasks다. 이번 protocol은
  요청 사양대로 `40+40`의 정확히 5 tasks다.
- 공개 trainer의 seed loop에도 불구하고 `_set_random()`은 항상 1을 사용한다. 이번
  구현은 지정 seed 1993/1996/1997을 실제 Python/NumPy/Torch RNG에 적용한다. 이는
  알고리즘 변경은 아니지만 공식 실행 코드와의 재현성 호환 수정이다.
- 공개 stage-2는 class당 256개와 old-task mean scale을 코드에 고정한다. 이번에는
  값을 `replay_samples_per_class=256`으로 명시했으며 scale 규칙은 그대로 유지했다.
- 공개 ImageNet-A loader에는 실제 경로 대신 placeholder만 있다. 데이터 경로는
  `--data-root`로 받도록 변경했다.

따라서 E0는 공개 CDS 알고리즘(SCC+APR+single Gaussian replay)을 유지하지만, 논문
수치와 bitwise 동일한 실행이라고 주장하지 않는다.

## 확인된 ImageNet-A protocol

현재 로컬 root `${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a`를 read-only로 검사했다.

- train/test 모두 동일한 200개 ImageFolder class를 가진다.
- train 5,981장, test 1,519장으로 합계 7,500장이다.
- class별 train 2--86장, test 1--26장이다.
- train/test의 상대 경로, basename, 파일 내용 SHA-256 교집합은 모두 0이다.
- content 검증을 포함한 현재 split SHA-256은 train
  `0141af10b0ae34375f768bb4fc9fff8af06ae54d3274c379f9ea4fe690d26cf6`, test
  `4dd8c00cb855e24776ccb871420d896e302ccfd4ef5276b75a08c281d9d678b1`이다.
- 코드가 매 실행마다 file content SHA-256을 포함한 split SHA-256, 클래스별 표본 수,
  전체 class order, task별 원 ID/새 ID/class name을 `protocol_manifest.json`에 쓴다.

이 5,981/1,519 split의 생성 근거는 CDS 저장소나 논문에서 확인되지 않았다. 따라서
이 데이터로 얻는 결과는 **동일한 로컬 5-task protocol의 E0--E7 비교**이며, CVPR 표
수치의 엄밀한 재현으로 표시하면 안 된다.

## FCS PIC 대응

FCS 논문 식 (8)은 두 augmented view의 normalized feature를 쓰는 supervised
contrastive loss이고, 식 (9)은 key 집합에 old prototype을 추가한다. 공식 코드의
대응은 다음과 같다.

- `SupContrastive.forward`: positive mask와 negative mask를 사용한
  `log(1 + sum_neg * sum_pos)` 형태.
- `FCS._compute_il2a_loss`: query/key를 L2-normalize하고 `contrast_T`(공개 config에서
  보통 0.07)로 나눈다.
- incremental task에서는 stored prototype에 isotropic noise `_radius`를 더한 feature를
  key에 연결한다. 이 key는 old class label을 가지므로 current query에는 negative다.
- prototype은 calibration network를 통과한 후 classifier loss에도 사용된다. 이 경로와
  FCN을 제외한 단순 point-distance loss는 FCS 전체나 PIC 재현이 아니다.

이번 E0--E7에는 PIC/FCN을 넣지 않았다. 향후 별도 loss adapter로 추가할 수 있도록
Structure 원 loss와 separation loss를 `_init_train`에서 분리했다.

## Multi-center 구현

`local_stats[class_id]`는 각 component의 mean, full covariance, count, weight,
covariance별 tau, birth task, feature-space ID, rank, regularization/fallback을 저장한다.

- 현재 class의 deterministic K-means++만 사용한다. 별도 CPU generator, `n_init=10`,
  최대 100 iteration이다.
- 초기 K는 `min(requested_K, unique_count, floor(N/min_cluster_size))`이고, 빈 cluster나
  2개 미만 cluster가 남으면 K를 줄여 다시 수행한다. singleton class covariance는
  `0 + 1e-3 I`다.
- 원본 class mean/covariance는 SCC와 APR용으로 그대로 별도 유지한다.
- local APR는 old class mean으로 계산한 동일한 `Delta_k`와 동일한 covariance rank-1
  addition을 해당 class의 모든 local covariance에 한 번 적용한다.
- 보정 후 class-level moments와 local mixture moments를 강제로 일치시키지 않는다.
  SCC 입력도 local center로 바꾸지 않는다.
- replay 256개는 largest-remainder로 weight에 배분한다. 모든 component label은 원래
  class이고 출력 class 수는 늘지 않는다.
- `num_prototypes=1`이면 stage-2는 원본 single-Gaussian 경로를 사용한다.

float32 full covariance의 이론 저장량은 component당 약 2.25 MiB다. 200 classes,
3 components면 covariance만 약 1.32 GiB이고 full Cholesky cache도 old region 수에
비례해 추가된다. 실제 class/region/factor memory와 peak CUDA memory를 log에 남긴다.

## Separation과 epoch APR cache

region의 안정화된 full covariance에서 isotropic은 `trace(S)/d`, diagonal은 `diag(S)`,
full은 Cholesky solve를 사용한다. training iteration에서 inverse를 계산하지 않는다.
Cholesky가 jitter escalation 후에도 실패하면 full matrix eigenvalue clipping 후 다시
Cholesky하며 diagonal 실험으로 바꾸지 않고 fallback 횟수를 기록한다.

tau는 class/cluster 생성 당시 실제 current-task train feature의 squared Mahalanobis
distance 95 percentile을 isotropic/diagonal/full별로 따로 저장한다. 후속 APR에서
mean/covariance가 바뀌어도 tau는 고정한다. 이는 in-sample, birth-time 근사이며 후속
task에서 95% coverage를 보장하지 않는다.

각 incremental epoch 시작 시:

1. task 시작 때 저장한 이전 task final statistics를 source로 사용한다.
2. task 시작 시 두 adapter가 동일하므로 원본의 `now_task_mean/covs`를 reference로
   고정한다. 직전 student가 task 전환에서 다음 frozen teacher와 student에 복사됨을
   코드에서 확인했다.
3. 현재 epoch 시작 student로 동일 current-task train/test-transform feature 통계를
   no-grad로 계산한다.
4. source에 APR를 한 번 적용해 old cache를 만들며, 직전 epoch cache를 source로 쓰지
   않는다.
5. raw student feature에 `mean(max_j ReLU(1 - D2_j/tau_j))`를 추가한다.

cache 통계 추출 전후 RNG 상태와 train/eval mode를 복원한다. separation off 또는
`sep_lambda=0`이면 cache/factorization 경로 자체를 건너뛴다.

## 산출 파일

각 `outputs/imageneta/E*/seed_*/` 아래에 resolved config, source/run metadata, class order,
protocol manifest, sample-weighted epoch loss, K-means summary/detail/assignment, cache 및 APR
diagnostics, task/class accuracy, accuracy matrix, summary, training log와
`checkpoint_latest.pt`를 기록한다. 정확한 필드 정의는 `RECORDING_SCHEMA_KO.md`에 있다.
checkpoint에는 model, class/local statistics, optimizer, scheduler, RNG, APR source와
reference가 포함되어 epoch 경계 또는 task 경계에서 재개할 수 있다.

task/seed 요약에는 학습 시간, CUDA allocated/reserved peak, 메인 프로세스 RSS peak,
영구 class/local 통계 크기와 separation factor-cache 크기를 구분해 기록한다. RSS는
Linux `ru_maxrss` 기반이므로 DataLoader worker별 peak를 합산한 값은 아니다.
