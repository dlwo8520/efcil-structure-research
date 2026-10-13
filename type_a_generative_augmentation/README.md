# Type A current-task 생성 증강 pilot

사용자 명세(2026-10-08)를 별도 폴더에 구현한 코드다. 기존 vanilla Structure, B1 실험과 결과는 수정하지 않는다. 이 문서는 실행 안내이며 실험 결과 보고서가 아니다.

현재 모드는 사용자 변경 요청에 따른 **사람 검수 없는 exploratory pilot**이다. 기존 audit·1,192개 생성 artifact는 그대로 재사용한다. 원 generation recipe의 `config.json`은 보존하고, 현재 운영 모드는 `exploratory_config.json`, 자동 규칙은 `sanity_policy.json`으로 별도 고정한다. Generated image의 label fidelity, 특히 CAR의 차종·연식 일치는 검증된 것으로 주장하지 않는다.

## 실행 순서

1. `cds` 환경 활성화 → 생성 전용 가상환경 설치.
2. GPU 번호를 입력해 audit → parent 고정 → 생성 순차 실행. **생성 후 자동 종료하며 학습하지 않는다.**
3. 모든 생성 이미지에 사전 고정한 자동 sanity 규칙을 적용한다. Contact sheet는 결과 확인용 artifact일 뿐 선택에 사용하지 않는다.
4. 동일하거나 다른 GPU 번호를 입력해 R/S/G 및 GM continuation을 순차 실행. 완료 후 `final_report.md` 생성.

### 1. 환경 준비 + audit/생성 백그라운드 실행

```bash
cd ${RESEARCH_WORKSPACE}/type_a_generative_augmentation
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
bash setup_generator_env.sh
mkdir -p logs
read -rp '사용할 GPU 번호: ' PILOT_GPU
nohup bash run_feasibility.sh "$PILOT_GPU" > "logs/feasibility_gpu${PILOT_GPU}.log" 2>&1 &
echo $! > "logs/feasibility_gpu${PILOT_GPU}.pid"
tail -f "logs/feasibility_gpu${PILOT_GPU}.log"
```

`tail`에서 Ctrl+C는 로그 보기만 종료한다. 위 코드는 다른 실험을 중단하거나 GPU를 자동 예약하지 않는다. 해당 GPU가 비었는지 사용자가 확인한 뒤 실행한다. GPU 물리 번호는 `CUDA_VISIBLE_DEVICES`로 지정하고 내부에서는 `cuda:0`만 사용한다. Queue lock은 이 새 폴더에서의 중복 실행만 방지한다.

원 `cds`에는 diffusers/transformers/accelerate가 없다. 설치 스크립트는 `.venv_generator`에만 패키지를 설치하며 원 `cds`의 PyTorch를 공유한다. 설치에는 네트워크가 필요할 수 있지만 이미지 생성은 로컬 모델만 읽으며 다운로드하지 않는다. GPU 학습은 원 `cds`에서 실행한다.

### 2. 자동 sanity — 사람 검수 불필요

```bash
cd ${RESEARCH_WORKSPACE}/type_a_generative_augmentation
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
.venv_generator/bin/python -u run.py --stage auto-sanity --gpu 4
python run.py --stage quality-check
```

`run_matched.sh GPU번호`는 이 단계를 학습 전에 자동 실행한다. 이미 실행된 sanity 결과는 입력·정책·산출물·채택 이미지 hash를 확인하고 재사용한다. CPU로 수행하려면 `--gpu`를 생략한다. 이 단계는 생성 전용 `.venv_generator`를 사용하지만 본 학습은 원 `cds` 환경이다.

- 생성 실패/원 safety checker 차단, 파일 없음·decode 실패, 예상 외 크기, generated hash 변경: 제외.
- 동일 parent를 생성 때와 같은 512×512 padding으로 맞춘 뒤 near-duplicate: block SSIM≥.995 **AND** MAE≤.01 **AND** dHash distance≤2.
- 극단적 단색 붕괴: 평균 channel std≤.005 **AND** edge energy≤.002.
- 거친 semantic mismatch proxy: frozen CLIP global cosine<.25 **AND** 중심 70% crop cosine<.25.
- 극단적 구조 소실 proxy: 중심 cosine<.35 **AND** edge ratio<.1 **AND** 중심 block SSIM<.2, 원본 중심 edge energy≥.02.

규칙은 downstream 결과나 similarity 분포를 보고 조정하지 않는다. 분포 상·하위 몇 %를 버리는 방식도 아니다. 생성에 이미 사용한 SD1.5의 frozen CLIP vision/projection weights를 재사용하며 별도 모델 다운로드·학습은 없다. Baseline class confidence, test, Type A sample, 인간의 개별 선택을 사용하지 않는다. `quality_review.csv`의 수동 accept/reject 값은 현재 모드에서 완전히 무시하며 해당 파일을 조작하지 않는다.

고정 규칙을 통과한 이미지는 모두 사용한다. 제외 parent는 R/S/G auxiliary bank에서 공통으로 제외해 비교를 맞춘다. 전체 real anchor train은 그대로다. 자동 제외는 disk에서 삭제하는 것이 아니라 학습 eligibility만 바꾸며 모든 생성 artifact는 보존한다. Current class의 usable parent가 0개이면 조용히 class를 버리지 않고 HOLD한다. 원 real parent가 손상되면 real anchor도 학습 불가능하므로 HOLD한다.

`READY_EXPLORATORY`는 기술적 실행 가능 상태이지 label 검증 PASS가 아니다. 모든 record에 label-fidelity-unverified를 명시한다. 위 object/structure-loss 규칙은 segmentation/object-preservation 인증이 아니며, 객체 소실·극단 변형도 일부 놓칠 수 있다. 특히 CAR fine-grained identity 보존은 보장하지 않는다. 자동 filtering 자체의 distribution bias도 완전히 제거되었다고 주장하지 않는다.

기록: `automatic_sanity.csv`(모든 parent/generated pair), `automatic_sanity_summary.json`(전체/채택/제외 similarity 통계), `automatic_parent_counts.csv`(class별 공통 수), `automatic_sanity_report.md`(규칙·제외 사유·한계). Original `generation_quality.md`의 HUMAN HOLD는 과거 시도 기록이며 현재 실행 gate가 아니다. Contact sheet는 원본 그대로 보존한다.

### 3. 학습 + 보고서 백그라운드 실행

```bash
cd ${RESEARCH_WORKSPACE}/type_a_generative_augmentation
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
mkdir -p logs
read -rp '학습에 사용할 GPU 번호: ' PILOT_GPU
nohup bash run_matched.sh "$PILOT_GPU" > "logs/matched_gpu${PILOT_GPU}.log" 2>&1 &
echo $! > "logs/matched_gpu${PILOT_GPU}.pid"
tail -f "logs/matched_gpu${PILOT_GPU}.log"
```

IA10 → CAR10 순서로, 각각 T1 R/S/G(GM 통계 복사) → T9/T10 R/S/G(GM 통계 분기) → GM T10을 순차 실행한다. V는 기존 checkpoint/trajectory 참조이므로 재학습하지 않는다. GM은 T1/T9에서 별도 representation 학습을 하지 않지만, T9의 별도 CA와 real-only T10 continuation은 수행한다.

완료된 branch는 보존하고 건너뛴다. T9 task-end checkpoint가 있으면 T10부터 재개한다. Epoch 단위 재개는 구현하지 않았다. Epoch 중 중단된 task는 해당 task를 처음부터 재실행한다. 이미 저장된 immutable task snapshot은 덮어쓰지 않는다. 폴더 전체를 다른 곳으로 옮기는 실행 방식은 현재 지원하지 않는다.

보고서만 다시 생성:

```bash
python run.py --stage report
```

## 실험 정의와 원 코드 사용 범위

| 조건 | T1/T9 학습 source | Current statistics | T10 |
|---|---|---|---|
| V | 원 vanilla | 원 real | 기존 결과 참조 |
| R | real anchor + ordinary real parent | 전체 real train | real-only |
| S | real anchor + strong real parent | 전체 real train | real-only |
| G | real anchor + generated parent | 전체 real train | real-only |
| GM | G representation 공유 | current real+generated | 독립 real-only continuation |

원 `models.CDS_adapter.Learner`의 optimizer parameter group, scheduler, freeze 정책, CosFace CE, SCC, APR, class-statistics, CA를 상속한다. CE는 real/aux에 각각 0.5, SCC는 real에만 적용한다. Sequential backward accumulation으로 activation memory는 B 단위, effective image 수는 2B, update 수는 원 U를 유지한다. Real anchor transform RNG, epoch shuffle, auxiliary parent 선택을 조건별로 동일하게 고정하고 hash·샘플 수·update 수·LR로 검증한다. 원 per-epoch test 평가는 생략하며 task-end 평가만 한다. Test로 early stopping/recipe 조정을 하지 않는다.

S는 원 RandomResizedCrop/RandomHorizontalFlip 뒤에 RandAugment(2,9), ToTensor/원 Normalize 뒤에 RandomErasing(p=.25)를 적용한다. G에는 R과 같은 공통 input transform을 적용한다. 원 구조의 optimizer 설정은 `training/.../resolved_config.json`에 복사한다.

Strong transform 순서 오류는 수정했다. 실제 Structure의 `_train_trsf`에 ToTensor가 이미 포함되어 있어 예전 구현은 RandAugment를 float tensor 뒤에 붙였다. 현재는 `base.trsf`에서 tensor conversion 위치를 찾아 그 **앞**에 RandAugment를 삽입한다. Crop/flip/normalization, real anchor, R/G의 ordinary transform, parent schedule, optimizer budget은 그대로다. 실제 IA/CAR 구성과 강제 Solarize 회귀 테스트를 추가했다. 완료된 R checkpoint는 재학습하지 않고 보존하며 미완료 S task부터 시작한다.

T1은 원 Structure에서도 CA를 실행하지 않는다. 따라서 GM T1은 G와 **head와 예측까지 동일**하고 저장 moments만 달라진다. T1에 CA를 새로 추가하지 않았다. T9에서는 G의 pre-CA bundle, head, old APR moments, CPU/CUDA RNG를 공유하고 current moments만 mixed로 바꿔 원 CA(256/class, 원 5 epoch)를 두 번 실행한다. GM의 pooled covariance는 between-source mean scatter를 포함하고 concatenation covariance와 검산한다. Original mean scaling/jitter 및 sample 수는 그대로다. APR displacement anchor는 real로 유지한다.

G/GM representation 일치는 T1/T9의 fork 시점에만 보장한다. 다른 stored memory에서 이어가는 T10에는 SCC/APR/CA 효과로 representation도 달라질 수 있다. 생성 bank를 learner 부모 목록에서 T10 시작 전에 제거한다. Offline 생성 이미지 보관은 재현 artifact이며 다음 task replay 입력이 아니다. Gaussian feature replay는 그대로 사용하므로 엄밀한 replay-free라고 부르지 않는다.

## Generator 사전 고정

- Model: `runwayml/stable-diffusion-v1-5`.
- Revision: `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
- Local: `${RESEARCH_WORKSPACE}/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14`.
- FP16, 원 checkpoint scheduler 유지, img2img strength=.2 / inference steps=30 / guidance=7.5 / 512×512.
- Parent는 current real train에서 class당 최대 16개, 1 variation/parent. T1과 T9만 생성한다. Generic object-preserving prompt에 고정 lighting/background 변화를 주며 미래 class 이름·test·Type A sample은 사용하지 않는다.
- Parent seed=20261008; 각 생성의 seed/prompt/path/hash를 manifest에 기록한다. Safety checker는 유지하며 차단 이미지는 기술적 제외한다.
- Generator 파일별 SHA256/bytes와 실제 scheduler·패키지 버전을 실행 시 기록한다. 로컬 파일 존재는 확인했지만, **실제 pipeline 로드와 이미지 생성은 아직 검증하지 않았다.**

일반 img2img는 객체 보호 mask를 적용한 background-only 편집과 다르다. Prompt/낮은 strength가 label 보존을 보장하지 않기 때문에 사람 검수가 필수다. 30은 전체 scheduler inference step 설정이며 strength .2에서는 pipeline이 그 일부만 실행한다.

API 근거: [Diffusers 0.30.3 img2img](https://huggingface.co/docs/diffusers/v0.30.3/en/api/pipelines/stable_diffusion/img2img), [Torchvision RandAugment](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandAugment.html).

## Audit와 평가

Fixed vanilla retention A=IA10 327/CAR10 907. T1..T9 birth post-CA checkpoint의 local/all prediction과 margin, balanced current real train support를 기록한다. n=min(64,min class count), k=min(5,n−1), subsample seed1993/1994/1995, train reference는 LOO다. n<2는 support 비교 불가로 명시한다. 같은 class birth-correct comparator가 없으면 누락 class를 표시한다. A-within/A-cross 및 support는 diagnostic-only이며 generation 대상 선택에 연결하지 않는다.

`parent_distribution.csv`는 이미 고정한 parent에 대한 train-only feature cluster 분포(구면 k-means K≤4)다. 과대표집 여부를 기술할 뿐 이를 보고 parent를 다시 뽑지 않는다.

평가: seen/macro/old/new, new local, T9 cohort after T10, T9 correct→wrong/wrong→correct, fixed A/B1/B2/C 및 original-final-correct repair/harm/net, CE/margin, 오류의 old/new 방향과 raw N. V trajectory에는 logits가 없으므로 V의 CE/margin/local은 ‘없음’으로 기록한다. 다른 조건들은 scaled cosine score로 CE/margin을 계산하며 training CosFace margin은 평가에 적용하지 않는다.

Primary G−S; secondary G−R; statistics GM−G. G−V는 추가 exposure/compute가 다르므로 생성 source의 독립 증거로 해석하지 않는다. 두 dataset 모두 명세의 수치 기준을 만족해야 full trajectory 확장을 추천한다. 명세의 ‘G가 R도 넘어야 함’은 실행 전에 **T1/T9 birth 평균 G−R>0 및 T10 seen G−R>0**으로 구체화했다. 실패는 현재 recipe의 STOP이며 모든 coverage/generation 가설의 반박이 아니다. GM−G가 좋아도 G의 expansion 기준을 우회하지 않는다.

## 산출물 / 코드 위치

| 파일 | 용도 |
|---|---|
| `audit.py` | Birth support, class-matched/LOO audit |
| `generation.py`, `automatic_sanity.py` | Train parent, diffusion 생성, 자동 exploratory gate |
| `matched_views.py`, `learner.py`, `training.py` | Matched stream, 원 Structure 상속, GM fork/continuation |
| `moments.py` | Unbiased mixed covariance |
| `parent_distribution.py`, `results.py` | Descriptive train clusters, paired 결과 및 보고서 |
| `birth_support_audit.csv`, `audit/{protocol}/report.md` | Audit 실행 후 생성 |
| `generation_manifest.csv`, `quality_review.csv`, `generation_quality.md` | 생성/과거 human-gate 기록 보존; 현재 수동 선택에 미사용 |
| `exploratory_config.json`, `sanity_policy.json`, `automatic_sanity_report.md` | 현재 mode, 고정 자동 규칙, 기술적 검사 결과 |
| `config.json`, `generator_resource.json`, `generator_runtime_config.json` | 고정 recipe/실제 weights·scheduler |
| `training/{protocol}/{T1,T9_T10}/{R,S,G,GM}/` | 독립 checkpoints / trace / predictions |
| `per_sample_predictions.csv`, `metrics.json` | 최종 paired 결과 |
| `statistics_comparison.json`, `resource_usage.json`, `final_report.md` | Fork 검증, 비용, 최종 보고서 |

G branch resource time은 통계 추출과 GM fork CA 비용도 포함한다. GM T10 continuation 비용은 별도로 기록한다. 각 class generated fraction과 mean/covariance 변화도 저장한다.

현재 workspace root는 Git repository가 아니므로 무단으로 repository/branch를 만들지 않았다. 요청 branch 이름 `exp/type-a-current-generative-augmentation`은 config metadata에 기록하고 새 폴더로 격리했다.

## CPU 검증

```bash
cd ${RESEARCH_WORKSPACE}
CUDA_VISIBLE_DEVICES='' ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python -m unittest discover -s type_a_generative_augmentation/tests -v
```

합성 테스트 및 import 회귀 검증: 동일 R/S/G 학습량·anchor/parent hash, real-only SCC, G/GM 동일 CA 난수와 state 복원, mixed covariance의 source scatter 및 m=1, 미검수 gate 차단 및 이미지 변경 차단, fixed A harm, retention/CE 집계, support LOO, 실제 실행 폴더의 fresh-process 원 Structure data/learner import. 실제 dataset/GPU/diffusion 성공이나 label fidelity를 이 테스트만으로 주장하지 않는다.

### Import 오류 수정 기록

처음 제공한 local `data.py`가 원 Structure의 namespace package `data/`를 가려서 `No module named 'data.data'; 'data' is not a package`가 발생했다. Local module을 `matched_views.py`로 변경했다. 기존 Structure source는 수정하지 않았다. Preflight에 실제 data manager/class-order 및 learner import 검증을 추가했고, wrapper는 생성 전용 환경의 pipeline import도 CPU에서 검사한다.

`cds`의 preflight에서 generation dependencies가 False로 표시되어도 생성 전용 환경이 따로 있으므로 그 자체는 설치 실패가 아니다. 바로 다음 줄의 `Generator environment/imports PASS`가 생성 환경 검증이다. 이미 생성이 완료됐다면 재생성/재설치 없이 `run_matched.sh`로 이어간다.
