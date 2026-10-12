# 실제 실험 흐름

## 1. Queue

`run_gpu1.sh`가 `run_queue.py --gpu 1`을 호출한다. queue는 exclusive lock을 잡은 뒤 protocol마다 `run_structure.py`, `run_audit.py`를 순서대로 실행한다.

## 2. Training

`run_structure.py`는 `oracle_audit.config.protocol_config()`로 selective sweep protocol과 vanilla Structure method 설정을 합친다. 이후 base repository의 `trainer.CDS_train()`을 호출하되 model factory만 `AuditSnapshotLearner`로 교체한다.

`AuditSnapshotLearner`가 원 Structure와 다르게 수행하는 관찰 작업은 다음 두 가지다.

1. 원 `_stage2_compact_classifier()` 호출 직전에 classifier state와 RNG를 `pre_ca_classifier/task_XX.pt`로 저장한다.
2. 원 task-end checkpoint를 `task_checkpoints/task_XX.pt`로 보존한다.

실제 학습, APR, SCC, classifier adaptation 계산은 base Structure 코드에 위임된다.

정식 공통 설정:

```text
seed = 1993
GPU = physical 1, process logical cuda:0
init epochs = 30
incremental epochs = 15
classifier adaptation epochs = 5
synthetic samples per class = 256
optimizer = SGD
init_lr = 0.01
weight_decay = 0.0005
scale = 20
APR = true
SCC = true
multi_center = false
```

Protocol partition:

```text
IA5  = [40, 40, 40, 40, 40]
CAR5 = [40, 39, 39, 39, 39]
```

## 3. Oracle statistics

`run_audit.py` → `oracle_audit.pipeline.audit_protocol()`이 task checkpoint를 순서대로 읽는다.

각 checkpoint에서 model을 freeze하고 `eval()`/`torch.no_grad()`로 해당 시점의 train/test sample을 다시 forward한다. train split feature로 class별 current oracle mean과 centered covariance를 계산한다.

```text
oracle mean = current-checkpoint feature의 class sample mean
oracle covariance = unbiased centered sample covariance + matched jitter
```

old raw sample은 이 post-hoc statistics 계산에만 사용한다.

## 4. APR ablation

old class별로 다음 세 값을 비교한다.

```text
pre-APR = checkpoint.task_source_means
APR     = checkpoint.class_means
oracle  = current checkpoint에서 old train sample을 forward한 mean
```

연속 checkpoint가 있으므로 oracle one-step drift, APR one-step drift, one-step error, cumulative error도 계산한다.

## 5. A–D classifier replacement

final task에서만 수행한다.

```text
A Structure          old mean=Structure, old cov=Structure
B Oracle Mean        old mean=oracle,    old cov=Structure
C Oracle Covariance  old mean=Structure, old cov=oracle
D Full Oracle        old mean=oracle,    old cov=oracle
```

current/new class statistics는 네 조건 모두 checkpoint Structure statistics로 고정한다. classifier 초기화와 RNG는 `pre_ca_classifier/task_05.pt`에서 읽는다. 네 조건은 epoch, sample 수, SGD, LR, weight decay, cosine scheduler, normalization, evaluation feature를 공유한다.

`oracle_head.py`는 checkpoint의 여러 task head weight를 하나의 `DiagnosticCosineHead.weight` tensor로 합쳐 별도 head-only adaptation을 수행한다. 이 구현 차이가 exact reproduction 불일치의 주요 검토 대상이다.

## 6. Evaluation

같은 frozen test feature에 대해 다음을 저장한다.

- seen/cumulative, old, new accuracy
- old-only, new-only accuracy
- old→new, old→wrong-old, new→old, new→wrong-new
- new sample의 strongest-old margin
- old prototype error의 classifier boundary 방향 성분

## 7. Output

CSV는 `run_artifacts/full_outputs/{IA5,CAR5}/seed_1993/`에 있다. 보고서는 해당 폴더의 `structure_prototype_oracle_audit.md`다.
