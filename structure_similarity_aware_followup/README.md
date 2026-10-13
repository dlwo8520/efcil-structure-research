# Structure Similarity-Aware Follow-up

`structure_similarity_aware_followup_experiments.md`의 B0/B1/S0-S4를 구현한 독립 extension이다. 다음 두 폴더는 읽기 전용 의존성으로 사용하며 수정하지 않는다.

- `${RESEARCH_WORKSPACE}/structure_multicenter_separation`
- `${RESEARCH_WORKSPACE}/structure_e3_followup`

## 구현된 실험

| ID | 실제 loss 변경 |
|---|---|
| B0 | Structure baseline |
| B1 | 기존 diagonal E3 + AM-Drift |
| S0 | B0와 같은 학습, old-model pair similarity/confusion/forgetting logging |
| S1 | AM-Drift tau × similarity relaxation |
| S2 | S0 p75 이상 class는 adapted relative margin, 나머지는 AM-Drift absolute branch |
| S3 | S2 + per-sample similarity-aware SCC |
| S4 | sigmoid gate로 AM-Drift absolute와 relative margin 혼합 |

Structure의 실제 E3가 dimensionless `relu(1-min(D²/tau))`를 사용하므로 S1-S4의 absolute branch도 이 스케일을 유지한다. 문서의 raw hinge와 다른 이 선택은 B1과 loss scale을 통제하기 위한 것이며, 원 논문 구현과 adaptation의 구분은 [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md)에 기록했다.

S2/S4의 hard old `k*`는 frozen previous-task space에서 선택한다. Margin의 실제 cosine은 현재 feature space를 맞추기 위해 epoch current prototype과 APR-corrected old prototype을 사용한다. S4 gate는 이 hard pair에 하나만 적용한다.

## 환경 및 빠른 검사

```bash
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd ${RESEARCH_WORKSPACE}/structure_similarity_aware_followup

python run.py \
  --config configs/B0.json configs/B1.json configs/S0.json configs/S1.json configs/S2.json configs/S3.json configs/S4.json \
  --gpu 1 \
  --validate-only

python scripts/smoke_test.py
```

Smoke test는 dataset/pretrained ViT를 읽지 않는 CPU 축소 검사다. S1-S4 forward/backward, S3 학습 epoch, 필수 CSV 및 평가 logging을 확인한다.

실제 ImageNet-A/ViT 결합 확인용 `configs/smoke_S4.json`은 protocol을 유지한 5 tasks×1 epoch의 비과학적 smoke 전용 설정이며 main 결과에 포함하지 않는다.

## GPU 1 pilot 실행

S2-S4의 beta는 같은 seed S0 결과가 필요하다. 아래 runner가 `B0 → B1 → S0 → S1 → S2 → S3 → S4` 순서를 보장한다.

```bash
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd ${RESEARCH_WORKSPACE}/structure_similarity_aware_followup

python scripts/run_experiments.py \
  --stage pilot \
  --gpu 1 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
```

`pilot`은 seed 1993만, `final`은 1993/1996/1997을 실행한다. 완료된 run은 건너뛰고 checkpoint가 있는 미완료 run은 기본적으로 resume한다.

### 백그라운드

```bash
mkdir -p logs
nohup python scripts/run_experiments.py \
  --stage pilot \
  --gpu 1 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
  > logs/pilot_gpu1.log 2>&1 &
echo $! > logs/pilot_gpu1.pid
```

```bash
tail -f logs/pilot_gpu1.log
python scripts/status.py
nvidia-smi -i 1
```

## 최종 3-seed 실행

```bash
python scripts/run_experiments.py \
  --stage final \
  --gpu 1 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
```

주요 결과는 `outputs/imageneta/<METHOD>/seed_<SEED>/`에 저장된다. 각 run에는 source/config/command snapshot, pair similarity, confusion, class forgetting, method별 loss 진단, feature/geometry/gradient 진단이 포함된다.
