# PTM-Guided 실험 실행 방법

## 1. 현재 준비 상태

- GPU 3: NVIDIA RTX A6000 48GB, 병렬 script dry-run 점검 시 유휴 상태
- GPU 4: NVIDIA RTX 6000 Ada Generation, 순차 실행 fallback에 사용 가능
- ImageNet-A processed CIL split: `${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a`
  - 200 classes, train 5,981 / test 1,519
- Cars-196: `${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196`
  - 기존 `${RESEARCH_WORKSPACE}/cds-main/data/cars196`을 가리키는 symlink
  - 196 classes, train 8,144 / test 8,041
- ViT-B/16-IN21K pretrained weight: Hugging Face/timm cache에 존재

ImageNet-R, OmniBenchmark, ObjectNet processed split은 아직 준비되지 않았다. Phase 1에는 필요 없다.

## 2. 환경

권장 환경은 전달된 `cds.yaml`이다.

마지막 검증 시점에는 `cds` 환경이 아직 없고 `base`, `clip` 환경만 있다. 따라서 아래 첫 줄은 현재 머신에서 한 번 실행해야 한다.

```bash
cd ${RESEARCH_WORKSPACE}/cdsp_handoff_20260813
conda env create -f cds.yaml   # 현재 머신에서 한 번만
conda activate cds
```

현재 설치된 별도 `clip` 환경에서도 import/unit/smoke 검증은 수행하지만, 논문 비교 실험은 `cds.yaml`의 PyTorch 2.4.1/timm 0.6.12 조합을 권장한다.

## 3. Dataset 환경변수

```bash
export IMAGENETA_ROOT=${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
export CARS196_ROOT=${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196
```

Phase 1 사전 검증:

```bash
python scripts/check_ptm_guided_datasets.py --phase1
```

## 4. 단일 method GPU 3 실행 명령

ImageNet-A 10-task Adapter Baseline부터 실행한다.

```bash
cd ${RESEARCH_WORKSPACE}/cdsp_handoff_20260813
conda env create -f cds.yaml   # 최초 1회; 이미 만든 뒤에는 이 줄을 생략
conda activate cds
export IMAGENETA_ROOT=${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
CUDA_VISIBLE_DEVICES=3 python main.py \
  --config exps_jsons/ptm_guided_imageneta_10task_adapter_baseline.json
```

`CUDA_VISIBLE_DEVICES=3` 때문에 프로세스 안에서는 physical GPU 3이 logical `cuda:0`이며 JSON의 `"device": ["0"]`이 맞다. Phase 1 전체 실행에는 아래 병렬 script를 사용한다.

## 5. Phase 1: GPU 3 method 병렬 실행

권장 script는 한 데이터셋의 세 method를 physical GPU 3에서 동시에 실행한다. 각 method 내부의 seed 1993, 1996, 1997은 순차 실행된다. ImageNet-A 세 method가 모두 정상 종료된 뒤 Cars-196 세 method를 시작한다.

```bash
cd ${RESEARCH_WORKSPACE}/cdsp_handoff_20260813
conda activate cds
bash scripts/run_phase1_gpu3_parallel.sh
```

터미널 출력에는 `[imageneta/adapter_baseline]`과 같은 접두사가 붙는다. 별도 console log는 `logs/ptm_guided/parallel_gpu3/<실행시각>/`에 저장된다. method 하나라도 실패하면 남은 method가 끝날 때까지 기다린 후 Cars-196으로 넘어가지 않고 종료한다.

실제 학습 없이 실행 구성을 확인하려면:

```bash
bash scripts/run_phase1_gpu3_parallel.sh --dry-run
```

GPU 4에서 모든 조건을 하나씩 실행하는 기존 순차 script도 보존되어 있다.

```bash
bash scripts/run_phase1_gpu4.sh
```

## 6. Smoke test

Production JSON을 바꾸지 않고 한 task, 한 epoch, train/stat/test 최대 두 batch, checkpoint off로 실행한다.

```bash
cd ${RESEARCH_WORKSPACE}/cdsp_handoff_20260813
conda activate cds
export CARS196_ROOT=${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196
CUDA_VISIBLE_DEVICES=4 python main.py --smoke-test \
  --config exps_jsons/ptm_guided_cars196_10task_ptm_place_move.json
```

Smoke 결과는 `metrics.json`에 `status=SMOKE_TEST`로 표시되어 mean ± std 집계에서 제외된다.

## 7. Full benchmark

나머지 processed split을 준비한 뒤 환경변수를 지정한다.

```bash
export IMAGENETR_ROOT=/path/to/imagenet-r
export OBJECTNET_ROOT=/path/to/licensed/objectnet-cil
export OMNIBENCHMARK_ROOT=/path/to/omnibenchmark
```

Phase 1에서 완료한 ImageNet-A/Cars-196 10-task를 중복 실행하지 않고, 남은 7개 main-method config를 순차 실행한다. 전체 benchmark를 처음부터 수행하려면 Phase 1 script를 먼저 실행한 뒤 아래 script를 실행한다.

```bash
cd ${RESEARCH_WORKSPACE}/cdsp_handoff_20260813
conda activate cds
bash scripts/run_full_benchmark_gpu4.sh
```

ObjectNet은 공식 라이선스에 model parameter tuning 제한이 있으므로 다운로드/실행 전에 processed CIL split의 사용 권한을 반드시 확인한다.

## 8. 결과 MD 갱신

```bash
python scripts/summarize_ptm_guided_results.py
```

출력: `RESULTS_PTM_GUIDED_VS_STRUCTURE.md`

## 9. Checkpoint 용량

Default config는 task별 resume checkpoint를 저장한다. 저장 공간이 부족하면 실행 전에 해당 JSON 또는 base config에서 다음 값을 `false`로 바꾼다.

```json
"save_checkpoints": false
```

Phase 1 전체는 task/seed/method 수가 많으므로 checkpoint 사용 시 수십 GB가 필요할 수 있다.
