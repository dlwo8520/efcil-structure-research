# PTM-Guided EFCIL 최종 인수인계

작성일: 2026-08-19 KST  
최종 전달 파일: `${RESEARCH_WORKSPACE}/PTM_.zip`

이 문서는 최종 인수인계 기준 문서다. 이전의 날짜가 붙은 PTM handoff 문서보다 이 문서를 우선한다.

## 1. 먼저 읽을 파일

1. `PTM_METHOD.md`: 방법론, 수식, 비교 조건
2. `RESULTS_PTM_GUIDED_VS_STRUCTURE.md`: 최종 수치, 모든 stage/task matrix, 진단, 그래프
3. `RUN_PTM_GUIDED.md`: 환경 구성과 실행 명령
4. `PTM_GUIDED_EXPERIMENT_PLAN.md`: 구현 배경과 protocol

## 2. 최종 완료 상태

총 30개 run이 모두 `status=COMPLETE`다.

| Dataset | Tasks | Methods | Seeds | 상태 |
|---|---:|---|---|---|
| ImageNet-A | 5 | PTM Place+Move | 1993/1996/1997 | 완료 |
| ImageNet-A | 10 | Adapter / Place Only / Place+Move | 각 1993/1996/1997 | 완료 |
| ImageNet-A | 20 | PTM Place+Move | 1993/1996/1997 | 완료 |
| Cars-196 | 5 | PTM Place+Move | 1993/1996/1997 | 완료 |
| Cars-196 | 10 | Adapter / Place Only / Place+Move | 각 1993/1996/1997 | 완료 |
| Cars-196 | 20 | PTM Place+Move | 1993/1996/1997 | 완료 |

ObjectNet, OmniBenchmark, ImageNet-R 10-task는 processed dataset이 준비되지 않아 아직 실행하지 않았다.

## 3. 핵심 결과

모든 값은 3 seeds의 mean ± sample standard deviation이다.

| Dataset | Tasks | Method | Last-Acc | Inc-Acc | Structure 대비 Last | Structure 대비 Inc |
|---|---:|---|---:|---:|---:|---:|
| ImageNet-A | 5 | PTM Place+Move | 62.76 ± 0.94 | 70.08 ± 0.97 | -3.89 | -4.01 |
| ImageNet-A | 10 | PTM Place+Move | 58.15 ± 1.87 | 66.99 ± 2.01 | -6.43 | -4.85 |
| ImageNet-A | 20 | PTM Place+Move | 51.53 ± 1.58 | 61.27 ± 2.50 | -10.77 | -8.63 |
| Cars-196 | 5 | PTM Place+Move | 70.43 ± 0.63 | 81.10 ± 0.45 | -13.09 | -6.15 |
| Cars-196 | 10 | PTM Place+Move | 57.97 ± 2.29 | 72.37 ± 1.71 | -22.26 | -13.16 |
| Cars-196 | 20 | PTM Place+Move | 43.76 ± 3.22 | 60.04 ± 1.65 | -22.38 | -14.07 |

10-task 내부 ablation에서는 Cars-196의 PTM Place+Move가 Adapter Baseline보다 Last-Acc +1.43%p, Inc-Acc +0.72%p 높고 forgetting은 2.55%p 낮았다. ImageNet-A에서는 Place+Move가 Adapter Baseline보다 Last-Acc -0.64%p, Inc-Acc -0.16%p였다.

해석상 중요한 점은 PTM constraint가 일부 조건에서 Adapter Baseline 대비 drift/forgetting을 개선했지만, 모든 완료 protocol의 최종 성능은 Structure 논문 reference보다 낮다는 것이다.

## 4. 그래프

`figures/ptm_guided/`에 다음 결과가 있다.

- `all_tasks_final_vs_structure.png`: ImageNet-A/Cars-196 5/10/20-task 최종 성능 vs Structure
- `phase1_final_vs_structure.png`: 10-task 세 method ablation vs Structure
- `*_vs_adapter_task_heatmap.png`: 10-task PTM method vs Adapter Baseline
- `*_5task_*_retention_heatmap.png`: 5-task 학습 이후 유지·망각
- `*_20task_*_retention_heatmap.png`: 20-task 학습 이후 유지·망각

초록/빨강의 기준은 그래프마다 다르다.

- 최종 막대그래프: Structure 대비 향상/하락
- 10-task heatmap: Adapter Baseline 대비 향상/하락
- 5/20-task heatmap: 각 task 최초 정확도 대비 이후 개선/망각

## 5. 코드 지도

| 경로 | 역할 |
|---|---|
| `main.py`, `trainer.py` | config 실행, seed/task loop, 진행률/ETA/시간/GPU memory |
| `models/PTM_guided_adapter.py` | PTM-guided learner |
| `utils/ptm_guidance.py` | frozen PTM prototype와 anchor memory |
| `utils/ptm_metrics.py` | 정형 결과 기록 |
| `data/data_manager.py` | dataset/task protocol |
| `exps_jsons/ptm_guided_*.json` | 실행 config |
| `scripts/run_phase1_gpu3_parallel.sh` | GPU 한 장에서 method 3개 병렬 실행 |
| `scripts/check_ptm_guided_datasets.py` | dataset 사전 검사 |
| `scripts/summarize_ptm_guided_results.py` | MD와 PNG 재생성 |
| `references/structure_cvpr2026_results.json` | Structure reference와 출처 |

## 6. 환경과 외부 데이터

권장 환경:

```bash
cd /path/to/cdsp_handoff_20260813
conda env create -f cds.yaml
conda activate cds
```

현재 머신의 dataset 경로:

```bash
export IMAGENETA_ROOT=${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
export CARS196_ROOT=${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196
python scripts/check_ptm_guided_datasets.py --phase1
```

데이터셋, ViT pretrained cache는 ZIP에 포함되지 않는다.

## 7. 실행과 결과 재생성

GPU 3에서 Phase 1 전체 실행:

```bash
PHYSICAL_GPU=3 bash scripts/run_phase1_gpu3_parallel.sh
```

단일 config 실행:

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u main.py \
  --config exps_jsons/ptm_guided_imageneta_5task_ptm_place_move.json
```

결과 MD와 모든 그래프 재생성:

```bash
python scripts/summarize_ptm_guided_results.py
```

집계기는 `metrics.json`에서 `status=COMPLETE`이고 task 수가 config와 일치하는 run만 사용한다.

## 8. 결과 파일 구조

각 run은 다음 경로에 있다.

```text
results/ptm_guided/<dataset>/<N>_tasks/<method>/seed_<seed>/
├── metrics.json
├── task_metrics.csv
├── task_accuracy_matrix.csv
├── class_accuracy.csv
├── loss_components.csv
├── ptm_geometry_drift.csv
└── anchor_drift.csv
```

`metrics.json`은 최종 결과와 모든 task history를 포함한다. CSV는 독립 분석과 재시각화에 사용할 수 있다.

## 9. ZIP 포함 및 제외 범위

포함:

- 실행 가능한 전체 source code와 JSON config
- `PTM_METHOD.md`, `RESULTS_PTM_GUIDED_VS_STRUCTURE.md`, `PTM_HANDOFF.md`
- 모든 완료 run의 JSON/CSV 정형 결과
- PNG 그래프와 실행 로그
- environment, run script, reference

제외:

- task별 `.pth` checkpoint 330개, 약 117.42 GiB
- dataset과 pretrained model cache
- `.git`, `__pycache__`, `.pyc`, 기존 ZIP
- 이전 중간상태 handoff인 `HANDOFF_PTM_GUIDED_20260814.md`

checkpoint를 제외해도 최종 수치, task matrix, class accuracy, loss/drift 분석 및 MD/PNG 재생성에는 문제가 없다. 다른 머신에서 중단 task를 resume하려면 checkpoint를 별도로 전달해야 한다.

## 10. 전달 후 검증 체크리스트

1. `PTM_METHOD.md`와 결과 문서를 읽어 비교 기준을 확인한다.
2. `cds.yaml`로 환경을 준비한다.
3. dataset 환경변수를 설정하고 dataset check를 통과시킨다.
4. `python scripts/summarize_ptm_guided_results.py`로 MD/PNG가 재생성되는지 확인한다.
5. 새 실험 전 seed, task protocol, 저장 공간과 `save_checkpoints` 설정을 확인한다.
6. Structure 수치는 repository 재실행값이 아니라 논문 reference임을 구분한다.

## 11. Reference

- Xin Zhang, Liang Bai, Guanchao Wang, and Xian Yang. “Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure.” *CVPR 2026*. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Exemplar-Free_Class_Incremental_Learning_via_Preserving_Class-Discriminative_Structure_CVPR_2026_paper.html)

