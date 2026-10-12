# PTM-Guided EFCIL 실험 인수인계

작성 시각: 2026-08-14 14:00 KST  
프로젝트 기준 경로: `${RESEARCH_WORKSPACE}/cdsp_handoff_20260813`

## 1. 전달 목적

이 패키지는 Structure(CVPR 2026) 공식 CDS 코드 기반으로 구현한 exemplar-free class-incremental learning 실험을 다른 환경에서 검토·재현·연장하기 위한 전달본이다. 다음 항목을 함께 보존한다.

- PTM-Guided Adapter 구현 전체와 실행 config
- ImageNet-A/Cars-196에서 생성된 정형 결과(`metrics.json`, stage/task matrix CSV, drift/loss CSV)
- 실행 로그와 결과 그래프
- 수식, 방법론 차이, task별 결과가 포함된 `RESULTS_PTM_GUIDED_VS_STRUCTURE.md`
- Structure 논문 reference 수치와 출처

데이터셋, pretrained cache, task별 `.pth` checkpoint는 용량 때문에 ZIP에 포함하지 않는다.

## 2. 실험 상태

### 완료

- ImageNet-A 10-task: Adapter Baseline / PTM Place Only / PTM Place+Move, 각 3 seeds
- Cars-196 10-task: Adapter Baseline / PTM Place Only / PTM Place+Move, 각 3 seeds
- 위 Phase 1 총 18 runs의 stage별 결과, task accuracy matrix, loss/drift 진단, 시간/GPU memory 측정

### 패키징 시 실행 중

GPU 3에서 아래 두 config가 동시에 실행 중이었다.

- ImageNet-A 5-task `ptm_place_move`
- ImageNet-A 20-task `ptm_place_move`

최종 전달 snapshot(14:00 KST)에는 5-task와 20-task 모두 seed 1993/1996 완료분이 들어 있고, 두 config의 seed 1997은 실행을 시작한 상태였다. 전달 ZIP은 실행 디렉터리를 직접 압축하지 않고 별도 staging snapshot을 만든 뒤 집계하므로, `RESULTS_PTM_GUIDED_VS_STRUCTURE.md`에는 snapshot 안에서 `status=COMPLETE`인 run만 반영된다. 진행 중 결과는 최종 성능 집계에서 제외되며, 각 `metrics.json`의 `status`가 판정 기준이다.

### 미완료 또는 데이터 미준비

- Cars-196 5-task / 20-task `ptm_place_move`
- ObjectNet 10-task `ptm_place_move`
- OmniBenchmark 10-task `ptm_place_move`
- ImageNet-R 10-task `ptm_place_move`

정확한 snapshot별 완료 seed는 결과 문서의 `최종 결과 요약`과 `아직 실행되지 않은 실험` 절을 확인한다.

## 3. 방법론 요약

Frozen ViT-B/16-IN21K backbone 위의 bottleneck adapter만 학습한다. 공통 classification loss에 frozen PTM prototype placement와 old-anchor functional movement constraint를 단계적으로 더한다.

$$
\begin{aligned}
\text{Adapter Baseline}:\quad &\mathcal L=\mathcal L_{\mathrm{cls}},\\
\text{PTM Place Only}:\quad &\mathcal L=\mathcal L_{\mathrm{cls}}+\lambda_{\mathrm{place}}(e)\mathcal L_{\mathrm{place}},\\
\text{PTM Place+Move}:\quad &\mathcal L=\mathcal L_{\mathrm{cls}}+\lambda_{\mathrm{place}}(e)\mathcal L_{\mathrm{place}}+\lambda_{\mathrm{move}}\mathcal L_{\mathrm{move}}.
\end{aligned}
$$

Placement는 new class의 adapted feature mean을 adapter-off frozen PTM prototype 쪽으로 유도한다.

$$
p_c=\operatorname{Norm}\left(\frac{1}{|D_c|}\sum_{x\in D_c}f_{\theta_0}(x;\mathrm{adapter\ off})\right),\qquad
\mathcal L_{\mathrm{place}}=\frac1{|C_B|}\sum_{c\in C_B}\left(1-\cos(\bar z_c,p_c)\right).
$$

Movement는 old class/layer 평균 anchor에서 직전 adapter response와 현재 response의 차이를 soft하게 제한한다.

$$
\mathcal L_{\mathrm{move}}=\frac1{|\mathcal A_{\mathrm{old}}|}\sum_{(\ell,c)\in\mathcal A_{\mathrm{old}}}
\operatorname{SmoothL1}\left(A_{\ell,\phi^t}(a_{\ell,c}),r^{t-1}_{\ell,c}\right).
$$

Structure는 SCC로 sample-to-old-prototype 관계를 보존하고 APR 및 synthetic feature 기반 classifier adaptation을 사용한다. 이 구현은 APR/SCC/CA를 사용하지 않고 new-class PTM placement와 old-anchor adapter response 안정성을 직접 제약한다. 전체 유도식과 비교는 `RESULTS_PTM_GUIDED_VS_STRUCTURE.md` 2절을 참고한다.

## 4. 핵심 파일 지도

| 경로 | 역할 |
|---|---|
| `models/PTM_guided_adapter.py` | PTM-guided learner, placement/movement loss, checkpoint save/load |
| `utils/ptm_guidance.py` | frozen PTM prototype와 layer anchor/response memory |
| `utils/ptm_metrics.py` | JSON/CSV 결과 기록과 run status 관리 |
| `trainer.py` | seed/task 진행률, ETA, wall time, GPU memory 로그 |
| `data/data_manager.py` | dataset root 및 task protocol 검증 |
| `exps_jsons/ptm_guided_*.json` | method/dataset/task별 실행 config |
| `scripts/run_phase1_gpu3_parallel.sh` | GPU 3 한 장에서 method 3개 병렬, dataset 순차 실행 |
| `scripts/check_ptm_guided_datasets.py` | dataset 구조 사전 검사 |
| `scripts/summarize_ptm_guided_results.py` | COMPLETE run 집계, 결과 MD와 그래프 생성 |
| `scripts/plot_ptm_guided_results.py` | Structure 비교 막대그래프와 task heatmap 생성 |
| `references/structure_cvpr2026_results.json` | 논문 reference 수치와 출처 |
| `RESULTS_PTM_GUIDED_VS_STRUCTURE.md` | 방법론 수식, 전체 task 결과, 진단, 그래프, 미완료 목록 |
| `PTM_GUIDED_EXPERIMENT_PLAN.md` | 구현 의도와 protocol |
| `RUN_PTM_GUIDED.md` | 실행 절차 |

## 5. 환경과 외부 자산

현재 머신의 검증 환경은 다음과 같다.

```bash
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd ${RESEARCH_WORKSPACE}/cdsp_handoff_20260813
```

환경을 새로 만들 때는 다음을 사용한다.

```bash
conda env create -f cds.yaml
conda activate cds
```

데이터 경로:

```bash
export IMAGENETA_ROOT=${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
export CARS196_ROOT=${RESEARCH_WORKSPACE}/datasets/ptm_guided/cars196
```

- ImageNet-A: processed CIL split, 200 classes, train 5,981 / test 1,519
- Cars-196: `${RESEARCH_WORKSPACE}/cds-main/data/cars196`을 가리키는 symlink 사용
- ViT-B/16-IN21K weight: timm/Hugging Face cache 사용
- 데이터셋과 pretrained cache는 ZIP 외부 자산이다.

검사:

```bash
python scripts/check_ptm_guided_datasets.py --phase1
```

## 6. 재현 및 후속 실행

### Phase 1 전체: GPU 3, method 3개 병렬

```bash
cd ${RESEARCH_WORKSPACE}/cdsp_handoff_20260813
conda activate cds
PHYSICAL_GPU=3 bash scripts/run_phase1_gpu3_parallel.sh
```

dry-run:

```bash
PHYSICAL_GPU=3 bash scripts/run_phase1_gpu3_parallel.sh --dry-run
```

이 script는 ImageNet-A의 세 method를 동시에 끝낸 뒤 Cars-196의 세 method를 동시에 실행한다. 각 method 내부 seed 1993/1996/1997은 순차 처리된다.

### 단일 config: GPU 3

```bash
CUDA_VISIBLE_DEVICES=3 PYTHONUNBUFFERED=1 python -u main.py \
  --config exps_jsons/ptm_guided_imageneta_5task_ptm_place_move.json
```

주의: production config에는 세 seed가 들어 있으므로 위 명령을 다시 실행하면 완료 seed부터 다시 시작한다. 특정 seed/task checkpoint에서 재개하려면 별도 JSON에 해당 seed 하나와 `resume_checkpoint`를 지정해야 한다.

예시 필드:

```json
{
  "base_config": "ptm_guided_imageneta_20task_ptm_place_move.json",
  "seed": [1996],
  "resume_checkpoint": "results/ptm_guided/imageneta/20_tasks/ptm_place_move/seed_1996/checkpoints/checkpoint_task_16.pth"
}
```

`base_config` 상대경로는 새 JSON이 있는 디렉터리를 기준으로 해석된다. checkpoint 번호는 실제 디렉터리에서 가장 큰 파일을 확인하고 지정한다. 전달 ZIP에는 `.pth`가 없으므로, 다른 머신에서는 해당 checkpoint를 별도로 복사하지 않는 한 seed 처음부터 실행해야 한다.

### 결과 문서와 그래프 갱신

```bash
python scripts/summarize_ptm_guided_results.py
```

집계기는 `status=COMPLETE`만 사용하며 `IN_PROGRESS`와 `SMOKE_TEST`를 제외한다.

## 7. 결과 해석 시 주의

- Structure 수치는 논문 table에서 옮긴 reference이며 이 repository에서 재실행한 값이 아니다.
- 최종 막대그래프의 초록/빨강은 Structure Last/Inc-Acc 대비 향상/하락이다.
- task heatmap의 초록/빨강은 Structure task matrix가 공개되어 있지 않아 동일 protocol의 Adapter Baseline 대비 차이다. 대각선 파란 셀은 각 task의 최초 정확도다.
- 평균과 표준편차는 COMPLETE seed만 대상으로 하며, 3개 미만이면 `Seeds` 열에서 `n/3`으로 표시된다.
- Phase 1에서 PTM constraint는 Cars-196 10-task의 Adapter Baseline 대비 성능과 forgetting을 개선했지만 Structure reference보다 낮다. ImageNet-A 10-task에서는 drift 감소가 정확도 개선으로 이어지지 않았다.

## 8. ZIP 포함/제외 범위

포함:

- Python/shell 코드, JSON config, `cds.yaml`
- 모든 Markdown 문서와 reference JSON
- COMPLETE 및 snapshot 당시 진행 중 run의 JSON/CSV
- PNG 결과 그래프
- 실행 로그

제외:

- `results/**/checkpoints/*.pth` 및 모든 `.pth`: 227개, 약 80.75 GiB
- `__pycache__`, `.pyc`, `.git`
- 데이터셋과 pretrained model cache
- 기존 ZIP 파일

checkpoint를 제외해도 최종 수치, task matrix, class accuracy, drift/loss 진단은 JSON/CSV로 남아 있으므로 결과 검토와 문서 재생성에는 문제가 없다.

## 9. 전달 후 확인 체크리스트

1. ZIP 압축 해제 후 `HANDOFF_PTM_GUIDED_20260814.md`와 `RESULTS_PTM_GUIDED_VS_STRUCTURE.md`를 먼저 읽는다.
2. `conda env create -f cds.yaml` 또는 기존 `cds` 환경을 준비한다.
3. dataset root 환경변수를 설정하고 `check_ptm_guided_datasets.py --phase1`을 통과시킨다.
4. `python scripts/summarize_ptm_guided_results.py`로 포함된 정형 결과에서 MD/PNG가 재생성되는지 확인한다.
5. 새 실험 전에 config의 seed, dataset/task protocol, `save_checkpoints` 및 저장 공간을 확인한다.
6. 논문 비교 시 `reference`와 `executed result`를 구분한다.

## 10. References

1. Xin Zhang, Liang Bai, Guanchao Wang, and Xian Yang. “Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure.” *CVPR 2026*. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Exemplar-Free_Class_Incremental_Learning_via_Preserving_Class-Discriminative_Structure_CVPR_2026_paper.html)
2. Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
3. RevisitingCIL processed dataset protocol. [GitHub](https://github.com/LAMDA-CL/RevisitingCIL)
