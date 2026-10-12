# CDSP EFCIL Drift-Prevention 실험 인수인계

## 1. 패키지 목적

이 패키지는 공식 CDS 코드 위에 추가한 **Class-Discriminative Subspace Protection (CDSP)** 진단 실험의 소스, 설정, 완료 로그 및 구조화 결과를 인수인계하기 위한 것이다.

실험 질문은 다음과 같다.

> Exemplar-Free Class-Incremental Learning에서 old class를 구별하는 데 중요한 representation subspace를 새 task 학습 중 보호하면, 저장된 class prototype의 drift와 forgetting을 줄일 수 있는가?

공식 CDS는 발생한 drift를 APR과 SCC로 보정한다. 이번 진단은 APR/SCC/synthetic classifier adaptation을 모두 끄고, adapter update가 보호 subspace를 침범하지 않게 제한한 뒤 **보정하지 않은 stored prototype**으로 평가한다.

이 구현은 KeepLoRA의 protected-subspace 철학에서 영감을 받았지만 KeepLoRA 자체 구현은 아니다. CDS adapter가 선형 LoRA `BAx`가 아니라 `up_proj(ReLU(down_proj(x)))` 구조이므로, LoRA의 `Delta W` 대신 각 adapter의 `down_proj.weight`에 `D Q = 0` 제약을 적용한다.

## 2. 완료 상태

- Dataset/protocol: CIFAR-100, 10 initial classes + 이후 task당 10 classes, 총 10 tasks
- Backbone: `pretrained_vit_b16_224_in21k_adapter`
- Seed: 1993 한 번
- 완료 조건: Prototype Baseline, Task Subspace, Inter Subspace, Inter + Intra
- Primary evaluation: 모든 seen class를 후보로 사용하는 cosine prototype classifier
- 실행 GPU: physical GPU 3 (`CUDA_VISIBLE_DEVICES=3`), 프로세스 내부 logical `cuda:0`
- 결과 문서: [`RESULTS_CDSP.md`](RESULTS_CDSP.md)

네 조건 모두 Task 0부터 Task 9까지 정상 종료했다.

## 3. 핵심 결과

| Method | Avg incremental | Final | Old | New | Forgetting | Prototype drift | Pairwise drift |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prototype Baseline | 93.145 | 89.20 | 88.64 | **94.20** | 6.38 | 0.070491 | 0.047917 |
| Task Subspace | 93.138 | 89.37 | 88.99 | 92.80 | **4.91** | **0.034317** | **0.025408** |
| Inter Subspace | 93.172 | 89.35 | 88.87 | 93.70 | 5.89 | 0.063638 | 0.041754 |
| Inter + Intra | **93.199** | **89.46** | **89.04** | 93.20 | 5.00 | 0.040492 | 0.028202 |

이 seed에서는 세 protection 조건 모두 baseline보다 final/old accuracy가 소폭 높고 forgetting이 낮았다. Task와 Inter + Intra는 drift를 크게 줄였지만 new-class accuracy도 낮아져 stability-plasticity trade-off가 확인됐다. 차이가 0.15-0.26%p로 작고 단일 seed 결과이므로 통계적 우위를 주장할 수 없다.

Inter-only basis는 Task 9에도 layer별 rank 3-68로 작았다. Task와 Inter + Intra basis는 여러 후반 layer에서 feature dimension 768까지 포화되어 over-protection 가능성이 있다. 세 protection 조건의 최대 post-training constraint violation은 각각 `8.75e-6`, `3.75e-8`, `1.20e-5`였다.

각 조건의 전체 task elapsed 합은 약 1.42시간, peak allocated GPU memory는 약 4.66-4.67 GiB였다.

## 4. 구현 개요

### 4.1 통계 수집과 memory

각 ViT MLP adapter의 `down_proj` 입력에 forward pre-hook을 걸고 CLS token만 수집한다. 수집은 task 종료 후 current-task train split에 대해서만 수행하며, tensor는 즉시 CPU로 이동한다. 개별 feature나 old train sample은 보관하지 않고 다음 compressed statistics만 유지한다.

- layer별 class prototype
- 최종 feature class prototype
- streaming centered task scatter
- streaming within-class scatter
- 압축된 orthonormal basis

### 4.2 비교 조건

| Method | Protected space |
|---|---|
| Prototype Baseline | 없음 |
| Task Subspace | label을 쓰지 않은 generic task-feature subspace, energy 0.95 |
| Inter Subspace | 누적 class prototype의 centered inter-class subspace, energy 0.95 |
| Inter + Intra | inter-class basis와 within-class residual basis(energy 0.90)의 union |

모든 조건에서 raw replay, synthetic replay, APR, SCC, prototype rectification, old-sample distillation 및 oracle task ID를 사용하지 않는다.

### 4.3 Hard constraint

Layer `l`의 adapter down projection을 `D_l`, protected orthonormal basis를 `Q_l`이라 하면 다음을 유지한다.

```text
D_l Q_l = 0
D_l <- D_l - (D_l Q_l) Q_l^T
```

제약은 task 시작 전 weight, backward gradient, optimizer step 후 weight와 SGD momentum에 적용된다. Protection 조건에서는 adapter bias도 0으로 유지하고 freeze한다.

## 5. 코드 지도

| 파일 | 역할 |
|---|---|
| `models/CDSP_adapter.py` | CDSP learner, 학습/평가, 통계 수집, checkpoint save/resume |
| `utils/protected_subspace.py` | streaming scatter, SVD/eigendecomposition, basis memory, gradient/weight/momentum projection |
| `utils/prototype_classifier.py` | cosine prototype classifier와 evaluation-only drift metric |
| `utils/cil_metrics.py` | JSON/CSV 결과 writer |
| `utils/inc_net.py` | adapter network 연결부 |
| `utils/model_factory.py` | `model_name=cdsp` learner 등록 |
| `network/classifier.py` | classifier 관련 보조 변경 |
| `trainer.py` | task/seed/전체 진행률, ETA, runtime, peak GPU memory logging, resume 진입점 |
| `main.py` | `--config` 인자 처리 |
| `exps_jsons/cifar100_*.json` | 네 controlled condition 설정 |

`models/CDS_adapter.py`는 공식 CDS 경로이며, CDSP 진단은 별도 `models/CDSP_adapter.py`에 격리했다.

## 6. 환경 구성

```bash
cd /path/to/cdsp_handoff_20260813
conda env create -f cds.yaml
conda activate cds
```

주요 버전은 Python 3.9.19, PyTorch 2.4.1, torchvision 0.19.1, timm 0.6.12, NumPy 1.23.5다. 정확한 목록은 `cds.yaml`을 사용한다.

CIFAR-100 데이터와 pretrained ViT weight/cache는 배포 크기 및 라이선스 문제 때문에 ZIP에 포함하지 않았다. 첫 실행 시 기존 loader와 timm이 필요한 데이터를 준비하며, 폐쇄망에서는 데이터와 pretrained cache를 별도로 전달해야 한다.

## 7. 재현 명령

아래 명령에서 physical GPU 3만 노출되므로 JSON의 `"device": ["0"]`은 올바른 설정이다. 여기서 `0`은 노출된 장치 안의 logical index다.

```bash
cd /path/to/cdsp_handoff_20260813
conda activate cds

CUDA_VISIBLE_DEVICES=3 python main.py --config exps_jsons/cifar100_proto_baseline.json
CUDA_VISIBLE_DEVICES=3 python main.py --config exps_jsons/cifar100_task_subspace.json
CUDA_VISIBLE_DEVICES=3 python main.py --config exps_jsons/cifar100_inter_subspace.json
CUDA_VISIBLE_DEVICES=3 python main.py --config exps_jsons/cifar100_inter_intra_subspace.json
```

표준 출력과 `logs/cdsp/.../*.log`에는 다음 정보가 지속적으로 기록된다.

- 현재 task와 epoch
- epoch별 loss/accuracy
- epoch/task/seed/전체 elapsed time과 ETA
- protected rank 및 constraint violation
- peak allocated/reserved GPU memory
- prototype/head accuracy, forgetting, drift

동시에 실행하면 동일 GPU의 메모리와 runtime 측정이 왜곡될 수 있으므로 한 조건씩 순차 실행하는 편이 안전하다.

## 8. 결과 파일

각 방법의 결과는 다음 구조를 사용한다.

```text
results/cdsp/cifar224/<method>/seed_1993/
├── metrics.json
├── task_metrics.csv
├── task_accuracy_matrix.csv
├── class_accuracy.csv
├── subspace_stats.csv
└── prototype_drift.csv
```

- `metrics.json`: config snapshot과 task별 핵심 metric
- `task_metrics.csv`: seen-class accuracy, old/new, forgetting, drift, 시간/메모리
- `task_accuracy_matrix.csv`: 학습 시점별 각 task accuracy
- `class_accuracy.csv`: class별 accuracy
- `subspace_stats.csv`: layer/task별 rank와 constraint violation
- `prototype_drift.csv`: old class별 prototype cosine drift

완료 로그는 `logs/cdsp/cifar224/0/10/`에 있다. 상세 표와 해석은 `RESULTS_CDSP.md`를 먼저 확인한다.

## 9. Checkpoint와 resume

원래 실험 디렉터리에는 task별 resumable checkpoint 40개, 총 약 15 GiB가 생성됐다. 인수인계 ZIP은 전송 가능한 크기를 유지하기 위해 이 checkpoint들을 제외했다. 따라서 ZIP만 받은 환경에서는 처음부터 실행해야 한다.

새 실행에서 checkpoint 생성을 원하지 않으면 네 JSON 설정의 다음 값을 바꾼다.

```json
"save_checkpoints": false
```

Checkpoint를 생성한 실행은 config에 다음 필드를 추가해 이어갈 수 있다.

```json
"resume_checkpoint": "results/cdsp/cifar224/<method>/seed_1993/checkpoints/checkpoint_task_XX.pth"
```

Checkpoint에는 model/optimizer-independent continual state, prototype/subspace memory와 누적 metric이 저장되며, 완료된 다음 task부터 계속한다.

## 10. 알려진 한계와 다음 실험

1. Seed 1993 한 번뿐이므로 최소 3-5 seed mean/std가 필요하다.
2. Official CDS(APR+SCC+synthetic CA)는 이번 네 조건과 함께 재실행하지 않았다.
3. Task/Intra basis가 후반 layer에서 rank 768로 포화된다. Global rank budget, layer별 cap, forgetting factor 또는 incremental SVD가 필요하다.
4. Inter-only의 energy threshold 및 보호 layer 선택 ablation이 필요하다.
5. 최종 accuracy 차이가 작으므로 class-order seed와 초기 task variation을 분리해서 확인해야 한다.
6. Old test images는 drift 진단에만 사용했다. 학습 통계나 prototype update에는 사용하지 않았다.

추천 후속 실험 순서는 `inter_subspace` multi-seed, inter energy sweep, last-4-layer protection, rank-capped inter+intra 순이다.

## 11. 패키지 포함/제외 범위

포함:

- 실행에 필요한 Python source tree
- `cds.yaml`과 CDSP JSON config 네 개
- `README.md`, `RESULTS_CDSP.md`, 본 인수인계 문서
- 네 조건의 JSON/CSV 결과와 완료 로그
- 공식 CDS pipeline 그림

제외:

- 약 15 GiB의 task별 `.pth` checkpoint
- CIFAR-100 archive/extracted dataset
- pretrained model cache
- `__pycache__`, `.pytest_cache`, `.git`
- 별도 PCMD/SGDS 진단 코드와 결과

원본 코드의 provenance는 전달받은 `cds-main.zip`이다. 공식 CDS 부분의 배포 조건은 원 저장소의 라이선스/고지를 확인해야 한다.

## 12. References

1. Xin Zhang, Liang Bai, Guanchao Wang, and Xian Yang. “Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure.” *CVPR 2026*, pp. 17979-17988. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Exemplar-Free_Class_Incremental_Learning_via_Preserving_Class-Discriminative_Structure_CVPR_2026_paper.html)
2. Mao-Lin Luo, Zi-Hao Zhou, Yi-Lin Zhang, Yuanyu Wan, Min-Ling Zhang, and Tong Wei. “KeepLoRA: Continual Learning with Residual Gradient Adaptation.” *ICLR 2026*. [OpenReview PDF](https://openreview.net/pdf?id=T3Vc5fkTzV)
3. Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
4. Alex Krizhevsky. “Learning Multiple Layers of Features from Tiny Images.” Technical report, 2009. [CIFAR technical report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf) · [official CIFAR page](https://www.cs.toronto.edu/~kriz/cifar.html)

## 13. 인수인계 체크리스트

- [ ] `conda env create -f cds.yaml` 성공
- [ ] CIFAR-100 및 pretrained ViT weight 접근 가능
- [ ] `CUDA_VISIBLE_DEVICES=3`에서 PyTorch가 logical `cuda:0`을 인식
- [ ] Baseline의 첫 epoch 로그와 ETA 출력 확인
- [ ] 결과 디렉터리에 JSON/CSV 여섯 종류 생성 확인
- [ ] Multi-seed 실행 전 저장 공간을 고려해 `save_checkpoints` 여부 결정
