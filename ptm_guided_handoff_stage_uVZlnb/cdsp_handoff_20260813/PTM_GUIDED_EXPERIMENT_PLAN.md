# PTM-Guided Adapter 실험 계획 및 구현

## 1. 가설

ViT-B/16-IN21K의 frozen visual geometry를 new class의 hard target이 아닌 초기 placement prior로 사용한다. Trainable nonlinear bottleneck adapter는 classification objective로 필요한 discriminative refinement를 수행하고, 이후 task에서는 old class가 위치했던 layer anchor에서 adapter의 기능적 response가 과도하게 변하지 않도록 soft하게 제한한다.

```text
Frozen PTM geometry
  -> new-class PTM prototype guide
  -> bottleneck adapter refinement
  -> class/layer anchor 저장
  -> 이후 task에서 old-anchor response movement 억제
```

현재 trainable module은 LoRA가 아니라 다음 adapter다.

```text
g_l(x) = up_proj(ReLU(down_proj(x)))
```

## 2. Frozen PTM reference

`network/vision_transformer_adapter.py`에 모든 adapter를 명시적으로 우회하는 `use_adapter=False` 경로를 추가했다. Forward 중 global placement 값을 변경하지 않으며, PTM feature는 `torch.no_grad()`에서 한 번의 backbone pass로 계산한다.

새 class `c`가 처음 등장할 때 current-task train split의 deterministic test transform을 사용해 다음 normalized prototype을 계산하고 CPU에 보관한다.

```math
\mu_c^{PTM} = \operatorname{normalize}\left(
\frac{1}{N_c}\sum_{i:y_i=c} f_{PTM}(x_i)
\right)
```

Raw image와 individual feature는 보관하지 않는다.

## 3. Placement guidance

Mini-batch 안에서 관측된 class별 current feature mean과 저장된 PTM prototype의 cosine distance를 사용한다.

```math
L_{place}=\frac{1}{|C_B|}\sum_{c\in C_B}
\left(1-\cos(\mu^{cur}_{c,B},\mu_c^{PTM})\right)
```

기본 weight는 `1.0 -> 0.0` cosine decay다. 따라서 초반에는 PTM 위치를 참고하고 후반에는 current classification에 맞게 자유롭게 refine한다. 관련 값은 모두 JSON에서 변경할 수 있다.

## 4. Old-anchor response stability

Task 종료 후 active adapter `down_proj`에 실제로 들어간 CLS representation의 class mean `q_{u,c}`를 CPU에 저장한다. `u`는 adapter unit이며 MLP placement에서는 layer별 MLP adapter, attention placement에서는 layer별 K/V adapter다.

다음 task 시작 시 현재 branch의 이전 response를 snapshot한다.

```math
r_{u,c}^{prev}=\operatorname{stopgrad}(g_u^{prev}(q_{u,c}))
```

학습 중에는 dropout을 제외한 실제 nonlinear residual response를 비교한다.

```math
L_{move}=\frac{1}{|A|}\sum_{u,c}
\operatorname{SmoothL1}(g_u^{cur}(q_{u,c}), r_{u,c}^{prev})
```

이는 parameter나 old representation 전체를 freeze하지 않으며, old class anchor에서의 기능적 변화만 soft하게 제한한다. Task 0에는 old anchor가 없어 정확히 0이다.

## 5. 최종 objective와 비교 조건

```math
L=L_{cls}+\lambda_{place}(e)L_{place}+\lambda_{move}L_{move}
```

`L_cls`는 기존 CosFace/AngularPenalty objective를 유지한다.

| Method | Classification | PTM placement | Old movement |
|---|---:|---:|---:|
| Adapter Baseline | O | X | X |
| PTM Place Only | O | O | X |
| PTM Place + Move | O | O | O |

APR, SCC, synthetic classifier adaptation, raw replay, old raw image 저장, old-sample distillation 및 CDSP hard projection은 모두 비활성화한다.

## 6. Benchmark protocol

| Dataset | Classes | Tasks | Split |
|---|---:|---:|---|
| ImageNet-A | 200 | 5 / 10 / 20 | 40 / 20 / 10 |
| Cars-196 | 196 | 5 / 10 / 20 | 마지막 task에 remainder |
| ObjectNet | 200 | 10 | 20 |
| OmniBenchmark | 300 | 10 | 30 |
| ImageNet-R | 200 | 10 | 20 |

기본 seed는 1993, 1996, 1997이다. Cars-196 class 수는 ImageFolder에서 196인지 검증하며, 10-task protocol은 `20×9 + 16`이다.

Dataset별 epoch, learning rate, batch size, margin 및 adapter placement는 전달받은 Structure/CDS 공식 config를 유지했다. 추가된 주요 변수는 placement/movement weight뿐이다.

## 7. Metric 및 결과

각 seed에 다음 파일을 저장한다.

```text
results/ptm_guided/<dataset>/<tasks>_tasks/<method>/seed_<seed>/
├── metrics.json
├── task_metrics.csv
├── task_accuracy_matrix.csv
├── class_accuracy.csv
├── ptm_geometry_drift.csv
├── anchor_drift.csv
└── loss_components.csv
```

- Last-Acc: 마지막 stage의 모든 seen-class accuracy
- Inc-Acc: 각 stage seen-class accuracy의 평균
- PTM prototype drift: current class prototype과 stored PTM prototype의 cosine distance
- PTM geometry drift: current/PTM prototype cosine matrix의 off-diagonal 차이
- Anchor drift: 저장 당시 response와 현재 response의 cosine/SmoothL1 차이
- 추가 기록: old/new accuracy, forgetting, loss component, runtime, peak GPU memory

`scripts/summarize_ptm_guided_results.py`는 완료된 3 seeds만 읽어 mean ± sample std와 Structure reference 차이를 갱신한다. Smoke 결과는 논문 비교 집계에서 제외한다.

## 8. 구현 파일

| File | Role |
|---|---|
| `models/PTM_guided_adapter.py` | 독립 learner, 학습/평가/checkpoint |
| `utils/ptm_guidance.py` | PTM prototype, placement loss, adapter anchor/response movement |
| `utils/ptm_metrics.py` | JSON/CSV writer |
| `network/vision_transformer_adapter.py` | 명시적 adapter bypass 및 single adapted forward |
| `data/data.py` | config/environment 기반 dataset root와 class 검증 |
| `references/structure_cvpr2026_results.json` | 논문 reference 수치 |
| `scripts/summarize_ptm_guided_results.py` | mean ± std 및 비교 MD 생성 |

## 9. 해석 시 주의

- Structure는 exemplar-free이며 APR/SCC와 stored class statistics, rectified Gaussian prototype distribution의 synthetic feature를 이용한 classifier adaptation을 수행한다.
- PTM-guided 방법은 drift를 사후 보정하기보다 new class에 weak PTM prior를 주고 old anchor의 이후 기능적 이동을 soft하게 억제한다.
- ObjectNet 공식 배포본은 parameter tuning 금지 조항이 있는 특수 라이선스다. 해당 CIL processed split 사용 권한과 논문 protocol의 라이선스 준수 여부를 실행자가 확인해야 한다.
- 실행 전 결과는 `PENDING`이며 임의 수치를 채우지 않는다.

## 10. References

1. Xin Zhang et al. “Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure.” CVPR 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Exemplar-Free_Class_Incremental_Learning_via_Preserving_Class-Discriminative_Structure_CVPR_2026_paper.html
2. Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words.” ICLR 2021. https://arxiv.org/abs/2010.11929
3. RevisitingCIL processed dataset protocol. https://github.com/LAMDA-CL/RevisitingCIL
