# StoredAPR RP-ProtoCLAD 비교 보고서

실험은 기존 `RP_C_P020_L030`을 보존한 별도 ablation이다. 새 방식은 `mean(ReLU(z_birth R)) + ReLU(mu_APR R) - ReLU(mu_birth R)`를 old class에 사용한다.
원공간 Structure/APR/SCC/classifier adaptation과 Top-P 선택은 변경하지 않았다.

## 성능 (단위: percentage point)

| Protocol | Run | 상태 | Inc | Last | Forgetting | Old | New | ΔStruct Inc | ΔOldRP Inc |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| IA5 | Structure | COMPLETE | 74.598 | 66.425 | 7.844 | 66.987 | 63.838 | +0.000 | -0.106 |
| IA5 | C_P020_L030 | COMPLETE | 74.640 | 66.557 | 7.864 | 67.147 | 63.838 | +0.042 | -0.064 |
| IA5 | RP_C_P020_L030 | COMPLETE | 74.704 | 66.820 | 7.310 | 67.468 | 63.838 | +0.106 | +0.000 |
| IA5 | StoredAPR_RP_C_P020_L030 | COMPLETE | 74.632 | 66.294 | 7.867 | 67.228 | 61.993 | +0.034 | -0.072 |
| CAR5 | Structure | COMPLETE | 84.929 | 81.196 | 4.431 | 80.987 | 82.055 | +0.000 | — |
| CAR5 | C_P020_L030 | COMPLETE | 84.905 | 81.184 | 4.450 | 81.002 | 81.928 | -0.024 | — |
| CAR5 | RP_C_P020_L030 | NOT_STARTED | — | — | — | — | — | — | — |
| CAR5 | StoredAPR_RP_C_P020_L030 | COMPLETE | 84.938 | 81.221 | 4.387 | 81.049 | 81.928 | +0.009 | — |

전체 항목의 `ΔStruct_*`, `ΔOldRP_*`는 [`outputs/comparison_metrics.csv`](outputs/comparison_metrics.csv)에 저장한다.

## Prototype 및 geometry diagnostic

| Protocol | Birth rel-L2 (oldstyle vs true) | Stored vs oldstyle rel-L2 | APR Δq norm | Selected RP cosine | RP gap | Gradient cos(cls, RP) |
|---|---:|---:|---:|---:|---:|---:|
| IA5 | 0.425 | 0.561 | 7.149 | 0.626 | 0.081 | -0.032 |
| CAR5 | 0.396 | 0.514 | 8.427 | 0.678 | 0.038 | -0.014 |

`birth_relative_l2`의 분모는 정확한 `q_birth`; `stored_vs_oldstyle_relative_l2`의 분모는 해당 epoch의 old-style prototype이다.
Geometry와 accuracy 사이의 인과관계는 이 두 run만으로 확정할 수 없다.

## 핵심 질문

- Q1–Q7: IA5/CAR5의 새 run과 기존 RP run이 모두 완료되기 전이므로 PENDING. 부분 결과로 결론 내리지 않는다.

## 원자료

각 새 run의 `task_summary.csv`, `class_accuracy.csv`, `stored_rp_prototype_diagnostics.csv`, `rp_pair_geometry_diagnostics.csv`, `rp_gap_diagnostics.csv`, `rp_gradient_diagnostics.csv`, `rp_confusion_diagnostics.csv`, `rp_selected_confusion.csv`를 보존한다.
