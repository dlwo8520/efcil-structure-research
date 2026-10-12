# RP-ProtoCLAD Task-5 결과 보고서

> 생성 시각: 2026-09-21 15:18 KST. 파일 기반 부분 결과 보고서이며, `—`는 아직 완료 결과가 없다는 뜻이다.

2026-09-21 14:47 KST에 종료된 첫 GPU 2 RP queue에서는 IA5 RP 두 run이 완료됐고 CAR5 RP 두 run은 parent 미완료로 DEFER되어 학습을 시작하지 않았다 (`failures=0`, `deferred=2`; 큐의 비영(2) 종료 코드는 대기 중인 parent가 있다는 뜻이다). 아래 판정은 생성 시점의 파일시스템 상태를 다시 확인한 결과다.

## 방법 구분

Prior work idea: fixed random projection을 이용한 고차원 nonlinear random feature representation.

Our experimental adaptation: Structure의 classifier/SCC/APR/classifier adaptation은 원 feature 공간에 유지하고, frozen RP를 Proto-CLAD old-new conflict regularization cosine에만 적용한다. 기존 논문의 직접 재현이 아니다.

Top-P conflict selection 역시 기존 original-space frozen similarity를 그대로 사용한다.

## 실행 상태

| Protocol | Run | Parent | Status | Action |
|---|---|---|---|---|
| IA5 | RP_C_P020_L030 | C_P020_L030 (COMPLETE) | COMPLETE | KEEP_COMPLETE |
| IA5 | RP_C_P020_L100 | C_P020_L100 (COMPLETE) | COMPLETE | KEEP_COMPLETE |
| CAR5 | RP_C_P020_L030 | C_P020_L030 (PARTIAL) | NOT_STARTED | WAIT_PARENT |
| CAR5 | RP_C_P020_L100 | C_P020_L100 (NOT_STARTED) | NOT_STARTED | WAIT_PARENT |

CAR5의 Structure 기준선은 `FAILED`이고, 두 Proto-CLAD parent는 각각 `PARTIAL`, `NOT_STARTED`다. CAR5 RP의 비어 있는 성능 셀은 실패한 RP 결과가 아니라 **미실행** 상태를 뜻한다.

## IA5

| Method | Inc | ΔStruct | ΔParent | Last | F↓ | Old | New | Guardrail |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Structure | 74.598 | +0.000 | — | 66.425 | 7.844 | 66.987 | 63.838 | PASS |
| C_P020_L030 | 74.640 | +0.042 | — | 66.557 | 7.864 | 67.147 | 63.838 | PASS |
| RP_C_P020_L030 | 74.704 | +0.106 | +0.064 | 66.820 | 7.310 | 67.468 | 63.838 | PASS |
| C_P020_L100 | 74.621 | +0.024 | — | 66.623 | 7.424 | 67.308 | 63.469 | PASS |
| RP_C_P020_L100 | 74.495 | -0.102 | -0.126 | 66.359 | 7.682 | 67.067 | 63.100 | FAIL |

전체 delta(Last/F/Old/New 포함)는 `outputs/metrics_per_seed.csv`에 저장한다.

## CAR5

| Method | Inc | ΔStruct | ΔParent | Last | F↓ | Old | New | Guardrail |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Structure | — | — | — | — | — | — | — | — |
| C_P020_L030 | — | — | — | — | — | — | — | — |
| RP_C_P020_L030 | — | — | — | — | — | — | — | — |
| C_P020_L100 | — | — | — | — | — | — | — | — |
| RP_C_P020_L100 | — | — | — | — | — | — | — | — |

전체 delta(Last/F/Old/New 포함)는 `outputs/metrics_per_seed.csv`에 저장한다.

## RP 전후 geometry

| Protocol | Run | Task | orig selected cosine | RP selected cosine | Δcos | orig gap | RP gap | Δgap | RP zero ratio | grad cos(cls, RP) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IA5 | RP_C_P020_L030 | 2 | 0.401 | 0.540 | +0.139 | 0.329 | 0.209 | -0.119 | 0.499 | -0.050 |
| IA5 | RP_C_P020_L030 | 3 | 0.404 | 0.537 | +0.133 | 0.287 | 0.187 | -0.100 | 0.503 | -0.068 |
| IA5 | RP_C_P020_L030 | 4 | 0.438 | 0.552 | +0.114 | 0.311 | 0.199 | -0.112 | 0.504 | -0.014 |
| IA5 | RP_C_P020_L030 | 5 | 0.454 | 0.560 | +0.106 | 0.283 | 0.184 | -0.099 | 0.505 | -0.000 |
| IA5 | RP_C_P020_L030 | ALL | 0.434 | 0.551 | +0.117 | 0.297 | 0.192 | -0.105 | 0.503 | -0.033 |
| IA5 | RP_C_P020_L100 | 2 | 0.396 | 0.524 | +0.128 | 0.330 | 0.213 | -0.117 | 0.499 | -0.057 |
| IA5 | RP_C_P020_L100 | 3 | 0.396 | 0.510 | +0.114 | 0.288 | 0.192 | -0.096 | 0.508 | -0.092 |
| IA5 | RP_C_P020_L100 | 4 | 0.427 | 0.513 | +0.086 | 0.312 | 0.206 | -0.106 | 0.512 | -0.043 |
| IA5 | RP_C_P020_L100 | 5 | 0.440 | 0.514 | +0.074 | 0.286 | 0.193 | -0.094 | 0.515 | -0.019 |
| IA5 | RP_C_P020_L100 | ALL | 0.423 | 0.514 | +0.091 | 0.299 | 0.199 | -0.100 | 0.508 | -0.053 |
| CAR5 | RP_C_P020_L030 | ALL | — | — | — | — | — | — | — | — |
| CAR5 | RP_C_P020_L100 | ALL | — | — | — | — | — | — | — | — |

## IA5 완료 결과의 상세 해석

아래 차이는 모두 퍼센트포인트(pp)다. `ΔParent`는 동일 λ의 원공간 Proto-CLAD 대비 변화이고, `ΔStruct`는 vanilla Structure 대비 변화다. Forgetting은 낮을수록 좋다.

| RP run | 비교 | ΔInc | ΔLast | ΔF↓ | ΔOld | ΔNew |
|---|---|---:|---:|---:|---:|---:|
| RP_C_P020_L030 | Parent | +0.064 | +0.263 | -0.554 | +0.321 | +0.000 |
| RP_C_P020_L030 | Structure | +0.106 | +0.395 | -0.535 | +0.481 | +0.000 |
| RP_C_P020_L100 | Parent | -0.126 | -0.263 | +0.258 | -0.240 | -0.369 |
| RP_C_P020_L100 | Structure | -0.102 | -0.066 | -0.162 | +0.080 | -0.738 |

`λ=0.30` RP는 parent 대비 Inc +0.064 pp, Last +0.263 pp, F −0.554 pp, Old +0.321 pp이며 New는 동일하다. Structure guardrail도 통과한다. 다만 효과 크기가 작고 단일 seed여서 유망한 **예비 결과**이지 일반화된 우월성의 증거는 아니다.

`λ=1.00` RP는 parent 대비 Inc −0.126 pp, Last −0.263 pp, F +0.258 pp, Old −0.240 pp, New −0.369 pp다. Structure 대비 New −0.738 pp로 −0.5 pp guardrail을 위반한다. 기존 New 손실을 회복하지 못했다.

### Task별 single-head accuracy

| λ | Task | Parent | RP | RP − Parent |
|---:|---:|---:|---:|---:|
| 0.30 | 1 | 83.889 | 83.889 | +0.000 |
| 0.30 | 2 | 79.370 | 79.083 | -0.287 |
| 0.30 | 3 | 74.154 | 74.256 | +0.103 |
| 0.30 | 4 | 69.231 | 69.471 | +0.240 |
| 0.30 | 5 | 66.557 | 66.820 | +0.263 |
| 1.00 | 1 | 83.889 | 83.889 | +0.000 |
| 1.00 | 2 | 79.656 | 79.370 | -0.287 |
| 1.00 | 3 | 73.949 | 73.949 | +0.000 |
| 1.00 | 4 | 68.990 | 68.910 | -0.080 |
| 1.00 | 5 | 66.623 | 66.359 | -0.263 |

### RP 자체의 geometry 및 근사 진단

`selected cosine`은 RP run 안에서 같은 class pair를 원 공간과 RP 공간에 각각 투영해 비교한 값이다. 두 공간의 cosine 척도가 달라 **RP가 학습 후 원공간 separation을 개선했다는 직접 증거로 해석하면 안 된다**. 아래 gap은 sampled current sample의 correct-new cosine − selected-old cosine이다.

| λ | selected Δcos (RP−orig) | gap Δ (RP−orig) | Pearson | Spearman | RP zero ratio | prototype cos(a,b) | prototype rel-L2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.30 | +0.117 | -0.105 | 0.980 | 0.966 | 0.503 | 0.910 | 0.433 |
| 1.00 | +0.091 | -0.100 | 0.973 | 0.956 | 0.508 | 0.903 | 0.449 |

두 RP run 모두 selected cosine이 RP 공간에서 더 높고 correct-new gap이 더 작다. 이 조작적 진단에서는 의도한 separation 증가가 관측되지 않았다. original/RP similarity의 Pearson·Spearman은 높아, 관계 구조 자체는 대체로 보존되지만 ReLU로 cosine의 절대 수준이 이동한다.

prototype 근사는 `a=ReLU(mean(z)R)`, `b=mean(ReLU(zR))`를 current class의 기존 cache feature로 계산했다. old class에서 직접 검증한 값은 아니다.

| λ | class-task rows | mean cosine(a,b) | mean relative-L2 | relative-L2 > 0.1 |
|---:|---:|---:|---:|---:|
| 0.30 | 160 | 0.910 | 0.433 | 160/160 |
| 1.00 | 160 | 0.903 | 0.449 | 160/160 |

mean relative-L2가 약 0.43–0.45이고 모든 320개 class-task row에서 0.1을 넘는다. `0.1`은 탐색적 참고선이며 사전 등록된 검정 임계값은 아니다. 그럼에도 `ReLU(mean(z)R)`을 과거 class의 sample-wise projected mean 대리값으로 취급하는 근사는 작지 않아 보인다.

### Parent와 같은 원공간에서의 sampled pair 비교

마지막 epoch의 동일 task/epoch/batch/sample/class pair만 대조했다. selection이 달라진 pair는 제외했다.

| λ | matched pairs | Parent orig cosine | RP run orig cosine | ΔRP−Parent |
|---:|---:|---:|---:|---:|
| 0.30 | 7634/7680 | 0.215 | 0.222 | +0.007 |
| 1.00 | 7582/7680 | 0.195 | 0.214 | +0.020 |

원공간 selected cosine도 RP run에서 parent보다 소폭 높다. 따라서 `λ=0.30`의 작은 accuracy 이득을 'old-new cosine separation 향상'으로 설명할 수는 없다.

### Gradient 및 최종 task confusion

gradient 값은 logging point에서 trainable network parameter에 대한 **raw loss component**의 평균이다. λ와 SCC 계수가 적용된 실제 합성 gradient norm은 아니므로 크기를 직접 동등 비교하면 안 된다.

| λ | ∥grad cls∥ | ∥grad SCC∥ | ∥grad RP∥ | cos(cls,RP) | cos(SCC,RP) |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 2.447 | 0.079 | 0.083 | -0.033 | -0.152 |
| 1.00 | 2.445 | 0.080 | 0.072 | -0.053 | -0.207 |

| λ | Run | old→new | new→old | selected old→new | selected new→old |
|---:|---|---:|---:|---:|---:|
| 0.30 | Parent | 103 | 86 | 72 | 68 |
| 0.30 | RP | 105 | 85 | 73 | 67 |
| 1.00 | Parent | 100 | 85 | 68 | 67 |
| 1.00 | RP | 99 | 86 | 69 | 68 |

selected confusion은 각 run의 task-5 Top-P pair 1,280개에 해당하는 오분류 count다. 두 방향 모두 변화가 1–2건 수준이어서 큰 구조적 혼동 개선의 증거는 아니다.

### Class-wise 최종 accuracy 변화

| λ | Old classes 개선/동일/악화 | New classes 개선/동일/악화 |
|---:|---:|---:|
| 0.30 | 7/150/3 | 1/38/1 |
| 1.00 | 4/149/7 | 1/37/2 |

### IA5에 한정한 Q1–Q8 판정

| 질문 | IA5 판정 | 근거 |
|---|---|---|
| Q1: λ=0.30 RP가 parent보다 좋은가? | YES (예비) | ΔInc +0.064 pp, ΔF −0.554 pp, ΔOld +0.321 pp, ΔNew 0; 단일 seed의 작은 차이 |
| Q2: λ=1.00 RP가 parent보다 좋은가? | NO | ΔInc −0.126 pp, ΔNew −0.369 pp, ΔF +0.258 pp |
| Q3: selected cosine/gap이 개선됐는가? | NO (기록된 지표 기준) | 두 run 모두 RP selected cosine 상승, correct-new gap 감소 |
| Q4: geometry 개선이 accuracy 개선으로 이어졌는가? | NO/해당 없음 | λ=0.30의 accuracy 이득에도 geometry 개선은 관측되지 않음 |
| Q5: λ=1.00의 New 손실이 회복됐는가? | NO | Parent 대비 New −0.369 pp, Structure 대비 −0.738 pp |
| Q6: Old/F 유지와 New 유지·개선이 동시에 일어났는가? | λ=0.30 YES, λ=1.00 NO | λ=0.30은 New 동일·Old/F 개선, λ=1.00은 모두 악화 |
| Q7: IA5와 CAR5에서 재현되는가? | INCONCLUSIVE | CAR5 RP 두 run 미실행 |
| Q8: nonlinear prototype approximation gap이 작은가? | NO (탐색적 판단) | mean relative-L2 0.43–0.45, cosine 약 0.90–0.91 |

### 검증 및 한계

- IA5 RP 두 run 모두 task 5/5 완료, checkpoint와 summary/metadata 일치, `recording_gaps=[]` 확인. RP projection 메타데이터 동일: **True**; checksum `849337f236a4df355a5d74c27d015c45a80c27cd4f7fc0e32957a02c7619aa2d`.
- 기존 IA5 Structure는 외부 완료 run을 재사용했다. Structure/parent/RP의 `class_order.json` 내용이 동일하며 seed=1993, data root와 5×40 task 설정을 맞췄다.
- accuracy 변화는 200-class IA5의 seed 하나에서 나온 관측치다. 분산·유의성·다른 dataset으로의 일반화는 판단할 수 없다.
- geometry는 서로 다른 특징 공간의 cosine을 비교한 서술적 진단이다. approximation은 현재 new class에서만 직접 검증 가능하며, old APR prototype에 대한 오차는 직접 측정할 수 없다.
- 위 결과로 RP dimension, seed, λ 등을 추가 탐색하지 않았다. CAR5 parent가 완료된 후 고정된 두 RP 설정만 후속 검증한다.

### 주요 근거 파일

- [GPU 2 queue 기록](logs/rp_task5_gpu2.log), [실행별 종료 코드](logs/rp_task5/exit_codes.csv)
- [λ=0.30 RP summary](outputs/IA5/RP_C_P020_L030/seed_1993/summary.json), [λ=1.00 RP summary](outputs/IA5/RP_C_P020_L100/seed_1993/summary.json)
- [원자료 성능·delta CSV](outputs/metrics_per_seed.csv), [실행 계획 및 parent audit](rp_experiment_plan.md)
- RP run별 `rp_pair_geometry_diagnostics.csv`, `rp_gap_diagnostics.csv`, `rp_prototype_approximation.csv`, `rp_gradient_diagnostics.csv`, `confusion_task_05.npz`는 각 output 디렉터리 안에 있다.

## 필수 질문 판정

- **Q1 — RP_C_P020_L030이 parent보다 좋아졌는가?: INCONCLUSIVE** — 두 dataset 결과가 모두 필요하다.
- **Q2 — RP_C_P020_L100이 parent보다 좋아졌는가?: INCONCLUSIVE** — 두 dataset 결과가 모두 필요하다.
- **Q3 — RP가 selected cosine 또는 correct-new gap을 개선했는가?: INCONCLUSIVE** — geometry diagnostic이 아직 완결되지 않았다.
- **Q4 — geometry 개선이 accuracy 개선으로 이어졌는가?: INCONCLUSIVE** — geometry/accuracy 결과가 아직 완결되지 않았다.
- **Q5 — lambda=1.0의 New 손실이 회복됐는가?: INCONCLUSIVE** — lambda=1.0 결과가 아직 완결되지 않았다.
- **Q6 — Old/F를 유지하며 New도 유지·개선했는가?: INCONCLUSIVE** — 4개 RP 결과가 모두 필요하다.
- **Q7 — IA5/CAR5에서 같은 방향으로 반복됐는가?: INCONCLUSIVE** — 두 dataset 결과가 아직 완결되지 않았다.
- **Q8 — nonlinear prototype approximation gap이 충분히 작았는가?: INCONCLUSIVE** — prototype approximation 결과가 아직 완결되지 않았다.

## 해석 원칙

RP 자체 효과는 Structure delta보다 동일 lambda parent의 `ΔParent_*`를 우선한다. Inc가 상승해도 New가 크게 하락하면 긍정 결과로 해석하지 않으며, Old만 상승하고 New가 하락하면 stability-plasticity 이동으로 기록한다.
