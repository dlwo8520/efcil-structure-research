# Similarity-Selective Old-New Separation 중간 결과 보고서

작성 기준 시각: 2026-09-21 11:47 KST  
Stage: Stage 1 development sweep, seed 1993  
스냅샷: 완료 60 / 108, 실패 4, partial 1, 대기 43  
주 비교 기준: **기존 vanilla Structure**

> 이 문서는 실행 도중의 중간 보고서다. IA5만 27개 sweep 설정이 모두 완료되었다. IA10은
> 4개 run이 실패했고, CAR5/CAR10은 sweep이 끝나지 않았다. IA5에는 동일 조건의 기존
> Structure 결과가 있어 정식 delta를 계산했다. IA10과 Cars에는 현재 protocol과 정확히 맞는
> Structure baseline이 없으므로 절대값만 제시한다.

> **2026-09-21 운영 변경:** 방법 탐색은 IA5/CAR5의 5-task만 수행한다. 기존 IA10 결과는
> 참고용으로 보존하되 추가 실행·실패 복구·matched Structure 생성을 하지 않으며, CAR10은
> 전체 HOLD한다. 실제 실행 계획은 `reduced_experiment_plan.md`를 따른다.

## 1. 현재 결론 요약

IA5를 C0 AM-Drift가 아니라 기존 Structure에 맞춰 다시 계산하면 결론은 다음과 같다.

- 기존 Structure: Inc `74.60`, Last `66.43`, Forgetting `7.84`, Old `66.99`, New `63.84`
- C0 AM-Drift: Structure 대비 Inc `+0.17`, Last `+0.26`, Forgetting `-0.87`, Old `+0.56`,
  New `-1.11 pp`. Stability는 좋아졌지만 New guardrail을 위반한다.
- A/B 설정은 모두 Structure 기준 New guardrail을 위반한다. Inc 최고는 A_P025의
  `+0.16 pp`이지만 New는 `-1.85 pp`다.
- Structure 기준 guardrail을 통과한 것은 Method C의 4개 설정뿐이다.
  `C_P020_L030`이 그중 Inc 최고로 `+0.04 pp`이며 Last `+0.13`, Forgetting `+0.02`,
  Old `+0.16`, New `0.00 pp`다.
- `C_P020_L100`도 Inc `+0.02`, Last `+0.20`, Forgetting `-0.42`, Old `+0.32`,
  New `-0.37 pp`로 guardrail을 통과한다.

따라서 현재 IA5의 가장 보수적인 후보는 `C_P020_L030`, stability까지 고려한 차선은
`C_P020_L100`이다. 다만 이득이 `0.02–0.04 pp`로 매우 작고 seed가 하나뿐이므로 Structure보다
우수하다고 확정할 수는 없다.

Cars-196의 기존 Structure 실행 결과는 있다. `pcds_structure_sgds_official`의 Cars-196
10-task official Structure run이며 Inc `84.86`, Last `79.62`다. 그러나 현재 sweep과 task
partition, 학습 seed, epochs, batch size, adapter placement 등이 달라 **참고값으로만** 사용할 수
있다. 현재 CAR5/CAR10의 정식 delta를 계산하려면 동일 조건 vanilla Structure run이 각각 필요하다.

## 2. 실행 완결성과 baseline 가용성

| Protocol | 완료 | 실패 | partial | 대기 | 동일 조건 Structure baseline | 비교 방식 |
|---|---:|---:|---:|---:|---|---|
| IA5 | 27/27 | 0 | 0 | 0 | 있음 | Structure 대비 정식 delta/guardrail |
| IA10 | 23/27 | 4 | 0 | 0 | 없음 | 절대값만 |
| CAR5 | 10/27 | 0 | 1 | 16 | 없음 | 완료분 절대값만 |
| CAR10 | 0/27 | 0 | 0 | 27 | 없음 | 결과 없음 |

보고서 시점의 활성 항목은 `CAR5/B_P010_A050`, Task 5/5다. 상태 파일은 `partial`이지만
queue와 GPU 1 학습 프로세스는 살아 있다.

IA5 기준선은
`structure_e3_followup/outputs/imageneta/B0_STRUCTURE/seed_1993`의 `B0_STRUCTURE`다.
이 결과는 원래 E0 Structure 결과와도 동일하다. 현재 IA5와 다음 항목이 일치한다.

- ImageNet-A train/test 파일 구성 및 class별 sample 수
- class order 전체 200개와 seed 1993
- task partition `40 × 5`
- ViT-B/16 MLP adapter, SCC, APR, classifier adaptation
- init/incremental epochs `30/15`, batch size 48, learning rate 0.01

Structure만 `old_new_separation=false`, `adaptive_margin_mode=none`이며, 이 차이가 비교하려는
방법 효과다. 정확도와 forgetting 단위는 `%`, delta 단위는 percentage point다. `ΔF<0`이
개선이다.

## 3. IA5 — Structure 기준 전체 결과

### 3.1 Main table

| Run | Method | P | α/λ | Inc | ΔInc | Last | ΔLast | F↓ | ΔF | Old | ΔOld | New | ΔNew | Guardrail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Structure** | vanilla | — | — | 74.60 | 0.00 | 66.43 | 0.00 | 7.84 | 0.00 | 66.99 | 0.00 | 63.84 | 0.00 | ✓ |
| C0 | AM-Drift | — | — | **74.77** | **+0.17** | 66.69 | +0.26 | 6.97 | -0.87 | 67.55 | +0.56 | 62.73 | -1.11 | ✗ |
| A_P005 | A | .05 | — | 74.67 | +0.07 | 66.69 | +0.26 | 7.07 | -0.77 | 67.63 | +0.64 | 62.36 | -1.48 | ✗ |
| A_P010 | A | .10 | — | 74.55 | -0.05 | 66.43 | +0.00 | 7.27 | -0.57 | 67.39 | +0.40 | 61.99 | -1.85 | ✗ |
| A_P015 | A | .15 | — | 74.55 | -0.05 | 66.62 | +0.20 | 7.05 | -0.79 | 67.47 | +0.48 | 62.73 | -1.11 | ✗ |
| A_P020 | A | .20 | — | 74.58 | -0.02 | 66.23 | -0.20 | 7.41 | -0.44 | 67.15 | +0.16 | 61.99 | -1.85 | ✗ |
| A_P025 | A | .25 | — | 74.76 | +0.16 | 66.49 | +0.07 | 7.45 | -0.40 | 67.47 | +0.48 | 61.99 | -1.85 | ✗ |
| B_P005_A025 | B | .05 | .25 | 74.56 | -0.04 | 66.36 | -0.07 | 7.28 | -0.56 | 67.15 | +0.16 | 62.73 | -1.11 | ✗ |
| B_P005_A050 | B | .05 | .50 | 74.40 | -0.20 | 66.49 | +0.07 | 6.98 | -0.86 | 67.39 | +0.40 | 62.36 | -1.48 | ✗ |
| B_P005_A100 | B | .05 | 1.00 | 74.56 | -0.04 | 66.43 | +0.00 | 7.21 | -0.63 | 67.31 | +0.32 | 62.36 | -1.48 | ✗ |
| B_P010_A025 | B | .10 | .25 | 74.60 | +0.00 | 66.43 | +0.00 | 7.30 | -0.54 | 67.23 | +0.24 | 62.73 | -1.11 | ✗ |
| B_P010_A050 | B | .10 | .50 | 74.41 | -0.19 | 66.36 | -0.07 | 7.01 | -0.84 | 67.23 | +0.24 | 62.36 | -1.48 | ✗ |
| B_P010_A100 | B | .10 | 1.00 | 74.50 | -0.10 | 66.49 | +0.07 | 7.01 | -0.83 | 67.31 | +0.32 | 62.73 | -1.11 | ✗ |
| B_P020_A025 | B | .20 | .25 | 74.58 | -0.02 | 66.56 | +0.13 | 6.80 | -1.05 | 67.63 | +0.64 | 61.62 | -2.21 | ✗ |
| B_P020_A050 | B | .20 | .50 | 74.58 | -0.02 | **66.89** | **+0.46** | **6.67** | **-1.17** | **67.95** | **+0.96** | 61.99 | -1.85 | ✗ |
| B_P020_A100 | B | .20 | 1.00 | 74.55 | -0.04 | 66.36 | -0.07 | 7.01 | -0.83 | 67.23 | +0.24 | 62.36 | -1.48 | ✗ |
| C_P005_L010 | C | .05 | .10 | 74.54 | -0.06 | 66.62 | +0.20 | 7.52 | -0.33 | 67.23 | +0.24 | **63.84** | +0.00 | ✓ |
| C_P005_L030 | C | .05 | .30 | 74.68 | +0.08 | 66.62 | +0.20 | 7.45 | -0.39 | 67.39 | +0.40 | 63.10 | -0.74 | ✗ |
| C_P005_L100 | C | .05 | 1.00 | 74.57 | -0.03 | 66.49 | +0.07 | 7.42 | -0.43 | 67.31 | +0.32 | 62.73 | -1.11 | ✗ |
| C_P005_L300 | C | .05 | 3.00 | 74.49 | -0.11 | 66.62 | +0.20 | 7.23 | -0.62 | 67.55 | +0.56 | 62.36 | -1.48 | ✗ |
| C_P010_L010 | C | .10 | .10 | 74.60 | +0.00 | 66.56 | +0.13 | 7.27 | -0.57 | 67.31 | +0.32 | 63.10 | -0.74 | ✗ |
| C_P010_L030 | C | .10 | .30 | 74.67 | +0.07 | 66.16 | -0.26 | 8.14 | +0.29 | 66.99 | +0.00 | 62.36 | -1.48 | ✗ |
| C_P010_L100 | C | .10 | 1.00 | 74.60 | +0.00 | 66.56 | +0.13 | 7.31 | -0.53 | 67.55 | +0.56 | 61.99 | -1.85 | ✗ |
| C_P010_L300 | C | .10 | 3.00 | 74.68 | +0.08 | 66.75 | +0.33 | 7.01 | -0.83 | 67.71 | +0.72 | 62.36 | -1.48 | ✗ |
| C_P020_L010 | C | .20 | .10 | 74.53 | -0.07 | 66.62 | +0.20 | 7.26 | -0.58 | 67.23 | +0.24 | **63.84** | +0.00 | ✓ |
| **C_P020_L030** | **C** | **.20** | **.30** | **74.64** | **+0.04** | **66.56** | **+0.13** | **7.86** | **+0.02** | **67.15** | **+0.16** | **63.84** | **+0.00** | **✓** |
| **C_P020_L100** | **C** | **.20** | **1.00** | **74.62** | **+0.02** | **66.62** | **+0.20** | **7.42** | **-0.42** | **67.31** | **+0.32** | **63.47** | **-0.37** | **✓** |
| C_P020_L300 | C | .20 | 3.00 | 74.68 | +0.08 | 66.62 | +0.20 | 7.21 | -0.63 | 67.47 | +0.48 | 62.73 | -1.11 | ✗ |

Guardrail은 Structure 대비 `ΔOld ≥ -0.5`, `ΔNew ≥ -0.5`, `ΔF ≤ +0.5 pp`다. 통과는
세 지표의 손실이 허용 범위 안이라는 뜻이며 우수성을 보장하지 않는다. C0와 A/B는
Forgetting/Old를 대체로 개선하지만 New 손실 때문에 모두 탈락한다.

### 3.2 대표 설정의 task별 정확도

| Run | T1 | T2 | T3 | T4 | T5 |
|---|---:|---:|---:|---:|---:|
| Structure | 83.89 | 79.37 | 74.15 | 69.15 | 66.43 |
| C0 | 83.89 | 79.37 | 74.26 | **69.63** | 66.69 |
| A_P025 | 83.89 | 79.37 | 74.56 | 69.47 | 66.49 |
| B_P020_A050 | 83.89 | 78.80 | 73.95 | 69.39 | **66.89** |
| C_P020_L030 | 83.89 | 79.37 | 74.15 | 69.23 | 66.56 |
| C_P020_L100 | 83.89 | 79.66 | 73.95 | 68.99 | 66.62 |

T1은 old-new separation이 활성화되기 전이라 동일하다. 변형들은 일부 후반 task에서
Structure를 넘지만, final New 정확도를 함께 보면 stability–plasticity trade-off가 확인된다.

## 4. IA5 방법별 진단

### 4.1 Method A — Selective AM-Drift

| P | mean selected similarity | active separation ratio | mean pair loss | ΔInc vs Structure | ΔNew vs Structure |
|---:|---:|---:|---:|---:|---:|
| .05 | .619 | .188 | .0268 | +0.07 | -1.48 |
| .10 | .558 | .174 | .0250 | -0.05 | -1.85 |
| .15 | .519 | .171 | .0247 | -0.05 | -1.11 |
| .20 | .487 | .170 | .0228 | -0.02 | -1.85 |
| .25 | .465 | .160 | .0216 | +0.16 | -1.85 |

P가 커질수록 낮은-similarity old class가 선택 집합에 들어가 mean similarity와 평균 pair
loss가 감소한다. A_P025는 Inc가 가장 높지만 New 손실이 커서 Structure를 안정적으로
대체할 수 없다. Method A의 어느 설정도 Structure 기준 guardrail을 통과하지 못했다.

### 4.2 Method B — Similarity-Enlarged Boundary

아래 진단은 final task/final epoch sampled pair 기준이다.

| P | α | mean τ scale | intrusion ratio | mean sep loss | ΔInc | ΔLast | ΔF |
|---:|---:|---:|---:|---:|---:|---:|---:|
| .05 | .25 | 1.035 | .191 | .0298 | -0.04 | -0.07 | -0.56 |
| .05 | .50 | 1.071 | .216 | .0340 | -0.20 | +0.07 | -0.86 |
| .05 | 1.00 | 1.143 | .264 | .0458 | -0.04 | +0.00 | -0.63 |
| .10 | .25 | 1.038 | .186 | .0280 | +0.00 | +0.00 | -0.54 |
| .10 | .50 | 1.077 | .210 | .0322 | -0.19 | -0.07 | -0.84 |
| .10 | 1.00 | 1.152 | .254 | .0444 | -0.10 | +0.07 | -0.83 |
| .20 | .25 | 1.041 | .171 | .0243 | -0.02 | +0.13 | -1.05 |
| .20 | .50 | 1.082 | .190 | .0289 | -0.02 | +0.46 | **-1.17** |
| .20 | 1.00 | 1.164 | .243 | .0427 | -0.04 | -0.07 | -0.83 |

α 증가에 따라 tau scale, intrusion, separation loss가 증가해 구현 의도는 확인됐다.
모든 B 설정이 Structure보다 Forgetting을 줄였지만 New는 `-1.11`에서 `-2.21 pp` 낮다.
즉 enlarged boundary는 stability를 높이는 대신 plasticity를 해쳤다.

### 4.3 Method C — Proto-CLAD

| P | selected cosine λ=.10 → 3.00 | cosine gap λ=.10 → 3.00 | IA5 best Inc setting |
|---:|---:|---:|---|
| .05 | .310 → .227 | .045 → .062 | C_P005_L030: 74.68 |
| .10 | .276 → .200 | .045 → .060 | C_P010_L300: 74.68 |
| .20 | .237 → .166 | .045 → .056 | C_P020_L300: 74.68 |

λ가 커질수록 selected old prototype cosine은 낮아지고 correct-new cosine gap은 커졌다.
classification–Proto-CLAD gradient cosine도 λ 증가와 함께 더 음수가 되는 경향을 보였다.
Geometry 변화가 정확도 향상을 보장하지는 않았지만, Structure 기준 guardrail 통과 설정은
Method C에서만 나왔다. 그중 `P=.20, λ=.30–1.00`이 가장 균형적이다.

## 5. IA10 중간 결과 — matched Structure 없음

현재 보유한 vanilla Structure는 IA5 `40 × 5` 결과뿐이다. IA10은 `20 × 10`이므로 IA5
결과를 기준선으로 재사용할 수 없다. 아래는 최종 10-task까지 완료된 23개 run의 절대값이며,
delta나 guardrail 판정을 하지 않는다.

| Run | Method | Inc | Last | F↓ | Old | New |
|---|---|---:|---:|---:|---:|---:|
| A_P005 | A | 74.13 | 65.17 | 8.35 | 64.95 | 67.74 |
| A_P010 | A | 74.15 | 64.98 | 8.71 | 64.66 | 68.55 |
| A_P020 | A | 74.09 | **65.44** | 8.19 | **65.16** | 68.55 |
| A_P025 | A | 74.09 | 65.17 | 8.15 | 64.95 | 67.74 |
| B_P005_A025 | B | 74.14 | 65.37 | **7.93** | 65.02 | 69.35 |
| B_P005_A100 | B | 74.09 | 64.98 | 8.44 | 64.66 | 68.55 |
| B_P010_A025 | B | 74.22 | 65.37 | 8.37 | 65.09 | 68.55 |
| B_P010_A050 | B | 74.07 | 65.24 | 8.12 | 64.87 | 69.35 |
| B_P020_A025 | B | 74.06 | 65.11 | 8.97 | 64.59 | **70.97** |
| B_P020_A050 | B | 74.15 | 65.11 | 8.56 | 64.73 | 69.35 |
| **B_P020_A100** | **B** | **74.30** | 65.17 | 8.08 | 65.02 | 66.94 |
| C_P005_L010 | C | 73.66 | 64.65 | 9.21 | 64.44 | 66.94 |
| C_P005_L030 | C | 73.81 | 64.85 | 8.62 | 64.73 | 66.13 |
| C_P005_L100 | C | 74.04 | 64.91 | 9.23 | 64.73 | 66.94 |
| C_P005_L300 | C | 74.06 | 65.24 | 8.92 | 64.95 | 68.55 |
| C_P010_L010 | C | 73.80 | 64.52 | 9.14 | 64.30 | 66.94 |
| C_P010_L030 | C | 73.96 | 65.04 | 8.61 | 64.87 | 66.94 |
| C_P010_L100 | C | 74.04 | 64.71 | 9.35 | 64.52 | 66.94 |
| **C_P010_L300** | **C** | **74.34** | **65.31** | 9.18 | **65.16** | 66.94 |
| C_P020_L010 | C | 73.85 | 64.98 | 8.76 | 64.80 | 66.94 |
| C_P020_L030 | C | 73.75 | 64.71 | 8.62 | 64.44 | 67.74 |
| C_P020_L100 | C | 73.83 | 64.58 | 9.25 | 64.37 | 66.94 |
| C_P020_L300 | C | 74.06 | 64.85 | 9.47 | 64.52 | 68.55 |

완료 run 중 Inc 최고는 C_P010_L300 `74.34`, B 계열 최고는 B_P020_A100 `74.30`이다.
하지만 Structure보다 좋은지는 판단할 수 없다. C0를 포함한 4개 실패도 selection bias를 만들 수 있다.

모든 27개가 공통 완료한 Task 1–5 prefix의 대표 절대값은 다음과 같다. 이 값도 `20 × 10`
protocol의 중간 지점이며 IA5 Structure와 비교하지 않는다.

| Run | Inc5 | Last5 | F5↓ | Old5 | New5 |
|---|---:|---:|---:|---:|---:|
| C0 AM-Drift | 80.08 | 73.96 | 6.06 | 76.07 | 63.64 |
| A_P005 | 79.96 | 74.08 | 6.05 | 75.93 | 65.03 |
| A_P025 | 79.99 | 74.20 | 6.26 | 76.22 | 64.34 |
| B_P010_A025 | 80.11 | 74.32 | 6.38 | 76.22 | 65.03 |
| B_P010_A100* | 80.13 | 74.55 | 6.09 | 76.65 | 64.34 |
| B_P020_A100 | **80.26** | 74.67 | 6.25 | **76.65** | 65.03 |
| C_P005_L030 | 79.63 | 74.32 | 6.32 | 75.79 | **67.13** |
| C_P005_L100 | 80.06 | **74.79** | 7.05 | 76.36 | **67.13** |
| C_P010_L300 | 80.21 | 74.55 | 7.34 | 76.22 | 66.43 |

`*`는 Task 6 classifier adaptation에서 실패한 run이다.

## 6. CAR5 현재 완료분 — matched Structure 없음

현재 CAR5의 C0는 **AM-Drift**이며 vanilla Structure가 아니다. 따라서 C0 delta를 baseline
delta로 사용하지 않고 완료 run의 절대값만 기록한다.

| Run | Method | Inc | Last | F↓ | Old | New |
|---|---|---:|---:|---:|---:|---:|
| C0 | AM-Drift | 84.89 | **81.30** | 4.03 | 81.13 | 81.99 |
| A_P005 | A | 84.90 | **81.30** | **3.94** | **81.17** | 81.80 |
| A_P010 | A | 84.92 | 81.17 | 4.11 | 81.05 | 81.67 |
| A_P015 | A | 84.90 | 81.13 | 4.23 | 81.00 | 81.67 |
| A_P020 | A | **84.97** | 81.20 | 4.14 | 81.08 | 81.67 |
| A_P025 | A | 84.89 | 81.17 | 4.15 | 81.05 | 81.67 |
| B_P005_A025 | B | 84.91 | 81.25 | 3.95 | **81.17** | 81.55 |
| B_P005_A050 | B | 84.89 | 81.18 | 4.00 | 81.06 | 81.67 |
| B_P005_A100 | B | 84.92 | **81.30** | 4.03 | 81.11 | **82.06** |
| B_P010_A025 | B | 84.91 | 81.21 | 4.06 | 81.10 | 81.67 |

완료분의 절대값 차이는 작지만, Structure 기준선이 없으므로 현재 어느 설정도 “Structure보다
개선”이라고 표현할 수 없다. 나머지 B/C가 끝나기 전 방법 순위도 잠정적이다.

### 6.1 기존 Cars Structure 결과의 위치

기존 결과는 다음 경로에 있다.

`pcds_structure_sgds_official/outputs/cars196_10task_official_structure/seed_0/pcds_structure`

| Run | Tasks | Inc | Last | F↓ | Old | New |
|---|---:|---:|---:|---:|---:|---:|
| Official Structure (PCDS) | 10 | 84.86 | 79.62 | 8.79 | 79.12 | 85.15 |

이 결과와 현재 sweep은 class order seed 1993 및 Cars-196 train/test split은 공유하지만,
아래 조건이 다르다.

| 조건 | 기존 official Structure | 현재 CAR5/CAR10 sweep |
|---|---|---|
| task partition | `20 × 9 + 16` | CAR5 `40 + 39 × 4`; CAR10 `20 × 6 + 19 × 4` |
| training seed | 0 | 1993 |
| feature epochs | 70 / 40 | 30 / 15 |
| batch size | 32 | 48 |
| initial LR | .005 | .01 |
| adapter placement | attention | MLP |
| margin | .3 | .2 |

따라서 official Structure 수치는 문헌/구현 sanity reference로는 유효하지만, 현재 CAR5 또는
CAR10 표에 delta를 붙이는 기준선으로는 부적합하다.

## 7. Cross-protocol 해석 — 잠정

### 7.1 5-task와 10-task 중 어디에서 선택적 방법이 유리한가?

아직 확정할 수 없다. 동일 조건 Structure 비교가 가능한 것은 IA5뿐이다. IA10은 높은 절대
Inc를 보인 후보가 있지만 matched Structure가 없어 상대 개선 여부를 판단할 수 없다.

### 7.2 Cars-196에서 선택적 방법이 더 유효한가?

아직 답할 수 없다. 현재 CAR5 run끼리의 차이는 작고 sweep도 미완료다. 기존 official Cars
Structure는 protocol이 달라 직접 비교할 수 없다. CAR5/CAR10 동일 조건 Structure run이
완료된 뒤에만 dataset 효과를 주장할 수 있다.

### 7.3 현재 가장 강한 신호는 무엇인가?

IA5에서 AM-Drift와 Method B는 Forgetting/Old를 개선하는 대신 New를 낮춘다. Method C의
`P=.20, λ=.30–1.00`만 이 trade-off를 제한하면서 Structure 수준의 New를 유지했다. 다만
Inc 향상은 최대 `+0.04 pp`여서 추가 seed 검증이 필요하다.

## 8. 실행 시간과 자원

| Protocol | 완료 run | 평균 학습 시간 | 범위 | 최대 CUDA allocated | 최대 main RSS |
|---|---:|---:|---:|---:|---:|
| IA5 | 27 | 48.9 min | 45.1–52.1 min | 5.85 GiB | 4.10 GiB |
| IA10 | 23 | 65.3 min | 62.3–67.5 min | 5.85 GiB | 4.27 GiB |
| CAR5 | 10 | 94.0 min | 90.8–97.2 min | 5.85 GiB | 4.10 GiB |

IA10의 기록된 feature-training 합계는 IA5의 약 `1.34×`, CAR5는 IA5의 약 `1.92×`다.

## 9. IA10 실패 4건

| Run | 성공 완료 범위 | 실패 위치 | 오류 |
|---|---|---|---|
| C0 | Task 1–5 | Task 6 classifier adaptation | covariance matrix not PositiveDefinite |
| A_P015 | Task 1–5 | Task 6 classifier adaptation | 동일 |
| B_P005_A050 | Task 1–5 | Task 6 classifier adaptation | 동일 |
| B_P010_A100 | Task 1–5 | Task 6 classifier adaptation | 동일 |

직접 원인은 `models/base.py::_stage2_compact_classifier`에서 768×768 full covariance가
positive definite 조건을 만족하지 못한 것이다. 결과 누락이 method-dependent selection bias를
만들 수 있으므로 결과는 참고용으로만 보존한다. 이번 reduced Stage 1에서는 네 checkpoint를
재개하지 않는다.

## 10. 다음 조치

1. 기존 108-run queue는 중단하고 완료·partial·failed 결과를 모두 보존한다.
2. IA5는 Structure와 선택된 네 방법이 모두 완료되어 재실행하지 않는다.
3. CAR5에서만 `Structure`, `B_P020_A100`, `C_P020_L030`, `C_P020_L100`을 새로 실행한다.
4. IA10과 CAR10은 HOLD하며 어떤 GPU 작업도 추가하지 않는다.
5. CAR5 완료 후 모든 delta와 guardrail을 Structure 기준으로 다시 계산한다.
6. seed 1996/1997과 10-task 검증은 5-task 최종 후보 1–2개를 정한 뒤 별도로 계획한다.

## 11. 재현 가능한 원자료

- IA5 Structure 기준선:
  `structure_e3_followup/outputs/imageneta/B0_STRUCTURE/seed_1993/{summary.json,task_summary.csv}`
- 기존 Cars official Structure:
  `pcds_structure_sgds_official/outputs/cars196_10task_official_structure/seed_0/pcds_structure/`
- 현재 sweep 전체 절대 지표: `structure_selective_similarity_sweep/outputs/metrics_per_seed.csv`
  (현재 snapshot의 `delta_*` 열은 C0 기준이라 사용하지 않음; reduced 실행 이후 exporter는
  Structure 기준으로 생성)
- run별 설정/상태: `resolved_config.json`, `run_metadata.json`, `summary.json`
- task/class 성능: `task_accuracy_matrix.csv`, `class_accuracy.csv`
- 방법별 진단: `selective_am_diagnostics.csv`, `enlarged_boundary_diagnostics.csv`,
  `proto_clad_diagnostics.csv`
- confusion/loss/gradient/resource: `confusion_task_XX.npz`, `loss_diagnostics.csv`,
  `selective_gradient_diagnostics.csv`, `resource_diagnostics.csv`

FROM CLAD CONCEPT는 Top-P conflict selection, class-aware similarity, disentanglement이다.
Selective AM-Drift, similarity-enlarged tau, APR prototype 기반 Proto-CLAD는 exemplar-free
Structure를 위한 adaptation이며 CLAD 원 논문의 직접 재현이 아니다.
