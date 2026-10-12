# StoredAPR RP-ProtoCLAD 최종 실행 보고서

작성 기준: 2026-09-23 KST  
새 run: `StoredAPR_RP_C_P020_L030`  
범위: IA5·CAR5, seed 1993, P=0.20, λ=0.30, RP `[768, 2048]`, RP seed 1993

## 1. 실행 상태

| Protocol | 상태 | Completed tasks | 학습 시간 |
|---|---|---:|---:|
| IA5 | COMPLETE | 5/5 | 45.3 min |
| CAR5 | COMPLETE | 5/5 | 93.2 min |

GPU 2 queue는 IA5 다음 CAR5를 순차 실행해 `failures=0`으로 종료했다. Queue wall time은
대략 2시간 23분이며, task 학습 시간의 합은 약 2시간 19분이다.

- Queue log: `logs/stored_apr_gpu2_queue.log`
- IA5 log: `logs/IA5__StoredAPR_RP_C_P020_L030__seed_1993.log`
- CAR5 log: `logs/CAR5__StoredAPR_RP_C_P020_L030__seed_1993.log`

각 run은 `summary.json`, `run_metadata.json`, 5개 `task_final` 행,
`checkpoint_latest.pt`, `stored_rp_birth_prototypes.pt`를 갖는다. 기존 RP 코드와 output은
덮어쓰지 않았다.

## 2. 방법 확인

실제로 사용한 old projected prototype은 다음과 같다.

```text
q_birth(c) = mean_i(ReLU(z_i R))
q_old(c,t,e) = q_birth(c)
               + ReLU(mu_APR(c,t,e) R)
               - ReLU(mu_birth(c) R)
```

Top-P conflict selection, 원공간 APR/SCC/classifier adaptation, covariance와 class mean은
기존 RP run과 동일하다. Class별 birth statistic만 별도 저장하며 raw exemplar는 저장하지 않는다.

## 3. 성능 결과

### 절대값

| Protocol | Run | Inc | Last | F↓ | Old | New |
|---|---|---:|---:|---:|---:|---:|
| IA5 | Structure | 74.598 | 66.425 | 7.844 | 66.987 | 63.838 |
| IA5 | C_P020_L030 | 74.640 | 66.557 | 7.864 | 67.147 | 63.838 |
| IA5 | 기존 RP_C_P020_L030 | **74.704** | **66.820** | **7.310** | **67.468** | **63.838** |
| IA5 | StoredAPR RP | 74.632 | 66.294 | 7.867 | 67.228 | 61.993 |
| CAR5 | Structure | 84.929 | 81.196 | 4.431 | 80.987 | **82.055** |
| CAR5 | C_P020_L030 | 84.905 | 81.184 | 4.450 | 81.002 | 81.928 |
| CAR5 | 기존 RP_C_P020_L030 | — | — | — | — | — |
| CAR5 | StoredAPR RP | **84.938** | **81.221** | **4.387** | **81.049** | 81.928 |

### StoredAPR의 delta

| Protocol | 비교 기준 | ΔInc | ΔLast | ΔF | ΔOld | ΔNew |
|---|---|---:|---:|---:|---:|---:|
| IA5 | Structure | +0.034 | -0.132 | +0.022 | +0.240 | -1.845 |
| IA5 | C_P020_L030 | -0.008 | -0.263 | +0.003 | +0.080 | -1.845 |
| IA5 | 기존 RP_C_P020_L030 | **-0.072** | **-0.527** | **+0.557** | **-0.240** | **-1.845** |
| CAR5 | Structure | +0.009 | +0.025 | -0.044 | +0.062 | -0.127 |
| CAR5 | C_P020_L030 | +0.033 | +0.037 | -0.062 | +0.046 | +0.000 |
| CAR5 | 기존 RP_C_P020_L030 | — | — | — | — | — |

ΔF는 음수가 개선이다. IA5에서는 StoredAPR이 기존 RP보다 모든 핵심 지표에서 불리했고,
특히 New가 `-1.845 pp` 낮았다. CAR5에서는 Structure와 C 대비 Inc/Last/Old/F가 모두 아주
조금 좋아졌지만 New는 Structure보다 `-0.127 pp` 낮았다. CAR5 기존 RP run이 없어
StoredAPR의 직접적인 old-RP ablation 결론은 아직 낼 수 없다.

## 4. Task별 accuracy

### IA5

| Run | T1 | T2 | T3 | T4 | T5 |
|---|---:|---:|---:|---:|---:|
| Structure | 83.889 | 79.370 | 74.154 | 69.151 | 66.425 |
| 기존 RP | 83.889 | 79.083 | 74.256 | 69.471 | 66.820 |
| StoredAPR RP | 83.889 | **79.513** | 74.154 | 69.311 | 66.294 |

### CAR5

| Run | T1 | T2 | T3 | T4 | T5 |
|---|---:|---:|---:|---:|---:|
| Structure | 88.962 | 86.616 | 85.184 | 82.689 | 81.196 |
| C_P020_L030 | 88.962 | 86.646 | 85.122 | 82.611 | 81.184 |
| StoredAPR RP | 88.962 | 86.585 | **85.204** | **82.720** | **81.221** |

## 5. Prototype 및 geometry diagnostic

| Protocol | Birth cos(A,B) | Birth rel-L2 | q_current vs old-style cos | q_current vs old-style rel-L2 | APR Δq norm | q_current 음수 비율 |
|---|---:|---:|---:|---:|---:|---:|
| IA5 | 0.913 | 0.425 | 0.915 | 0.561 | 7.149 | 0.000010 |
| CAR5 | 0.924 | 0.396 | 0.925 | 0.514 | 8.427 | 0.000000 |

여기서 birth의 A는 `ReLU(mean(z)R)`, B는 `mean(ReLU(zR))`다. Birth rel-L2가
IA5 `0.425`, CAR5 `0.396`이므로 기존 old-style 근사가 작지 않다는 관측이 재확인됐다.
새 방식은 B를 직접 저장하므로 birth 시점의 이 근사를 사용하지 않는다. APR Δq norm이 0이
아니므로 이후 epoch의 APR drift correction도 실제로 적용됐다.

### IA5 기존 RP와 geometry 비교

| Run | Selected old RP cosine | Correct-new RP gap | Δgap(RP−original) | grad cos(cls, RP) |
|---|---:|---:|---:|---:|
| 기존 RP | 0.551 | 0.192 | -0.105 | -0.033 |
| StoredAPR RP | 0.626 | 0.081 | -0.215 | -0.032 |

StoredAPR은 selected old prototype과 current sample의 cosine을 더 높이고 correct-new gap을
더 작게 만들었다. 같은 run에서 New accuracy가 `-1.845 pp` 낮아진 것과 방향상 일치한다.
그러나 단일 seed의 관측적 동반 변화이므로 geometry가 accuracy 저하의 원인이라고 확정할
수는 없다.

Final-task IA5 confusion에서도 기존 RP 대비 new→old 오류율이 `31.365% → 32.841%`로
증가했고, old→new 오류율은 `8.413% → 8.173%`로 소폭 감소했다. 즉 IA5에서는 plasticity
손실과 stability의 작은 이동이 동시에 나타났다.

## 6. 연구 질문에 대한 답

1. **Q1 — 기존 RP보다 Inc를 개선했는가?**  
   IA5에서는 아니다(`-0.072 pp`). CAR5 기존 RP 결과가 없어 CAR5 직접 비교는 미정이다.

2. **Q2 — 어떤 성능이 변했는가?**  
   IA5 기존 RP 대비 Last `-0.527`, F `+0.557`(악화), Old `-0.240`, New `-1.845`다.
   CAR5는 Structure 대비 Inc `+0.009`, Last `+0.025`, F `-0.044`, Old `+0.062`,
   New `-0.127`이다.

3. **Q3 — `ReLU(mean(z)R)` 근사 문제를 줄였는가?**  
   구현 정의상 birth prototype에서는 제거했다. 모든 sample을 먼저 RP+ReLU한 뒤 평균한
   `q_birth`를 저장했다. 다만 근사 제거가 IA5 성능 개선으로 이어지지는 않았다.

4. **Q4 — APR drift가 정상 반영됐는가?**  
   그렇다. 매 epoch `ReLU(mu_APR R)-ReLU(mu_birth R)`를 더했고 평균 Δq norm은 IA5
   `7.149`, CAR5 `8.427`이었다. Zero-drift 및 checkpoint round-trip sanity test도 통과했다.

5. **Q5 — New를 희생하지 않고 Old/F가 개선됐는가?**  
   IA5에서는 아니다. CAR5는 Structure 대비 Old/F가 소폭 개선됐지만 New가 `-0.127 pp`
   낮아 “희생 없음”이라고 엄격히 말할 수 없다. 0.5 pp 허용 guardrail 기준으로는 통과한다.

6. **Q6 — Geometry 변화가 accuracy 변화와 연결되는가?**  
   IA5에서 smaller gap과 증가한 new→old confusion, 낮아진 New가 함께 관측됐다. 연관은
   있지만 단일 seed 결과이므로 인과관계는 미확정이다.

7. **Q7 — IA5와 CAR5에서 같은 방향인가?**  
   Structure 대비 Inc는 둘 다 아주 작게 증가했지만, IA5의 New 손실은 크고 CAR5의 변화는
   작다. 따라서 실질적인 stability–plasticity 효과가 동일하다고 보기는 어렵다.

## 7. 결론과 다음 단계

- 정확한 sample-wise birth RP prototype 저장과 APR displacement 결합은 의도대로 동작했다.
- 그러나 **IA5에서는 기존 RP보다 명확히 불리**해 현재 결과만으로 기존 RP를 대체할 근거가 없다.
- CAR5는 Structure/C보다 미세하게 개선됐지만 차이가 `0.01–0.06 pp` 수준이며 기존 RP
  comparator가 없다.
- 가장 필요한 다음 실험은 새로운 sweep이 아니라 **동일 설정의 CAR5
  `RP_C_P020_L030` 한 건**이다. 이 결과가 있어야 CAR5에서 StoredAPR 자체의 효과를
  분리할 수 있다.
- 그 전에는 Multi-View, 추가 λ/RP seed/dimension sweep으로 범위를 넓히지 않는 것이 좋다.

## 8. 제한 사항

- 각 protocol seed 하나이므로 분산과 유의성을 평가할 수 없다.
- IA5와 CAR5의 class 수·표본 구조가 달라 rel-L2나 Δq norm의 절대 크기를 직접 비교하면 안 된다.
- `q_birth + delta_q`는 수식상 음수가 가능하다. 실제 평균 음수 비율은 IA5 약 `0.001%`,
  CAR5 `0%`로 매우 작았으며 사양대로 추가 ReLU를 적용하지 않았다.

