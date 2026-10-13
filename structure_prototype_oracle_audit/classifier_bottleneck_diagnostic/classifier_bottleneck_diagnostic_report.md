# IA5/CAR5 Classifier Bottleneck Diagnostic — Combined

## 통합 결론 — YES / NO / INCONCLUSIVE

| 질문 | 통합 답 | IA5 | CAR5 |
|---|---|---|---|
| Q1. Oracle NCM은 Full Oracle Gaussian CA보다 높은가? | **NO** | NO | NO |
| Q2. Actual old feature CA는 matched Gaussian CA보다 높은가? | **YES** | YES | YES |
| Q3. Gaussian→Actual이 New→Old confusion을 줄이는가? | **INCONCLUSIVE** | INCONCLUSIVE | INCONCLUSIVE |
| Q4. Actual CA에도 New-only gap이 크게 남는가? | **INCONCLUSIVE** | YES | NO |
| Q5. 단일 bias로 Old를 크게 잃지 않고 New를 회복하는가? | **NO** | NO | NO |
| Q6. Structure/D/R이 서로 다른 bias frontier를 만드는가? | **YES** | YES | YES |
| Q7. CA 이후 weight가 class mean 방향에서 멀어지는가? | **NO** | NO | NO |
| Q8. G/R 차이를 optimization/fitting 차이로 설명할 수 있는가? | **INCONCLUSIVE** | INCONCLUSIVE | INCONCLUSIVE |

## Final-task 핵심 delta

| Protocol | Tasks | Oracle NCM − D Seen | R-fixed − G-fixed Seen | ΔNew→Old (pp) | R New-only gap |
|---|---|---:|---:|---:|---:|
| IA5 | `[2, 3, 4, 5]` | -4.5425 | 0.3292 | 0.0000 | 5.5351 |
| CAR5 | `[2, 3, 4, 5]` | -16.5278 | 0.8954 | 0.0634 | 1.6487 |

## 전 transition 일관성

| Protocol | Task | R−G Seen (pp) | Oracle-NCM−D Seen (pp) |
|---|---:|---:|---:|
| IA5 | 2 | 0.4298 | -3.8682 |
| IA5 | 3 | 0.6154 | -4.4103 |
| IA5 | 4 | 1.0417 | -3.6859 |
| IA5 | 5 | 0.3292 | -4.5425 |
| CAR5 | 2 | 0.3369 | -13.8132 |
| CAR5 | 3 | 0.3283 | -15.4115 |
| CAR5 | 4 | 0.8199 | -15.6714 |
| CAR5 | 5 | 0.8954 | -16.5278 |

## 원인 분리 해석

- Actual old feature는 모든 transition에서 fixed Gaussian보다 약간 높았지만 최대 이득이 1.05 pp 이하였다. 따라서 old Gaussian replacement가 일부 손실은 만들지만, 현재 관측된 병목의 주원인이라는 강한 증거는 아니다.
- Oracle-NCM은 모든 transition에서 Full Oracle Gaussian CA보다 낮았다. 따라서 현재 frozen feature에서 prototype classifier 정보가 CA보다 월등한데 classifier fitting이 이를 잃는다는 가설은 지지되지 않는다.
- Actual 교체는 final-task New→Old confusion을 줄이지 않았다. 단일 old/new bias도 Old 손실 1 pp 이내에서 큰 New 회복을 만들지 못했다.
- 남는 후보는 CA objective, classifier form, optimization과 feature-space class overlap이지만, 이번 진단만으로 어느 하나를 원인으로 확정하지 않는다.

## IA10/CAR10 확장 권고

**RECOMMEND (별도 후속 실행, 이번 실행에는 미포함).** IA5/CAR5 전 transition에서 R−G 방향과 NCM−D 방향이 모두 일관되어, 장기 sequence에서 작은 G/R 차이가 누적되는지만 IA10/CAR10으로 검증할 근거가 생겼다. 다만 현재 결론을 내기 위해 긴 protocol이 필수인 것은 아니다.

Optional G/R Linear는 실행하지 않았다. G/R 차이가 모든 transition에서 작고 같은 방향이어서 old source의 주효과 판단은 모호하지 않았으며, 이번 범위에서 새로운 classifier 실험을 추가할 필요가 없었다.

## 주의

Actual>Gaussian은 multimodality를 단독 원인으로 증명하지 않으며, Actual≈Gaussian도 representation 원인을 증명하지 않는다. Bias sweep은 test-oracle diagnostic이고 정식 EF-CIL result가 아니다.
