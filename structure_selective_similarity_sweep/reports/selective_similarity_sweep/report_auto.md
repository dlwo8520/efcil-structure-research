# Similarity-Selective Old-New Separation 결과 보고서

Primary baseline: **vanilla Structure**. 모든 delta는 동일 protocol/seed의 Structure 대비 값이다.

## 실행 완결성

| Protocol | Completed metric rows |
|---|---:|
| IA5 | 28 |
| CAR5 | 15 |
| IA10 | 23 |
| CAR10 | 0 |

## IA5 Stage-1

| Run | Method | P | alpha/lambda | Inc | ΔInc | Last | ΔLast | F↓ | ΔF | Old | ΔOld | New | ΔNew | Guardrail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Structure | Structure | — | — | 74.598 | 0.000 | 66.425 | 0.000 | 7.844 | 0.000 | 66.987 | 0.000 | 63.838 | 0.000 | True |
| A_P005 | A | 0.050 | — | 74.670 | 0.072 | 66.689 | 0.263 | 7.070 | -0.775 | 67.628 | 0.641 | 62.362 | -1.476 | False |
| B_P020_A100 | B | 0.200 | 1.000 | 74.554 | -0.043 | 66.359 | -0.066 | 7.010 | -0.834 | 67.228 | 0.240 | 62.362 | -1.476 | False |
| C_P020_L030 | C | 0.200 | 0.300 | 74.640 | 0.042 | 66.557 | 0.132 | 7.864 | 0.019 | 67.147 | 0.160 | 63.838 | 0.000 | True |
| C_P020_L100 | C | 0.200 | 1.000 | 74.621 | 0.024 | 66.623 | 0.197 | 7.424 | -0.421 | 67.308 | 0.321 | 63.469 | -0.369 | True |

## CAR5 Stage-1

| Run | Method | P | alpha/lambda | Inc | ΔInc | Last | ΔLast | F↓ | ΔF | Old | ΔOld | New | ΔNew | Guardrail |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Structure | Structure | — | — | 84.929 | 0.000 | 81.196 | 0.000 | 4.431 | 0.000 | 80.987 | 0.000 | 82.055 | 0.000 | True |
| A_P005 | A | 0.050 | — | 84.902 | -0.028 | 81.296 | 0.099 | 3.938 | -0.493 | 81.173 | 0.186 | 81.801 | -0.254 | True |
| B_P020_A100 | B | 0.200 | 1.000 | 84.920 | -0.009 | 81.246 | 0.050 | 4.076 | -0.355 | 81.049 | 0.062 | 82.055 | 0.000 | True |
| C_P020_L030 | C | 0.200 | 0.300 | 84.905 | -0.024 | 81.184 | -0.012 | 4.450 | 0.018 | 81.002 | 0.015 | 81.928 | -0.127 | True |
| C_P020_L100 | C | 0.200 | 1.000 | 84.854 | -0.076 | 81.184 | -0.012 | 4.369 | -0.063 | 81.018 | 0.031 | 81.864 | -0.190 | True |

## 해석 원칙

- FROM CLAD CONCEPT: Top-P conflict selection, class-aware similarity, disentanglement.
- ADAPTED FOR EXEMPLAR-FREE STRUCTURE: APR prototype, Selective AM-Drift, similarity-enlarged tau, Proto-CLAD.
- Reduced Stage 1은 IA5/CAR5, seed 1993의 방법 탐색 결과입니다.
- IA10 기존 결과는 참고용으로만 보존하고 CAR10과 함께 HOLD합니다.
- 추가 seed와 10-task 검증은 5-task 최종 후보 1–2개를 정한 뒤 별도로 계획합니다.
