# IA5/CAR5 T5 Classifier Procedure Diagnostic — Final

## 첫 페이지 결론

| 질문 | IA5 | CAR5 |
|---|---|---|
| Q1. 기존 G/R exposure 불균형? | **YES** | **YES** |
| Q2. G256으로 회복? | **YES** | **YES** |
| Q3. R256 balance 회복? | **YES** | **YES** |
| Q4. R256 > G256? | **INCONCLUSIVE** | **YES** |
| Q5. 5-epoch cosine 충분? | **INCONCLUSIVE** | **INCONCLUSIVE** |
| Q6. Converged > 5ep? | **NO** | **NO** |
| Q7. Linear no-bias > cosine? | **NO** | **NO** |
| Q8. Linear bias 추가 개선? | **YES** | **YES** |
| Q9. 다음 우선순위 | **YES — sampling/weighting** | **YES — sampling/weighting** |

## 핵심 delta

| Protocol | G256−Gfixed | R256−Rfixed | R256−G256 | Converged−5ep | Linear-no-bias−Converged | Bias gain | Priority |
|---|---:|---:|---:|---:|---:|---:|---|
| IA5 | 3.8183 | 3.4233 | -0.0658 | -0.9217 | -6.4516 | 0.7900 | sampling/weighting |
| CAR5 | 5.7207 | 5.3725 | 0.5472 | -0.1865 | -17.9331 | 0.7710 | sampling/weighting |

## T2–T4 확장 권고

**HOLD** — 두 protocol의 우선 병목 판정이 같아 T5 원인 분리 결론이 명확하다. 현재 단계에서는 자동 확장하지 않는다.

## 범위와 주의

이 결과는 actual old/current training feature를 사용하는 diagnostic oracle probe다. 정식 replay-free CIL method 결과가 아니며 actual feature를 main training이나 checkpoint에 전달하지 않았다. T2–T4, IA10/CAR10 및 신규 방법론은 실행하지 않았다.
