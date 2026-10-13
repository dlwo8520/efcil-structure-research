# Structure Similarity-Aware 후속 실험 보고서

공통 완료 seed: 1993 (n=1). 정확도와 forgetting은 %, 차이는 paired percentage point다. n=1이면 SD는 N/A이며 관측 pilot 결과로만 해석한다.

## 1. Main Performance

| Method | n | Inc-Acc | Delta_Inc-Acc | Last-Acc | Delta_Last-Acc | Forgetting | Delta_Forgetting | Old | Delta_Old | New | Delta_New |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | 1 | 74.60 (SD N/A) | 0.00 (SD N/A) | 66.43 (SD N/A) | 0.00 (SD N/A) | 7.84 (SD N/A) | 0.00 (SD N/A) | 66.99 (SD N/A) | 0.00 (SD N/A) | 63.84 (SD N/A) | 0.00 (SD N/A) |
| B1 | 1 | 74.77 (SD N/A) | +0.17 (SD N/A) | 66.69 (SD N/A) | +0.26 (SD N/A) | 6.97 (SD N/A) | -0.87 (SD N/A) | 67.55 (SD N/A) | +0.56 (SD N/A) | 62.73 (SD N/A) | -1.11 (SD N/A) |
| S0 | 1 | 74.60 (SD N/A) | 0.00 (SD N/A) | 66.43 (SD N/A) | 0.00 (SD N/A) | 7.84 (SD N/A) | 0.00 (SD N/A) | 66.99 (SD N/A) | 0.00 (SD N/A) | 63.84 (SD N/A) | 0.00 (SD N/A) |
| S1 | 1 | 74.70 (SD N/A) | +0.10 (SD N/A) | 66.62 (SD N/A) | +0.20 (SD N/A) | 7.40 (SD N/A) | -0.45 (SD N/A) | 67.47 (SD N/A) | +0.48 (SD N/A) | 62.73 (SD N/A) | -1.11 (SD N/A) |
| S2 | 1 | 74.45 (SD N/A) | -0.15 (SD N/A) | 66.62 (SD N/A) | +0.20 (SD N/A) | 7.24 (SD N/A) | -0.60 (SD N/A) | 67.31 (SD N/A) | +0.32 (SD N/A) | 63.47 (SD N/A) | -0.37 (SD N/A) |
| S3 | 1 | 74.50 (SD N/A) | -0.10 (SD N/A) | 66.36 (SD N/A) | -0.07 (SD N/A) | 8.26 (SD N/A) | +0.42 (SD N/A) | 66.91 (SD N/A) | -0.08 (SD N/A) | 63.84 (SD N/A) | 0.00 (SD N/A) |
| S4 | 1 | 74.55 (SD N/A) | -0.05 (SD N/A) | 66.36 (SD N/A) | -0.07 (SD N/A) | 8.12 (SD N/A) | +0.27 (SD N/A) | 66.99 (SD N/A) | 0.00 (SD N/A) | 63.47 (SD N/A) | -0.37 (SD N/A) |

![stability-plasticity](figures/stability_plasticity.png)

오른쪽 위가 old/new를 동시에 개선한 영역이다. 점의 위치는 공통 seed에서 계산한 paired delta 평균이며 유의성을 뜻하지 않는다.

## 2. S0: Similar old-new가 진짜 더 문제인가?

| Similarity group | n | old accuracy drop | forgetting | old->new confusion | new->old confusion |
|---|---|---|---|---|---|
| Low | 100 | 1.405 | 1.405 | 0.0774 | 0.0011 |
| Mid | 200 | 4.756 | 4.756 | 0.1372 | 0.0022 |
| High | 100 | 4.054 | 4.054 | 0.1412 | 0.0035 |
| Top1 | 112 | 6.100 | 6.100 | 0.1528 | 0.0035 |
| Top5 | 285 | 4.415 | 4.415 | 0.1274 | 0.0027 |

![S0 relationship](figures/s0_similarity_relationship.png)

각 old class와 현재 new classes의 최대 pre-training cosine을 사용했다. 상관은 원인으로 해석하지 않는다. 원자료는 [s0_old_class_records.csv](tables/s0_old_class_records.csv), top-1 사례는 [s0_top1_pair_examples.csv](tables/s0_top1_pair_examples.csv)다.

## 3. S1: Similarity-aware tau

![S1 tau](figures/s1_similarity_tau.png)

S1 진단 row 수는 240,000개다. similarity가 높을수록 tau scale이 낮아지는지와 B1 대비 New/Old/Forgetting을 함께 판단한다.

## 4. S2: Relative margin

![S2 gap](figures/s2_similarity_gap.png)

S2 진단 row 수는 67,650개다. 높은 similarity에서 cosine gap과 margin violation, new→hard-old confusion을 함께 확인한다.

## 5. S3: Similarity-aware SCC

![S3 SCC](figures/s3_similarity_scc.png)

S3 진단 row 수는 67,650개다. SCC 완화로 New가 회복되더라도 Old/Forgetting이 크게 악화되면 성공으로 보지 않는다.

## 6. S4: Soft hybrid

![S4 gate](figures/s4_similarity_gate.png)

S4 진단 row 수는 67,650개다. hard threshold S2 대비 trade-off와 seed 방향 반복을 확인한다.

## 7. Stability-Plasticity

위 stability-plasticity 그림의 x는 ΔOld, y는 ΔNew다. Inc/Last/Forgetting과 함께 판단하며 단일 headline metric만으로 성공을 선언하지 않는다.

## 8. Feature norm / Gradient

![feature norm](figures/feature_norm.png)

E3처럼 norm inflation이 재발하는지 확인한다. gradient cosine 요약은 [gradient_cosine_summary.csv](tables/gradient_cosine_summary.csv)에 저장했다.

## 9. 의미 있는 class pair 사례

S0의 top-1 pair와 method별 `relative_margin_diagnostics.csv`, `cross_task_confusion.csv`, `exports/class_accuracy.csv`를 class ID로 결합할 수 있다. 최소 5개 사례 선정은 공통 3-seed에서 개선/악화 방향이 반복되는 pair만 대상으로 한다.

## 10. 연구 질문 결론

- RQ1 Similar pair가 더 충돌하는가: S0 similarity group의 forgetting/confusion 차이로 판단한다.
- RQ2 similarity-aware tau가 new 손실을 줄이는가: S1 vs B1의 paired New와 Old/F를 함께 본다.
- RQ3 relative margin이 나은가: S2 vs B1/S1의 gap, confusion, main metric을 본다.
- RQ4 SCC 완화가 plasticity를 회복하는가: S3 vs S2 paired New/Old/F와 gradient를 본다.
- RQ5 soft hybrid가 안정적인가: S4 vs S1/S2의 seed별 paired delta 분산을 본다.

각 결론은 저장된 공통 seed 수가 3보다 작으면 pilot 관측으로 제한한다.

## 재생성

```bash
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
python ${RESEARCH_WORKSPACE}/structure_similarity_aware_followup/analysis/build_report.py \
  --output-root ${RESEARCH_WORKSPACE}/structure_similarity_aware_followup/outputs \
  --report-dir ${RESEARCH_WORKSPACE}/structure_similarity_aware_followup/reports/similarity_aware_followup
```
