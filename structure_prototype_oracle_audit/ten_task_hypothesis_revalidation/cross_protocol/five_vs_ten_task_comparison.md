# Five-task vs Ten-task Structure Revalidation

## Q1-Q13 concise answers

The table below keeps prototype tracking, inter-class crowding, and temporal self-similarity as distinct quantities.

| Metric | IA5 | IA10 | CAR5 | CAR10 |
|---|---:|---:|---:|---:|
| Final tracking cosine | 0.9944 | 0.9927 | 0.9925 | 0.9885 |
| Final tracking relative L2 | 0.1007 | 0.1100 | 0.1087 | 0.1328 |
| Final nearest-rival cosine | 0.6913 | 0.7364 | 0.7762 | 0.8256 |
| Final covariance relative error | 0.2984 | 0.3529 | 0.2939 | 0.4034 |
| Oracle D-A (pp) | -0.3292 | -0.0658 | 0.6716 | 1.1068 |
| R256-G256 (pp) | -0.0658 | 0.0658 | 0.5472 | 0.8954 |
| Top-20 error share | 0.0882 | 0.0840 | 0.1276 | 0.1330 |
| Wrong-sample test purity | 0.1251 | 0.1190 | 0.1687 | 0.1690 |

## Hypothesis status

| Hypothesis | 5-task evidence | 10-task evidence | Final status |
|---|---|---|---|
| Prototype mean tracking error is the main bottleneck | IA5=0.1007; CAR5=0.1087 | IA10=0.1100; CAR10=0.1328 | **NOT SUPPORTED** |
| Covariance mismatch is the main bottleneck | IA5=0.2984; CAR5=0.2939 | IA10=0.3529; CAR10=0.4034, but C-A is only +0.132/+0.075pp | **PARTIALLY SUPPORTED** |
| Gaussian approximation loss is material | IA5=-0.0658; CAR5=0.5472 | IA10=0.0658; CAR10=0.8954 | **NOT SUPPORTED** |
| A small set of directional pairs dominates error | IA5=0.0882; CAR5=0.1276 | IA10=0.0840; CAR10=0.1330 | **NOT SUPPORTED** |
| Incremental pair degradation dominates recovery | Prior T4->T5 diagnostic reused; see pair_boundary_risk_diagnostic | IA10: degradation=931, improvement=1214; CAR10: degradation=1493, improvement=2397 | **NOT SUPPORTED** |
| Stored gamma predicts newly degraded relations | Previously weak at T4->T5 | Positive-delta captured mass: IA10=0.0069; CAR10=0.0385-0.0435 | **WEAKENED** |
| Broad generalization / coverage is the stronger explanation | IA5=0.1251; CAR5=0.1687 | IA10=0.1190; CAR10=0.1690 | **SUPPORTED** |

## Interpretation guardrails

Observed 5/10 differences are descriptive, not proof of causal transition accumulation: task size, class count, and age distributions differ. New-only gaps also include the easier-candidate-set effect and are not calibration evidence by themselves. Oracle actual data are diagnostic-only.

수동 교차검증과 Q1-Q13의 상세 해석은 [`../ten_task_hypothesis_revalidation_final_report.md`](../ten_task_hypothesis_revalidation_final_report.md)를 따른다.
