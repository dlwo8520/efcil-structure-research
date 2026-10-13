# Paired Source Prediction Input Audit

- Seed: 1993
- GPU: physical 4, process-local cuda:0
- Existing R256/G256 sample-level logits and trained weights: absent; only aggregate accuracies are available.
- h_A/h_B are reproduced from the exact prior phase-3 code path and checked against its aggregate accuracy.
- h_C uses the same fixed exposure, new bank, pre-CA head and optimization controls.
- Coverage support B/C is averaged over its three independent draws per canonical sample ID.

## IA10

- checkpoint: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/task_checkpoints/task_10.pt`
- feature_cache: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/.cache/model_10_on_task_10.npz`
- accuracy: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/g256_r256_final_task.csv`
- support_A: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/query_support_A.csv`
- support_B: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/query_support_B.csv`
- support_C: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/query_support_C.csv`
- support_delta: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/paired_support_delta.csv`

## CAR10

- checkpoint: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/task_checkpoints/task_10.pt`
- feature_cache: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/.cache/model_10_on_task_10.npz`
- accuracy: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/g256_r256_final_task.csv`
- support_A: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/query_support_A.csv`
- support_B: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/query_support_B.csv`
- support_C: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/query_support_C.csv`
- support_delta: OK — `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/paired_support_delta.csv`
