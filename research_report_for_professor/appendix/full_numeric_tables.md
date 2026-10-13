# 주요 수치 전체표

자동 집계: `build_report.py --stage all`. Accuracy %, delta pp, 원문 ratio/count/CE는 구분한다. 모두 model seed 1993이며 bank/draw seed를 model replicate로 간주하지 않는다.

## memory_age

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | mean_relative_L2 | 0.000002 | ratio | age=0; pooled class-task records |
| IA5 | mean_relative_L2 | 0.066891 | ratio | age=1; pooled class-task records |
| IA5 | mean_relative_L2 | 0.100968 | ratio | age=2; pooled class-task records |
| IA5 | mean_relative_L2 | 0.129730 | ratio | age=3; pooled class-task records |
| IA5 | mean_relative_L2 | 0.156988 | ratio | age=4; pooled class-task records |
| CAR5 | mean_relative_L2 | 0.000001 | ratio | age=0; pooled class-task records |
| CAR5 | mean_relative_L2 | 0.065609 | ratio | age=1; pooled class-task records |
| CAR5 | mean_relative_L2 | 0.105698 | ratio | age=2; pooled class-task records |
| CAR5 | mean_relative_L2 | 0.147218 | ratio | age=3; pooled class-task records |
| CAR5 | mean_relative_L2 | 0.187675 | ratio | age=4; pooled class-task records |
| IA10 | mean_relative_L2 | 0.000002 | ratio | age=0; pooled class-task records |
| IA10 | mean_relative_L2 | 0.057474 | ratio | age=1; pooled class-task records |
| IA10 | mean_relative_L2 | 0.080493 | ratio | age=2; pooled class-task records |
| IA10 | mean_relative_L2 | 0.099881 | ratio | age=3; pooled class-task records |
| IA10 | mean_relative_L2 | 0.117721 | ratio | age=4; pooled class-task records |
| IA10 | mean_relative_L2 | 0.133399 | ratio | age=5; pooled class-task records |
| IA10 | mean_relative_L2 | 0.149644 | ratio | age=6; pooled class-task records |
| IA10 | mean_relative_L2 | 0.164742 | ratio | age=7; pooled class-task records |
| IA10 | mean_relative_L2 | 0.177156 | ratio | age=8; pooled class-task records |
| IA10 | mean_relative_L2 | 0.204598 | ratio | age=9; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.000001 | ratio | age=0; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.056544 | ratio | age=1; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.084328 | ratio | age=2; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.110481 | ratio | age=3; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.135178 | ratio | age=4; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.160734 | ratio | age=5; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.186910 | ratio | age=6; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.211580 | ratio | age=7; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.238935 | ratio | age=8; pooled class-task records |
| CAR10 | mean_relative_L2 | 0.269991 | ratio | age=9; pooled class-task records |

원본: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/prototype_tracking_by_age.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/prototype_tracking_by_age.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/prototype_tracking_by_age.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/prototype_tracking_by_age.csv`

## oracle_CA

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | A_STRUCTURE | 66.425280 | % | final seen micro; diagnostic head |
| IA5 | B_ORACLE_MEAN | 66.030283 | % | final seen micro; diagnostic head |
| IA5 | C_ORACLE_COVARIANCE | 66.359447 | % | final seen micro; diagnostic head |
| IA5 | D_FULL_ORACLE | 66.096116 | % | final seen micro; diagnostic head |
| IA5 | CHECKPOINT_MAIN | 66.425280 | % | final seen micro; diagnostic head |
| CAR5 | A_STRUCTURE | 81.196369 | % | final seen micro; diagnostic head |
| CAR5 | B_ORACLE_MEAN | 81.507275 | % | final seen micro; diagnostic head |
| CAR5 | C_ORACLE_COVARIANCE | 81.221241 | % | final seen micro; diagnostic head |
| CAR5 | D_FULL_ORACLE | 81.867927 | % | final seen micro; diagnostic head |
| CAR5 | CHECKPOINT_MAIN | 81.196369 | % | final seen micro; diagnostic head |
| IA10 | A_STRUCTURE | 64.647795 | % | final seen micro; diagnostic head |
| IA10 | B_ORACLE_MEAN | 64.581962 | % | final seen micro; diagnostic head |
| IA10 | C_ORACLE_COVARIANCE | 64.779460 | % | final seen micro; diagnostic head |
| IA10 | D_FULL_ORACLE | 64.581962 | % | final seen micro; diagnostic head |
| IA10 | CHECKPOINT_MAIN | 64.647795 | % | final seen micro; diagnostic head |
| CAR10 | A_STRUCTURE | 78.957841 | % | final seen micro; diagnostic head |
| CAR10 | B_ORACLE_MEAN | 79.492600 | % | final seen micro; diagnostic head |
| CAR10 | C_ORACLE_COVARIANCE | 79.032459 | % | final seen micro; diagnostic head |
| CAR10 | D_FULL_ORACLE | 80.064669 | % | final seen micro; diagnostic head |
| CAR10 | CHECKPOINT_MAIN | 78.957841 | % | final seen micro; diagnostic head |

원본: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/oracle_abcd_final_task.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/oracle_abcd_final_task.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/oracle_abcd_final_task.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/oracle_abcd_final_task.csv`

## matched_source

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | G256_FIXED | 66.688611 | % | 256/class/epoch; old source changed; current bank fixed |
| IA5 | R256_FIXED | 66.622778 | % | 256/class/epoch; old source changed; current bank fixed |
| CAR5 | G256_FIXED | 81.718692 | % | 256/class/epoch; old source changed; current bank fixed |
| CAR5 | R256_FIXED | 82.265887 | % | 256/class/epoch; old source changed; current bank fixed |
| IA10 | G256_FIXED | 64.713627 | % | 256/class/epoch; old source changed; current bank fixed |
| IA10 | R256_FIXED | 64.779460 | % | 256/class/epoch; old source changed; current bank fixed |
| CAR10 | G256_FIXED | 79.728889 | % | 256/class/epoch; old source changed; current bank fixed |
| CAR10 | R256_FIXED | 80.624300 | % | 256/class/epoch; old source changed; current bank fixed |

원본: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/g256_r256_final_task.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/g256_r256_final_task.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/g256_r256_final_task.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/g256_r256_final_task.csv`

## memory_tracking

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | prototype_relative_l2 | 0.098692 | ratio | all old class-task records; train oracle |
| IA5 | frobenius_relative_error | 0.303006 | ratio | all old class-task records; train oracle |
| CAR5 | prototype_relative_l2 | 0.106371 | ratio | all old class-task records; train oracle |
| CAR5 | frobenius_relative_error | 0.288396 | ratio | all old class-task records; train oracle |
| IA10 | prototype_relative_l2 | 0.108565 | ratio | all old class-task records; train oracle |
| IA10 | frobenius_relative_error | 0.374102 | ratio | all old class-task records; train oracle |
| CAR10 | prototype_relative_l2 | 0.127057 | ratio | all old class-task records; train oracle |
| CAR10 | frobenius_relative_error | 0.390922 | ratio | all old class-task records; train oracle |

원본: `structure_prototype_oracle_audit/scientific_audit/CAR10/seed_1993/covariance_oracle_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/CAR10/seed_1993/prototype_oracle_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/CAR5/seed_1993/covariance_oracle_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/CAR5/seed_1993/prototype_oracle_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/IA10/seed_1993/covariance_oracle_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/IA10/seed_1993/prototype_oracle_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/IA5/seed_1993/covariance_oracle_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/IA5/seed_1993/prototype_oracle_audit.csv`

## APR

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | improved_records | 400 | count | old class-task records; total=400 |
| CAR5 | improved_records | 392 | count | old class-task records; total=394 |
| IA10 | improved_records | 900 | count | old class-task records; total=900 |
| CAR10 | improved_records | 889 | count | old class-task records; total=894 |

원본: `structure_prototype_oracle_audit/scientific_audit/CAR10/seed_1993/apr_oracle_drift_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/CAR5/seed_1993/apr_oracle_drift_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/IA10/seed_1993/apr_oracle_drift_audit.csv`, `structure_prototype_oracle_audit/scientific_audit/IA5/seed_1993/apr_oracle_drift_audit.csv`

## classifier_probe

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | STRUCTURE_CA | 66.425280 | % | same frozen features; validation-selected; diagnostic |
| IA5 | COSINE_5EP | 66.491113 | % | same frozen features; validation-selected; diagnostic |
| IA5 | COSINE_CONVERGED | 65.569454 | % | same frozen features; validation-selected; diagnostic |
| IA5 | LINEAR_NO_BIAS | 59.117841 | % | same frozen features; validation-selected; diagnostic |
| IA5 | LINEAR_WITH_BIAS | 59.907834 | % | same frozen features; validation-selected; diagnostic |
| CAR5 | STRUCTURE_CA | 81.196369 | % | same frozen features; validation-selected; diagnostic |
| CAR5 | COSINE_5EP | 81.619202 | % | same frozen features; validation-selected; diagnostic |
| CAR5 | COSINE_CONVERGED | 81.432658 | % | same frozen features; validation-selected; diagnostic |
| CAR5 | LINEAR_NO_BIAS | 63.499565 | % | same frozen features; validation-selected; diagnostic |
| CAR5 | LINEAR_WITH_BIAS | 64.270613 | % | same frozen features; validation-selected; diagnostic |

원본: `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/CAR5_T5/classifier_probe_accuracy.csv`, `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/IA5_T5/classifier_probe_accuracy.csv`

## sampling_fairness

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | FULL_ORACLE_GAUSSIAN_CA | 66.096116 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| IA5 | G_FIXED | 62.870309 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| IA5 | R_FIXED | 63.199473 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| IA5 | G256_FIXED | 66.688611 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| IA5 | R256_FIXED | 66.622778 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| CAR5 | FULL_ORACLE_GAUSSIAN_CA | 81.867927 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| CAR5 | G_FIXED | 75.998010 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| CAR5 | R_FIXED | 76.893421 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| CAR5 | G256_FIXED | 81.718692 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |
| CAR5 | R256_FIXED | 82.265887 | % | G_FIXED/R_FIXED exposure imbalance; do not compare as source-only effect |

원본: `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/CAR5_T5/g256_r256_accuracy.csv`, `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/IA5_T5/g256_r256_accuracy.csv`

## pretrained_final

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | PRETRAINED_train_accuracy | 89.701953 | % | same split/optimizer/budget; test not used for selection |
| IA10 | PRETRAINED_test_accuracy | 50.427913 | % | same split/optimizer/budget; test not used for selection |
| IA10 | PRETRAINED_class_macro_accuracy | 40.742504 | % | same split/optimizer/budget; test not used for selection |
| IA10 | PRETRAINED_support_margin_micro | -0.011712 | cosine gap | same split/optimizer/budget; test not used for selection |
| IA10 | STRUCTURE_FINAL_train_accuracy | 90.791367 | % | same split/optimizer/budget; test not used for selection |
| IA10 | STRUCTURE_FINAL_test_accuracy | 61.816985 | % | same split/optimizer/budget; test not used for selection |
| IA10 | STRUCTURE_FINAL_class_macro_accuracy | 50.205427 | % | same split/optimizer/budget; test not used for selection |
| IA10 | STRUCTURE_FINAL_support_margin_micro | 0.022510 | cosine gap | same split/optimizer/budget; test not used for selection |
| CAR10 | PRETRAINED_train_accuracy | 92.164292 | % | same split/optimizer/budget; test not used for selection |
| CAR10 | PRETRAINED_test_accuracy | 56.585002 | % | same split/optimizer/budget; test not used for selection |
| CAR10 | PRETRAINED_class_macro_accuracy | 56.535219 | % | same split/optimizer/budget; test not used for selection |
| CAR10 | PRETRAINED_support_margin_micro | -0.009728 | cosine gap | same split/optimizer/budget; test not used for selection |
| CAR10 | STRUCTURE_FINAL_train_accuracy | 99.378600 | % | same split/optimizer/budget; test not used for selection |
| CAR10 | STRUCTURE_FINAL_test_accuracy | 80.810844 | % | same split/optimizer/budget; test not used for selection |
| CAR10 | STRUCTURE_FINAL_class_macro_accuracy | 80.701859 | % | same split/optimizer/budget; test not used for selection |
| CAR10 | STRUCTURE_FINAL_support_margin_micro | 0.033297 | cosine gap | same split/optimizer/budget; test not used for selection |

원본: `structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/probe_results.csv`, `structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/probe_results.csv`

## coverage_source

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | delta_B_minus_A | -0.007001 | cosine margin | equal per-class reference count; pooled 3 Gaussian draws, not model seeds |
| IA10 | delta_C_minus_B | -0.004746 | cosine margin | equal per-class reference count; pooled 3 Gaussian draws, not model seeds |
| IA10 | actual_support_margin | 0.022510 | cosine margin | overall; classifier_correct= |
| CAR10 | delta_B_minus_A | -0.010422 | cosine margin | equal per-class reference count; pooled 3 Gaussian draws, not model seeds |
| CAR10 | delta_C_minus_B | -0.012000 | cosine margin | equal per-class reference count; pooled 3 Gaussian draws, not model seeds |
| CAR10 | actual_support_margin | 0.033297 | cosine margin | overall; classifier_correct= |

원본: `structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/paired_support_delta.csv`, `structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/support_summary.csv`, `structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/paired_support_delta.csv`, `structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/support_summary.csv`

## trajectory

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | A | 327 | count | birth<=T9; task-end checkpoints only |
| IA10 | B | 155 | count | birth<=T9; task-end checkpoints only |
| IA10 | C | 13 | count | birth<=T9; task-end checkpoints only |
| IA10 | B1_restored | 126 | count | candidate set = all classes seen at birth |
| IA10 | checkpoint_final_accuracy | 64.713627 | % | task_end checkpoint direct inference; not pre-after_task summary |
| CAR10 | A | 907 | count | birth<=T9; task-end checkpoints only |
| CAR10 | B | 560 | count | birth<=T9; task-end checkpoints only |
| CAR10 | C | 81 | count | birth<=T9; task-end checkpoints only |
| CAR10 | B1_restored | 473 | count | candidate set = all classes seen at birth |
| CAR10 | checkpoint_final_accuracy | 78.957841 | % | task_end checkpoint direct inference; not pre-after_task summary |

원본: `error_trajectory_diagnostic/CAR10/sample_trajectory.csv`, `error_trajectory_diagnostic/IA10/sample_trajectory.csv`

## rank_preserving

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | C0_repair | 0 | count | T10 whole seen micro |
| IA10 | C0_harm | 0 | count | T10 whole seen micro |
| IA10 | C0_net | 0 | count | T10 whole seen micro |
| IA10 | C0_delta_pp | 0.000000 | pp | T10 whole seen micro |
| IA10 | C0_independent_Gaussian_CE | 0.431093 | CE | independent synthetic bank; NOT held-out image accuracy |
| IA10 | C1_repair | 2 | count | T10 whole seen micro |
| IA10 | C1_harm | 1 | count | T10 whole seen micro |
| IA10 | C1_net | 1 | count | T10 whole seen micro |
| IA10 | C1_delta_pp | 0.065833 | pp | T10 whole seen micro |
| IA10 | C1_independent_Gaussian_CE | 0.430838 | CE | independent synthetic bank; NOT held-out image accuracy |
| IA10 | C2_repair | 7 | count | T10 whole seen micro |
| IA10 | C2_harm | 15 | count | T10 whole seen micro |
| IA10 | C2_net | -8 | count | T10 whole seen micro |
| IA10 | C2_delta_pp | -0.526662 | pp | T10 whole seen micro |
| IA10 | C2_independent_Gaussian_CE | 0.322080 | CE | independent synthetic bank; NOT held-out image accuracy |
| CAR10 | C0_repair | 0 | count | T10 whole seen micro |
| CAR10 | C0_harm | 0 | count | T10 whole seen micro |
| CAR10 | C0_net | 0 | count | T10 whole seen micro |
| CAR10 | C0_delta_pp | 0.000000 | pp | T10 whole seen micro |
| CAR10 | C0_independent_Gaussian_CE | 0.626735 | CE | independent synthetic bank; NOT held-out image accuracy |
| CAR10 | C1_repair | 5 | count | T10 whole seen micro |
| CAR10 | C1_harm | 12 | count | T10 whole seen micro |
| CAR10 | C1_net | -7 | count | T10 whole seen micro |
| CAR10 | C1_delta_pp | -0.087054 | pp | T10 whole seen micro |
| CAR10 | C1_independent_Gaussian_CE | 0.626540 | CE | independent synthetic bank; NOT held-out image accuracy |
| CAR10 | C2_repair | 66 | count | T10 whole seen micro |
| CAR10 | C2_harm | 103 | count | T10 whole seen micro |
| CAR10 | C2_net | -37 | count | T10 whole seen micro |
| CAR10 | C2_delta_pp | -0.460142 | pp | T10 whole seen micro |
| CAR10 | C2_independent_Gaussian_CE | 0.447319 | CE | independent synthetic bank; NOT held-out image accuracy |

원본: `b1_ranking_preserving_correction/CAR10/metrics.json`, `b1_ranking_preserving_correction/IA10/metrics.json`

## cross_task_CA

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | repair_n | 3 | count | internal paired control; S0 parity FAIL |
| IA10 | harm_n | 6 | count | internal paired control; S0 parity FAIL |
| IA10 | net_gain_n | -3 | count | internal paired control; S0 parity FAIL |
| IA10 | delta_accuracy | -0.197498 | pp | internal paired control; S0 parity FAIL |
| CAR10 | repair_n | 56 | count | internal paired control; S0 parity FAIL |
| CAR10 | harm_n | 50 | count | internal paired control; S0 parity FAIL |
| CAR10 | net_gain_n | 6 | count | internal paired control; S0 parity FAIL |
| CAR10 | delta_accuracy | 0.074618 | pp | internal paired control; S0 parity FAIL |

원본: `cross_task_ca_reweighting/CAR10/summary.json`, `cross_task_ca_reweighting/IA10/summary.json`

## local_all_audit

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | T9_N | 147.000000 | count | T9_all; S->G |
| IA10 | T9_local_repair | 1.000000 | count | T9_all; S->G |
| IA10 | T9_local_harm | 0.000000 | count | T9_all; S->G |
| IA10 | T9_local_net | 1.000000 | count | T9_all; S->G |
| IA10 | T9_local_delta_pp | 0.680272 | pp | T9_all; S->G |
| IA10 | T9_all_repair | 2.000000 | count | T9_all; S->G |
| IA10 | T9_all_harm | 2.000000 | count | T9_all; S->G |
| IA10 | T9_all_net | 0.000000 | count | T9_all; S->G |
| IA10 | T9_all_delta_pp | 0.000000 | pp | T9_all; S->G |
| IA10 | T9_reference_macro_all_accuracy_pct | 50.048993 | % | T9_all; S->G |
| IA10 | T9_target_macro_all_accuracy_pct | 51.874389 | % | T9_all; S->G |
| IA10 | T9_N | 27.000000 | count | T9_A_within; S->G |
| IA10 | T9_local_repair | 0.000000 | count | T9_A_within; S->G |
| IA10 | T9_local_harm | 0.000000 | count | T9_A_within; S->G |
| IA10 | T9_local_net | 0.000000 | count | T9_A_within; S->G |
| IA10 | T9_local_delta_pp | 0.000000 | pp | T9_A_within; S->G |
| IA10 | T9_all_repair | 0.000000 | count | T9_A_within; S->G |
| IA10 | T9_all_harm | 0.000000 | count | T9_A_within; S->G |
| IA10 | T9_all_net | 0.000000 | count | T9_A_within; S->G |
| IA10 | T9_all_delta_pp | 0.000000 | pp | T9_A_within; S->G |
| IA10 | T9_reference_macro_all_accuracy_pct | 0.000000 | % | T9_A_within; S->G |
| IA10 | T9_target_macro_all_accuracy_pct | 0.000000 | % | T9_A_within; S->G |
| IA10 | T9_N | 34.000000 | count | T9_A_cross; S->G |
| IA10 | T9_local_repair | 0.000000 | count | T9_A_cross; S->G |
| IA10 | T9_local_harm | 0.000000 | count | T9_A_cross; S->G |
| IA10 | T9_local_net | 0.000000 | count | T9_A_cross; S->G |
| IA10 | T9_local_delta_pp | 0.000000 | pp | T9_A_cross; S->G |
| IA10 | T9_all_repair | 1.000000 | count | T9_A_cross; S->G |
| IA10 | T9_all_harm | 0.000000 | count | T9_A_cross; S->G |
| IA10 | T9_all_net | 1.000000 | count | T9_A_cross; S->G |
| IA10 | T9_all_delta_pp | 2.941176 | pp | T9_A_cross; S->G |
| IA10 | T9_reference_macro_all_accuracy_pct | 0.000000 | % | T9_A_cross; S->G |
| IA10 | T9_target_macro_all_accuracy_pct | 3.571429 | % | T9_A_cross; S->G |
| IA10 | T9_N | 86.000000 | count | T9_non_A; S->G |
| IA10 | T9_local_repair | 1.000000 | count | T9_non_A; S->G |
| IA10 | T9_local_harm | 0.000000 | count | T9_non_A; S->G |
| IA10 | T9_local_net | 1.000000 | count | T9_non_A; S->G |
| IA10 | T9_local_delta_pp | 1.162791 | pp | T9_non_A; S->G |
| IA10 | T9_all_repair | 1.000000 | count | T9_non_A; S->G |
| IA10 | T9_all_harm | 2.000000 | count | T9_non_A; S->G |
| IA10 | T9_all_net | -1.000000 | count | T9_non_A; S->G |
| IA10 | T9_all_delta_pp | -1.162791 | pp | T9_non_A; S->G |
| IA10 | T9_reference_macro_all_accuracy_pct | 96.862745 | % | T9_non_A; S->G |
| IA10 | T9_target_macro_all_accuracy_pct | 95.882353 | % | T9_non_A; S->G |
| IA10 | T10_N | 147.000000 | count | T9_all; S->G |
| IA10 | T10_local_repair | 1.000000 | count | T9_all; S->G |
| IA10 | T10_local_harm | 1.000000 | count | T9_all; S->G |
| IA10 | T10_local_net | 0.000000 | count | T9_all; S->G |
| IA10 | T10_local_delta_pp | 0.000000 | pp | T9_all; S->G |
| IA10 | T10_all_repair | 1.000000 | count | T9_all; S->G |
| IA10 | T10_all_harm | 1.000000 | count | T9_all; S->G |
| IA10 | T10_all_net | 0.000000 | count | T9_all; S->G |
| IA10 | T10_all_delta_pp | 0.000000 | pp | T9_all; S->G |
| IA10 | T10_reference_macro_all_accuracy_pct | 50.048993 | % | T9_all; S->G |
| IA10 | T10_target_macro_all_accuracy_pct | 49.731532 | % | T9_all; S->G |
| IA10 | T10_N | 27.000000 | count | T9_A_within; S->G |
| IA10 | T10_local_repair | 1.000000 | count | T9_A_within; S->G |
| IA10 | T10_local_harm | 1.000000 | count | T9_A_within; S->G |
| IA10 | T10_local_net | 0.000000 | count | T9_A_within; S->G |
| IA10 | T10_local_delta_pp | 0.000000 | pp | T9_A_within; S->G |
| IA10 | T10_all_repair | 0.000000 | count | T9_A_within; S->G |
| IA10 | T10_all_harm | 0.000000 | count | T9_A_within; S->G |
| IA10 | T10_all_net | 0.000000 | count | T9_A_within; S->G |
| IA10 | T10_all_delta_pp | 0.000000 | pp | T9_A_within; S->G |
| IA10 | T10_reference_macro_all_accuracy_pct | 0.000000 | % | T9_A_within; S->G |
| IA10 | T10_target_macro_all_accuracy_pct | 0.000000 | % | T9_A_within; S->G |
| IA10 | T10_N | 34.000000 | count | T9_A_cross; S->G |
| IA10 | T10_local_repair | 0.000000 | count | T9_A_cross; S->G |
| IA10 | T10_local_harm | 0.000000 | count | T9_A_cross; S->G |
| IA10 | T10_local_net | 0.000000 | count | T9_A_cross; S->G |
| IA10 | T10_local_delta_pp | 0.000000 | pp | T9_A_cross; S->G |
| IA10 | T10_all_repair | 1.000000 | count | T9_A_cross; S->G |
| IA10 | T10_all_harm | 0.000000 | count | T9_A_cross; S->G |
| IA10 | T10_all_net | 1.000000 | count | T9_A_cross; S->G |
| IA10 | T10_all_delta_pp | 2.941176 | pp | T9_A_cross; S->G |
| IA10 | T10_reference_macro_all_accuracy_pct | 0.000000 | % | T9_A_cross; S->G |
| IA10 | T10_target_macro_all_accuracy_pct | 2.380952 | % | T9_A_cross; S->G |
| IA10 | T10_N | 86.000000 | count | T9_non_A; S->G |
| IA10 | T10_local_repair | 0.000000 | count | T9_non_A; S->G |
| IA10 | T10_local_harm | 0.000000 | count | T9_non_A; S->G |
| IA10 | T10_local_net | 0.000000 | count | T9_non_A; S->G |
| IA10 | T10_local_delta_pp | 0.000000 | pp | T9_non_A; S->G |
| IA10 | T10_all_repair | 0.000000 | count | T9_non_A; S->G |
| IA10 | T10_all_harm | 1.000000 | count | T9_non_A; S->G |
| IA10 | T10_all_net | -1.000000 | count | T9_non_A; S->G |
| IA10 | T10_all_delta_pp | -1.162791 | pp | T9_non_A; S->G |
| IA10 | T10_reference_macro_all_accuracy_pct | 96.862745 | % | T9_non_A; S->G |
| IA10 | T10_target_macro_all_accuracy_pct | 95.392157 | % | T9_non_A; S->G |
| CAR10 | T9_N | 758.000000 | count | T9_all; S->G |
| CAR10 | T9_local_repair | 2.000000 | count | T9_all; S->G |
| CAR10 | T9_local_harm | 7.000000 | count | T9_all; S->G |
| CAR10 | T9_local_net | -5.000000 | count | T9_all; S->G |
| CAR10 | T9_local_delta_pp | -0.659631 | pp | T9_all; S->G |
| CAR10 | T9_all_repair | 8.000000 | count | T9_all; S->G |
| CAR10 | T9_all_harm | 12.000000 | count | T9_all; S->G |
| CAR10 | T9_all_net | -4.000000 | count | T9_all; S->G |
| CAR10 | T9_all_delta_pp | -0.527704 | pp | T9_all; S->G |
| CAR10 | T9_reference_macro_all_accuracy_pct | 82.277425 | % | T9_all; S->G |
| CAR10 | T9_target_macro_all_accuracy_pct | 81.791043 | % | T9_all; S->G |
| CAR10 | T9_N | 42.000000 | count | T9_A_within; S->G |
| CAR10 | T9_local_repair | 1.000000 | count | T9_A_within; S->G |
| CAR10 | T9_local_harm | 2.000000 | count | T9_A_within; S->G |
| CAR10 | T9_local_net | -1.000000 | count | T9_A_within; S->G |
| CAR10 | T9_local_delta_pp | -2.380952 | pp | T9_A_within; S->G |
| CAR10 | T9_all_repair | 0.000000 | count | T9_A_within; S->G |
| CAR10 | T9_all_harm | 0.000000 | count | T9_A_within; S->G |
| CAR10 | T9_all_net | 0.000000 | count | T9_A_within; S->G |
| CAR10 | T9_all_delta_pp | 0.000000 | pp | T9_A_within; S->G |
| CAR10 | T9_reference_macro_all_accuracy_pct | 0.000000 | % | T9_A_within; S->G |
| CAR10 | T9_target_macro_all_accuracy_pct | 0.000000 | % | T9_A_within; S->G |
| CAR10 | T9_N | 77.000000 | count | T9_A_cross; S->G |
| CAR10 | T9_local_repair | 1.000000 | count | T9_A_cross; S->G |
| CAR10 | T9_local_harm | 2.000000 | count | T9_A_cross; S->G |
| CAR10 | T9_local_net | -1.000000 | count | T9_A_cross; S->G |
| CAR10 | T9_local_delta_pp | -1.298701 | pp | T9_A_cross; S->G |
| CAR10 | T9_all_repair | 0.000000 | count | T9_A_cross; S->G |
| CAR10 | T9_all_harm | 1.000000 | count | T9_A_cross; S->G |
| CAR10 | T9_all_net | -1.000000 | count | T9_A_cross; S->G |
| CAR10 | T9_all_delta_pp | -1.298701 | pp | T9_A_cross; S->G |
| CAR10 | T9_reference_macro_all_accuracy_pct | 2.314815 | % | T9_A_cross; S->G |
| CAR10 | T9_target_macro_all_accuracy_pct | 1.851852 | % | T9_A_cross; S->G |
| CAR10 | T9_N | 639.000000 | count | T9_non_A; S->G |
| CAR10 | T9_local_repair | 0.000000 | count | T9_non_A; S->G |
| CAR10 | T9_local_harm | 3.000000 | count | T9_non_A; S->G |
| CAR10 | T9_local_net | -3.000000 | count | T9_non_A; S->G |
| CAR10 | T9_local_delta_pp | -0.469484 | pp | T9_non_A; S->G |
| CAR10 | T9_all_repair | 8.000000 | count | T9_non_A; S->G |
| CAR10 | T9_all_harm | 11.000000 | count | T9_non_A; S->G |
| CAR10 | T9_all_net | -3.000000 | count | T9_non_A; S->G |
| CAR10 | T9_all_delta_pp | -0.469484 | pp | T9_non_A; S->G |
| CAR10 | T9_reference_macro_all_accuracy_pct | 96.750910 | % | T9_non_A; S->G |
| CAR10 | T9_target_macro_all_accuracy_pct | 96.354636 | % | T9_non_A; S->G |
| CAR10 | T10_N | 758.000000 | count | T9_all; S->G |
| CAR10 | T10_local_repair | 1.000000 | count | T9_all; S->G |
| CAR10 | T10_local_harm | 9.000000 | count | T9_all; S->G |
| CAR10 | T10_local_net | -8.000000 | count | T9_all; S->G |
| CAR10 | T10_local_delta_pp | -1.055409 | pp | T9_all; S->G |
| CAR10 | T10_all_repair | 8.000000 | count | T9_all; S->G |
| CAR10 | T10_all_harm | 14.000000 | count | T9_all; S->G |
| CAR10 | T10_all_net | -6.000000 | count | T9_all; S->G |
| CAR10 | T10_all_delta_pp | -0.791557 | pp | T9_all; S->G |
| CAR10 | T10_reference_macro_all_accuracy_pct | 81.237770 | % | T9_all; S->G |
| CAR10 | T10_target_macro_all_accuracy_pct | 80.518439 | % | T9_all; S->G |
| CAR10 | T10_N | 42.000000 | count | T9_A_within; S->G |
| CAR10 | T10_local_repair | 1.000000 | count | T9_A_within; S->G |
| CAR10 | T10_local_harm | 2.000000 | count | T9_A_within; S->G |
| CAR10 | T10_local_net | -1.000000 | count | T9_A_within; S->G |
| CAR10 | T10_local_delta_pp | -2.380952 | pp | T9_A_within; S->G |
| CAR10 | T10_all_repair | 0.000000 | count | T9_A_within; S->G |
| CAR10 | T10_all_harm | 0.000000 | count | T9_A_within; S->G |
| CAR10 | T10_all_net | 0.000000 | count | T9_A_within; S->G |
| CAR10 | T10_all_delta_pp | 0.000000 | pp | T9_A_within; S->G |
| CAR10 | T10_reference_macro_all_accuracy_pct | 0.000000 | % | T9_A_within; S->G |
| CAR10 | T10_target_macro_all_accuracy_pct | 0.000000 | % | T9_A_within; S->G |
| CAR10 | T10_N | 77.000000 | count | T9_A_cross; S->G |
| CAR10 | T10_local_repair | 0.000000 | count | T9_A_cross; S->G |
| CAR10 | T10_local_harm | 1.000000 | count | T9_A_cross; S->G |
| CAR10 | T10_local_net | -1.000000 | count | T9_A_cross; S->G |
| CAR10 | T10_local_delta_pp | -1.298701 | pp | T9_A_cross; S->G |
| CAR10 | T10_all_repair | 1.000000 | count | T9_A_cross; S->G |
| CAR10 | T10_all_harm | 2.000000 | count | T9_A_cross; S->G |
| CAR10 | T10_all_net | -1.000000 | count | T9_A_cross; S->G |
| CAR10 | T10_all_delta_pp | -1.298701 | pp | T9_A_cross; S->G |
| CAR10 | T10_reference_macro_all_accuracy_pct | 10.052910 | % | T9_A_cross; S->G |
| CAR10 | T10_target_macro_all_accuracy_pct | 8.201058 | % | T9_A_cross; S->G |
| CAR10 | T10_N | 639.000000 | count | T9_non_A; S->G |
| CAR10 | T10_local_repair | 0.000000 | count | T9_non_A; S->G |
| CAR10 | T10_local_harm | 6.000000 | count | T9_non_A; S->G |
| CAR10 | T10_local_net | -6.000000 | count | T9_non_A; S->G |
| CAR10 | T10_local_delta_pp | -0.938967 | pp | T9_non_A; S->G |
| CAR10 | T10_all_repair | 7.000000 | count | T9_non_A; S->G |
| CAR10 | T10_all_harm | 12.000000 | count | T9_non_A; S->G |
| CAR10 | T10_all_net | -5.000000 | count | T9_non_A; S->G |
| CAR10 | T10_all_delta_pp | -0.782473 | pp | T9_non_A; S->G |
| CAR10 | T10_reference_macro_all_accuracy_pct | 95.014557 | % | T9_non_A; S->G |
| CAR10 | T10_target_macro_all_accuracy_pct | 94.318642 | % | T9_non_A; S->G |

원본: `generation_local_all_transition_audit/CAR10/t10_transition_summary.csv`, `generation_local_all_transition_audit/CAR10/t9_transition_summary.csv`, `generation_local_all_transition_audit/IA10/t10_transition_summary.csv`, `generation_local_all_transition_audit/IA10/t9_transition_summary.csv`

## birth_support

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | A-within_n | 169 | count | birth<=T9; common-class macro own support; bank seed 1993 only |
| IA10 | A-within_matched_classes | 81 | count | birth<=T9; common-class macro own support; bank seed 1993 only |
| IA10 | A-within_a | 0.207733 | cosine gap | birth<=T9; common-class macro own support; bank seed 1993 only |
| IA10 | A-within_correct | 0.320396 | cosine gap | birth<=T9; common-class macro own support; bank seed 1993 only |
| IA10 | A-cross_n | 158 | count | birth<=T9; common-class macro own support; bank seed 1993 only |
| IA10 | A-cross_matched_classes | 76 | count | birth<=T9; common-class macro own support; bank seed 1993 only |
| IA10 | A-cross_a | 0.255513 | cosine gap | birth<=T9; common-class macro own support; bank seed 1993 only |
| IA10 | A-cross_correct | 0.364054 | cosine gap | birth<=T9; common-class macro own support; bank seed 1993 only |
| IA10 | all_A_class_matched_own_support_delta | -0.109804 | cosine similarity | all A; common-class macro; single bank seed 1993 |
| CAR10 | A-within_n | 411 | count | birth<=T9; common-class macro own support; bank seed 1993 only |
| CAR10 | A-within_matched_classes | 129 | count | birth<=T9; common-class macro own support; bank seed 1993 only |
| CAR10 | A-within_a | 0.394591 | cosine gap | birth<=T9; common-class macro own support; bank seed 1993 only |
| CAR10 | A-within_correct | 0.541665 | cosine gap | birth<=T9; common-class macro own support; bank seed 1993 only |
| CAR10 | A-cross_n | 496 | count | birth<=T9; common-class macro own support; bank seed 1993 only |
| CAR10 | A-cross_matched_classes | 122 | count | birth<=T9; common-class macro own support; bank seed 1993 only |
| CAR10 | A-cross_a | 0.470697 | cosine gap | birth<=T9; common-class macro own support; bank seed 1993 only |
| CAR10 | A-cross_correct | 0.565189 | cosine gap | birth<=T9; common-class macro own support; bank seed 1993 only |
| CAR10 | all_A_class_matched_own_support_delta | -0.118014 | cosine similarity | all A; common-class macro; single bank seed 1993 |

원본: `type_a_generative_augmentation/audit/CAR10/class_matched_support.csv`, `type_a_generative_augmentation/audit/IA10/class_matched_support.csv`, `type_a_generative_augmentation/birth_support_audit.csv`

## generation

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | T1_T1_G-S_n | 175 | count | overall |
| IA10 | T1_T1_G-S_repair | 5 | count | overall |
| IA10 | T1_T1_G-S_harm | 3 | count | overall |
| IA10 | T1_T1_G-S_net | 2 | count | overall |
| IA10 | T1_T1_G-S_delta_pp | 1.142857 | pp | overall |
| IA10 | T1_T1_G-S_n | 175 | count | new |
| IA10 | T1_T1_G-S_repair | 5 | count | new |
| IA10 | T1_T1_G-S_harm | 3 | count | new |
| IA10 | T1_T1_G-S_net | 2 | count | new |
| IA10 | T1_T1_G-S_delta_pp | 1.142857 | pp | new |
| IA10 | T1_T1_G-S_n | 0 | count | T9 |
| IA10 | T1_T1_G-S_repair | 0 | count | T9 |
| IA10 | T1_T1_G-S_harm | 0 | count | T9 |
| IA10 | T1_T1_G-S_net | 0 | count | T9 |
| IA10 | T1_T1_GM-G_n | 175 | count | overall |
| IA10 | T1_T1_GM-G_repair | 0 | count | overall |
| IA10 | T1_T1_GM-G_harm | 0 | count | overall |
| IA10 | T1_T1_GM-G_net | 0 | count | overall |
| IA10 | T1_T1_GM-G_delta_pp | 0.000000 | pp | overall |
| IA10 | T1_T1_GM-G_n | 175 | count | new |
| IA10 | T1_T1_GM-G_repair | 0 | count | new |
| IA10 | T1_T1_GM-G_harm | 0 | count | new |
| IA10 | T1_T1_GM-G_net | 0 | count | new |
| IA10 | T1_T1_GM-G_delta_pp | 0.000000 | pp | new |
| IA10 | T1_T1_GM-G_n | 0 | count | T9 |
| IA10 | T1_T1_GM-G_repair | 0 | count | T9 |
| IA10 | T1_T1_GM-G_harm | 0 | count | T9 |
| IA10 | T1_T1_GM-G_net | 0 | count | T9 |
| IA10 | T9_T10_T9_G-S_n | 1395 | count | overall |
| IA10 | T9_T10_T9_G-S_repair | 16 | count | overall |
| IA10 | T9_T10_T9_G-S_harm | 18 | count | overall |
| IA10 | T9_T10_T9_G-S_net | -2 | count | overall |
| IA10 | T9_T10_T9_G-S_delta_pp | -0.143369 | pp | overall |
| IA10 | T9_T10_T9_G-S_n | 147 | count | new |
| IA10 | T9_T10_T9_G-S_repair | 2 | count | new |
| IA10 | T9_T10_T9_G-S_harm | 2 | count | new |
| IA10 | T9_T10_T9_G-S_net | 0 | count | new |
| IA10 | T9_T10_T9_G-S_delta_pp | 0.000000 | pp | new |
| IA10 | T9_T10_T9_G-S_n | 147 | count | T9 |
| IA10 | T9_T10_T9_G-S_repair | 2 | count | T9 |
| IA10 | T9_T10_T9_G-S_harm | 2 | count | T9 |
| IA10 | T9_T10_T9_G-S_net | 0 | count | T9 |
| IA10 | T9_T10_T9_G-S_delta_pp | 0.000000 | pp | T9 |
| IA10 | T9_T10_T9_GM-G_n | 1395 | count | overall |
| IA10 | T9_T10_T9_GM-G_repair | 4 | count | overall |
| IA10 | T9_T10_T9_GM-G_harm | 8 | count | overall |
| IA10 | T9_T10_T9_GM-G_net | -4 | count | overall |
| IA10 | T9_T10_T9_GM-G_delta_pp | -0.286738 | pp | overall |
| IA10 | T9_T10_T9_GM-G_n | 147 | count | new |
| IA10 | T9_T10_T9_GM-G_repair | 1 | count | new |
| IA10 | T9_T10_T9_GM-G_harm | 1 | count | new |
| IA10 | T9_T10_T9_GM-G_net | 0 | count | new |
| IA10 | T9_T10_T9_GM-G_delta_pp | 0.000000 | pp | new |
| IA10 | T9_T10_T9_GM-G_n | 147 | count | T9 |
| IA10 | T9_T10_T9_GM-G_repair | 1 | count | T9 |
| IA10 | T9_T10_T9_GM-G_harm | 1 | count | T9 |
| IA10 | T9_T10_T9_GM-G_net | 0 | count | T9 |
| IA10 | T9_T10_T9_GM-G_delta_pp | 0.000000 | pp | T9 |
| IA10 | T9_T10_T10_G-S_n | 1519 | count | overall |
| IA10 | T9_T10_T10_G-S_repair | 7 | count | overall |
| IA10 | T9_T10_T10_G-S_harm | 11 | count | overall |
| IA10 | T9_T10_T10_G-S_net | -4 | count | overall |
| IA10 | T9_T10_T10_G-S_delta_pp | -0.263331 | pp | overall |
| IA10 | T9_T10_T10_G-S_n | 124 | count | new |
| IA10 | T9_T10_T10_G-S_repair | 0 | count | new |
| IA10 | T9_T10_T10_G-S_harm | 1 | count | new |
| IA10 | T9_T10_T10_G-S_net | -1 | count | new |
| IA10 | T9_T10_T10_G-S_delta_pp | -0.806452 | pp | new |
| IA10 | T9_T10_T10_G-S_n | 147 | count | T9 |
| IA10 | T9_T10_T10_G-S_repair | 1 | count | T9 |
| IA10 | T9_T10_T10_G-S_harm | 1 | count | T9 |
| IA10 | T9_T10_T10_G-S_net | 0 | count | T9 |
| IA10 | T9_T10_T10_G-S_delta_pp | 0.000000 | pp | T9 |
| IA10 | T9_T10_T10_GM-G_n | 1519 | count | overall |
| IA10 | T9_T10_T10_GM-G_repair | 9 | count | overall |
| IA10 | T9_T10_T10_GM-G_harm | 5 | count | overall |
| IA10 | T9_T10_T10_GM-G_net | 4 | count | overall |
| IA10 | T9_T10_T10_GM-G_delta_pp | 0.263331 | pp | overall |
| IA10 | T9_T10_T10_GM-G_n | 124 | count | new |
| IA10 | T9_T10_T10_GM-G_repair | 0 | count | new |
| IA10 | T9_T10_T10_GM-G_harm | 0 | count | new |
| IA10 | T9_T10_T10_GM-G_net | 0 | count | new |
| IA10 | T9_T10_T10_GM-G_delta_pp | 0.000000 | pp | new |
| IA10 | T9_T10_T10_GM-G_n | 147 | count | T9 |
| IA10 | T9_T10_T10_GM-G_repair | 2 | count | T9 |
| IA10 | T9_T10_T10_GM-G_harm | 4 | count | T9 |
| IA10 | T9_T10_T10_GM-G_net | -2 | count | T9 |
| IA10 | T9_T10_T10_GM-G_delta_pp | -1.360544 | pp | T9 |
| CAR10 | T1_T1_G-S_n | 838 | count | overall |
| CAR10 | T1_T1_G-S_repair | 16 | count | overall |
| CAR10 | T1_T1_G-S_harm | 7 | count | overall |
| CAR10 | T1_T1_G-S_net | 9 | count | overall |
| CAR10 | T1_T1_G-S_delta_pp | 1.073986 | pp | overall |
| CAR10 | T1_T1_G-S_n | 838 | count | new |
| CAR10 | T1_T1_G-S_repair | 16 | count | new |
| CAR10 | T1_T1_G-S_harm | 7 | count | new |
| CAR10 | T1_T1_G-S_net | 9 | count | new |
| CAR10 | T1_T1_G-S_delta_pp | 1.073986 | pp | new |
| CAR10 | T1_T1_G-S_n | 0 | count | T9 |
| CAR10 | T1_T1_G-S_repair | 0 | count | T9 |
| CAR10 | T1_T1_G-S_harm | 0 | count | T9 |
| CAR10 | T1_T1_G-S_net | 0 | count | T9 |
| CAR10 | T1_T1_GM-G_n | 838 | count | overall |
| CAR10 | T1_T1_GM-G_repair | 0 | count | overall |
| CAR10 | T1_T1_GM-G_harm | 0 | count | overall |
| CAR10 | T1_T1_GM-G_net | 0 | count | overall |
| CAR10 | T1_T1_GM-G_delta_pp | 0.000000 | pp | overall |
| CAR10 | T1_T1_GM-G_n | 838 | count | new |
| CAR10 | T1_T1_GM-G_repair | 0 | count | new |
| CAR10 | T1_T1_GM-G_harm | 0 | count | new |
| CAR10 | T1_T1_GM-G_net | 0 | count | new |
| CAR10 | T1_T1_GM-G_delta_pp | 0.000000 | pp | new |
| CAR10 | T1_T1_GM-G_n | 0 | count | T9 |
| CAR10 | T1_T1_GM-G_repair | 0 | count | T9 |
| CAR10 | T1_T1_GM-G_harm | 0 | count | T9 |
| CAR10 | T1_T1_GM-G_net | 0 | count | T9 |
| CAR10 | T9_T10_T9_G-S_n | 7262 | count | overall |
| CAR10 | T9_T10_T9_G-S_repair | 32 | count | overall |
| CAR10 | T9_T10_T9_G-S_harm | 47 | count | overall |
| CAR10 | T9_T10_T9_G-S_net | -15 | count | overall |
| CAR10 | T9_T10_T9_G-S_delta_pp | -0.206555 | pp | overall |
| CAR10 | T9_T10_T9_G-S_n | 758 | count | new |
| CAR10 | T9_T10_T9_G-S_repair | 8 | count | new |
| CAR10 | T9_T10_T9_G-S_harm | 12 | count | new |
| CAR10 | T9_T10_T9_G-S_net | -4 | count | new |
| CAR10 | T9_T10_T9_G-S_delta_pp | -0.527704 | pp | new |
| CAR10 | T9_T10_T9_G-S_n | 758 | count | T9 |
| CAR10 | T9_T10_T9_G-S_repair | 8 | count | T9 |
| CAR10 | T9_T10_T9_G-S_harm | 12 | count | T9 |
| CAR10 | T9_T10_T9_G-S_net | -4 | count | T9 |
| CAR10 | T9_T10_T9_G-S_delta_pp | -0.527704 | pp | T9 |
| CAR10 | T9_T10_T9_GM-G_n | 7262 | count | overall |
| CAR10 | T9_T10_T9_GM-G_repair | 30 | count | overall |
| CAR10 | T9_T10_T9_GM-G_harm | 38 | count | overall |
| CAR10 | T9_T10_T9_GM-G_net | -8 | count | overall |
| CAR10 | T9_T10_T9_GM-G_delta_pp | -0.110162 | pp | overall |
| CAR10 | T9_T10_T9_GM-G_n | 758 | count | new |
| CAR10 | T9_T10_T9_GM-G_repair | 13 | count | new |
| CAR10 | T9_T10_T9_GM-G_harm | 25 | count | new |
| CAR10 | T9_T10_T9_GM-G_net | -12 | count | new |
| CAR10 | T9_T10_T9_GM-G_delta_pp | -1.583113 | pp | new |
| CAR10 | T9_T10_T9_GM-G_n | 758 | count | T9 |
| CAR10 | T9_T10_T9_GM-G_repair | 13 | count | T9 |
| CAR10 | T9_T10_T9_GM-G_harm | 25 | count | T9 |
| CAR10 | T9_T10_T9_GM-G_net | -12 | count | T9 |
| CAR10 | T9_T10_T9_GM-G_delta_pp | -1.583113 | pp | T9 |
| CAR10 | T9_T10_T10_G-S_n | 8041 | count | overall |
| CAR10 | T9_T10_T10_G-S_repair | 29 | count | overall |
| CAR10 | T9_T10_T10_G-S_harm | 51 | count | overall |
| CAR10 | T9_T10_T10_G-S_net | -22 | count | overall |
| CAR10 | T9_T10_T10_G-S_delta_pp | -0.273598 | pp | overall |
| CAR10 | T9_T10_T10_G-S_n | 779 | count | new |
| CAR10 | T9_T10_T10_G-S_repair | 4 | count | new |
| CAR10 | T9_T10_T10_G-S_harm | 5 | count | new |
| CAR10 | T9_T10_T10_G-S_net | -1 | count | new |
| CAR10 | T9_T10_T10_G-S_delta_pp | -0.128370 | pp | new |
| CAR10 | T9_T10_T10_G-S_n | 758 | count | T9 |
| CAR10 | T9_T10_T10_G-S_repair | 8 | count | T9 |
| CAR10 | T9_T10_T10_G-S_harm | 14 | count | T9 |
| CAR10 | T9_T10_T10_G-S_net | -6 | count | T9 |
| CAR10 | T9_T10_T10_G-S_delta_pp | -0.791557 | pp | T9 |
| CAR10 | T9_T10_T10_GM-G_n | 8041 | count | overall |
| CAR10 | T9_T10_T10_GM-G_repair | 36 | count | overall |
| CAR10 | T9_T10_T10_GM-G_harm | 44 | count | overall |
| CAR10 | T9_T10_T10_GM-G_net | -8 | count | overall |
| CAR10 | T9_T10_T10_GM-G_delta_pp | -0.099490 | pp | overall |
| CAR10 | T9_T10_T10_GM-G_n | 779 | count | new |
| CAR10 | T9_T10_T10_GM-G_repair | 2 | count | new |
| CAR10 | T9_T10_T10_GM-G_harm | 2 | count | new |
| CAR10 | T9_T10_T10_GM-G_net | 0 | count | new |
| CAR10 | T9_T10_T10_GM-G_delta_pp | 0.000000 | pp | new |
| CAR10 | T9_T10_T10_GM-G_n | 758 | count | T9 |
| CAR10 | T9_T10_T10_GM-G_repair | 11 | count | T9 |
| CAR10 | T9_T10_T10_GM-G_harm | 23 | count | T9 |
| CAR10 | T9_T10_T10_GM-G_net | -12 | count | T9 |
| CAR10 | T9_T10_T10_GM-G_delta_pp | -1.583113 | pp | T9 |

원본: `type_a_generative_augmentation/metrics.json`

## structure_multicenter_separation

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | E0_Inc_Acc | 74.597657 | % | single seed; within-study comparator |
| IA5 | E0_Last_Acc | 66.425280 | % | single seed; within-study comparator |
| IA5 | E0_Forgetting_pp | 7.844444 | pp | single seed; within-study comparator |
| IA5 | E1_Inc_Acc | 74.464567 | % | single seed; within-study comparator |
| IA5 | E1_Last_Acc | 66.227781 | % | single seed; within-study comparator |
| IA5 | E1_Forgetting_pp | 7.945143 | pp | single seed; within-study comparator |
| IA5 | E2_Inc_Acc | 74.283571 | % | single seed; within-study comparator |
| IA5 | E2_Last_Acc | 65.964450 | % | single seed; within-study comparator |
| IA5 | E2_Forgetting_pp | 8.063044 | pp | single seed; within-study comparator |
| IA5 | E3_Inc_Acc | 74.531915 | % | single seed; within-study comparator |
| IA5 | E3_Last_Acc | 66.491113 | % | single seed; within-study comparator |
| IA5 | E3_Forgetting_pp | 7.533670 | pp | single seed; within-study comparator |
| IA5 | E4_Inc_Acc | 74.597657 | % | single seed; within-study comparator |
| IA5 | E4_Last_Acc | 66.425280 | % | single seed; within-study comparator |
| IA5 | E4_Forgetting_pp | 7.844444 | pp | single seed; within-study comparator |
| IA5 | E5_Inc_Acc | 74.509774 | % | single seed; within-study comparator |
| IA5 | E5_Last_Acc | 66.425280 | % | single seed; within-study comparator |
| IA5 | E5_Forgetting_pp | 7.795712 | pp | single seed; within-study comparator |
| IA5 | E6_Inc_Acc | 74.467466 | % | single seed; within-study comparator |
| IA5 | E6_Last_Acc | 65.766952 | % | single seed; within-study comparator |
| IA5 | E6_Forgetting_pp | 7.945961 | pp | single seed; within-study comparator |
| IA5 | E7_Inc_Acc | 74.464567 | % | single seed; within-study comparator |
| IA5 | E7_Last_Acc | 66.227781 | % | single seed; within-study comparator |
| IA5 | E7_Forgetting_pp | 7.945143 | pp | single seed; within-study comparator |

원본: `structure_multicenter_separation/reports/e0_comparison_20260916/tables/metrics_per_seed.csv`

## structure_e3_followup

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | B0_STRUCTURE_Inc_Acc | 74.597657 | % | single seed; within-study comparator |
| IA5 | B0_STRUCTURE_Last_Acc | 66.425280 | % | single seed; within-study comparator |
| IA5 | B0_STRUCTURE_Forgetting_pp | 7.844444 | pp | single seed; within-study comparator |
| IA5 | B0_STRUCTURE_final_old_accuracy | 66.987179 | % | single seed; within-study comparator |
| IA5 | B0_STRUCTURE_final_new_accuracy | 63.837638 | % | single seed; within-study comparator |
| IA5 | L01_Inc_Acc | 74.690708 | % | single seed; within-study comparator |
| IA5 | L01_Last_Acc | 66.359447 | % | single seed; within-study comparator |
| IA5 | L01_Forgetting_pp | 7.220850 | pp | single seed; within-study comparator |
| IA5 | L01_final_old_accuracy | 67.307692 | % | single seed; within-study comparator |
| IA5 | L01_final_new_accuracy | 61.992620 | % | single seed; within-study comparator |
| IA5 | L03_Inc_Acc | 74.545081 | % | single seed; within-study comparator |
| IA5 | L03_Last_Acc | 66.556945 | % | single seed; within-study comparator |
| IA5 | L03_Forgetting_pp | 7.044070 | pp | single seed; within-study comparator |
| IA5 | L03_final_old_accuracy | 67.387821 | % | single seed; within-study comparator |
| IA5 | L03_final_new_accuracy | 62.730627 | % | single seed; within-study comparator |
| IA5 | L05_Inc_Acc | 74.554748 | % | single seed; within-study comparator |
| IA5 | L05_Last_Acc | 66.359447 | % | single seed; within-study comparator |
| IA5 | L05_Forgetting_pp | 7.084365 | pp | single seed; within-study comparator |
| IA5 | L05_final_old_accuracy | 67.307692 | % | single seed; within-study comparator |
| IA5 | L05_final_new_accuracy | 61.992620 | % | single seed; within-study comparator |
| IA5 | L10_Inc_Acc | 74.531915 | % | single seed; within-study comparator |
| IA5 | L10_Last_Acc | 66.491113 | % | single seed; within-study comparator |
| IA5 | L10_Forgetting_pp | 7.533670 | pp | single seed; within-study comparator |
| IA5 | L10_final_old_accuracy | 67.387821 | % | single seed; within-study comparator |
| IA5 | L10_final_new_accuracy | 62.361624 | % | single seed; within-study comparator |
| IA5 | FCS_PIC_Inc_Acc | 74.171760 | % | single seed; within-study comparator |
| IA5 | FCS_PIC_Last_Acc | 65.108624 | % | single seed; within-study comparator |
| IA5 | FCS_PIC_Forgetting_pp | 12.161715 | pp | single seed; within-study comparator |
| IA5 | FCS_PIC_final_old_accuracy | 64.743590 | % | single seed; within-study comparator |
| IA5 | FCS_PIC_final_new_accuracy | 66.789668 | % | single seed; within-study comparator |
| IA5 | E3_NORM_Inc_Acc | 74.578477 | % | single seed; within-study comparator |
| IA5 | E3_NORM_Last_Acc | 66.556945 | % | single seed; within-study comparator |
| IA5 | E3_NORM_Forgetting_pp | 7.613980 | pp | single seed; within-study comparator |
| IA5 | E3_NORM_final_old_accuracy | 67.227564 | % | single seed; within-study comparator |
| IA5 | E3_NORM_final_new_accuracy | 63.468635 | % | single seed; within-study comparator |
| IA5 | E3_AHN_Inc_Acc | 74.442965 | % | single seed; within-study comparator |
| IA5 | E3_AHN_Last_Acc | 66.227781 | % | single seed; within-study comparator |
| IA5 | E3_AHN_Forgetting_pp | 7.687296 | pp | single seed; within-study comparator |
| IA5 | E3_AHN_final_old_accuracy | 66.907051 | % | single seed; within-study comparator |
| IA5 | E3_AHN_final_new_accuracy | 63.099631 | % | single seed; within-study comparator |
| IA5 | CEOS50_Inc_Acc | 74.038648 | % | single seed; within-study comparator |
| IA5 | CEOS50_Last_Acc | 66.161949 | % | single seed; within-study comparator |
| IA5 | CEOS50_Forgetting_pp | 11.556417 | pp | single seed; within-study comparator |
| IA5 | CEOS50_final_old_accuracy | 65.544872 | % | single seed; within-study comparator |
| IA5 | CEOS50_final_new_accuracy | 69.003690 | % | single seed; within-study comparator |
| IA5 | CEOS100_Inc_Acc | 71.321641 | % | single seed; within-study comparator |
| IA5 | CEOS100_Last_Acc | 61.290323 | % | single seed; within-study comparator |
| IA5 | CEOS100_Forgetting_pp | 21.238736 | pp | single seed; within-study comparator |
| IA5 | CEOS100_final_old_accuracy | 58.253205 | % | single seed; within-study comparator |
| IA5 | CEOS100_final_new_accuracy | 75.276753 | % | single seed; within-study comparator |
| IA5 | AM_VARIANCE_Inc_Acc | 74.406813 | % | single seed; within-study comparator |
| IA5 | AM_VARIANCE_Last_Acc | 66.556945 | % | single seed; within-study comparator |
| IA5 | AM_VARIANCE_Forgetting_pp | 6.794121 | pp | single seed; within-study comparator |
| IA5 | AM_VARIANCE_final_old_accuracy | 67.548077 | % | single seed; within-study comparator |
| IA5 | AM_VARIANCE_final_new_accuracy | 61.992620 | % | single seed; within-study comparator |
| IA5 | AM_AGE_Inc_Acc | 74.422044 | % | single seed; within-study comparator |
| IA5 | AM_AGE_Last_Acc | 66.491113 | % | single seed; within-study comparator |
| IA5 | AM_AGE_Forgetting_pp | 6.996757 | pp | single seed; within-study comparator |
| IA5 | AM_AGE_final_old_accuracy | 67.467949 | % | single seed; within-study comparator |
| IA5 | AM_AGE_final_new_accuracy | 61.992620 | % | single seed; within-study comparator |
| IA5 | AM_DRIFT_Inc_Acc | 74.766990 | % | single seed; within-study comparator |
| IA5 | AM_DRIFT_Last_Acc | 66.688611 | % | single seed; within-study comparator |
| IA5 | AM_DRIFT_Forgetting_pp | 6.974950 | pp | single seed; within-study comparator |
| IA5 | AM_DRIFT_final_old_accuracy | 67.548077 | % | single seed; within-study comparator |
| IA5 | AM_DRIFT_final_new_accuracy | 62.730627 | % | single seed; within-study comparator |

원본: `structure_e3_followup/reports/e3_followup_comparison_20260917/tables/metrics_per_seed.csv`

## structure_similarity_aware_followup

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | B0_Inc-Acc | 74.597657 | % | single seed; raw unrounded export |
| IA5 | B0_Last-Acc | 66.425280 | % | single seed; raw unrounded export |
| IA5 | B0_Forgetting | 7.844444 | pp | single seed; raw unrounded export |
| IA5 | B0_Old | 66.987179 | % | single seed; raw unrounded export |
| IA5 | B0_New | 63.837638 | % | single seed; raw unrounded export |
| IA5 | B1_Inc-Acc | 74.766990 | % | single seed; raw unrounded export |
| IA5 | B1_Last-Acc | 66.688611 | % | single seed; raw unrounded export |
| IA5 | B1_Forgetting | 6.974950 | pp | single seed; raw unrounded export |
| IA5 | B1_Old | 67.548077 | % | single seed; raw unrounded export |
| IA5 | B1_New | 62.730627 | % | single seed; raw unrounded export |
| IA5 | S0_Inc-Acc | 74.597657 | % | single seed; raw unrounded export |
| IA5 | S0_Last-Acc | 66.425280 | % | single seed; raw unrounded export |
| IA5 | S0_Forgetting | 7.844444 | pp | single seed; raw unrounded export |
| IA5 | S0_Old | 66.987179 | % | single seed; raw unrounded export |
| IA5 | S0_New | 63.837638 | % | single seed; raw unrounded export |
| IA5 | S1_Inc-Acc | 74.701259 | % | single seed; raw unrounded export |
| IA5 | S1_Last-Acc | 66.622778 | % | single seed; raw unrounded export |
| IA5 | S1_Forgetting | 7.395284 | pp | single seed; raw unrounded export |
| IA5 | S1_Old | 67.467949 | % | single seed; raw unrounded export |
| IA5 | S1_New | 62.730627 | % | single seed; raw unrounded export |
| IA5 | S2_Inc-Acc | 74.446324 | % | single seed; raw unrounded export |
| IA5 | S2_Last-Acc | 66.622778 | % | single seed; raw unrounded export |
| IA5 | S2_Forgetting | 7.241284 | pp | single seed; raw unrounded export |
| IA5 | S2_Old | 67.307692 | % | single seed; raw unrounded export |
| IA5 | S2_New | 63.468635 | % | single seed; raw unrounded export |
| IA5 | S3_Inc-Acc | 74.497952 | % | single seed; raw unrounded export |
| IA5 | S3_Last-Acc | 66.359447 | % | single seed; raw unrounded export |
| IA5 | S3_Forgetting | 8.263131 | pp | single seed; raw unrounded export |
| IA5 | S3_Old | 66.907051 | % | single seed; raw unrounded export |
| IA5 | S3_New | 63.837638 | % | single seed; raw unrounded export |
| IA5 | S4_Inc-Acc | 74.547696 | % | single seed; raw unrounded export |
| IA5 | S4_Last-Acc | 66.359447 | % | single seed; raw unrounded export |
| IA5 | S4_Forgetting | 8.115202 | pp | single seed; raw unrounded export |
| IA5 | S4_Old | 66.987179 | % | single seed; raw unrounded export |
| IA5 | S4_New | 63.468635 | % | single seed; raw unrounded export |

원본: `structure_similarity_aware_followup/outputs/imageneta/metrics_per_seed.csv`

## structure_similarity_control_experiments

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | C0_Inc-Acc | 74.766990 | % | single seed; raw unrounded export |
| IA5 | C0_Last-Acc | 66.688611 | % | single seed; raw unrounded export |
| IA5 | C0_Forgetting | 6.974950 | pp | single seed; raw unrounded export |
| IA5 | C0_Old | 67.548077 | % | single seed; raw unrounded export |
| IA5 | C0_New | 62.730627 | % | single seed; raw unrounded export |
| IA5 | C1_Inc-Acc | 74.531671 | % | single seed; raw unrounded export |
| IA5 | C1_Last-Acc | 66.359447 | % | single seed; raw unrounded export |
| IA5 | C1_Forgetting | 7.096374 | pp | single seed; raw unrounded export |
| IA5 | C1_Old | 67.387821 | % | single seed; raw unrounded export |
| IA5 | C1_New | 61.623616 | % | single seed; raw unrounded export |
| IA5 | C2_Inc-Acc | 74.609042 | % | single seed; raw unrounded export |
| IA5 | C2_Last-Acc | 66.688611 | % | single seed; raw unrounded export |
| IA5 | C2_Forgetting | 6.890623 | pp | single seed; raw unrounded export |
| IA5 | C2_Old | 67.467949 | % | single seed; raw unrounded export |
| IA5 | C2_New | 63.099631 | % | single seed; raw unrounded export |
| IA5 | C3_Inc-Acc | 74.433594 | % | single seed; raw unrounded export |
| IA5 | C3_Last-Acc | 66.359447 | % | single seed; raw unrounded export |
| IA5 | C3_Forgetting | 7.166996 | pp | single seed; raw unrounded export |
| IA5 | C3_Old | 67.147436 | % | single seed; raw unrounded export |
| IA5 | C3_New | 62.730627 | % | single seed; raw unrounded export |
| IA5 | C4_Inc-Acc | 74.588427 | % | single seed; raw unrounded export |
| IA5 | C4_Last-Acc | 66.425280 | % | single seed; raw unrounded export |
| IA5 | C4_Forgetting | 7.242282 | pp | single seed; raw unrounded export |
| IA5 | C4_Old | 67.227564 | % | single seed; raw unrounded export |
| IA5 | C4_New | 62.730627 | % | single seed; raw unrounded export |

원본: `structure_similarity_control_experiments/outputs/imageneta/metrics_per_seed.csv`

## reduced_selective

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | A_P005_Inc-Acc | 74.669747 | % | completed 5-task selected matrix only |
| IA5 | A_P005_Last-Acc | 66.688611 | % | completed 5-task selected matrix only |
| IA5 | A_P005_Forgetting | 7.069723 | pp | completed 5-task selected matrix only |
| IA5 | A_P005_old_accuracy | 67.628205 | % | final task single-head |
| IA5 | A_P005_new_accuracy | 62.361624 | % | final task single-head |
| IA5 | B_P020_A100_Inc-Acc | 74.554492 | % | completed 5-task selected matrix only |
| IA5 | B_P020_A100_Last-Acc | 66.359447 | % | completed 5-task selected matrix only |
| IA5 | B_P020_A100_Forgetting | 7.010496 | pp | completed 5-task selected matrix only |
| IA5 | B_P020_A100_old_accuracy | 67.227564 | % | final task single-head |
| IA5 | B_P020_A100_new_accuracy | 62.361624 | % | final task single-head |
| IA5 | C_P020_L030_Inc-Acc | 74.640015 | % | completed 5-task selected matrix only |
| IA5 | C_P020_L030_Last-Acc | 66.556945 | % | completed 5-task selected matrix only |
| IA5 | C_P020_L030_Forgetting | 7.863930 | pp | completed 5-task selected matrix only |
| IA5 | C_P020_L030_old_accuracy | 67.147436 | % | final task single-head |
| IA5 | C_P020_L030_new_accuracy | 63.837638 | % | final task single-head |
| IA5 | C_P020_L100_Inc-Acc | 74.621386 | % | completed 5-task selected matrix only |
| IA5 | C_P020_L100_Last-Acc | 66.622778 | % | completed 5-task selected matrix only |
| IA5 | C_P020_L100_Forgetting | 7.423665 | pp | completed 5-task selected matrix only |
| IA5 | C_P020_L100_old_accuracy | 67.307692 | % | final task single-head |
| IA5 | C_P020_L100_new_accuracy | 63.468635 | % | final task single-head |
| CAR5 | Structure_Inc-Acc | 84.929241 | % | completed 5-task selected matrix only |
| CAR5 | Structure_Last-Acc | 81.196369 | % | completed 5-task selected matrix only |
| CAR5 | Structure_Forgetting | 4.431443 | pp | completed 5-task selected matrix only |
| CAR5 | Structure_old_accuracy | 80.987005 | % | final task single-head |
| CAR5 | Structure_new_accuracy | 82.054534 | % | final task single-head |
| CAR5 | A_P005_Inc-Acc | 84.901657 | % | completed 5-task selected matrix only |
| CAR5 | A_P005_Last-Acc | 81.295859 | % | completed 5-task selected matrix only |
| CAR5 | A_P005_Forgetting | 3.938160 | pp | completed 5-task selected matrix only |
| CAR5 | A_P005_old_accuracy | 81.172649 | % | final task single-head |
| CAR5 | A_P005_new_accuracy | 81.800888 | % | final task single-head |
| CAR5 | B_P020_A100_Inc-Acc | 84.920252 | % | completed 5-task selected matrix only |
| CAR5 | B_P020_A100_Last-Acc | 81.246114 | % | completed 5-task selected matrix only |
| CAR5 | B_P020_A100_Forgetting | 4.076100 | pp | completed 5-task selected matrix only |
| CAR5 | B_P020_A100_old_accuracy | 81.048886 | % | final task single-head |
| CAR5 | B_P020_A100_new_accuracy | 82.054534 | % | final task single-head |
| CAR5 | C_P020_L030_Inc-Acc | 84.905096 | % | completed 5-task selected matrix only |
| CAR5 | C_P020_L030_Last-Acc | 81.183932 | % | completed 5-task selected matrix only |
| CAR5 | C_P020_L030_Forgetting | 4.449508 | pp | completed 5-task selected matrix only |
| CAR5 | C_P020_L030_old_accuracy | 81.002475 | % | final task single-head |
| CAR5 | C_P020_L030_new_accuracy | 81.927711 | % | final task single-head |
| CAR5 | C_P020_L100_Inc-Acc | 84.853696 | % | completed 5-task selected matrix only |
| CAR5 | C_P020_L100_Last-Acc | 81.183932 | % | completed 5-task selected matrix only |
| CAR5 | C_P020_L100_Forgetting | 4.368696 | pp | completed 5-task selected matrix only |
| CAR5 | C_P020_L100_old_accuracy | 81.017946 | % | final task single-head |
| CAR5 | C_P020_L100_new_accuracy | 81.864299 | % | final task single-head |

원본: `structure_selective_similarity_sweep/outputs/CAR5/A_P005/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/CAR5/A_P005/seed_1993/task_summary.csv`, `structure_selective_similarity_sweep/outputs/CAR5/B_P020_A100/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/CAR5/B_P020_A100/seed_1993/task_summary.csv`, `structure_selective_similarity_sweep/outputs/CAR5/C_P020_L030/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/CAR5/C_P020_L030/seed_1993/task_summary.csv`, `structure_selective_similarity_sweep/outputs/CAR5/C_P020_L100/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/CAR5/C_P020_L100/seed_1993/task_summary.csv`, `structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/task_summary.csv`, `structure_selective_similarity_sweep/outputs/IA5/A_P005/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/IA5/A_P005/seed_1993/task_summary.csv`, `structure_selective_similarity_sweep/outputs/IA5/B_P020_A100/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/IA5/B_P020_A100/seed_1993/task_summary.csv`, `structure_selective_similarity_sweep/outputs/IA5/C_P020_L030/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/IA5/C_P020_L030/seed_1993/task_summary.csv`, `structure_selective_similarity_sweep/outputs/IA5/C_P020_L100/seed_1993/summary.json`, `structure_selective_similarity_sweep/outputs/IA5/C_P020_L100/seed_1993/task_summary.csv`

## structure_rp_protoclad_task5

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA5 | RP_C_P020_L030_Inc-Acc | 74.703965 | % | completed canonical run |
| IA5 | RP_C_P020_L030_Last-Acc | 66.820276 | % | completed canonical run |
| IA5 | RP_C_P020_L030_Forgetting | 7.309552 | pp | completed canonical run |
| IA5 | RP_C_P020_L100_Inc-Acc | 74.495388 | % | completed canonical run |
| IA5 | RP_C_P020_L100_Last-Acc | 66.359447 | % | completed canonical run |
| IA5 | RP_C_P020_L100_Forgetting | 7.682102 | pp | completed canonical run |

원본: `structure_rp_protoclad_task5/outputs/IA5/RP_C_P020_L030/seed_1993/summary.json`, `structure_rp_protoclad_task5/outputs/IA5/RP_C_P020_L100/seed_1993/summary.json`

## structure_stored_apr_rp

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| CAR5 | StoredAPR_RP_C_P020_L030_Inc-Acc | 84.938382 | % | completed canonical run |
| CAR5 | StoredAPR_RP_C_P020_L030_Last-Acc | 81.221241 | % | completed canonical run |
| CAR5 | StoredAPR_RP_C_P020_L030_Forgetting | 4.387206 | pp | completed canonical run |
| IA5 | StoredAPR_RP_C_P020_L030_Inc-Acc | 74.632028 | % | completed canonical run |
| IA5 | StoredAPR_RP_C_P020_L030_Last-Acc | 66.293614 | % | completed canonical run |
| IA5 | StoredAPR_RP_C_P020_L030_Forgetting | 7.866575 | pp | completed canonical run |

원본: `structure_stored_apr_rp/outputs/CAR5/StoredAPR_RP_C_P020_L030/seed_1993/summary.json`, `structure_stored_apr_rp/outputs/IA5/StoredAPR_RP_C_P020_L030/seed_1993/summary.json`

## FeCAM_CEOS

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | F0_seen_class_accuracy | 64.713627 | % | final frozen features; matched comparator per condition |
| IA10 | F0_old_accuracy | 64.516129 | % | final frozen features; matched comparator per condition |
| IA10 | F0_new_accuracy | 66.935484 | % | final frozen features; matched comparator per condition |
| IA10 | F1_seen_class_accuracy | 60.763660 | % | final frozen features; matched comparator per condition |
| IA10 | F1_old_accuracy | 61.075269 | % | final frozen features; matched comparator per condition |
| IA10 | F1_new_accuracy | 57.258065 | % | final frozen features; matched comparator per condition |
| IA10 | F2_seen_class_accuracy | 59.907834 | % | final frozen features; matched comparator per condition |
| IA10 | F2_old_accuracy | 60.143369 | % | final frozen features; matched comparator per condition |
| IA10 | F2_new_accuracy | 57.258065 | % | final frozen features; matched comparator per condition |
| IA10 | F3_seen_class_accuracy | 60.368664 | % | final frozen features; matched comparator per condition |
| IA10 | F3_old_accuracy | 60.501792 | % | final frozen features; matched comparator per condition |
| IA10 | F3_new_accuracy | 58.870968 | % | final frozen features; matched comparator per condition |
| IA10 | C0_matched_stored_gaussian_seen_class_accuracy | 64.384463 | % | final frozen features; matched comparator per condition |
| IA10 | C0_matched_stored_gaussian_old_accuracy | 64.229391 | % | final frozen features; matched comparator per condition |
| IA10 | C0_matched_stored_gaussian_new_accuracy | 66.129032 | % | final frozen features; matched comparator per condition |
| IA10 | C1_ceos_only_seen_class_accuracy | 63.726136 | % | final frozen features; matched comparator per condition |
| IA10 | C1_ceos_only_old_accuracy | 63.870968 | % | final frozen features; matched comparator per condition |
| IA10 | C1_ceos_only_new_accuracy | 62.096774 | % | final frozen features; matched comparator per condition |
| IA10 | c0_synthetic_margin_q05 | 0.573918 | logit margin | final frozen features; matched comparator per condition |
| IA10 | c0_synthetic_margin_q10 | 1.149731 | logit margin | final frozen features; matched comparator per condition |
| IA10 | c0_synthetic_margin_negative_fraction | 0.023090 | ratio | final frozen features; matched comparator per condition |
| IA10 | ceos_synthetic_margin_q05 | 0.341677 | logit margin | final frozen features; matched comparator per condition |
| IA10 | ceos_synthetic_margin_q10 | 0.914571 | logit margin | final frozen features; matched comparator per condition |
| IA10 | ceos_synthetic_margin_negative_fraction | 0.032096 | ratio | final frozen features; matched comparator per condition |
| CAR10 | F0_seen_class_accuracy | 78.957841 | % | final frozen features; matched comparator per condition |
| CAR10 | F0_old_accuracy | 78.683558 | % | final frozen features; matched comparator per condition |
| CAR10 | F0_new_accuracy | 81.514763 | % | final frozen features; matched comparator per condition |
| CAR10 | F1_seen_class_accuracy | 62.666335 | % | final frozen features; matched comparator per condition |
| CAR10 | F1_old_accuracy | 62.407050 | % | final frozen features; matched comparator per condition |
| CAR10 | F1_new_accuracy | 65.083440 | % | final frozen features; matched comparator per condition |
| CAR10 | F2_seen_class_accuracy | 74.592712 | % | final frozen features; matched comparator per condition |
| CAR10 | F2_old_accuracy | 74.125585 | % | final frozen features; matched comparator per condition |
| CAR10 | F2_new_accuracy | 78.947368 | % | final frozen features; matched comparator per condition |
| CAR10 | F3_seen_class_accuracy | 77.129710 | % | final frozen features; matched comparator per condition |
| CAR10 | F3_old_accuracy | 77.141283 | % | final frozen features; matched comparator per condition |
| CAR10 | F3_new_accuracy | 77.021823 | % | final frozen features; matched comparator per condition |
| CAR10 | C0_matched_stored_gaussian_seen_class_accuracy | 78.559881 | % | final frozen features; matched comparator per condition |
| CAR10 | C0_matched_stored_gaussian_old_accuracy | 78.394382 | % | final frozen features; matched comparator per condition |
| CAR10 | C0_matched_stored_gaussian_new_accuracy | 80.102696 | % | final frozen features; matched comparator per condition |
| CAR10 | C1_ceos_only_seen_class_accuracy | 78.087303 | % | final frozen features; matched comparator per condition |
| CAR10 | C1_ceos_only_old_accuracy | 78.490774 | % | final frozen features; matched comparator per condition |
| CAR10 | C1_ceos_only_new_accuracy | 74.326059 | % | final frozen features; matched comparator per condition |
| CAR10 | c0_synthetic_margin_q05 | 0.147021 | logit margin | final frozen features; matched comparator per condition |
| CAR10 | c0_synthetic_margin_q10 | 0.624115 | logit margin | final frozen features; matched comparator per condition |
| CAR10 | c0_synthetic_margin_negative_fraction | 0.039901 | ratio | final frozen features; matched comparator per condition |
| CAR10 | ceos_synthetic_margin_q05 | -0.074803 | logit margin | final frozen features; matched comparator per condition |
| CAR10 | ceos_synthetic_margin_q10 | 0.405177 | logit margin | final frozen features; matched comparator per condition |
| CAR10 | ceos_synthetic_margin_negative_fraction | 0.056254 | ratio | final frozen features; matched comparator per condition |

원본: `fecam_boundary_followup/ceos/CAR10/metrics.json`, `fecam_boundary_followup/ceos/IA10/metrics.json`, `fecam_boundary_followup/fecam/CAR10/metrics.json`, `fecam_boundary_followup/fecam/IA10/metrics.json`

## pair_transition

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | positive_degradation_mass | 25.000000 | boundary violation count | T1->T2 |
| IA10 | improvement_mass | 46.000000 | boundary violation count | T1->T2 |
| IA10 | positive_degradation_mass | 29.000000 | boundary violation count | T2->T3 |
| IA10 | improvement_mass | 30.000000 | boundary violation count | T2->T3 |
| IA10 | positive_degradation_mass | 46.000000 | boundary violation count | T3->T4 |
| IA10 | improvement_mass | 111.000000 | boundary violation count | T3->T4 |
| IA10 | positive_degradation_mass | 44.000000 | boundary violation count | T4->T5 |
| IA10 | improvement_mass | 94.000000 | boundary violation count | T4->T5 |
| IA10 | positive_degradation_mass | 61.000000 | boundary violation count | T5->T6 |
| IA10 | improvement_mass | 92.000000 | boundary violation count | T5->T6 |
| IA10 | positive_degradation_mass | 129.000000 | boundary violation count | T6->T7 |
| IA10 | improvement_mass | 166.000000 | boundary violation count | T6->T7 |
| IA10 | positive_degradation_mass | 184.000000 | boundary violation count | T7->T8 |
| IA10 | improvement_mass | 186.000000 | boundary violation count | T7->T8 |
| IA10 | positive_degradation_mass | 197.000000 | boundary violation count | T8->T9 |
| IA10 | improvement_mass | 262.000000 | boundary violation count | T8->T9 |
| IA10 | positive_degradation_mass | 216.000000 | boundary violation count | T9->T10 |
| IA10 | improvement_mass | 227.000000 | boundary violation count | T9->T10 |
| CAR10 | positive_degradation_mass | 43.000000 | boundary violation count | T1->T2 |
| CAR10 | improvement_mass | 106.000000 | boundary violation count | T1->T2 |
| CAR10 | positive_degradation_mass | 66.000000 | boundary violation count | T2->T3 |
| CAR10 | improvement_mass | 125.000000 | boundary violation count | T2->T3 |
| CAR10 | positive_degradation_mass | 96.000000 | boundary violation count | T3->T4 |
| CAR10 | improvement_mass | 185.000000 | boundary violation count | T3->T4 |
| CAR10 | positive_degradation_mass | 142.000000 | boundary violation count | T4->T5 |
| CAR10 | improvement_mass | 238.000000 | boundary violation count | T4->T5 |
| CAR10 | positive_degradation_mass | 171.000000 | boundary violation count | T5->T6 |
| CAR10 | improvement_mass | 223.000000 | boundary violation count | T5->T6 |
| CAR10 | positive_degradation_mass | 146.000000 | boundary violation count | T6->T7 |
| CAR10 | improvement_mass | 284.000000 | boundary violation count | T6->T7 |
| CAR10 | positive_degradation_mass | 238.000000 | boundary violation count | T7->T8 |
| CAR10 | improvement_mass | 368.000000 | boundary violation count | T7->T8 |
| CAR10 | positive_degradation_mass | 299.000000 | boundary violation count | T8->T9 |
| CAR10 | improvement_mass | 354.000000 | boundary violation count | T8->T9 |
| CAR10 | positive_degradation_mass | 292.000000 | boundary violation count | T9->T10 |
| CAR10 | improvement_mass | 514.000000 | boundary violation count | T9->T10 |

원본: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/transition_summary.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/transition_summary.csv`

## pair_concentration

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | top20_share | 0.083955 | ratio | T10 directed error pairs |
| CAR10 | top20_share | 0.132979 | ratio | T10 directed error pairs |

원본: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/error_concentration_by_task.csv`, `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/error_concentration_by_task.csv`

## JS_consistency

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | control_t9_new | 57.823129 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| IA10 | control_t10_seen | 64.450296 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| IA10 | control_t10_newly_old_t9 | 57.823129 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| IA10 | control_retention_t9 | 0.000000 | pp | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| IA10 | js_t9_new | 57.823129 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| IA10 | js_t10_seen | 64.450296 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| IA10 | js_t10_newly_old_t9 | 57.823129 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| IA10 | js_retention_t9 | 0.000000 | pp | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| CAR10 | control_t9_new | 81.794195 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| CAR10 | control_t10_seen | 78.634498 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| CAR10 | control_t10_newly_old_t9 | 80.738786 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| CAR10 | control_retention_t9 | -1.055409 | pp | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| CAR10 | js_t9_new | 81.530343 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| CAR10 | js_t10_seen | 78.634498 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| CAR10 | js_t10_newly_old_t9 | 80.606860 | % | same two-view supervised control; lambda=.1, T=1 unscaled cosine |
| CAR10 | js_retention_t9 | -0.923483 | pp | same two-view supervised control; lambda=.1, T=1 unscaled cosine |

원본: `variation_consistency_pilot/cross_protocol/summary.csv`

## paired_source

| Protocol | Metric | Value | Unit | Scope |
|---|---|---:|---|---|
| IA10 | AB_repair_count | 30.000000 | count | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| IA10 | AB_harm_count | 29.000000 | count | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| IA10 | AB_net_gain_pp | 0.065833 | pp | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| IA10 | AB_class_macro_delta_pp | -0.974659 | pp | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| IA10 | BC_repair_count | 25.000000 | count | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| IA10 | BC_harm_count | 20.000000 | count | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| IA10 | BC_net_gain_pp | 0.329164 | pp | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| IA10 | BC_class_macro_delta_pp | -0.029151 | pp | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| CAR10 | AB_repair_count | 180.000000 | count | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| CAR10 | AB_harm_count | 108.000000 | count | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| CAR10 | AB_net_gain_pp | 0.895411 | pp | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| CAR10 | AB_class_macro_delta_pp | 0.870167 | pp | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| CAR10 | BC_repair_count | 229.000000 | count | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| CAR10 | BC_harm_count | 135.000000 | count | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| CAR10 | BC_net_gain_pp | 1.169009 | pp | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |
| CAR10 | BC_class_macro_delta_pp | 1.181704 | pp | A actual; B oracle Gaussian; C stored Gaussian; identical current bank |

원본: `structure_prototype_oracle_audit/paired_source_prediction_diagnostic/cross_protocol/final_summary.csv`

