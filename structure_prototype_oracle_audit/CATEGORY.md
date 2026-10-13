# Prototype, moment, source and head audits

**Category:** Oracle and source diagnostics

**Repository path:** `structure_prototype_oracle_audit/`

## Purpose

Locate memory and decision bottlenecks.

## Evidence and status

multiple completed diagnostics; protocol-specific. Interpret this state using the linked analysis and saved run records.

## Method identity

- Experiment ID: `EFCIL-013`
- Question: Locate memory and decision bottlenecks.
- Origin: diagnostic-only old/oracle access.
- Baseline: package-specific Structure/CDS or explicitly recorded control.
- Limitations: single-seed and protocol-specific evidence; diagnostic boundaries in report.

## Read first

- [structure_prototype_oracle_audit/scientific_audit/structure_oracle_scientific_final_report_20260928.md](scientific_audit/structure_oracle_scientific_final_report_20260928.md)
- [experiments/EFCIL-013.md](../experiments/EFCIL-013.md)
- [structure_prototype_oracle_audit/run_scientific_queue.py](run_scientific_queue.py)
- [structure_prototype_oracle_audit/run_audit.py](run_audit.py)
- [structure_prototype_oracle_audit/run_representation_relation_diagnostic.py](run_representation_relation_diagnostic.py)

## Contents

| Entry | Role |
|---|---|
| [.gitignore](.gitignore) | Preserved source or evidence file |
| [README.md](README.md) | Original package overview |
| [ca_sampling_audit.md](ca_sampling_audit.md) | Preserved source or evidence file |
| [classifier_bottleneck_diagnostic](classifier_bottleneck_diagnostic) | Original subdirectory; consult its saved code, configs or evidence |
| [classifier_bottleneck_diagnostic_plan.md](classifier_bottleneck_diagnostic_plan.md) | Preserved source or evidence file |
| [classifier_diagnostic](classifier_diagnostic) | Original subdirectory; consult its saved code, configs or evidence |
| [classifier_procedure](classifier_procedure) | Original subdirectory; consult its saved code, configs or evidence |
| [classifier_procedure_diagnostic](classifier_procedure_diagnostic) | Original subdirectory; consult its saved code, configs or evidence |
| [classifier_procedure_diagnostic_gpu2](classifier_procedure_diagnostic_gpu2) | Original subdirectory; consult its saved code, configs or evidence |
| [classifier_procedure_diagnostic_plan.md](classifier_procedure_diagnostic_plan.md) | Preserved source or evidence file |
| [coverage_generalization](coverage_generalization) | Original subdirectory; consult its saved code, configs or evidence |
| [coverage_generalization_diagnostic](coverage_generalization_diagnostic) | Original subdirectory; consult its saved code, configs or evidence |
| [generate_report.py](generate_report.py) | Preserved source or evidence file |
| [logs](logs) | Saved analysis or execution evidence; completion is record-specific |
| [oracle_audit](oracle_audit) | Method implementation |
| [outputs](outputs) | Saved analysis or execution evidence; completion is record-specific |
| [outputs_revision_smoke](outputs_revision_smoke) | Original subdirectory; consult its saved code, configs or evidence |
| [outputs_smoke](outputs_smoke) | Original subdirectory; consult its saved code, configs or evidence |
| [outputs_smoke_verify2](outputs_smoke_verify2) | Original subdirectory; consult its saved code, configs or evidence |
| [pair_boundary_risk](pair_boundary_risk) | Original subdirectory; consult its saved code, configs or evidence |
| [pair_boundary_risk_diagnostic](pair_boundary_risk_diagnostic) | Original subdirectory; consult its saved code, configs or evidence |
| [pair_boundary_risk_diagnostic_plan.md](pair_boundary_risk_diagnostic_plan.md) | Preserved source or evidence file |
| [paired_source_prediction](paired_source_prediction) | Original subdirectory; consult its saved code, configs or evidence |
| [paired_source_prediction_diagnostic](paired_source_prediction_diagnostic) | Original subdirectory; consult its saved code, configs or evidence |
| [reports](reports) | Saved analysis or execution evidence; completion is record-specific |
| [representation_relation](representation_relation) | Original subdirectory; consult its saved code, configs or evidence |
| [representation_relation_diagnostic](representation_relation_diagnostic) | Original subdirectory; consult its saved code, configs or evidence |
| [representation_relation_diagnostic_plan.md](representation_relation_diagnostic_plan.md) | Preserved source or evidence file |
| [run_audit.py](run_audit.py) | Preserved source or evidence file |
| [run_classifier_diagnostic.py](run_classifier_diagnostic.py) | Preserved source or evidence file |
| [run_classifier_procedure_diagnostic.py](run_classifier_procedure_diagnostic.py) | Preserved source or evidence file |
| [run_classifier_procedure_gpu2.sh](run_classifier_procedure_gpu2.sh) | Preserved source or evidence file |
| [run_coverage_generalization_diagnostic.py](run_coverage_generalization_diagnostic.py) | Preserved source or evidence file |
| [run_coverage_generalization_gpu4.sh](run_coverage_generalization_gpu4.sh) | Preserved source or evidence file |
| [run_gpu1.sh](run_gpu1.sh) | Preserved source or evidence file |
| [run_pair_boundary_risk_diagnostic.py](run_pair_boundary_risk_diagnostic.py) | Preserved source or evidence file |
| [run_pair_boundary_risk_gpu4.sh](run_pair_boundary_risk_gpu4.sh) | Preserved source or evidence file |
| [run_paired_source_prediction_diagnostic.py](run_paired_source_prediction_diagnostic.py) | Preserved source or evidence file |
| [run_paired_source_prediction_gpu4.sh](run_paired_source_prediction_gpu4.sh) | Preserved source or evidence file |
| [run_queue.py](run_queue.py) | Preserved source or evidence file |
| [run_representation_relation_diagnostic.py](run_representation_relation_diagnostic.py) | Preserved source or evidence file |
| [run_representation_relation_gpu2.sh](run_representation_relation_gpu2.sh) | Preserved source or evidence file |
| [run_scientific_gpu1.sh](run_scientific_gpu1.sh) | Preserved source or evidence file |
| [run_scientific_queue.py](run_scientific_queue.py) | Preserved source or evidence file |
| [run_structure.py](run_structure.py) | Preserved source or evidence file |
| [run_ten_task_hypothesis_revalidation.py](run_ten_task_hypothesis_revalidation.py) | Preserved source or evidence file |
| [run_ten_task_hypothesis_revalidation_gpu4.sh](run_ten_task_hypothesis_revalidation_gpu4.sh) | Preserved source or evidence file |
| [scientific_audit](scientific_audit) | Original subdirectory; consult its saved code, configs or evidence |
| [scientific_status.py](scientific_status.py) | Preserved source or evidence file |
| [scripts](scripts) | Launch, analysis or validation commands |
| [status.py](status.py) | Preserved source or evidence file |
| [structure_covariance_code_audit.md](structure_covariance_code_audit.md) | Preserved source or evidence file |
| [structure_oracle_audit_revision_plan.md](structure_oracle_audit_revision_plan.md) | Preserved source or evidence file |
| [structure_prototype_audit_plan.md](structure_prototype_audit_plan.md) | Preserved source or evidence file |
| [ten_task_hypothesis_revalidation](ten_task_hypothesis_revalidation) | Original subdirectory; consult its saved code, configs or evidence |
| [ten_task_revalidation](ten_task_revalidation) | Original subdirectory; consult its saved code, configs or evidence |
| [tests](tests) | Existing checks |
| [training_outputs](training_outputs) | Original subdirectory; consult its saved code, configs or evidence |

## Category navigation

[Full categorized repository map](../docs/CATEGORIES.md)
