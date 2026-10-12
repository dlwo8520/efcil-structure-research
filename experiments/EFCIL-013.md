# EFCIL-013: Moment/source/head/representation diagnostics

Project: `structure_prototype_oracle_audit`. Status at packaging: **multiple completed diagnostics; protocol-specific**.

## Hypothesis and problem

Locate memory and decision bottlenecks.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: diagnostic-only old/oracle access. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../structure_prototype_oracle_audit/scientific_audit/structure_oracle_scientific_final_report_20260928.md).

## Code, configurations and environment

Entrypoints: [structure_prototype_oracle_audit/run_scientific_queue.py](../structure_prototype_oracle_audit/run_scientific_queue.py), [structure_prototype_oracle_audit/run_audit.py](../structure_prototype_oracle_audit/run_audit.py), [structure_prototype_oracle_audit/run_representation_relation_diagnostic.py](../structure_prototype_oracle_audit/run_representation_relation_diagnostic.py), [structure_prototype_oracle_audit/run_ten_task_hypothesis_revalidation.py](../structure_prototype_oracle_audit/run_ten_task_hypothesis_revalidation.py), [structure_prototype_oracle_audit/run_paired_source_prediction_diagnostic.py](../structure_prototype_oracle_audit/run_paired_source_prediction_diagnostic.py), [structure_prototype_oracle_audit/run_classifier_procedure_diagnostic.py](../structure_prototype_oracle_audit/run_classifier_procedure_diagnostic.py), [structure_prototype_oracle_audit/run_coverage_generalization_diagnostic.py](../structure_prototype_oracle_audit/run_coverage_generalization_diagnostic.py), [structure_prototype_oracle_audit/run_classifier_diagnostic.py](../structure_prototype_oracle_audit/run_classifier_diagnostic.py), [structure_prototype_oracle_audit/run_structure.py](../structure_prototype_oracle_audit/run_structure.py), [structure_prototype_oracle_audit/run_queue.py](../structure_prototype_oracle_audit/run_queue.py), [structure_prototype_oracle_audit/run_pair_boundary_risk_diagnostic.py](../structure_prototype_oracle_audit/run_pair_boundary_risk_diagnostic.py).

Configuration files: no standalone config catalogue; inspect CLI/source and resolved run metadata.

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `structure_prototype_oracle_audit/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
