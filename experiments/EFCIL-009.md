# EFCIL-009: High/low/random-off controls

Project: `structure_similarity_control_experiments`. Status at packaging: **completed IA5 single-seed pilot**.

## Hypothesis and problem

Disentangle similarity from constraint count.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: controlled custom ablations. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../structure_similarity_control_experiments/reports/additional_control/report.md).

## Code, configurations and environment

Entrypoints: [structure_similarity_control_experiments/run.py](../structure_similarity_control_experiments/run.py).

Configuration files: [structure_similarity_control_experiments/configs/C4.json](../structure_similarity_control_experiments/configs/C4.json), [structure_similarity_control_experiments/configs/C2.json](../structure_similarity_control_experiments/configs/C2.json), [structure_similarity_control_experiments/configs/C0.json](../structure_similarity_control_experiments/configs/C0.json), [structure_similarity_control_experiments/configs/C1.json](../structure_similarity_control_experiments/configs/C1.json), [structure_similarity_control_experiments/configs/smoke_C3.json](../structure_similarity_control_experiments/configs/smoke_C3.json), [structure_similarity_control_experiments/configs/C3.json](../structure_similarity_control_experiments/configs/C3.json).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `structure_similarity_control_experiments/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
