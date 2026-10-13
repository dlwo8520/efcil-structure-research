# EFCIL-001: Early CIFAR diagnostics / KeepLoRA-style / PCDS

Project: `projects/efcil_direction_diagnostic`. Status at packaging: **implemented; baseline-specific outputs**.

## Hypothesis and problem

Diagnose representation and classifier limitations.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: custom reproduction/adaptation. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../projects/efcil_direction_diagnostic/README.md).

## Code, configurations and environment

Entrypoints: [projects/efcil_direction_diagnostic/train_keeplora.py](../projects/efcil_direction_diagnostic/train_keeplora.py), [projects/efcil_direction_diagnostic/analyze.py](../projects/efcil_direction_diagnostic/analyze.py), [projects/efcil_direction_diagnostic/train.py](../projects/efcil_direction_diagnostic/train.py), [projects/efcil_direction_diagnostic/analyze_keeplora.py](../projects/efcil_direction_diagnostic/analyze_keeplora.py), [projects/efcil_direction_diagnostic/analyze_pcds.py](../projects/efcil_direction_diagnostic/analyze_pcds.py), [projects/efcil_direction_diagnostic/train_pcds.py](../projects/efcil_direction_diagnostic/train_pcds.py).

Configuration files: [projects/efcil_direction_diagnostic/configs/pcds_efcil_cifar100_10task.yaml](../projects/efcil_direction_diagnostic/configs/pcds_efcil_cifar100_10task.yaml), [projects/efcil_direction_diagnostic/configs/keeplora_efcil_sanity.yaml](../projects/efcil_direction_diagnostic/configs/keeplora_efcil_sanity.yaml), [projects/efcil_direction_diagnostic/configs/cifar100_10task.yaml](../projects/efcil_direction_diagnostic/configs/cifar100_10task.yaml), [projects/efcil_direction_diagnostic/configs/sanity.yaml](../projects/efcil_direction_diagnostic/configs/sanity.yaml), [projects/efcil_direction_diagnostic/configs/keeplora_efcil_cifar100_10task.yaml](../projects/efcil_direction_diagnostic/configs/keeplora_efcil_cifar100_10task.yaml).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `projects/efcil_direction_diagnostic/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
