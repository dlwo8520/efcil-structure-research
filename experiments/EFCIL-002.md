# EFCIL-002: CDSP and PCDS+SGDS exploration

Project: `cds-main`. Status at packaging: **completed and variant-specific records**.

## Hypothesis and problem

Protect discriminative subspaces and test activation guidance.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: upstream CDS plus custom changes. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../cds-main/RESULTS_CDSP.md).

## Code, configurations and environment

Entrypoints: [cds-main/trainer.py](../cds-main/trainer.py), [cds-main/main.py](../cds-main/main.py), [cds-main/analyze_cars196_pcds_sgds.py](../cds-main/analyze_cars196_pcds_sgds.py), [cds-main/analyze_pcds_structure_sgds.py](../cds-main/analyze_pcds_structure_sgds.py).

Configuration files: [cds-main/exps_jsons/cars196_pcds_structure_sgds.json](../cds-main/exps_jsons/cars196_pcds_structure_sgds.json), [cds-main/exps_jsons/cifar100_inter_subspace.json](../cds-main/exps_jsons/cifar100_inter_subspace.json), [cds-main/exps_jsons/adapter_car196.json](../cds-main/exps_jsons/adapter_car196.json), [cds-main/exps_jsons/cifar100_pcds_structure_sgds.json](../cds-main/exps_jsons/cifar100_pcds_structure_sgds.json), [cds-main/exps_jsons/adapter_cifar224.json](../cds-main/exps_jsons/adapter_cifar224.json), [cds-main/exps_jsons/cifar100_inter_intra_subspace.json](../cds-main/exps_jsons/cifar100_inter_intra_subspace.json), [cds-main/exps_jsons/cifar100_task_subspace.json](../cds-main/exps_jsons/cifar100_task_subspace.json), [cds-main/exps_jsons/cars196_pcds.json](../cds-main/exps_jsons/cars196_pcds.json), [cds-main/exps_jsons/cifar100_proto_baseline.json](../cds-main/exps_jsons/cifar100_proto_baseline.json), [cds-main/exps_jsons/adapter_ob.json](../cds-main/exps_jsons/adapter_ob.json), [cds-main/exps_jsons/adapter_imageneta.json](../cds-main/exps_jsons/adapter_imageneta.json), [cds-main/exps_jsons/adapter_objectnet.json](../cds-main/exps_jsons/adapter_objectnet.json), [cds-main/exps_jsons/adapter_imagenetr.json](../cds-main/exps_jsons/adapter_imagenetr.json).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `cds-main/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
