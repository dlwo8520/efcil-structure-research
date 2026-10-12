# EFCIL-003: Structure+SGDS local adaptation

Project: `pcds_structure_sgds_official`. Status at packaging: **recorded local results**.

## Hypothesis and problem

Test semantic activation guidance.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: modified upstream, not full SGDS. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../pcds_structure_sgds_official/RESULTS_STRUCTURE_SGDS.md).

## Code, configurations and environment

Entrypoints: [pcds_structure_sgds_official/trainer.py](../pcds_structure_sgds_official/trainer.py), [pcds_structure_sgds_official/main.py](../pcds_structure_sgds_official/main.py).

Configuration files: [pcds_structure_sgds_official/exps_jsons/adapter_car196.json](../pcds_structure_sgds_official/exps_jsons/adapter_car196.json), [pcds_structure_sgds_official/exps_jsons/adapter_cifar224.json](../pcds_structure_sgds_official/exps_jsons/adapter_cifar224.json), [pcds_structure_sgds_official/exps_jsons/cars196_structure_pcds_seed0.json](../pcds_structure_sgds_official/exps_jsons/cars196_structure_pcds_seed0.json), [pcds_structure_sgds_official/exps_jsons/cars196_structure_pcds_sgds_seed0.json](../pcds_structure_sgds_official/exps_jsons/cars196_structure_pcds_sgds_seed0.json), [pcds_structure_sgds_official/exps_jsons/adapter_ob.json](../pcds_structure_sgds_official/exps_jsons/adapter_ob.json), [pcds_structure_sgds_official/exps_jsons/adapter_imageneta.json](../pcds_structure_sgds_official/exps_jsons/adapter_imageneta.json), [pcds_structure_sgds_official/exps_jsons/adapter_objectnet.json](../pcds_structure_sgds_official/exps_jsons/adapter_objectnet.json), [pcds_structure_sgds_official/exps_jsons/adapter_imagenetr.json](../pcds_structure_sgds_official/exps_jsons/adapter_imagenetr.json).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `pcds_structure_sgds_official/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
