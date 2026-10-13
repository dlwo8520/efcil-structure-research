# EFCIL-006: E0–E7 multi-center/Mahalanobis

Project: `structure_multicenter_separation`. Status at packaging: **completed records and inactive-loss limitations**.

## Hypothesis and problem

Test local regions and old/new separation.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: upstream CDS plus proposed region code. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../structure_multicenter_separation/PROJECT_ANALYSIS_KO.md).

## Code, configurations and environment

Entrypoints: [structure_multicenter_separation/trainer.py](../structure_multicenter_separation/trainer.py), [structure_multicenter_separation/main.py](../structure_multicenter_separation/main.py).

Configuration files: [structure_multicenter_separation/exps_jsons/adapter_car196.json](../structure_multicenter_separation/exps_jsons/adapter_car196.json), [structure_multicenter_separation/exps_jsons/adapter_cifar224.json](../structure_multicenter_separation/exps_jsons/adapter_cifar224.json), [structure_multicenter_separation/exps_jsons/adapter_ob.json](../structure_multicenter_separation/exps_jsons/adapter_ob.json), [structure_multicenter_separation/exps_jsons/adapter_imageneta.json](../structure_multicenter_separation/exps_jsons/adapter_imageneta.json), [structure_multicenter_separation/exps_jsons/adapter_objectnet.json](../structure_multicenter_separation/exps_jsons/adapter_objectnet.json), [structure_multicenter_separation/exps_jsons/adapter_imagenetr.json](../structure_multicenter_separation/exps_jsons/adapter_imagenetr.json), [structure_multicenter_separation/exps_jsons/imageneta_5task/E5.json](../structure_multicenter_separation/exps_jsons/imageneta_5task/E5.json), [structure_multicenter_separation/exps_jsons/imageneta_5task/E6.json](../structure_multicenter_separation/exps_jsons/imageneta_5task/E6.json), [structure_multicenter_separation/exps_jsons/imageneta_5task/E2.json](../structure_multicenter_separation/exps_jsons/imageneta_5task/E2.json), [structure_multicenter_separation/exps_jsons/imageneta_5task/E7.json](../structure_multicenter_separation/exps_jsons/imageneta_5task/E7.json), [structure_multicenter_separation/exps_jsons/imageneta_5task/E4.json](../structure_multicenter_separation/exps_jsons/imageneta_5task/E4.json), [structure_multicenter_separation/exps_jsons/imageneta_5task/E1.json](../structure_multicenter_separation/exps_jsons/imageneta_5task/E1.json), [structure_multicenter_separation/exps_jsons/imageneta_5task/E0.json](../structure_multicenter_separation/exps_jsons/imageneta_5task/E0.json), [structure_multicenter_separation/exps_jsons/imageneta_5task/E3.json](../structure_multicenter_separation/exps_jsons/imageneta_5task/E3.json).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `structure_multicenter_separation/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
