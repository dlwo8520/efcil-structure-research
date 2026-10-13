# EFCIL-011: Auxiliary random-projection Proto-CLAD

Project: `structure_rp_protoclad_task5`. Status at packaging: **IA5 two completed; CAR5 old RP not started**.

## Hypothesis and problem

Test RP only in auxiliary conflict geometry.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: custom adaptation; not full RP-paper reproduction. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../structure_rp_protoclad_task5/rp_protoclad_report.md).

## Code, configurations and environment

Entrypoints: [structure_rp_protoclad_task5/run.py](../structure_rp_protoclad_task5/run.py).

Configuration files: [structure_rp_protoclad_task5/configs/RP_C_P020_L100_IA5.json](../structure_rp_protoclad_task5/configs/RP_C_P020_L100_IA5.json), [structure_rp_protoclad_task5/configs/RP_C_P020_L030_CAR5.json](../structure_rp_protoclad_task5/configs/RP_C_P020_L030_CAR5.json), [structure_rp_protoclad_task5/configs/RP_C_P020_L030_IA5.json](../structure_rp_protoclad_task5/configs/RP_C_P020_L030_IA5.json), [structure_rp_protoclad_task5/configs/RP_C_P020_L100_CAR5.json](../structure_rp_protoclad_task5/configs/RP_C_P020_L100_CAR5.json).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `structure_rp_protoclad_task5/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
