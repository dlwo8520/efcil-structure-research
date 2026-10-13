# EFCIL-012: Stored birth RP mean plus APR shift

Project: `structure_stored_apr_rp`. Status at packaging: **IA5/CAR5 complete; no common benefit**.

## Hypothesis and problem

Reduce nonlinear projected-prototype approximation.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: custom extension. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../structure_stored_apr_rp/reports/stored_apr_rp_final_20260923/report.md).

## Code, configurations and environment

Entrypoints: [structure_stored_apr_rp/run_all.py](../structure_stored_apr_rp/run_all.py), [structure_stored_apr_rp/run.py](../structure_stored_apr_rp/run.py).

Configuration files: [structure_stored_apr_rp/configs/CAR5.json](../structure_stored_apr_rp/configs/CAR5.json), [structure_stored_apr_rp/configs/IA5.json](../structure_stored_apr_rp/configs/IA5.json).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `structure_stored_apr_rp/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
