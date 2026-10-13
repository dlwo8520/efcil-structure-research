# EFCIL-007: E3/PIC/CEOS/adaptive margins

Project: `structure_e3_followup`. Status at packaging: **13 IA5 pilot variants complete**.

## Hypothesis and problem

Resolve new loss versus old stability.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: custom extensions; adapted prior ideas. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../structure_e3_followup/reports/e3_followup_comparison_20260917/report.md).

## Code, configurations and environment

Entrypoints: [structure_e3_followup/run.py](../structure_e3_followup/run.py).

Configuration files: [structure_e3_followup/configs/L03.json](../structure_e3_followup/configs/L03.json), [structure_e3_followup/configs/E3_ahn.json](../structure_e3_followup/configs/E3_ahn.json), [structure_e3_followup/configs/L01.json](../structure_e3_followup/configs/L01.json), [structure_e3_followup/configs/AM_drift.json](../structure_e3_followup/configs/AM_drift.json), [structure_e3_followup/configs/B1_e3.json](../structure_e3_followup/configs/B1_e3.json), [structure_e3_followup/configs/AM_variance.json](../structure_e3_followup/configs/AM_variance.json), [structure_e3_followup/configs/common.json](../structure_e3_followup/configs/common.json), [structure_e3_followup/configs/FCS_pic.json](../structure_e3_followup/configs/FCS_pic.json), [structure_e3_followup/configs/L05.json](../structure_e3_followup/configs/L05.json), [structure_e3_followup/configs/CEOS100.json](../structure_e3_followup/configs/CEOS100.json), [structure_e3_followup/configs/AM_age.json](../structure_e3_followup/configs/AM_age.json), [structure_e3_followup/configs/L10.json](../structure_e3_followup/configs/L10.json), [structure_e3_followup/configs/B0_structure.json](../structure_e3_followup/configs/B0_structure.json), [structure_e3_followup/configs/CEOS50.json](../structure_e3_followup/configs/CEOS50.json), [structure_e3_followup/configs/E3_norm.json](../structure_e3_followup/configs/E3_norm.json).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `structure_e3_followup/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
