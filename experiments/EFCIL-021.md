# EFCIL-021: BOB-full versus S-full IA10

Project: `ia10_bob_vs_strong_full`. Status at packaging: **interrupted_or_failed at BOB GPU smoke; no full result**.

## Hypothesis and problem

Test full cumulative trajectories with current-only synthetic views.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: official BOB vendor plus custom CIL integration. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../ia10_bob_vs_strong_full/run_manifest.json).

## Code, configurations and environment

Entrypoints: [ia10_bob_vs_strong_full/run.py](../ia10_bob_vs_strong_full/run.py).

Configuration files: no standalone config catalogue; inspect CLI/source and resolved run metadata.

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `ia10_bob_vs_strong_full/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
