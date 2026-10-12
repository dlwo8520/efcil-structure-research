# EFCIL-004: PTM-guided geometry

Project: `cdsp_handoff_20260813`. Status at packaging: **recorded results; paper numbers separated**.

## Hypothesis and problem

Compare pretrained anchors and Structure.

## Baseline and method difference

package-specific Structure/CDS or explicitly recorded control. Provenance: custom adaptation. The linked methodology report records the actual intervention and its difference from the comparator; no extra loss/metric is implied.

Method, interpretation, limitations and historical next steps: [source report](../cdsp_handoff_20260813/RESULTS_PTM_GUIDED_VS_STRUCTURE.md).

## Code, configurations and environment

Entrypoints: [cdsp_handoff_20260813/trainer.py](../cdsp_handoff_20260813/trainer.py), [cdsp_handoff_20260813/main.py](../cdsp_handoff_20260813/main.py).

Configuration files: [cdsp_handoff_20260813/exps_jsons/ptm_guided_cars196_10task_adapter_baseline.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_cars196_10task_adapter_baseline.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_imageneta_10task_ptm_place_move.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_imageneta_10task_ptm_place_move.json), [cdsp_handoff_20260813/exps_jsons/cifar100_inter_subspace.json](../cdsp_handoff_20260813/exps_jsons/cifar100_inter_subspace.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_imageneta_10task_ptm_place_only.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_imageneta_10task_ptm_place_only.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_cars196_5task_ptm_place_move.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_cars196_5task_ptm_place_move.json), [cdsp_handoff_20260813/exps_jsons/cifar100_inter_intra_subspace.json](../cdsp_handoff_20260813/exps_jsons/cifar100_inter_intra_subspace.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_cars196_10task_ptm_place_only.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_cars196_10task_ptm_place_only.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_omnibenchmark_10task_ptm_place_move.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_omnibenchmark_10task_ptm_place_move.json), [cdsp_handoff_20260813/exps_jsons/cifar100_task_subspace.json](../cdsp_handoff_20260813/exps_jsons/cifar100_task_subspace.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_cars196_10task_ptm_place_move.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_cars196_10task_ptm_place_move.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_imageneta_5task_ptm_place_move.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_imageneta_5task_ptm_place_move.json), [cdsp_handoff_20260813/exps_jsons/cifar100_proto_baseline.json](../cdsp_handoff_20260813/exps_jsons/cifar100_proto_baseline.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_base.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_base.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_imagenetr_10task_ptm_place_move.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_imagenetr_10task_ptm_place_move.json), [cdsp_handoff_20260813/exps_jsons/ptm_guided_imageneta_20task_ptm_place_move.json](../cdsp_handoff_20260813/exps_jsons/ptm_guided_imageneta_20task_ptm_place_move.json).

Environment: legacy Structure uses `environments/requirements-structure.txt`; early CIFAR diagnostic and generator dependencies are separate. See [reproduction](../docs/REPRODUCIBILITY.md). Dataset, partition, seed, optimizer and hyperparameters come from the specific `resolved_config.json` and protocol files. A family does not have one universal setting.

## Execution and evidence

Exact historical commands are preserved in the source README/report and shell scripts. Use `python ENTRY --help`, provide explicit dataset/checkpoint/output roots, and restore compressed tables before analysis. Do not rerun GPU experiments as part of index generation. New runs must use a new ID/output directory.

Results, original configs and runtime manifests remain under `cdsp_handoff_20260813/`. The [run index](../results/cil_run_index.csv) links recorded summary/config pairs; [results](../docs/RESULTS.md) separates comparison blocks. Evidence not in the numeric index, including diagnostic probes and repair/harm, is in the linked original report/raw files.

## Current follow-up

Follow the [current status](../docs/STATUS.md). Historical report suggestions are preserved as history and are not automatically adopted. Failed, partial, held and unimplemented hypotheses remain separate from completed numerical outcomes.
