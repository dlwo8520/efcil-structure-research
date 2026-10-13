# CDS, CDSP and PCDS+SGDS variants

**Category:** Baselines and subspace adaptation

**Repository path:** `cds-main/`

## Purpose

Protect discriminative subspaces and test activation guidance.

## Evidence and status

completed and variant-specific records. Interpret this state using the linked analysis and saved run records.

## Method identity

- Experiment ID: `EFCIL-002`
- Question: Protect discriminative subspaces and test activation guidance.
- Origin: upstream CDS plus custom changes.
- Baseline: package-specific Structure/CDS or explicitly recorded control.
- Limitations: single-seed and protocol-specific evidence; diagnostic boundaries in report.

## Read first

- [cds-main/RESULTS_CDSP.md](RESULTS_CDSP.md)
- [experiments/EFCIL-002.md](../experiments/EFCIL-002.md)
- [cds-main/trainer.py](trainer.py)
- [cds-main/main.py](main.py)
- [cds-main/analyze_cars196_pcds_sgds.py](analyze_cars196_pcds_sgds.py)
- [cds-main/exps_jsons/cars196_pcds_structure_sgds.json](exps_jsons/cars196_pcds_structure_sgds.json)
- [cds-main/exps_jsons/cifar100_inter_subspace.json](exps_jsons/cifar100_inter_subspace.json)
- [cds-main/exps_jsons/adapter_car196.json](exps_jsons/adapter_car196.json)

## Contents

| Entry | Role |
|---|---|
| [.gitignore](.gitignore) | Preserved source or evidence file |
| [CARS196_RUN.md](CARS196_RUN.md) | Preserved source or evidence file |
| [HANDOFF_CDSP.md](HANDOFF_CDSP.md) | Preserved source or evidence file |
| [README.md](README.md) | Original package overview |
| [RESULTS_CDSP.md](RESULTS_CDSP.md) | Preserved source or evidence file |
| [RESULTS_PCMD.md](RESULTS_PCMD.md) | Preserved source or evidence file |
| [analyze_cars196_pcds_sgds.py](analyze_cars196_pcds_sgds.py) | Preserved source or evidence file |
| [analyze_pcds_structure_sgds.py](analyze_pcds_structure_sgds.py) | Preserved source or evidence file |
| [cars196_static_check.py](cars196_static_check.py) | Preserved source or evidence file |
| [cds.yaml](cds.yaml) | Preserved source or evidence file |
| [data](data) | Original subdirectory; consult its saved code, configs or evidence |
| [exps_jsons](exps_jsons) | Experiment settings and protocol definitions |
| [logs](logs) | Saved analysis or execution evidence; completion is record-specific |
| [main.py](main.py) | Preserved source or evidence file |
| [models](models) | Method implementation |
| [network](network) | Original subdirectory; consult its saved code, configs or evidence |
| [results](results) | Saved analysis or execution evidence; completion is record-specific |
| [scripts](scripts) | Launch, analysis or validation commands |
| [sgds_static_check.py](sgds_static_check.py) | Preserved source or evidence file |
| [tests](tests) | Existing checks |
| [trainer.py](trainer.py) | Preserved source or evidence file |
| [utils](utils) | Method implementation |

## Category navigation

[Full categorized repository map](../docs/CATEGORIES.md)
