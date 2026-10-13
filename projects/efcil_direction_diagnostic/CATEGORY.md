# Early CIFAR / KeepLoRA-style / PCDS diagnostics

**Category:** Baselines and early diagnostics

**Repository path:** `projects/efcil_direction_diagnostic/`

## Purpose

Diagnose representation and classifier limitations.

## Evidence and status

implemented; baseline-specific outputs. Interpret this state using the linked analysis and saved run records.

## Method identity

- Experiment ID: `EFCIL-001`
- Question: Diagnose representation and classifier limitations.
- Origin: custom reproduction/adaptation.
- Baseline: package-specific Structure/CDS or explicitly recorded control.
- Limitations: single-seed and protocol-specific evidence; diagnostic boundaries in report.

## Read first

- [projects/efcil_direction_diagnostic/README.md](README.md)
- [experiments/EFCIL-001.md](../../experiments/EFCIL-001.md)
- [projects/efcil_direction_diagnostic/train_keeplora.py](train_keeplora.py)
- [projects/efcil_direction_diagnostic/analyze.py](analyze.py)
- [projects/efcil_direction_diagnostic/train.py](train.py)
- [projects/efcil_direction_diagnostic/configs/pcds_efcil_cifar100_10task.yaml](configs/pcds_efcil_cifar100_10task.yaml)
- [projects/efcil_direction_diagnostic/configs/keeplora_efcil_sanity.yaml](configs/keeplora_efcil_sanity.yaml)
- [projects/efcil_direction_diagnostic/configs/cifar100_10task.yaml](configs/cifar100_10task.yaml)

## Contents

| Entry | Role |
|---|---|
| [.gitignore](.gitignore) | Preserved source or evidence file |
| [PCDS_CODE_README.md](PCDS_CODE_README.md) | Preserved source or evidence file |
| [README.md](README.md) | Original package overview |
| [analyze.py](analyze.py) | Preserved source or evidence file |
| [analyze_keeplora.py](analyze_keeplora.py) | Preserved source or evidence file |
| [analyze_pcds.py](analyze_pcds.py) | Preserved source or evidence file |
| [configs](configs) | Experiment settings and protocol definitions |
| [efcil_diag](efcil_diag) | Original subdirectory; consult its saved code, configs or evidence |
| [outputs](outputs) | Saved analysis or execution evidence; completion is record-specific |
| [pcds_static_check.py](pcds_static_check.py) | Preserved source or evidence file |
| [requirements.txt](requirements.txt) | Preserved source or evidence file |
| [scripts](scripts) | Launch, analysis or validation commands |
| [tests](tests) | Existing checks |
| [train.py](train.py) | Preserved source or evidence file |
| [train_keeplora.py](train_keeplora.py) | Preserved source or evidence file |
| [train_pcds.py](train_pcds.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../../docs/CATEGORIES.md)
