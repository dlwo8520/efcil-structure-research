# Restoration, indexing and validation utilities

**Category:** Research utilities

**Repository path:** `tools/`

## Purpose

Supports restoration of compressed evidence, regeneration of indexes and verification of preserved exports. CIL additionally provides experiment registration, path materialization, intentional source-update recording and official BOB-vendor restoration.

## Evidence and status

These utilities manage evidence; they do not automatically retrain models or establish GPU reproduction.

## Read first

- [tools/restore_tables.py](restore_tables.py)
- [tools/rebuild_indexes.py](rebuild_indexes.py)
- [tools/validate_export.py](validate_export.py)
- [docs/REPRODUCIBILITY.md](../docs/REPRODUCIBILITY.md)
- [tools/new_experiment.py](new_experiment.py)
- [tools/record_source_update.py](record_source_update.py)
- [tools/materialize_config.py](materialize_config.py)
- [tools/restore_bob_vendor.py](restore_bob_vendor.py)
- [docs/WORKFLOW.md](../docs/WORKFLOW.md)

## Contents

| Entry | Role |
|---|---|
| [materialize_config.py](materialize_config.py) | Preserved source or evidence file |
| [new_experiment.py](new_experiment.py) | Preserved source or evidence file |
| [rebuild_indexes.py](rebuild_indexes.py) | Preserved source or evidence file |
| [record_source_update.py](record_source_update.py) | Preserved source or evidence file |
| [restore_bob_vendor.py](restore_bob_vendor.py) | Preserved source or evidence file |
| [restore_tables.py](restore_tables.py) | Preserved source or evidence file |
| [validate_export.py](validate_export.py) | Preserved source or evidence file |

## Category navigation

[Full categorized repository map](../docs/CATEGORIES.md)
