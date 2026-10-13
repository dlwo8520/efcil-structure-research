# Implementation audit

## Read-only dependency chain

- `structure_multicenter_separation`: trainer, Structure network, AM separation cache, data API
- `structure_e3_followup`: APR-corrected prototypes and AM-Drift tau refresh
- `structure_similarity_aware_followup`: frozen previous-model old/new similarity and common diagnostics
- `structure_selective_similarity_sweep`: all code added for this study

No file in the three dependency trees was changed.

## Protocol findings and corrections

The base ImageNet-A loader already accepts `data_root`. Its Cars-196 loader has a placeholder
absolute path and a 200-class order, so it is not safe for Cars-196. The base generic remainder
logic would also make a 20-class incremental Cars protocol end with 16 classes rather than the
specified `[20,20,20,20,20,20,19,19,19,19]`.

`SweepDataManager` therefore loads Cars-196 with `ImageFolder`, validates the train/test class
mapping, and enforces the exact four partitions. IA5/IA10 use the same run seed for their shared
order. An existing Cars-196 study already fixed its order with `class_order_seed=1993`; CAR5 and
CAR10 reuse that exact `RandomState(1993)` order as required by the specification. Manifests
verify no missing or duplicate class and save both class-order and partition hashes.

## Loss-scale decision

The experiment specification displays raw hinges `[tau-D2]+`, while the inherited AM-Drift code
uses the dimensionless max-relative-intrusion hinge `[1-D2/tau]+`. Applying Method A/B with a raw
hinge while retaining `lambda_sep=0.3` would change the optimization scale and invalidate the
controlled comparison. A/B therefore average the selected dimensionless pair hinges. The raw
hinge is also logged in `selective_pair_samples.csv` so the displayed formula can still be
inspected post hoc. C0 calls the inherited `legacy_absolute_relation` unchanged.

## CLAD attribution boundary

FROM CLAD CONCEPT:

- per-new-class Top-P conflict selection
- class-aware similarity
- disentanglement objective

ADAPTED FOR EXEMPLAR-FREE STRUCTURE:

- exemplar to APR-corrected old prototype
- Selective AM-Drift
- similarity-enlarged tau
- Proto-CLAD

Method C does not add AM-Drift loss. The inherited separation cache is constructed only because
the parent training loop uses it as the lifecycle hook for refreshed APR prototypes and common
geometry diagnostics; `sep_lambda` carries `lambda_clad` in that loop.

## Diagnostic storage

The pinned `cds` environment has neither `pyarrow` nor `fastparquet`. Required large logical
tables are therefore emitted as CSV with the same base names and schemas, and every run records
the choice in `diagnostic_storage.json`. No CSV payload is mislabeled as a Parquet file.

## Runtime estimates

The queue reads the prior IA5 C0 training time (55.5 minutes) and prior Cars-196 10-task Structure
elapsed time (203.6 minutes). IA10 is planned as 1.70× IA5 and CAR5 as 0.58× CAR10. These are
planning estimates, not measurements of the new methods. A sequential Stage 1 is approximately
212 GPU-hours (8.8 days), before failures or system contention.
