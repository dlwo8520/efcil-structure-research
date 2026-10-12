# PCDS CIFAR-100 Single-Head EFCIL Code

This package contains the code used for the `PCDS CIFAR-100 adaptation` run.
It is not an official CIFAR-100 number from the PCDS paper.

## Implemented method

- Adaptive Prototype Rectification (APR), paper Eq. (4)-(8)
- Structural Consistency Constraint (SCC), paper Eq. (9)
- Balanced Gaussian synthetic-feature sampling, paper Eq. (11)
- Classifier-only adaptation using the sampled features
- One expanding cosine classifier over every seen class
- Current-task image training only; no old raw-image replay

The feature extractor uses the released CDS FFN-output nonlinear bottleneck
adapter on the same `vit_base_patch16_224.augreg_in21k` backbone used by the
local KeepLoRA comparison.

## Main files

- `efcil_diag/pcds.py`: APR, SCC, prototype mean/covariance memory and stable sampling
- `efcil_diag/pcds_model.py`: ViT FFN adapter and expanding classifier model
- `efcil_diag/pcds_experiment.py`: sequential training, CA, evaluation and checkpoints
- `train_pcds.py`: training CLI
- `pcds_static_check.py`: protocol/GPU/data-leakage audit
- `analyze_pcds.py`: PCDS metrics and KeepLoRA comparison report
- `configs/pcds_efcil_cifar100_10task.yaml`: primary seed-0 configuration
- `tests/test_pcds.py`: APR/SCC/device/leakage/statistics tests

## Run

Edit absolute paths in the YAML when the package is moved to another machine.
The static check also expects the completed KeepLoRA config, class order and
summary paths specified in the YAML.

```bash
bash scripts/run_pcds_full.sh
```

Resume only from a saved task-boundary checkpoint:

```bash
bash scripts/resume_pcds_full.sh
```

Run tests without starting an experiment:

```bash
conda run --no-capture-output -n clip python -m pytest -q tests/test_pcds.py
```

## Interpretation of the completed local run

The local Full PCDS run obtained 89.26% final accuracy, but this should not be
interpreted as APR/SCC alone. At task 9, accuracy was 65.60% before synthetic
classifier adaptation and 89.26% after it. The primary run therefore shows the
effect of the complete PCDS pipeline, including balanced Gaussian classifier
retraining. APR-only, SCC-only and APR+SCC are supported by the CLI but were
not run in the primary comparison.
