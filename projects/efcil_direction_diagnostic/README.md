# EFCIL update-direction diagnostic

This repository implements the controlled experiment requested in the attached
research brief. It intentionally targets a narrow setting:

- CIFAR-100, ten disjoint tasks with ten classes per task
- exemplar-free training and no stored image paths
- a single expanding cosine classifier at inference
- one sequential ViT-B/16-IN21K model
- Query/Value LoRA deltas merged after every task
- physical GPU 1 only (`CUDA_VISIBLE_DEVICES=1`)
- the existing `/data/datasets/cifar100` class-directory copy; dataset download
  is disabled

The four methods share the same gradient-informed fixed-input-basis LoRA
parameterization. This removes the factorization confound that would arise if
vanilla trained both LoRA factors while the orthogonal condition froze one.

1. `vanilla_lora`: unconstrained input basis.
2. `small_update_lora`: vanilla direction, actual AdamW delta norm constrained
   to 25% of the saved `vanilla_lora` step norm.
3. `orthogonal_lora`: input basis selected from the complement of the stored
   old-class activation subspace.
4. `orthogonal_lora_norm_matched`: orthogonal direction, actual delta norm
   matched to the saved `small_update_lora` norm. This is the direct
   where-versus-how-much control.

For a PyTorch linear weight `W` with LoRA delta `B @ A`, protected input basis
`Q`, and `P = Q @ Q.T`, the projected full-weight gradient is

```text
G_perp = G @ (I - P) = G - (G @ Q) @ Q.T.
```

`A` is initialized from the leading right singular vectors of `G_perp` and is
frozen. Thus `A @ Q = 0`, and every trained effective update obeys
`delta_W @ Q = delta_B @ A @ Q = 0`.

The logged protected-component ratio is
`||delta_W @ Q @ Q.T||_F / ||delta_W||_F`; its residual is
`delta_W - delta_W @ Q @ Q.T`.

Mixed-precision training uses a conservative fixed initial loss scale. A
condition-dependent optimizer skip would invalidate exact norm matching, so a
detected scaler skip fails the run instead of silently advancing the schedule.

Only class means, singular directions/values, checkpoints, and evaluation
statistics are retained. Raw training images and replay samples are never
written to experiment outputs.

## KeepLoRA single-head EFCIL adaptation

The KeepLoRA experiment is implemented separately from the direction
diagnostic. It uses vision Q/K/V/O sites, paper-oriented first-gradient SVD
initialization, a frozen down-projection direction, B-only optimization,
task-end merge, pretrained-weight `Wp`, and residualized current-task feature
space `M`. The four parameter-matched conditions are `vanilla_lora`,
`keep_lora_wp_only`, `keep_lora_m_only`, and `keep_lora_full`.

Run the ten-task short validation before the normal ten-epoch experiment:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_keeplora_sanity_all.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/run_keeplora_full_all.sh
```

KeepLoRA outputs are written below `outputs/keeplora_efcil/`, so the existing
orthogonal diagnostic outputs are not modified.

## PCDS CIFAR-100 adaptation

The PCDS path implements paper Eq. (4)-(9) for Adaptive Prototype
Rectification and Structural Consistency, followed by balanced Gaussian
feature sampling (Eq. 11) to adapt only the expanding cosine classifier. It
uses the released CDS FFN-output bottleneck adapter and official default
hyperparameters, but retains this repository's CIFAR-100 class order,
ViT-B/16-IN21K checkpoint, single-head evaluation, and ten-epoch budget. Only
current-task images and stored class means/covariances are used during
training; old raw images are never replayed.

The primary script runs the protocol audit, the one requested `pcds_full`
condition, and then extends the completed KeepLoRA `RESULTS.md` with a combined
comparison. Run it only when physical GPU 1 is available:

```bash
bash scripts/run_pcds_full.sh
```

If a run stopped after a saved task boundary, resume it with:

```bash
bash scripts/resume_pcds_full.sh
```

The code also supports `baseline`, `apr_only`, `scc_only`, and `apr_scc`
through `train_pcds.py --method`, but the primary script intentionally runs
only Full PCDS.

## Commands

Run from this directory:

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n clip python train.py \
  --config configs/sanity.yaml --method vanilla_lora
```

The scripts under `scripts/` run the four conditions in the dependency order
needed by norm matching.

## Task-matrix analysis

`analyze.py` converts the saved lower-triangular single-head task accuracies
into `comparison/task_accuracy_matrix.csv`, `summary_metrics.csv`, four masked
heatmaps, and the corresponding sections in `RESULTS.md`. It performs no model
evaluation and does not require a GPU.

KeepLoRA's MTIL Transfer is the mean accuracy on a task before that task is
learned. It is reported as `N/A` here because an expanding CIL classifier has
no output rows for unseen future classes. The reported `Average` is therefore
the CIL Average Incremental Accuracy (the mean of stage-wise Seen Avg), not
KeepLoRA's full-matrix per-task Average. `Last` is final single-head accuracy.

```bash
conda run --no-capture-output -n clip python analyze.py \
  --config configs/cifar100_10task.yaml
```
