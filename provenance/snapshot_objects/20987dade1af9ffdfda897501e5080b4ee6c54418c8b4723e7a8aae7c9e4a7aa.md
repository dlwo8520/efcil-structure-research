# Reference and implementation audit

Audit date: 2026-09-16. The base repository was only read; it was not modified.
The commit hashes below identify the official repositories inspected before the
follow-up implementation.

## A. Reference code audit

| Method | Official source / audited commit | File and symbol | Original mechanism | Use here |
|---|---|---|---|---|
| Structure SCC | [lambor9973/cds](https://github.com/lambor9973/cds), `a0db2168663982a4cad24981231721a3518ddf29` | `models/CDS_adapter.py`, `LobachevskyRelativeGeomLoss` call; `utils/distill_loss.py` | Maintains class-discriminative relative geometry during incremental feature training. | Kept through the base learner without changing its equation or coefficient. |
| Structure APR | same | `utils/adjust_distribution.py`, `LobachevskyPrototypeCorrector.correct_prototypes` and `class_level_corrections` | Transfers old mean/covariance statistics into the current feature space using current-task before/after statistics. | Supplies epoch APR-corrected old centers/covariances to all follow-up objectives. |
| Structure replay | same | `models/base.py`, `BaseLearner._stage2_compact_classifier`; `utils/distribution_regions.py`, `sample_gaussian_mixture` | Samples synthetic features from stored class Gaussian statistics and retrains only the classifier. | Baseline preserved. CEOS changes only the old-class sample composition while holding class total fixed. |
| FCS PIC | [zhoujiahuan1991/CVPR2024-FCS](https://github.com/zhoujiahuan1991/CVPR2024-FCS), `41a4175b039bf44e06ed9e3665191285fe730606` | `models/fcs.py`, `SupContrastive` and `_compute_il2a_loss` | Normalized query/key views and same-label current samples are positives. Calibrated old prototypes are appended to the contrastive denominator as negatives. Official temperature is `0.07`. `SupContrastive` implements `log(1 + sum_neg exp(l_neg-l_pos))`. | **ADAPTED FOR STRUCTURE:** one-view Structure batches use the correct current epoch prototype as a guaranteed positive plus other same-label batch features. APR-corrected old centers are negatives. FCS calibration and the rest of FCS are not imported. |
| LUCIR cosine classifier | [hshustc/CVPR19_Incremental_Learning](https://github.com/hshustc/CVPR19_Incremental_Learning), `e5a90aed7640f3b00e5b1a8dfb5376c1628bfe6a` | `imagenet-class-incremental/modified_linear.py`, `CosineLinear` | L2-normalizes features and classifier weights before their dot product. | PIC/AHN geometry is computed with normalized features/prototypes. Structure's existing cosine classifier remains unchanged. |
| LUCIR hard-negative margin | same | `utils_incremental/incremental_train_and_eval_MR_LF.py`, margin-ranking block | For a reserved old sample, its ground-truth old score must exceed top-K new-class scores. Official ImageNet defaults are `K=2`, margin `0.5`, weight `1`. | **ADAPTED FOR STRUCTURE:** direction is reversed for a current new sample: correct current prototype must exceed the top-K old APR prototypes. It is never labeled as the original LUCIR objective. |
| CEOS nearest enemy and sampling | [HXuSz11/ACB_CEOS_CVPR2026_Findings](https://github.com/HXuSz11/ACB_CEOS_CVPR2026_Findings), `b1d7743b17729438ff8e81d98b6046fcb1f516f0` | `continual_learning/CEOS.py`, `expansive_oversampling` | Uses `cdist` to choose nearest current-feature enemies (`eos_k=1` default), samples `lambda=0.7+0.3U`, interpolates `lambda*old_prototype+(1-lambda)*enemy`, then L2-normalizes. Since lambda is above 0.5, the raw interpolant is closer to the old prototype than the enemy. | **ADAPTED FOR STRUCTURE:** starts at the APR-corrected old mean; because Structure's classifier is cosine-normalized, both endpoints are first put on the unit sphere. CEOS50/100 replace 50/100% of each old class's Gaussian count. Current classes stay Gaussian and total replay count is unchanged. ACB is excluded. |
| AMGC | [paper](https://arxiv.org/abs/2409.13275); no verified official repository | Paper equations for DBGC, variance enlarging and AMarX | AMGC enlarges covariance and derives a class-specific classifier margin including a variance term. It does not define E3's tau recalibration. | **AMGC-inspired extension only:** variance, age and APR-drift tau rules are separate fixed formulas using stored/current statistics, never old raw data. |

Primary papers: [Structure (CVPR 2026)](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Exemplar-Free_Class_Incremental_Learning_via_Preserving_Class-Discriminative_Structure_CVPR_2026_paper.html), [FCS (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Li_FCS_Feature_Calibration_and_Separation_for_Non-Exemplar_Class_Incremental_Learning_CVPR_2024_paper.html), [LUCIR (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Hou_Learning_a_Unified_Classifier_Incrementally_via_Rebalancing_CVPR_2019_paper.html), [CEOS (arXiv)](https://arxiv.org/abs/2606.05695), and [AMGC (arXiv)](https://arxiv.org/abs/2409.13275).

## B. Current E3 audit

Audited local base: `${RESEARCH_WORKSPACE}/structure_multicenter_separation` at the same
official Structure commit, with local experiment/logging changes. Those local changes are
preserved and not overwritten.

| Item | Exact base location | Observed behavior |
|---|---|---|
| Diagonal covariance | `utils/distribution_regions.py`, `SeparationCache.__init__` | Extracts each stored covariance diagonal and clamps it by `sep_cov_eigen_floor`. |
| Birth tau | `utils/distribution_regions.py`, `build_single_region` / `build_regions_for_class` and `tau_by_covariance` | Stores the birth-task percentile boundary per covariance view; the epoch cache copies it unchanged. |
| APR epoch cache | `models/CDS_adapter.py`, `Learner._refresh_separation_cache` | Collects only current-task features, computes current statistics, APR-corrects stored old statistics and constructs detached old regions in the current feature space. |
| D squared | `utils/distribution_regions.py`, `SeparationCache.loss` | For diagonal view: `sum_j (z_j-mu_kj)^2 / variance_kj`. |
| Intrusion | same | Finds minimum relative distance `rho=D squared/tau`; intrusion is `rho < 1`. |
| Separation loss | same | `mean relu(1-min_k rho_k)`, i.e. max relative intrusion. This is a dimensionless relative penalty, not the unnormalized `relu(tau-D squared)` expression. |
| Feature normalization | same and `models/CDS_adapter.py::_feature_space_id` | No normalization; raw 768-D student feature is used. |
| Old-data compliance | `models/CDS_adapter.py::_collect_current_features` and `_conpute_now_task_old_class_mean` | The misleadingly named latter method passes current new-task data through the previous adapter. No old raw sample is used by the E3 cache. |

The new `diagonal_region_loss(normalize=False)` has an explicit smoke assertion that its
scalar loss matches the base `SeparationCache.loss` on the same synthetic cache.

## C. Logging audit

| Area | Already in base | Added by this extension |
|---|---|---|
| Accuracy | task-final seen top-1/top-5, old/new accuracy, task accuracy matrix, class correct/total, Inc-Acc, Last-Acc, Forgetting | No duplicate evaluator; the base evaluator remains the source of truth. Later paired analysis must join identical seeds. |
| Loss | classification, SCC/feature/logit distillation, raw/weighted E3, total | PIC and AHN raw/weighted loss; normalized-diagonal flag; CEOS classifier CE. |
| Feature norm | mean only | mean, population std, p10/p25/p50/p75/p90/p95, max for every feature epoch. |
| E3 geometry | nearest D squared/rho mean and p05/p50/p95; intrusion mean | D squared, rho, selected tau and absolute `max(0,tau-D squared)` mean/std/p10/p25/p50/p75/p90/p95/max; nearest class; class-level intrusion/penalty records. `followup_task_end_diagnostics.jsonl` is the method-correct task-final source for E3-Norm/adaptive runs. |
| Cosine geometry | absent | nearest-old cosine, correct-current cosine, cosine gap, nearest-old Euclidean distance; PIC positive similarity and violations; AHN selected-class histogram. |
| Gradients | E3 norm on first batch only | configurable interval norms for classification/SCC/E3/PIC/AHN and every active pairwise cosine; raw batch rows plus epoch means. |
| Adaptive threshold | absent | birth tau, applied tau/scale, age, variance trace/mean/max, APR shift, selected count, intrusion and mean penalty by class/epoch. |
| CEOS | absent | per generated sample source/enemy, lambda, two distances, two cosines, dominance, old/enemy logits, gap, boundary proxy and old prediction; epoch summaries. |
| Resources/reproducibility | wall time, peak CUDA, statistics/cache memory, config and working-tree digest | source-only snapshots of base and extension, file hashes, config, command, git status/diff, Python/PyTorch/CUDA/GPU environment. |

## D. Implementation map

| Experiment | File / function | Modification | Reason | New logs |
|---|---|---|---|---|
| Lambda sweep | `followup/learner.py::_init_train`; `configs/L*.json` | Reuses exact raw diagonal objective at four coefficients. | Isolate strength from mechanism. | Expanded norms, intrusion and interval gradients. |
| PIC | `followup/losses.py::pic_loss`; `configs/FCS_pic.json` | Adds only the adapted PIC component and disables E3. | Point-prototype comparison without importing full FCS. | PIC loss, positive/old/current similarity, gap, violation. |
| AHN | `followup/losses.py::angular_hard_negative_loss`; `configs/E3_ahn.json` | Adds normalized top-2 old hard negatives to E3. | Test direct angular control and norm-evasion hypothesis. | AHN loss/violations/histogram and gradient conflicts. |
| E3-Norm | `followup/losses.py::diagonal_region_loss`; `configs/E3_norm.json` | Unit-sphere features/means and first-order diagonal covariance scaling. | Separate normalization from AHN; explicitly approximate. | Same E3 diagnostics plus normalized flag. |
| CEOS50/100 | `followup/ceos.py::sample_ceos`; `followup/stage2.py`; CEOS configs | Replaces a fixed fraction of old Gaussian samples only; preserves class total and excludes ACB. | Test boundary replay without pushing current features. | Per-sample and epoch CEOS diagnostics. |
| Adaptive tau | `followup/adaptive_margin.py::recalibrate_taus`; AM configs | Runs variance, age or drift formula independently. | Avoid an uninterpretable combined rule and avoid old exemplars. | Class/epoch tau inputs and outcomes. |
| Reproducibility | `followup/snapshot.py::create_run_snapshot` | Captures both source trees and run environment at learner creation. | Make dirty-tree runs reconstructable. | `source_snapshot/`, config, status, diff, environment, command. |

## Protocol safeguards

- Base files are imported, not patched or copied back.
- Follow-up cache uses the current task only plus stored/APR statistics.
- CEOS enemy candidates are current-task features only; no future task is visible.
- CEOS class totals equal Structure's `replay_samples_per_class` exactly.
- E3, PIC, AHN, CEOS and each adaptive rule have independent config flags; isolation
  constraints are also validated before training.
- No best-test epoch selection is added. The base fixed schedule and task-final evaluator remain.
- The report scaffold stays `pending` until all common seeds complete.
