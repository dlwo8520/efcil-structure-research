# Coverage + Generalization Diagnostic Audit

- Seed: `1993`
- Requested physical GPU: `4`; process-local device: `cuda:0`
- Protocol order: `['IA10', 'CAR10']`
- Gaussian draws: `3`; support k: `5`
- Bank cap: `none`
- Policy: frozen forward and offline diagnostic only; no Structure checkpoint is written.

## Pretrained initialization source

`pretrained_vit_b16_224_in21k_adapter` initializes the ViT backbone from the cached timm ImageNet-21K weights; newly created adapters have zero output projections.

| Cached source | Bytes | SHA-256 |
|---|---:|---|
| `${RESEARCH_WORKSPACE}/.cache/torch/hub/checkpoints/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz` | 410448610 | `a157033e309d33e79cbff2408dd77b4458014fcedc46f0ad853e2d08ecb493f6` |
| `${RESEARCH_WORKSPACE}/.cache/huggingface/hub/models--timm--vit_base_patch16_224.augreg_in21k/blobs/2d31bb5597c720912f8ea5e4f1de73909a6911798d78ae5d1b9d63a955fed049` | 410397786 | `2d31bb5597c720912f8ea5e4f1de73909a6911798d78ae5d1b9d63a955fed049` |

## IA10

- Final task/classes: `T10` / `200` (`180` old + `20` new)
- Final checkpoint: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/task_checkpoints/task_10.pt`
- Checkpoint fingerprint: size `1367010168`, edge SHA-256 `f12303d30fca90bb1cce99636a6e3451286c2dbf7098457c8caba8f3c3d3281a`
- Head rows / stored means / stored covariances: `200` / `200` / `200`
- Actual train samples per class min/median/max: `2/21.0/86`
- Test samples per class min/median/max: `1/5.0/26`
- Canonical train/test IDs agree with reused final cache: `True`

## CAR10

- Final task/classes: `T10` / `196` (`177` old + `19` new)
- Final checkpoint: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/task_checkpoints/task_10.pt`
- Checkpoint fingerprint: size `1345643768`, edge SHA-256 `f9abc1713f56b205d9d6d08a714e904ffe02b77298b025848b25bf9f275d2bb3`
- Head rows / stored means / stored covariances: `196` / `196` / `196`
- Actual train samples per class min/median/max: `24/42.0/68`
- Test samples per class min/median/max: `24/42.0/68`
- Canonical train/test IDs agree with reused final cache: `True`

## Metric-definition checks

- Classification margin is `true-class cosine score - strongest rival cosine score`; its negative fraction is treated as error-equivalent, not independent representation evidence.
- Support uses L2-normalized query/reference features. Gaussian samples are generated in raw feature space and normalized only before support computation.
- Support k is class-adaptive: `min(k, available class references)`; no duplicate or replacement references are introduced.
- The shared support implementation excludes every reference with the same canonical sample ID when reference/query sets overlap.
- Existing pair normalized entropy uses the nonzero directional-pair probability vector and divides Shannon entropy by `log(number of nonzero pairs)`.
- Existing stored-vs-oracle gamma analysis reports ranking correlation and target-specific Recall@K/captured mass; geometric agreement is not treated as predictive success.

## Missing inputs

MISSING INPUTS: none
