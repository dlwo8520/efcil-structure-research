# Automatic sanity — exploratory pilot

Exploratory pilot without human QC. Generated labels are inherited from parents, not verified. Automatic similarity/structure checks cannot guarantee CAR model/year identity or complete object retention. No individual human choices or classifier confidence are used. Fixed-rule filtering may still introduce distribution bias. Downstream G-S evaluates this fixed generator + automatic-sanity recipe, not all generative augmentation.

Status: READY_EXPLORATORY

All candidates are assessed by the same frozen rules. Human review file and contact sheets do not affect selection. Excluded images remain on disk as artifacts; only their training eligibility changes. R/S/G share the same accepted parent set.

| Protocol | Generated candidates | Accepted | Excluded | Reasons (non-exclusive) |
|---|---:|---:|---:|---|
| IA10 | 568 | 560 | 8 | {'generation_failed_or_safety_blocked': 8, 'collapsed_image': 8} |
| CAR10 | 624 | 622 | 2 | {'generation_failed_or_safety_blocked': 2, 'collapsed_image': 2} |

## Pair similarity statistics

Statistics include safety-blocked/black images in the all-decodable subset. High CLIP cosine does not certify fine-grained identity. Zero flags for a rule mean no threshold crossing, not that all objects are preserved.

| Protocol | Subset | N | Global cosine mean | Global q05 | Center cosine mean | Block SSIM mean | Pixel MAE mean |
|---|---|---:|---:|---:|---:|---:|---:|
| IA10 | all_decodable | 568 | 0.81683 | 0.65578 | 0.81884 | 0.68967 | 0.05205 |
| IA10 | accepted | 560 | 0.82159 | 0.67428 | 0.82377 | 0.69950 | 0.04622 |
| IA10 | excluded_decodable | 8 | 0.48331 | 0.38242 | 0.47334 | 0.00181 | 0.46018 |
| CAR10 | all_decodable | 624 | 0.88877 | 0.79639 | 0.90587 | 0.78079 | 0.03951 |
| CAR10 | accepted | 622 | 0.88981 | 0.79704 | 0.90678 | 0.78313 | 0.03817 |
| CAR10 | excluded_decodable | 2 | 0.56522 | 0.50876 | 0.62203 | 0.05401 | 0.45705 |

Thresholds are exploratory heuristics fixed before downstream outcomes, not calibrated label/segmentation guarantees. A conservative global+center CLIP mismatch or center edge collapse may flag gross changes; subtler object deletion/identity changes can pass. Block SSIM is a custom non-overlapping 8x8 RGB score, not standard Gaussian-window SSIM.

[Rules](sanity_policy.json) · [Per-image measurements](automatic_sanity.csv) · [Similarity statistics](automatic_sanity_summary.json) · [Class counts](automatic_parent_counts.csv) · [Contact sheets (artifact only)](contact_sheets/)
