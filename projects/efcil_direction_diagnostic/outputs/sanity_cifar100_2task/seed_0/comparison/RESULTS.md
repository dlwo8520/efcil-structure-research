# sanity_cifar100_2task: four-condition result

All accuracies are single-head class-incremental accuracies. Values are percentage points except drift, update geometry, and memory.

| method | final acc | avg incremental acc | final old acc | final new acc | forgetting | old prototype drift | mean update norm | mean protected ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_lora | 75.00 | 85.60 | 53.00 | 97.00 | 43.20 | 0.23108 | 2.8229e-01 | 2.4062e-01 |
| small_update_lora | 82.10 | 89.15 | 66.60 | 97.60 | 29.60 | 0.14942 | 2.5646e-01 | 2.4078e-01 |
| orthogonal_lora | 87.40 | 91.80 | 78.60 | 96.20 | 17.60 | 0.03693 | 2.6279e-01 | 2.4561e-08 |
| orthogonal_lora_norm_matched | 87.85 | 92.03 | 79.50 | 96.20 | 16.70 | 0.02028 | 2.7555e-01 | 1.9024e-08 |

## Controlled contrasts

- Direction at matched vanilla magnitude: orthogonal norm-matched minus vanilla = +12.85 points final accuracy.
- Magnitude with unconstrained direction: small-update minus vanilla = +7.10 points final accuracy.
- Natural combined effect: orthogonal minus vanilla = +12.40 points final accuracy.

## Integrity diagnostics

- `vanilla_lora`: max ||A Q||=2.835e+00, max protected update ratio=5.655e-01, norm-match saturated=0/8, peak CUDA allocated=3.56 GiB.
- `small_update_lora`: max ||A Q||=2.835e+00, max protected update ratio=5.651e-01, norm-match saturated=2/8, peak CUDA allocated=3.56 GiB.
- `orthogonal_lora`: max ||A Q||=2.784e-07, max protected update ratio=5.740e-08, norm-match saturated=0/8, peak CUDA allocated=3.56 GiB.
- `orthogonal_lora_norm_matched`: max ||A Q||=2.784e-07, max protected update ratio=5.728e-08, norm-match saturated=1/8, peak CUDA allocated=3.56 GiB.
