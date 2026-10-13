# sanity_cifar100_2task_controlled: four-condition result

All accuracies are single-head class-incremental accuracies. Values are percentage points except drift, update geometry, and memory.

| method | final acc | avg incremental acc | final old acc | final new acc | forgetting | old prototype drift | mean update norm | mean protected ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_lora | 70.05 | 83.02 | 42.60 | 97.50 | 53.40 | 0.34907 | 2.6410e-01 | 2.8395e-01 |
| small_update_lora | 92.00 | 93.85 | 89.00 | 95.00 | 6.70 | 0.10154 | 6.6025e-02 | 2.9066e-01 |
| orthogonal_lora | 79.65 | 87.83 | 62.30 | 97.00 | 33.70 | 0.07517 | 2.6451e-01 | 2.8191e-08 |
| orthogonal_lora_norm_matched | 92.75 | 94.23 | 91.10 | 94.40 | 4.60 | 0.02274 | 6.6025e-02 | 2.8844e-08 |

## Controlled contrasts

- Direction at matched small magnitude: orthogonal norm-matched minus small-update = +0.75 points final accuracy.
- Magnitude with unconstrained direction: small-update minus vanilla = +21.95 points final accuracy.
- Natural combined effect: orthogonal minus vanilla = +9.60 points final accuracy.

## Integrity diagnostics

- `vanilla_lora`: max ||A Q||=2.829e+00, max protected update ratio=5.746e-01, norm-match saturated=0/8, peak CUDA allocated=3.56 GiB.
- `small_update_lora`: max ||A Q||=2.888e+00, max protected update ratio=5.868e-01, norm-match saturated=0/8, peak CUDA allocated=3.56 GiB.
- `orthogonal_lora`: max ||A Q||=2.643e-07, max protected update ratio=5.648e-08, norm-match saturated=0/8, peak CUDA allocated=3.56 GiB.
- `orthogonal_lora_norm_matched`: max ||A Q||=2.557e-07, max protected update ratio=5.777e-08, norm-match saturated=0/8, peak CUDA allocated=3.56 GiB.
