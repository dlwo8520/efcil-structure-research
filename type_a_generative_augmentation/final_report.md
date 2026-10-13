# Type A current generative augmentation pilot

## Exploratory scope and label-fidelity limitation

Exploratory pilot without human QC. Generated labels are inherited from parents, not verified. Automatic similarity/structure checks cannot guarantee CAR model/year identity or complete object retention. No individual human choices or classifier confidence are used. Fixed-rule filtering may still introduce distribution bias. Downstream G-S evaluates this fixed generator + automatic-sanity recipe, not all generative augmentation.
Contact sheets are inspection artifacts only; no human per-image choices affect inclusion. All fixed-rule survivors are used in the common R/S/G parent bank. Promising pilot results recommend stricter label-preservation/human QC and additional seeds before any formal efficacy claim.

[Automatic sanity report](automatic_sanity_report.md) · [Fixed rules](sanity_policy.json) · [Similarity/counts](automatic_sanity_summary.json) · [Exploratory configuration](exploratory_config.json)

Current train parents only; fixed vanilla error subsets are diagnostic/evaluation labels. Primary comparison G−S, secondary G−R, statistics-only fork GM−G. T1 acquisition and T9→T10 suffix do not constitute full T1→T10 augmented training.

## Accuracy

| Protocol | Location/task/branch | N | Seen | Macro | New all-seen | New local | T9 cohort |
|---|---|---:|---:|---:|---:|---:|---:|
| IA10 | T1_T1_R | 175 | 85.143% | 73.034% | 85.143% | 85.143% | — |
| IA10 | T1_T1_S | 175 | 86.286% | 73.936% | 86.286% | 86.286% | — |
| IA10 | T1_T1_G | 175 | 87.429% | 77.611% | 87.429% | 87.429% | — |
| IA10 | T1_T1_GM | 175 | 87.429% | 77.611% | 87.429% | 87.429% | — |
| IA10 | T1_T1_V | 175 | 86.286% | 73.589% | 86.286% | — | — |
| IA10 | T9_T10_T9_R | 1395 | 65.018% | 54.680% | 57.143% | 82.993% | 57.143% |
| IA10 | T9_T10_T9_S | 1395 | 65.233% | 54.750% | 57.143% | 82.313% | 57.143% |
| IA10 | T9_T10_T9_G | 1395 | 65.090% | 55.626% | 57.143% | 82.993% | 57.143% |
| IA10 | T9_T10_T9_GM | 1395 | 64.803% | 55.410% | 57.143% | 83.673% | 57.143% |
| IA10 | T9_T10_T9_V | 1395 | 65.233% | 54.856% | 58.503% | — | 58.503% |
| IA10 | T9_T10_T10_R | 1519 | 64.582% | 53.427% | 66.129% | 85.484% | 57.823% |
| IA10 | T9_T10_T10_S | 1519 | 64.845% | 53.798% | 66.935% | 84.677% | 57.143% |
| IA10 | T9_T10_T10_G | 1519 | 64.582% | 53.777% | 66.129% | 85.484% | 57.143% |
| IA10 | T9_T10_T10_GM | 1519 | 64.845% | 53.649% | 66.129% | 85.484% | 55.782% |
| IA10 | T9_T10_T10_V | 1519 | 64.714% | 53.586% | 66.935% | — | 57.143% |
| CAR10 | T1_T1_R | 838 | 90.095% | 90.138% | 90.095% | 90.095% | — |
| CAR10 | T1_T1_S | 838 | 89.499% | 89.491% | 89.499% | 89.499% | — |
| CAR10 | T1_T1_G | 838 | 90.573% | 90.516% | 90.573% | 90.573% | — |
| CAR10 | T1_T1_GM | 838 | 90.573% | 90.516% | 90.573% | 90.573% | — |
| CAR10 | T1_T1_V | 838 | 90.215% | 90.235% | 90.215% | — | — |
| CAR10 | T9_T10_T9_R | 7262 | 79.358% | 79.441% | 81.662% | 94.327% | 81.662% |
| CAR10 | T9_T10_T9_S | 7262 | 79.510% | 79.594% | 81.926% | 94.063% | 81.926% |
| CAR10 | T9_T10_T9_G | 7262 | 79.303% | 79.392% | 81.398% | 93.404% | 81.398% |
| CAR10 | T9_T10_T9_GM | 7262 | 79.193% | 79.258% | 79.815% | 94.063% | 79.815% |
| CAR10 | T9_T10_T9_V | 7262 | 79.303% | 79.374% | 81.266% | — | 81.266% |
| CAR10 | T9_T10_T10_R | 8041 | 78.784% | 78.858% | 80.873% | 96.791% | 81.003% |
| CAR10 | T9_T10_T10_S | 8041 | 78.883% | 78.957% | 80.873% | 96.534% | 81.003% |
| CAR10 | T9_T10_T10_G | 8041 | 78.610% | 78.694% | 80.745% | 96.791% | 80.211% |
| CAR10 | T9_T10_T10_GM | 8041 | 78.510% | 78.585% | 80.745% | 96.791% | 78.628% |
| CAR10 | T9_T10_T10_V | 8041 | 78.958% | 79.059% | 81.515% | — | 82.190% |

## Paired comparisons

| Protocol | Comparison | Subset | N | Repair | Harm | Net | ΔAcc pp |
|---|---|---|---:|---:|---:|---:|---:|
| IA10 | T1_T1_G-S | overall | 175 | 5 | 3 | 2 | +1.143 |
| IA10 | T1_T1_G-S | new | 175 | 5 | 3 | 2 | +1.143 |
| IA10 | T1_T1_G-S | A | 19 | 2 | 0 | 2 | +10.526 |
| IA10 | T1_T1_G-S | B1 | 42 | 0 | 3 | -3 | -7.143 |
| IA10 | T1_T1_G-S | B2 | 5 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_G-S | C | 4 | 2 | 0 | 2 | +50.000 |
| IA10 | T1_T1_G-S | final_correct | 105 | 1 | 0 | 1 | +0.952 |
| IA10 | T1_T1_G-S | B | 47 | 0 | 3 | -3 | -6.383 |
| IA10 | T1_T1_G-R | overall | 175 | 6 | 2 | 4 | +2.286 |
| IA10 | T1_T1_G-R | new | 175 | 6 | 2 | 4 | +2.286 |
| IA10 | T1_T1_G-R | A | 19 | 2 | 0 | 2 | +10.526 |
| IA10 | T1_T1_G-R | B1 | 42 | 0 | 2 | -2 | -4.762 |
| IA10 | T1_T1_G-R | B2 | 5 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_G-R | C | 4 | 2 | 0 | 2 | +50.000 |
| IA10 | T1_T1_G-R | final_correct | 105 | 2 | 0 | 2 | +1.905 |
| IA10 | T1_T1_G-R | B | 47 | 0 | 2 | -2 | -4.255 |
| IA10 | T1_T1_GM-G | overall | 175 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_GM-G | new | 175 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_GM-G | A | 19 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_GM-G | B1 | 42 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_GM-G | B2 | 5 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_GM-G | C | 4 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_GM-G | final_correct | 105 | 0 | 0 | 0 | +0.000 |
| IA10 | T1_T1_GM-G | B | 47 | 0 | 0 | 0 | +0.000 |
| IA10 | T9_T10_T9_G-S | overall | 1395 | 16 | 18 | -2 | -0.143 |
| IA10 | T9_T10_T9_G-S | new | 147 | 2 | 2 | 0 | +0.000 |
| IA10 | T9_T10_T9_G-S | old | 1248 | 14 | 16 | -2 | -0.160 |
| IA10 | T9_T10_T9_G-S | T9 | 147 | 2 | 2 | 0 | +0.000 |
| IA10 | T9_T10_T9_G-S | A | 327 | 3 | 1 | 2 | +0.612 |
| IA10 | T9_T10_T9_G-S | B1 | 126 | 6 | 0 | 6 | +4.762 |
| IA10 | T9_T10_T9_G-S | B2 | 29 | 2 | 1 | 1 | +3.448 |
| IA10 | T9_T10_T9_G-S | C | 13 | 1 | 1 | 0 | +0.000 |
| IA10 | T9_T10_T9_G-S | final_correct | 900 | 4 | 15 | -11 | -1.222 |
| IA10 | T9_T10_T9_G-S | B | 155 | 8 | 1 | 7 | +4.516 |
| IA10 | T9_T10_T9_G-R | overall | 1395 | 15 | 14 | 1 | +0.072 |
| IA10 | T9_T10_T9_G-R | new | 147 | 2 | 2 | 0 | +0.000 |
| IA10 | T9_T10_T9_G-R | old | 1248 | 13 | 12 | 1 | +0.080 |
| IA10 | T9_T10_T9_G-R | T9 | 147 | 2 | 2 | 0 | +0.000 |
| IA10 | T9_T10_T9_G-R | A | 327 | 3 | 0 | 3 | +0.917 |
| IA10 | T9_T10_T9_G-R | B1 | 126 | 5 | 0 | 5 | +3.968 |
| IA10 | T9_T10_T9_G-R | B2 | 29 | 2 | 1 | 1 | +3.448 |
| IA10 | T9_T10_T9_G-R | C | 13 | 1 | 0 | 1 | +7.692 |
| IA10 | T9_T10_T9_G-R | final_correct | 900 | 4 | 13 | -9 | -1.000 |
| IA10 | T9_T10_T9_G-R | B | 155 | 7 | 1 | 6 | +3.871 |
| IA10 | T9_T10_T9_GM-G | overall | 1395 | 4 | 8 | -4 | -0.287 |
| IA10 | T9_T10_T9_GM-G | new | 147 | 1 | 1 | 0 | +0.000 |
| IA10 | T9_T10_T9_GM-G | old | 1248 | 3 | 7 | -4 | -0.321 |
| IA10 | T9_T10_T9_GM-G | T9 | 147 | 1 | 1 | 0 | +0.000 |
| IA10 | T9_T10_T9_GM-G | A | 327 | 1 | 1 | 0 | +0.000 |
| IA10 | T9_T10_T9_GM-G | B1 | 126 | 2 | 4 | -2 | -1.587 |
| IA10 | T9_T10_T9_GM-G | B2 | 29 | 0 | 0 | 0 | +0.000 |
| IA10 | T9_T10_T9_GM-G | C | 13 | 0 | 0 | 0 | +0.000 |
| IA10 | T9_T10_T9_GM-G | final_correct | 900 | 1 | 3 | -2 | -0.222 |
| IA10 | T9_T10_T9_GM-G | B | 155 | 2 | 4 | -2 | -1.290 |
| IA10 | T9_T10_T10_G-S | overall | 1519 | 7 | 11 | -4 | -0.263 |
| IA10 | T9_T10_T10_G-S | new | 124 | 0 | 1 | -1 | -0.806 |
| IA10 | T9_T10_T10_G-S | old | 1395 | 7 | 10 | -3 | -0.215 |
| IA10 | T9_T10_T10_G-S | T9 | 147 | 1 | 1 | 0 | +0.000 |
| IA10 | T9_T10_T10_G-S | A | 327 | 1 | 1 | 0 | +0.000 |
| IA10 | T9_T10_T10_G-S | B1 | 126 | 1 | 0 | 1 | +0.794 |
| IA10 | T9_T10_T10_G-S | B2 | 29 | 0 | 1 | -1 | -3.448 |
| IA10 | T9_T10_T10_G-S | C | 13 | 2 | 1 | 1 | +7.692 |
| IA10 | T9_T10_T10_G-S | final_correct | 900 | 3 | 7 | -4 | -0.444 |
| IA10 | T9_T10_T10_G-S | B | 155 | 1 | 1 | 0 | +0.000 |
| IA10 | T9_T10_T10_G-R | overall | 1519 | 11 | 11 | 0 | +0.000 |
| IA10 | T9_T10_T10_G-R | new | 124 | 0 | 0 | 0 | +0.000 |
| IA10 | T9_T10_T10_G-R | old | 1395 | 11 | 11 | 0 | +0.000 |
| IA10 | T9_T10_T10_G-R | T9 | 147 | 0 | 1 | -1 | -0.680 |
| IA10 | T9_T10_T10_G-R | A | 327 | 1 | 1 | 0 | +0.000 |
| IA10 | T9_T10_T10_G-R | B1 | 126 | 2 | 2 | 0 | +0.000 |
| IA10 | T9_T10_T10_G-R | B2 | 29 | 0 | 1 | -1 | -3.448 |
| IA10 | T9_T10_T10_G-R | C | 13 | 3 | 0 | 3 | +23.077 |
| IA10 | T9_T10_T10_G-R | final_correct | 900 | 5 | 7 | -2 | -0.222 |
| IA10 | T9_T10_T10_G-R | B | 155 | 2 | 3 | -1 | -0.645 |
| IA10 | T9_T10_T10_GM-G | overall | 1519 | 9 | 5 | 4 | +0.263 |
| IA10 | T9_T10_T10_GM-G | new | 124 | 0 | 0 | 0 | +0.000 |
| IA10 | T9_T10_T10_GM-G | old | 1395 | 9 | 5 | 4 | +0.287 |
| IA10 | T9_T10_T10_GM-G | T9 | 147 | 2 | 4 | -2 | -1.361 |
| IA10 | T9_T10_T10_GM-G | A | 327 | 2 | 0 | 2 | +0.612 |
| IA10 | T9_T10_T10_GM-G | B1 | 126 | 2 | 0 | 2 | +1.587 |
| IA10 | T9_T10_T10_GM-G | B2 | 29 | 0 | 0 | 0 | +0.000 |
| IA10 | T9_T10_T10_GM-G | C | 13 | 0 | 1 | -1 | -7.692 |
| IA10 | T9_T10_T10_GM-G | final_correct | 900 | 5 | 4 | 1 | +0.111 |
| IA10 | T9_T10_T10_GM-G | B | 155 | 2 | 0 | 2 | +1.290 |
| CAR10 | T1_T1_G-S | overall | 838 | 16 | 7 | 9 | +1.074 |
| CAR10 | T1_T1_G-S | new | 838 | 16 | 7 | 9 | +1.074 |
| CAR10 | T1_T1_G-S | A | 67 | 7 | 0 | 7 | +10.448 |
| CAR10 | T1_T1_G-S | B1 | 150 | 1 | 3 | -2 | -1.333 |
| CAR10 | T1_T1_G-S | B2 | 11 | 1 | 0 | 1 | +9.091 |
| CAR10 | T1_T1_G-S | C | 12 | 0 | 1 | -1 | -8.333 |
| CAR10 | T1_T1_G-S | final_correct | 598 | 7 | 3 | 4 | +0.669 |
| CAR10 | T1_T1_G-S | B | 161 | 2 | 3 | -1 | -0.621 |
| CAR10 | T1_T1_G-R | overall | 838 | 15 | 11 | 4 | +0.477 |
| CAR10 | T1_T1_G-R | new | 838 | 15 | 11 | 4 | +0.477 |
| CAR10 | T1_T1_G-R | A | 67 | 6 | 2 | 4 | +5.970 |
| CAR10 | T1_T1_G-R | B1 | 150 | 3 | 3 | 0 | +0.000 |
| CAR10 | T1_T1_G-R | B2 | 11 | 1 | 1 | 0 | +0.000 |
| CAR10 | T1_T1_G-R | C | 12 | 0 | 0 | 0 | +0.000 |
| CAR10 | T1_T1_G-R | final_correct | 598 | 5 | 5 | 0 | +0.000 |
| CAR10 | T1_T1_G-R | B | 161 | 4 | 4 | 0 | +0.000 |
| CAR10 | T1_T1_GM-G | overall | 838 | 0 | 0 | 0 | +0.000 |
| CAR10 | T1_T1_GM-G | new | 838 | 0 | 0 | 0 | +0.000 |
| CAR10 | T1_T1_GM-G | A | 67 | 0 | 0 | 0 | +0.000 |
| CAR10 | T1_T1_GM-G | B1 | 150 | 0 | 0 | 0 | +0.000 |
| CAR10 | T1_T1_GM-G | B2 | 11 | 0 | 0 | 0 | +0.000 |
| CAR10 | T1_T1_GM-G | C | 12 | 0 | 0 | 0 | +0.000 |
| CAR10 | T1_T1_GM-G | final_correct | 598 | 0 | 0 | 0 | +0.000 |
| CAR10 | T1_T1_GM-G | B | 161 | 0 | 0 | 0 | +0.000 |
| CAR10 | T9_T10_T9_G-S | overall | 7262 | 32 | 47 | -15 | -0.207 |
| CAR10 | T9_T10_T9_G-S | new | 758 | 8 | 12 | -4 | -0.528 |
| CAR10 | T9_T10_T9_G-S | old | 6504 | 24 | 35 | -11 | -0.169 |
| CAR10 | T9_T10_T9_G-S | T9 | 758 | 8 | 12 | -4 | -0.528 |
| CAR10 | T9_T10_T9_G-S | A | 907 | 4 | 2 | 2 | +0.221 |
| CAR10 | T9_T10_T9_G-S | B1 | 473 | 7 | 5 | 2 | +0.423 |
| CAR10 | T9_T10_T9_G-S | B2 | 87 | 5 | 4 | 1 | +1.149 |
| CAR10 | T9_T10_T9_G-S | C | 81 | 0 | 2 | -2 | -2.469 |
| CAR10 | T9_T10_T9_G-S | final_correct | 5714 | 16 | 34 | -18 | -0.315 |
| CAR10 | T9_T10_T9_G-S | B | 560 | 12 | 9 | 3 | +0.536 |
| CAR10 | T9_T10_T9_G-R | overall | 7262 | 42 | 46 | -4 | -0.055 |
| CAR10 | T9_T10_T9_G-R | new | 758 | 11 | 13 | -2 | -0.264 |
| CAR10 | T9_T10_T9_G-R | old | 6504 | 31 | 33 | -2 | -0.031 |
| CAR10 | T9_T10_T9_G-R | T9 | 758 | 11 | 13 | -2 | -0.264 |
| CAR10 | T9_T10_T9_G-R | A | 907 | 4 | 5 | -1 | -0.110 |
| CAR10 | T9_T10_T9_G-R | B1 | 473 | 8 | 7 | 1 | +0.211 |
| CAR10 | T9_T10_T9_G-R | B2 | 87 | 5 | 2 | 3 | +3.448 |
| CAR10 | T9_T10_T9_G-R | C | 81 | 0 | 2 | -2 | -2.469 |
| CAR10 | T9_T10_T9_G-R | final_correct | 5714 | 25 | 30 | -5 | -0.088 |
| CAR10 | T9_T10_T9_G-R | B | 560 | 13 | 9 | 4 | +0.714 |
| CAR10 | T9_T10_T9_GM-G | overall | 7262 | 30 | 38 | -8 | -0.110 |
| CAR10 | T9_T10_T9_GM-G | new | 758 | 13 | 25 | -12 | -1.583 |
| CAR10 | T9_T10_T9_GM-G | old | 6504 | 17 | 13 | 4 | +0.062 |
| CAR10 | T9_T10_T9_GM-G | T9 | 758 | 13 | 25 | -12 | -1.583 |
| CAR10 | T9_T10_T9_GM-G | A | 907 | 4 | 0 | 4 | +0.441 |
| CAR10 | T9_T10_T9_GM-G | B1 | 473 | 7 | 5 | 2 | +0.423 |
| CAR10 | T9_T10_T9_GM-G | B2 | 87 | 1 | 2 | -1 | -1.149 |
| CAR10 | T9_T10_T9_GM-G | C | 81 | 1 | 2 | -1 | -1.235 |
| CAR10 | T9_T10_T9_GM-G | final_correct | 5714 | 17 | 29 | -12 | -0.210 |
| CAR10 | T9_T10_T9_GM-G | B | 560 | 8 | 7 | 1 | +0.179 |
| CAR10 | T9_T10_T10_G-S | overall | 8041 | 29 | 51 | -22 | -0.274 |
| CAR10 | T9_T10_T10_G-S | new | 779 | 4 | 5 | -1 | -0.128 |
| CAR10 | T9_T10_T10_G-S | old | 7262 | 25 | 46 | -21 | -0.289 |
| CAR10 | T9_T10_T10_G-S | T9 | 758 | 8 | 14 | -6 | -0.792 |
| CAR10 | T9_T10_T10_G-S | A | 907 | 2 | 6 | -4 | -0.441 |
| CAR10 | T9_T10_T10_G-S | B1 | 473 | 7 | 8 | -1 | -0.211 |
| CAR10 | T9_T10_T10_G-S | B2 | 87 | 7 | 1 | 6 | +6.897 |
| CAR10 | T9_T10_T10_G-S | C | 81 | 0 | 3 | -3 | -3.704 |
| CAR10 | T9_T10_T10_G-S | final_correct | 5714 | 9 | 28 | -19 | -0.333 |
| CAR10 | T9_T10_T10_G-S | B | 560 | 14 | 9 | 5 | +0.893 |
| CAR10 | T9_T10_T10_G-R | overall | 8041 | 25 | 39 | -14 | -0.174 |
| CAR10 | T9_T10_T10_G-R | new | 779 | 2 | 3 | -1 | -0.128 |
| CAR10 | T9_T10_T10_G-R | old | 7262 | 23 | 36 | -13 | -0.179 |
| CAR10 | T9_T10_T10_G-R | T9 | 758 | 8 | 14 | -6 | -0.792 |
| CAR10 | T9_T10_T10_G-R | A | 907 | 2 | 3 | -1 | -0.110 |
| CAR10 | T9_T10_T10_G-R | B1 | 473 | 8 | 4 | 4 | +0.846 |
| CAR10 | T9_T10_T10_G-R | B2 | 87 | 3 | 2 | 1 | +1.149 |
| CAR10 | T9_T10_T10_G-R | C | 81 | 0 | 2 | -2 | -2.469 |
| CAR10 | T9_T10_T10_G-R | final_correct | 5714 | 10 | 25 | -15 | -0.263 |
| CAR10 | T9_T10_T10_G-R | B | 560 | 11 | 6 | 5 | +0.893 |
| CAR10 | T9_T10_T10_GM-G | overall | 8041 | 36 | 44 | -8 | -0.099 |
| CAR10 | T9_T10_T10_GM-G | new | 779 | 2 | 2 | 0 | +0.000 |
| CAR10 | T9_T10_T10_GM-G | old | 7262 | 34 | 42 | -8 | -0.110 |
| CAR10 | T9_T10_T10_GM-G | T9 | 758 | 11 | 23 | -12 | -1.583 |
| CAR10 | T9_T10_T10_GM-G | A | 907 | 3 | 4 | -1 | -0.110 |
| CAR10 | T9_T10_T10_GM-G | B1 | 473 | 16 | 6 | 10 | +2.114 |
| CAR10 | T9_T10_T10_GM-G | B2 | 87 | 0 | 1 | -1 | -1.149 |
| CAR10 | T9_T10_T10_GM-G | C | 81 | 0 | 1 | -1 | -1.235 |
| CAR10 | T9_T10_T10_GM-G | final_correct | 5714 | 15 | 30 | -15 | -0.263 |
| CAR10 | T9_T10_T10_GM-G | B | 560 | 16 | 7 | 9 | +1.607 |

## Decision

- IA10: **STOP_CURRENT_RECIPE**; checks: `{'birth_task_mean_G_minus_S_at_least_1pp': False, 'no_birth_loss_below_minus_0p5pp': True, 'T9_seen_not_below_minus_0p5pp': True, 'T9_cohort_after_T10_at_least_plus_0p5pp': False, 'T10_seen_does_not_drop': False, 'old_new_not_below_minus_1pp': True, 'better_than_R_birth_mean_and_T10_seen': False}`. Exploratory numerical screen only; not formal label/QC validation.
- CAR10: **STOP_CURRENT_RECIPE**; checks: `{'birth_task_mean_G_minus_S_at_least_1pp': False, 'no_birth_loss_below_minus_0p5pp': False, 'T9_seen_not_below_minus_0p5pp': True, 'T9_cohort_after_T10_at_least_plus_0p5pp': False, 'T10_seen_does_not_drop': False, 'old_new_not_below_minus_1pp': True, 'better_than_R_birth_mean_and_T10_seen': False}`. Exploratory numerical screen only; not formal label/QC validation.

## T9 birth → T10 transitions

| Protocol | Branch | N | Birth correct → wrong | Birth wrong → correct | Net |
|---|---|---:|---:|---:|---:|
| IA10 | V | 147 | 2 | 0 | -2 |
| IA10 | R | 147 | 1 | 2 | 1 |
| IA10 | S | 147 | 1 | 1 | 0 |
| IA10 | G | 147 | 2 | 2 | 0 |
| IA10 | GM | 147 | 4 | 2 | -2 |
| CAR10 | V | 758 | 16 | 23 | 7 |
| CAR10 | R | 758 | 23 | 18 | -5 |
| CAR10 | S | 758 | 21 | 14 | -7 |
| CAR10 | G | 758 | 23 | 14 | -9 |
| CAR10 | GM | 758 | 23 | 14 | -9 |

## Head diagnostics

CE uses scaled cosine logits (no training margin at evaluation); local scores are diagnostic, not oracle-task inference. V has no cached per-sample logits: CE/margins are unavailable, not zero.

| Protocol | Location/task/branch | Mean CE | Mean all margin | Mean local margin | New→old | New→new | Old→new | Old→old |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IA10 | T1_T1_R | 0.4955 | 3.5840 | 3.5840 | 0 | 26 | 0 | 0 |
| IA10 | T1_T1_S | 0.4932 | 3.5907 | 3.5907 | 0 | 24 | 0 | 0 |
| IA10 | T1_T1_G | 0.5457 | 3.2429 | 3.2429 | 0 | 22 | 0 | 0 |
| IA10 | T1_T1_GM | 0.5457 | 3.2429 | 3.2429 | 0 | 22 | 0 | 0 |
| IA10 | T1_T1_V | — | — | — | 0 | 24 | 0 | 0 |
| IA10 | T9_T10_T9_R | 1.4076 | 1.1348 | 3.8462 | 56 | 7 | 70 | 355 |
| IA10 | T9_T10_T9_S | 1.4084 | 1.1324 | 3.8412 | 56 | 7 | 73 | 349 |
| IA10 | T9_T10_T9_G | 1.4161 | 1.1088 | 3.7987 | 57 | 6 | 69 | 355 |
| IA10 | T9_T10_T9_GM | 1.4192 | 1.1083 | 3.7857 | 57 | 6 | 72 | 356 |
| IA10 | T9_T10_T9_V | — | — | — | 55 | 6 | 73 | 351 |
| IA10 | T9_T10_T10_R | 1.4632 | 1.1110 | 3.8371 | 41 | 1 | 51 | 445 |
| IA10 | T9_T10_T10_S | 1.4634 | 1.1099 | 3.8353 | 40 | 1 | 48 | 445 |
| IA10 | T9_T10_T10_G | 1.4694 | 1.0890 | 3.8026 | 41 | 1 | 48 | 448 |
| IA10 | T9_T10_T10_GM | 1.4726 | 1.0851 | 3.7879 | 41 | 1 | 47 | 445 |
| IA10 | T9_T10_T10_V | — | — | — | 40 | 1 | 50 | 445 |
| CAR10 | T1_T1_R | 0.3692 | 3.4865 | 3.4865 | 0 | 83 | 0 | 0 |
| CAR10 | T1_T1_S | 0.3699 | 3.5182 | 3.5182 | 0 | 88 | 0 | 0 |
| CAR10 | T1_T1_G | 0.3672 | 3.5141 | 3.5141 | 0 | 79 | 0 | 0 |
| CAR10 | T1_T1_GM | 0.3672 | 3.5141 | 3.5141 | 0 | 79 | 0 | 0 |
| CAR10 | T1_T1_V | — | — | — | 0 | 82 | 0 | 0 |
| CAR10 | T9_T10_T9_R | 1.0435 | 1.3829 | 3.4298 | 123 | 16 | 156 | 1204 |
| CAR10 | T9_T10_T9_S | 1.0442 | 1.3815 | 3.4272 | 118 | 19 | 157 | 1194 |
| CAR10 | T9_T10_T9_G | 1.0464 | 1.3758 | 3.4248 | 119 | 22 | 157 | 1205 |
| CAR10 | T9_T10_T9_GM | 1.0515 | 1.3656 | 3.4183 | 134 | 19 | 163 | 1195 |
| CAR10 | T9_T10_T9_V | — | — | — | 122 | 20 | 151 | 1210 |
| CAR10 | T9_T10_T10_R | 1.0835 | 1.3258 | 3.4889 | 145 | 4 | 185 | 1372 |
| CAR10 | T9_T10_T10_S | 1.0839 | 1.3248 | 3.4869 | 145 | 4 | 188 | 1361 |
| CAR10 | T9_T10_T10_G | 1.0857 | 1.3205 | 3.4848 | 146 | 4 | 185 | 1385 |
| CAR10 | T9_T10_T10_GM | 1.0907 | 1.3107 | 3.4793 | 146 | 4 | 190 | 1388 |
| CAR10 | T9_T10_T10_V | — | — | — | 137 | 7 | 183 | 1365 |

Both datasets must pass before extending to full trajectory. A failed recipe does not refute all coverage/generative augmentation hypotheses. Improved support is an association, and G−V includes added exposure/compute. G/GM representation equality applies at the statistics fork only; later T10 adaptation may diverge.

## Evidence

[IA10 birth audit](audit/IA10/report.md) · [CAR10 birth audit](audit/CAR10/report.md) · [Parent distribution](parent_distribution.csv) · [Automatic sanity](automatic_sanity_report.md) · [Configuration](config.json) · [Metrics](metrics.json) · [Sample predictions](per_sample_predictions.csv.gz) · [Statistics fork](statistics_comparison.json) · [Resources](resource_usage.json)
