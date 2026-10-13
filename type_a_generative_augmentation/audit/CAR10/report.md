# CAR10: Birth error location + real feature support audit

Fixed vanilla retention Type A: 907. Birth post-CA head; deterministic evaluation transform. Support is current-task actual train only (no Gaussian/old actual bank). Head margins are raw cosine scores. A-within/A-cross and class-matched support are associations, not proof of image coverage causality. This audit never selects generation parents or prompts.

| Task | A | A-within | A-cross | Balanced n/class | k | Support valid |
|---|---:|---:|---:|---:|---:|---|
| 1 | 67 | 67 | 0 | 33 | 5 | True |
| 2 | 67 | 44 | 23 | 33 | 5 | True |
| 3 | 86 | 58 | 28 | 29 | 5 | True |
| 4 | 83 | 40 | 43 | 28 | 5 | True |
| 5 | 85 | 37 | 48 | 24 | 5 | True |
| 6 | 131 | 55 | 76 | 38 | 5 | True |
| 7 | 103 | 25 | 78 | 30 | 5 | True |
| 8 | 166 | 43 | 123 | 35 | 5 | True |
| 9 | 119 | 42 | 77 | 34 | 5 | True |

Macro class-matched A−birth-correct own support by subsample seed: `{'1993': -0.11801420774640917, '1994': -0.1190860695854018, '1995': -0.11959288750268236}`.

Classes containing A but no birth-correct comparator (not silently counted as comparable): `[]`.

Error-location counts / positive-support versus local-head disagreement: `{'A-within': {'n': 411, 'support_valid_n': 411, 'positive_support_margin_local_head_wrong_n': 86}, 'A-cross': {'n': 496, 'support_valid_n': 496, 'positive_support_margin_local_head_wrong_n': 0}}`.

[Sample measurements](birth_support_audit.csv.gz) · [Class-matched and leave-one-out reference](class_matched_support.csv) · [Summary](summary.json)
