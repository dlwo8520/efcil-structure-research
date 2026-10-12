# IA10: Birth error location + real feature support audit

Fixed vanilla retention Type A: 327. Birth post-CA head; deterministic evaluation transform. Support is current-task actual train only (no Gaussian/old actual bank). Head margins are raw cosine scores. A-within/A-cross and class-matched support are associations, not proof of image coverage causality. This audit never selects generation parents or prompts.

| Task | A | A-within | A-cross | Balanced n/class | k | Support valid |
|---|---:|---:|---:|---:|---:|---|
| 1 | 19 | 19 | 0 | 10 | 5 | True |
| 2 | 21 | 15 | 6 | 6 | 5 | True |
| 3 | 24 | 12 | 12 | 4 | 3 | True |
| 4 | 37 | 16 | 21 | 6 | 5 | True |
| 5 | 42 | 18 | 24 | 2 | 1 | True |
| 6 | 31 | 14 | 17 | 2 | 1 | True |
| 7 | 45 | 25 | 20 | 7 | 5 | True |
| 8 | 47 | 23 | 24 | 4 | 3 | True |
| 9 | 61 | 27 | 34 | 4 | 3 | True |

Macro class-matched A−birth-correct own support by subsample seed: `{'1993': -0.10980440290378672, '1994': -0.10200662718021444, '1995': -0.11104575504681893}`.

Classes containing A but no birth-correct comparator (not silently counted as comparable): `[1, 4, 16, 26, 28, 57, 58, 85, 87, 88, 92, 94, 111, 115, 125, 128, 135, 147, 155, 159, 161, 170, 173]`.

Error-location counts / positive-support versus local-head disagreement: `{'A-within': {'n': 169, 'support_valid_n': 169, 'positive_support_margin_local_head_wrong_n': 34}, 'A-cross': {'n': 158, 'support_valid_n': 158, 'positive_support_margin_local_head_wrong_n': 0}}`.

[Sample measurements](birth_support_audit.csv) · [Class-matched and leave-one-out reference](class_matched_support.csv) · [Summary](summary.json)
