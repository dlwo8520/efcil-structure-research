# Packaging validation — 2026-10-12 KST

Validation used the copied repository and the installed legacy Structure environment. No real model training, GPU inference, generation or existing job interruption was performed.

| Check | Result | Scope |
|---|---:|---|
| Structure existing unit tests | 17 passed | statistics, Mahalanobis gradients, replay allocation, config, metrics and recording |
| RP existing tests | 3 passed | frozen/reproducible projection and auxiliary-only loss |
| StoredAPR tests | 6 passed | birth samplewise statistic, APR formula, recording and checkpoint restore |
| StoredAPR reused RP checks | 3 passed | unchanged auxiliary RP contracts |
| StoredAPR config checks | 2 passed | IA5/CAR5 configuration/projection metadata |
| BOB integration existing CPU tests | 18 passed | views, moments/SCC source, restart, QC, result analysis, logger compatibility; toy substitutes |
| Local/all analysis existing unit tests | 10 passed | W/C/K transitions, paired net, macro/micro and fixed subgroups |
| E3 config validation | passed | inherited AM-Drift config; no dataset or learner training |
| Shell scripts | 53 syntax passes | `bash -n`; runtime activation paths not asserted |
| Config inheritance | 66 valid parent links | non-snapshot JSON extends chains |
| Python source structure | 589 equal to originals | after path-helper normalization; no algorithm AST differences |
| Numeric evidence reaggregation | 854 / 854 equal | 91 stored source files, 30 generation-condition checks, four W/C/K cohorts |
| Index regeneration | 118 run summaries indexed | state/protocol separation, including smoke/incomplete |

The 57 existing test executions include three RP checks run once directly and once through StoredAPR. They are 54 distinct tests; count details are above. `provenance/numeric_validation.json`, `source_logic_validation.json` and `structure_validation.json` provide machine-readable evidence. The normal validator checks exported hashes, syntax and registry target paths, plus obvious credential/private-path patterns.

The original 854-row report was recomputed in memory from exported raw data with no changes to stored research conclusions. Four compressed tables needed for the check were locally restored; their original compressed form remains the published source. No figures or historical source reports were regenerated over original data during this check.

Not verified: all other test suites/legacy launchers, all historical models/weights, nested vendor checkout restoration, diffusion/Qwen generation, every training/environment path, source digest parity after relocation, complete GPU runs, independent model seeds or scientific novelty. The original BOB GPU smoke failure remains an unresolved research-run state even though its current CPU contracts pass.
