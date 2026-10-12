# Experiment families and method boundaries

IDs remain stable even if descriptions or status change. Original run IDs are retained below each family and in the generated run index. Config files define exact dataset/hyperparameters; documents define hypothesis/interpretation.

| ID | Family | Problem / hypothesis | Baseline / provenance | Inspected state |
|---|---|---|---|---|
| [EFCIL-001](../experiments/EFCIL-001.md) | Early CIFAR diagnostics / KeepLoRA-style / PCDS | Diagnose representation and classifier limitations | custom reproduction/adaptation | implemented; baseline-specific outputs |
| [EFCIL-002](../experiments/EFCIL-002.md) | CDSP and PCDS+SGDS exploration | Protect discriminative subspaces and test activation guidance | upstream CDS plus custom changes | completed and variant-specific records |
| [EFCIL-003](../experiments/EFCIL-003.md) | Structure+SGDS local adaptation | Test semantic activation guidance | modified upstream, not full SGDS | recorded local results |
| [EFCIL-004](../experiments/EFCIL-004.md) | PTM-guided geometry | Compare pretrained anchors and Structure | custom adaptation | recorded results; paper numbers separated |
| [EFCIL-005](../experiments/EFCIL-005.md) | RevisitingCIL inspection baseline | Inspect pretrained-model CIL reference | upstream reference | reference source; logs are not new proposal evidence |
| [EFCIL-006](../experiments/EFCIL-006.md) | E0–E7 multi-center/Mahalanobis | Test local regions and old/new separation | upstream CDS plus proposed region code | completed records and inactive-loss limitations |
| [EFCIL-007](../experiments/EFCIL-007.md) | E3/PIC/CEOS/adaptive margins | Resolve new loss versus old stability | custom extensions; adapted prior ideas | 13 IA5 pilot variants complete |
| [EFCIL-008](../experiments/EFCIL-008.md) | Similarity-aware S0–S4 | Relax constraints selectively for similar pairs | custom extension | completed IA5 single-seed pilot |
| [EFCIL-009](../experiments/EFCIL-009.md) | High/low/random-off controls | Disentangle similarity from constraint count | controlled custom ablations | completed IA5 single-seed pilot |
| [EFCIL-010](../experiments/EFCIL-010.md) | Selective SCC / separation / Proto-CLAD | Find a dataset-common balanced method | custom extension | reduced IA5/CAR5 complete; old full sweep held/partial |
| [EFCIL-011](../experiments/EFCIL-011.md) | Auxiliary random-projection Proto-CLAD | Test RP only in auxiliary conflict geometry | custom adaptation; not full RP-paper reproduction | IA5 two completed; CAR5 old RP not started |
| [EFCIL-012](../experiments/EFCIL-012.md) | Stored birth RP mean plus APR shift | Reduce nonlinear projected-prototype approximation | custom extension | IA5/CAR5 complete; no common benefit |
| [EFCIL-013](../experiments/EFCIL-013.md) | Moment/source/head/representation diagnostics | Locate memory and decision bottlenecks | diagnostic-only old/oracle access | multiple completed diagnostics; protocol-specific |
| [EFCIL-014](../experiments/EFCIL-014.md) | FeCAM boundary follow-up | Test covariance/boundary alternatives | custom FeCAM-inspired intervention | completed recorded follow-up; dataset-specific |
| [EFCIL-015](../experiments/EFCIL-015.md) | JS variation consistency | Improve acquisition/retention with matched views | custom intervention | completed pilot; null/negative scope |
| [EFCIL-016](../experiments/EFCIL-016.md) | Persistent error cohorts | Distinguish birth and later errors | test-defined diagnostic, no selector | completed analysis-only |
| [EFCIL-017](../experiments/EFCIL-017.md) | Cross-task Gaussian CA reweighting | Reduce later task competition | custom intervention | completed matched pilot; CA parity failure |
| [EFCIL-018](../experiments/EFCIL-018.md) | Conditional task score shift | Repair competition without changing local ranks | custom correction | completed matched pilot; negative real-test deltas |
| [EFCIL-019](../experiments/EFCIL-019.md) | Matched generic img2img / strong views / moments | Improve birth acquisition via generated variation | custom intervention; label fidelity unverified | completed pilot; STOP current recipe |
| [EFCIL-020](../experiments/EFCIL-020.md) | W/C/K paired local/all audit | Check whether competition hides local improvements | analysis-only | completed analysis; no shared persistent improvement |
| [EFCIL-021](../experiments/EFCIL-021.md) | BOB-full versus S-full IA10 | Test full cumulative trajectories with current-only synthetic views | official BOB vendor plus custom CIL integration | interrupted_or_failed at BOB GPU smoke; no full result |
