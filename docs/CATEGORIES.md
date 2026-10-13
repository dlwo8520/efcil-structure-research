# Categorized repository map

Folders are grouped by research method. Each guide briefly states purpose and evidence status. File-list messages use `method / artifact type` categories for code, configs, records, tables, figures and notes.

## Baselines and early diagnostics

| Path | Purpose / guide |
|---|---|
| `projects/efcil_direction_diagnostic/` | [Early CIFAR / KeepLoRA-style / PCDS diagnostics](../projects/efcil_direction_diagnostic/CATEGORY.md) |
| `projects/` | [Early diagnostic projects](../projects/README.md) |

## Baselines and subspace adaptation

| Path | Purpose / guide |
|---|---|
| `cds-main/` | [CDS, CDSP and PCDS+SGDS variants](../cds-main/CATEGORY.md) |
| `pcds_structure_sgds_official/` | [Structure+SGDS local adaptation](../pcds_structure_sgds_official/CATEGORY.md) |

## Pretrained anchors

| Path | Purpose / guide |
|---|---|
| `cdsp_handoff_20260813/` | [PTM-guided placement and movement](../cdsp_handoff_20260813/CATEGORY.md) |

## Upstream reference

| Path | Purpose / guide |
|---|---|
| `revisitingcil-inspect.uZlsNf/` | [RevisitingCIL inspection baseline](../revisitingcil-inspect.uZlsNf/CATEGORY.md) |

## Geometry and separation

| Path | Purpose / guide |
|---|---|
| `structure_multicenter_separation/` | [Multi-center and Mahalanobis separation](../structure_multicenter_separation/CATEGORY.md) |
| `structure_e3_followup/` | [E3, PIC, CEOS and adaptive margins](../structure_e3_followup/CATEGORY.md) |

## Similarity and pair controls

| Path | Purpose / guide |
|---|---|
| `structure_similarity_aware_followup/` | [Similarity-aware S0-S4 constraints](../structure_similarity_aware_followup/CATEGORY.md) |
| `structure_similarity_control_experiments/` | [High, low and random pair controls](../structure_similarity_control_experiments/CATEGORY.md) |
| `structure_selective_similarity_sweep/` | [Selective SCC, separation and Proto-CLAD](../structure_selective_similarity_sweep/CATEGORY.md) |

## Projected prototype memory

| Path | Purpose / guide |
|---|---|
| `structure_rp_protoclad_task5/` | [Auxiliary random-projection Proto-CLAD](../structure_rp_protoclad_task5/CATEGORY.md) |
| `structure_stored_apr_rp/` | [Stored birth RP means and APR shifts](../structure_stored_apr_rp/CATEGORY.md) |

## Oracle and source diagnostics

| Path | Purpose / guide |
|---|---|
| `structure_prototype_oracle_audit/` | [Prototype, moment, source and head audits](../structure_prototype_oracle_audit/CATEGORY.md) |

## Covariance and classifier interventions

| Path | Purpose / guide |
|---|---|
| `fecam_boundary_followup/` | [FeCAM boundary follow-up](../fecam_boundary_followup/README.md) |
| `cross_task_ca_reweighting/` | [Cross-task Gaussian CA reweighting](../cross_task_ca_reweighting/CATEGORY.md) |
| `b1_ranking_preserving_correction/` | [Ranking-preserving task-score correction](../b1_ranking_preserving_correction/CATEGORY.md) |

## Acquisition and augmentation

| Path | Purpose / guide |
|---|---|
| `variation_consistency_pilot/` | [JS variation-consistency pilot](../variation_consistency_pilot/README.md) |
| `type_a_generative_augmentation/` | [Generic img2img versus matched strong views](../type_a_generative_augmentation/CATEGORY.md) |
| `ia10_bob_vs_strong_full/` | [BOB-full versus S-full IA10 preparation](../ia10_bob_vs_strong_full/CATEGORY.md) |

## Error trajectory diagnostics

| Path | Purpose / guide |
|---|---|
| `error_trajectory_diagnostic/` | [Persistent birth and later-error cohorts](../error_trajectory_diagnostic/CATEGORY.md) |
| `generation_local_all_transition_audit/` | [Paired local/all W-C-K transition audit](../generation_local_all_transition_audit/README.md) |

## Historical bundles and handoffs

| Path | Purpose / guide |
|---|---|
| `0923_code_bundle/` | [Structure prototype oracle audit bundle](../0923_code_bundle/README.md) |
| `ptm_final_handoff_stage_bXo40S/` | [Final PTM handoff snapshot](../ptm_final_handoff_stage_bXo40S/README.md) |
| `ptm_guided_handoff_final_stage_VsYbp3/` | [PTM guided final-stage snapshot](../ptm_guided_handoff_final_stage_VsYbp3/README.md) |
| `ptm_guided_handoff_stage_uVZlnb/` | [PTM guided intermediate snapshot](../ptm_guided_handoff_stage_uVZlnb/README.md) |

## Research synthesis

| Path | Purpose / guide |
|---|---|
| `research_report_for_professor/` | [Evidence report, figures and appendices](../research_report_for_professor/README.md) |

## Research management

| Path | Purpose / guide |
|---|---|
| `research_log/` | [Dated research observations and template](../research_log/README.md) |

## Documentation

| Path | Purpose / guide |
|---|---|
| `docs/` | [Research navigation and reproducibility](README.md) |

## Environments

| Path | Purpose / guide |
|---|---|
| `environments/` | [Observed software environments](../environments/README.md) |

## Experiment management

| Path | Purpose / guide |
|---|---|
| `experiments/` | [Experiment registry and evidence cards](../experiments/README.md) |

## Provenance

| Path | Purpose / guide |
|---|---|
| `provenance/` | [Source lineage, hashes and excluded artifacts](../provenance/README.md) |

## Result indexes

| Path | Purpose / guide |
|---|---|
| `results/` | [Derived result tables and evidence indexes](../results/README.md) |

## Research utilities

| Path | Purpose / guide |
|---|---|
| `tools/` | [Restoration, indexing and validation utilities](../tools/README.md) |

## Root files

| File | Role |
|---|---|
| [.gitignore](../.gitignore) | Excludes heavy runtime artifacts, caches and credentials |
| [CHANGELOG.md](../CHANGELOG.md) | Evidence-based methodological and repository-management changes |
| [README.md](../README.md) | Research overview, methods, current evidence and category navigation |
| [THIRD_PARTY.md](../THIRD_PARTY.md) | Attribution and boundaries between upstream methods and local adaptations |
| [requirements.txt](../requirements.txt) | Dependency entry point; select the appropriate package-specific environment |

## Reading the evidence

Use original run records for scientific claims. Category snapshot commits organize existing files; they do not represent new experiments. Earlier history remains reachable through a merge, and preserved source/result blobs are unchanged.
