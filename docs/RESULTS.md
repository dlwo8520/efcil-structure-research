# Results: comparable blocks and diagnostic boundaries

Accuracy and forgetting are %, deltas pp. Model seed is 1993 in these main blocks. No cross-protocol average or global ranking is calculated. The [run index](../results/cil_run_index.csv) retains completed/incomplete/smoke distinctions and source/config paths. Source run IDs are namespaced by project; `B1` in different studies does not necessarily mean the same method.

## Reduced selective study: matched 5-task blocks

| Protocol | Run | Inc | Last | Forgetting ↓ | Old | New | ΔInc vs Structure |
|---|---|---:|---:|---:|---:|---:|---:|
| IA5 | Structure | 74.598 | 66.425 | 7.844 | 66.987 | 63.838 | 0.000 |
| IA5 | A_P005 | 74.670 | 66.689 | 7.070 | 67.628 | 62.362 | 0.072 |
| IA5 | B_P020_A100 | 74.554 | 66.359 | 7.010 | 67.228 | 62.362 | -0.043 |
| IA5 | C_P020_L030 | 74.640 | 66.557 | 7.864 | 67.147 | 63.838 | 0.042 |
| IA5 | C_P020_L100 | 74.621 | 66.623 | 7.424 | 67.308 | 63.469 | 0.024 |
| CAR5 | Structure | 84.929 | 81.196 | 4.431 | 80.987 | 82.055 | 0.000 |
| CAR5 | A_P005 | 84.902 | 81.296 | 3.938 | 81.173 | 81.801 | -0.028 |
| CAR5 | B_P020_A100 | 84.920 | 81.246 | 4.076 | 81.049 | 82.055 | -0.009 |
| CAR5 | C_P020_L030 | 84.905 | 81.184 | 4.450 | 81.002 | 81.928 | -0.024 |
| CAR5 | C_P020_L100 | 84.854 | 81.184 | 4.369 | 81.018 | 81.864 | -0.076 |

Source: [structure_selective_similarity_sweep/outputs/metrics_per_seed.csv](../structure_selective_similarity_sweep/outputs/metrics_per_seed.csv). A/B improve some stability measures but lose IA5 new accuracy; Method C preserves IA5 new accuracy better but does not improve CAR5 incremental accuracy. No common incremental-accuracy winner is established. See the [final reduced report](../structure_selective_similarity_sweep/reports/reduced_stage1_final_20260923/report.md).

## RP and stored birth prototypes

| Protocol | Method | Inc | Last | F ↓ | Old | New |
|---|---|---:|---:|---:|---:|---:|
| IA5 | RP_C_P020_L030 | 74.704 | 66.820 | 7.310 | 67.468 | 63.838 |
| IA5 | RP_C_P020_L100 | 74.495 | 66.359 | 7.682 | 67.067 | 63.100 |
| CAR5 | StoredAPR_RP_C_P020_L030 | 84.938 | 81.221 | 4.387 | 81.049 | 81.928 |
| IA5 | StoredAPR_RP_C_P020_L030 | 74.632 | 66.294 | 7.867 | 67.228 | 61.993 |

Old RP CAR5 runs have no completed result. StoredAPR CAR5 is not a substitute for that missing ablation. IA5 λ=.30 auxiliary RP has a small accuracy improvement; stronger λ and StoredAPR degrade aspects of new learning. See [RP report](../structure_rp_protoclad_task5/rp_protoclad_report.md) and [StoredAPR final report](../structure_stored_apr_rp/reports/stored_apr_rp_final_20260923/report.md).

## Later acquisition diagnostics and interventions

| Matched contrast / stage | IA10 | CAR10 | Unit / interpretation |
|---|---:|---:|---|
| Cross-task CA 2× minus its control, final seen | −0.197 | +0.075 | pp; original CA parity failed |
| Conditional task shift C2−C0, final seen | −0.527 | −0.460 | pp; local ranking preserved |
| Generated G−strong S, separate T1 new | +1.143 | +1.074 | pp; not a full T1→T10 trajectory |
| G−S, T9 new all-seen | 0.000 | −0.528 | pp; intervention from common T8 |
| G−S, T10 T9 cohort all-seen | 0.000 | −0.792 | pp; same cohort, real-only T10 |
| G−S, T9 cohort local net | +1 | −5 | sample counts; N=147 / 758 |
| G−S, T10 same cohort local net | 0 | −8 | sample counts; N=147 / 758 |

Each contrast uses its own internal control. Sources: [cross-task raw IA](../cross_task_ca_reweighting/IA10/summary.json), [CAR](../cross_task_ca_reweighting/CAR10/summary.json), [C2 IA](../b1_ranking_preserving_correction/IA10/metrics.json), [CAR](../b1_ranking_preserving_correction/CAR10/metrics.json), [generation report](../type_a_generative_augmentation/final_report.md), [paired local/all audit](../generation_local_all_transition_audit/final_report.md). Tables retain the original completed-record figures; index regeneration does not infer their mechanisms.

The [full numerical evidence table](../research_report_for_professor/key_results.csv) records oracle moments, source matching, probes, separation and error measures with units and precise scope. Oracle/actual old banks and test-defined subgroups are diagnostic-only and cannot enter a legal EFCIL method leaderboard. Original checkpoint/main versus post-after_task trajectory evaluation is kept separate: IA10 982 versus 983 correct in relevant records. The numerical mechanism of the one-sample discrepancy is not fully established.

## Latest incomplete branch

`ia10_bob_vs_strong_full/run_manifest.json` records `interrupted_or_failed`. The process-attempt list has successful environment/CPU checks and failed BOB GPU smoke. The README predates these attempts. No completed S-full or BOB-full result is reported here. Compatibility repair records are retained; their presence is not proof that the GPU failure was resolved.
