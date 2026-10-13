# Claim / Figure / Table source map

재현 명령: `${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python build_report.py --stage all` (CPU analysis only; no training/import of learners).

## 본문 claim / table routing

- Baseline / Table 1: `ten_task_hypothesis_revalidation/*/oracle_abcd_final_task.csv`, condition CHECKPOINT_MAIN. Class/task partition: 기존 run config / generation config.
- Branch B Table 2: cross_task_ca_reweighting/*/summary.json and b1_ranking_preserving_correction/*/metrics.json. Each comparator is internal.
- Branch A Table 3: birth_support_audit.csv, generation metrics.json, local/all transition summary CSV.
- Verdict Table 4: evidence_map.csv 23 rows; section-level evidence boundaries are explicit.
- baseline / SCC / APR / CosFace / CA equations: exact source files listed below (not an assumed conceptual replacement).

## Figures / query definitions

### fig01_research_flow

- Script/query: `build_report.py::collect / make_figures`
- Definition: Evidence synthesis; supported phenomenon is not a proven cause.

### fig02_age_tracking

- Script/query: `build_report.py::collect / make_figures`
- Definition: Left: pooled old class-task mean relative L2 (not a longitudinal independent replicate). Right: final seen micro B/C/D minus A, pp.
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/prototype_tracking_by_age.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/oracle_abcd_final_task.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/prototype_tracking_by_age.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/oracle_abcd_final_task.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/prototype_tracking_by_age.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/oracle_abcd_final_task.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/prototype_tracking_by_age.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/oracle_abcd_final_task.csv`

### fig03_classifier_source

- Script/query: `build_report.py::collect / make_figures`
- Definition: All changes are within their own matched experiment, not minus an unrelated baseline. Actual old features are diagnostic-only.
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/g256_r256_final_task.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/g256_r256_final_task.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/g256_r256_final_task.csv`
- Source: `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/g256_r256_final_task.csv`
- Source: `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/IA5_T5/classifier_probe_accuracy.csv`
- Source: `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/CAR5_T5/classifier_probe_accuracy.csv`

### fig04_representation_probe

- Script/query: `build_report.py::collect / make_figures`
- Definition: Selected cosine probe: identical split, optimizer and budget; historical train is not representation-unseen.
- Source: `structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/probe_results.csv`
- Source: `structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/probe_results.csv`

### fig05_error_trajectory

- Script/query: `build_report.py::collect / make_figures`
- Definition: Retained classes birth<=T9. A always wrong at task-end; B correct at birth/wrong finally; C remaining recovered-then-wrong. B1 final head with all birth-seen classes.
- Source: `error_trajectory_diagnostic/IA10/sample_trajectory.csv`
- Source: `error_trajectory_diagnostic/CAR10/sample_trajectory.csv`

### fig06_ranking_transfer

- Script/query: `build_report.py::collect / make_figures`
- Definition: C2 centered conditional task offset; CE and pp use separate axes. No seed uncertainty bars.
- Source: `b1_ranking_preserving_correction/IA10/metrics.json`
- Source: `b1_ranking_preserving_correction/CAR10/metrics.json`

### fig07_birth_support

- Script/query: `build_report.py::collect / make_figures`
- Definition: Birth<=T9; one bank seed 1993; A-within/A-cross own-support versus birth-correct in shared classes. Dots are classes, not replicates. Association, not causal evidence.
- Source: `type_a_generative_augmentation/birth_support_audit.csv`

### fig08_generation

- Script/query: `build_report.py::collect / make_figures`
- Definition: G-S; T1 experiment and T8->T9->T10 suffix are separate branches, not one full augmented trajectory.
- Source: `type_a_generative_augmentation/metrics.json`
- Source: `type_a_generative_augmentation/per_sample_predictions.csv`

### fig09_local_all_transition

- Script/query: `build_report.py::collect / make_figures`
- Definition: Fixed T9 cohort. W local wrong; C local correct/all wrong; K all correct. Shared 0-85% colour scale, cells raw counts.
- Source: `generation_local_all_transition_audit/IA10/t9_sample_state_transitions.csv`
- Source: `generation_local_all_transition_audit/IA10/t9_transition_summary.csv`
- Source: `generation_local_all_transition_audit/IA10/t10_transition_summary.csv`
- Source: `generation_local_all_transition_audit/CAR10/t9_sample_state_transitions.csv`
- Source: `generation_local_all_transition_audit/CAR10/t9_transition_summary.csv`
- Source: `generation_local_all_transition_audit/CAR10/t10_transition_summary.csv`

## Integrity / numeric discrepancies

- IA10 scientific CHECKPOINT_MAIN: 982/1519=64.647795%; task-end trajectory: 983/1519=64.713627%. Different evaluation timing (before/after `after_task`); do not silently equate. Exact 1-sample mechanism not proven here.
- Early unmatched G_FIXED/R_FIXED are superseded by G256/R256 for source-only inference; exposure restoration alone gives +3.818302/+5.720682 pp.
- Cross-task CA S0 weight parity failed (.011874 IA / .006949 CAR vs .001 tolerance). Internal paired results retained with caveat; no vanilla exact-reproduction claim.
- Coverage report marks Gaussianization NOT SUPPORTED under its decision rule although support gaps are negative; distinguish observed geometry loss from an unproven dominant downstream bottleneck.
- RP CAR5 original-RP comparator absent in final report: no fabricated direct CAR old-RP ablation.
- Original generation human HOLD is superseded only for exploratory auto-QC mode; label_fidelity_verified remains false.

## Read-only source fingerprints

| Source | Bytes | SHA-256 |
|---|---:|---|
| `b1_ranking_preserving_correction/CAR10/metrics.json` | 10882 | `3eb3a6c734de33369ba0fb9b9ea865ec4da5e74359d57e6db190b040317e0735` |
| `b1_ranking_preserving_correction/IA10/metrics.json` | 10780 | `9b12f23223ed13e67c8a7b6636cc1b441ad4882c910c7a3ae404d43e029e8556` |
| `b1_ranking_preserving_correction/core.py` | 5742 | `be0669d815a78ab56623b51ca7a6ae3be1a7987464fba4527a3f862a72312d76` |
| `cross_task_ca_reweighting/CAR10/summary.json` | 3309 | `4d5aaf89132ee3dbcddec94d5375f6948464144b578883b7398aa5e1226992d3` |
| `cross_task_ca_reweighting/IA10/summary.json` | 3331 | `bd0a22ae645fd02ff17dbb62e44a3e91fceedf044d6727ea92c750cb7be6add2` |
| `cross_task_ca_reweighting/losses.py` | 3368 | `e42465a05f68feae6d5235dff9af8e8ac71c2f1f4b5fa539de81911e106ab7f6` |
| `error_trajectory_diagnostic/CAR10/sample_trajectory.csv` | 1169315 | `d25f1a558007ab07c7838fdef9b789b84983bffe77b18064c61703a2bd03a184` |
| `error_trajectory_diagnostic/IA10/sample_trajectory.csv` | 221987 | `5840aa869d70941b697eed4ac7bf869d056e7b02b8ee959e2d6b9d2df3ad144a` |
| `error_trajectory_diagnostic/final_report.md` | 9981 | `b6c8baf9081b2d1db93801002311c449211632ff8bc74ff4d83a19c91bbdcf47` |
| `fecam_boundary_followup/ceos/CAR10/metrics.json` | 14175 | `8f30c9aec7a2c45d834468413d61adc18ebde5b0d68700fac9609487db95524e` |
| `fecam_boundary_followup/ceos/IA10/metrics.json` | 11876 | `7acd0a4702ed0780c5881883a73171206362e7fb4293006b9d0f1a5f0586be56` |
| `fecam_boundary_followup/ceos/run.py` | 7485 | `25b2f54d143277a4eeaeaba15b967197f5705bfd6621ee402c7f6f451d7edecb` |
| `fecam_boundary_followup/fecam/CAR10/metrics.json` | 26354 | `13a65e9d4946f46556fa6c2728c9bb8dcf2384ab121a946e61a50539f79403be` |
| `fecam_boundary_followup/fecam/IA10/metrics.json` | 21387 | `24fe267f01f6d4cbfae943ba825109ee9b145684247c254e1898f1c78bff384a` |
| `fecam_boundary_followup/fecam/run.py` | 5863 | `d38e7f0ea1b0c49e76eefbfb58e8abc54ef3e9de64aa03e0235e4487072acf30` |
| `generation_local_all_transition_audit/CAR10/t10_transition_summary.csv` | 2361 | `fe6ef949e548c3509f177a2a810b4b2aa9a71db4e2b24bb7f5411ca13e2f1ff7` |
| `generation_local_all_transition_audit/CAR10/t9_sample_state_transitions.csv` | 280525 | `5c72c392d33dbb23485ae3759190e3ee63c0f7d5ab66c605aef9bb1a4623853a` |
| `generation_local_all_transition_audit/CAR10/t9_transition_summary.csv` | 2362 | `a8add6a9996c462165d44a4f9ff4178012d166354bdcc96b563c8466292f1658` |
| `generation_local_all_transition_audit/IA10/t10_transition_summary.csv` | 1960 | `de15b9c58e4cf4033ed976a0a8c2e36b603cc6e9ebc2c6dbbbdf7ad596ae65b5` |
| `generation_local_all_transition_audit/IA10/t9_sample_state_transitions.csv` | 54626 | `05c38be98dfde7640f669f1588224e28ab1c6021af3649c1522e6a1a05eac4f2` |
| `generation_local_all_transition_audit/IA10/t9_transition_summary.csv` | 2044 | `1022951846bd7918105bf312b1bad762b3c45d27f5dd5e423bd81ae11206955d` |
| `generation_local_all_transition_audit/final_report.md` | 16538 | `e2c83119e937ed243dea36af88f303fc6a414120c29fed8fcfc838b7ac75433f` |
| `generation_local_all_transition_audit/run.py` | 43478 | `18ce7e9ab486d59e128468fb5dd08465612aa1376e03a3212ef18da033115b00` |
| `generation_local_all_transition_audit/run_manifest.json` | 8924 | `bd7b831c7baf67ea19864f142351e2c8793a008c973841111b6cea9a502ee5c2` |
| `structure_e3_followup/followup/adaptive_margin.py` | 3372 | `f8a441f782ceeb4c9762412a9624f114a750a467e0c64ff3de13c991749ac078` |
| `structure_e3_followup/followup/ceos.py` | 4012 | `6c77985aec83f8ae093c54005c81fc041a437f36588dc5d7842aa4691a0e9870` |
| `structure_e3_followup/followup/losses.py` | 11563 | `7ee986ebe60e872aefea6934eb3674c0272cecd291e3015ecd07a06eade0e482` |
| `structure_e3_followup/reports/e3_followup_comparison_20260917/tables/metrics_per_seed.csv` | 4444 | `02a220eb0b0da5aee0e9db3ae6b384b801cabbeec5c31352a9ea57397be4bb19` |
| `structure_multicenter_separation/exps_jsons/imageneta_5task/E0.json` | 1705 | `e3e9e85d89bcb78149ad1770bf3c5df31092cafb44a76990039feba2d6105168` |
| `structure_multicenter_separation/models/CDS_adapter.py` | 64623 | `c4278e55fe29fb9a0f7b9d4516b8aea037089046605f2927fe45c3ca18c56f87` |
| `structure_multicenter_separation/models/base.py` | 20514 | `9c954daf5ef89832b69ba1aa311fd4d3bbb91a8fe44256fa15d97708640c2f79` |
| `structure_multicenter_separation/network/classifier.py` | 3343 | `3806a98f8f820377096a314e505ed7b56ab2ee4eb463f7ad35722aadd24ae99e` |
| `structure_multicenter_separation/reports/e0_comparison_20260916/tables/metrics_per_seed.csv` | 2349 | `ed73ba9e6b848d31dc8504ee2cabd77d915c24844e28abe27f2a99c4fef41a2d` |
| `structure_multicenter_separation/utils/adjust_distribution.py` | 9600 | `6e943099d5bf003e1790b9f5d7b0c754c18487b54c4c07a273b41cbeaa080040` |
| `structure_multicenter_separation/utils/distill_loss.py` | 8321 | `26d3db91434b63d2745f52ad6d4bb7e3497ee4591f13b9bce4fb3e7f4272074a` |
| `structure_multicenter_separation/utils/distribution_regions.py` | 28223 | `23cebcf44f4921c761c9a84bb140c2ab2174761b8e3441846ccba602b0d91605` |
| `structure_multicenter_separation/utils/inc_net.py` | 8012 | `73ac3448cfd5cd2e96fcabe6a884cf7cb93a1a4aaae829baa9c6a9695414dba8` |
| `structure_multicenter_separation/utils/loss.py` | 2555 | `80140dece13e14e6a16426e82e5ed1d303b3ca144f23b8fbd02d78ec7cc3e0c4` |
| `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/CAR5_T5/classifier_probe_accuracy.csv` | 207817 | `2214fefec6ddb9642111ab82835afc21476be78d9b32b3d7b2887d3e87bb0e0b` |
| `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/CAR5_T5/g256_r256_accuracy.csv` | 103552 | `dd6711f5442356695aa12ea5742ae07adc3bb1df6740e315a29f24bdc5387163` |
| `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/IA5_T5/classifier_probe_accuracy.csv` | 202048 | `417023119fe3c120fa1b5f859b5bbafb843528b24e044306cc79d7cb78b439dc` |
| `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/IA5_T5/g256_r256_accuracy.csv` | 97389 | `0d673123c5275d6dd36d011e1ea665a9e9120dd62961a7d2b9b9dadad5abc042` |
| `structure_prototype_oracle_audit/classifier_procedure_diagnostic_gpu2/classifier_procedure_diagnostic_final_report_20260929.md` | 14966 | `8b08c971d93acebee62e6d8255da7e63a18b009767a4f8f26c69f9f1a2d6fb23` |
| `structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/paired_support_delta.csv` | 5109289 | `66753df72750b7db35ad650d0c78e2fe0997682adb9c94ba4c64af9c52ed0080` |
| `structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/probe_results.csv` | 3206 | `360e94ff903f2913a89f072ae5aa626c8f0338c1398e2dadf31f614c258a8ce4` |
| `structure_prototype_oracle_audit/coverage_generalization_diagnostic/CAR10/support_summary.csv` | 45192 | `4574c46de8e6c99a6755aee82445d68b75472547d431bc326453c8ef41502247` |
| `structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/paired_support_delta.csv` | 958013 | `74a49107cc9c1098aeed66e2382b62cbbbd02aa07170bf76a7f0d65f1a9138b4` |
| `structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/probe_results.csv` | 3296 | `4cb1a3d300a367d53a64f8a076328e871346c603f4360f4686f40673123971d2` |
| `structure_prototype_oracle_audit/coverage_generalization_diagnostic/IA10/support_summary.csv` | 53384 | `c2b8646c2cbfabbe21463f6357c686c45ed18f93ca798cd24af34263b74af47c` |
| `structure_prototype_oracle_audit/pair_boundary_risk/pipeline.py` | 38097 | `72ef87e3e377fd213828849acce821af73c20a9d9d3691fc5c5e6998a19d4665` |
| `structure_prototype_oracle_audit/paired_source_prediction_diagnostic/cross_protocol/final_summary.csv` | 1438 | `2d5887378b2e9f9627766da920ed2a4c3e102730a902fbcb15eb7796927b2137` |
| `structure_prototype_oracle_audit/scientific_audit/CAR10/seed_1993/apr_oracle_drift_audit.csv` | 259820 | `5e0417e5a9f6daba93dbda79ffee86b50aae40be2fd926d8755ca5c732ae98bc` |
| `structure_prototype_oracle_audit/scientific_audit/CAR10/seed_1993/ca_regression_check.json` | 24580 | `36b80bf2d5f3ab52b699cf9dabef48dbdb2e19f0b97f7f8171a88da014ea0a7d` |
| `structure_prototype_oracle_audit/scientific_audit/CAR10/seed_1993/covariance_oracle_audit.csv` | 773143 | `96ad8a5082c469ffb051e8aed1df82a19f1dc0b2b7a234a584c78d929e0de15f` |
| `structure_prototype_oracle_audit/scientific_audit/CAR10/seed_1993/prototype_oracle_audit.csv` | 568247 | `e602599f5f2a1a3336790252e7a084c0a600e09abf47119418783ed2969b7da3` |
| `structure_prototype_oracle_audit/scientific_audit/CAR5/seed_1993/apr_oracle_drift_audit.csv` | 114271 | `294a66aacb7d6be6cdf619c14f67359dda3081c5e2f4e017a51d6cd332f0c474` |
| `structure_prototype_oracle_audit/scientific_audit/CAR5/seed_1993/ca_regression_check.json` | 12170 | `097ed5a50e806f1360b36e190ecf6a83010b64b3856d36c8366e4fc60b7b21c7` |
| `structure_prototype_oracle_audit/scientific_audit/CAR5/seed_1993/covariance_oracle_audit.csv` | 415868 | `3076fadc6c7efe9d75fb586501abe45010972e9c4dc8501291ae1918863eaff1` |
| `structure_prototype_oracle_audit/scientific_audit/CAR5/seed_1993/prototype_oracle_audit.csv` | 305362 | `79ff7c1897758dbafad3257f19a98fcac284308aeaf94770a0511579789fe27c` |
| `structure_prototype_oracle_audit/scientific_audit/IA10/seed_1993/apr_oracle_drift_audit.csv` | 263552 | `87f45d12f85360ee33be5c91f9e25db4ea33fcd1b6be61ec5ab73a3c7128471b` |
| `structure_prototype_oracle_audit/scientific_audit/IA10/seed_1993/ca_regression_check.json` | 24745 | `9d090764f8720670809887d53b045a948de89fab5a5e255df108a12c91a5c42e` |
| `structure_prototype_oracle_audit/scientific_audit/IA10/seed_1993/covariance_oracle_audit.csv` | 765575 | `a56c04cc1f1f04c693b7b06a9d501a4086f531bee774a261d609a00cc56fce22` |
| `structure_prototype_oracle_audit/scientific_audit/IA10/seed_1993/prototype_oracle_audit.csv` | 561545 | `98a169bf1188f3b041fa339a3731869787bf0327dfbcae7b083fe35da1acda54` |
| `structure_prototype_oracle_audit/scientific_audit/IA5/seed_1993/apr_oracle_drift_audit.csv` | 116836 | `30c0414c0314cc69632ff2415089845e54960aa256c409cb1eefe4ef079e4b56` |
| `structure_prototype_oracle_audit/scientific_audit/IA5/seed_1993/ca_regression_check.json` | 12342 | `64b66f4c0827b4ae45aed207f49331383726ec640c89ac84a9da23e7e4fbd803` |
| `structure_prototype_oracle_audit/scientific_audit/IA5/seed_1993/covariance_oracle_audit.csv` | 415120 | `0ea66cc65570564cedfadf88fd56b9505538b623e435d0c6679f695c5ff6dc4b` |
| `structure_prototype_oracle_audit/scientific_audit/IA5/seed_1993/prototype_oracle_audit.csv` | 305628 | `78b67a4279ee559c95e429f170d53665d85126267c67a30ff5c905b03e2825d9` |
| `structure_prototype_oracle_audit/scientific_audit/structure_oracle_scientific_final_report_20260928.md` | 20160 | `3e8c8749de9199e1b81421db6f30b16adf76745f2cca6a024b4372ce26121b6b` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/error_concentration_by_task.csv` | 2558 | `e993ea3c77e4d211d85b886ff7c53cdae1e43a85273b287622356c7e2e9a5c23` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/g256_r256_final_task.csv` | 1267 | `60b33d4705570f411e87f8a44b09e5977370bde1f8a060f97e12edcf0cf043f9` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/oracle_abcd_final_task.csv` | 4310 | `93724ac0c1198dc60e97b213af2607a847a5cf5dbaae4ec16271d1c4db44c600` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/prototype_tracking_by_age.csv` | 2928 | `0abff389c5ebc58eb4404d44a06100d9ea191164d0d34fa364336c23a8d4710e` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/transition_summary.csv` | 1054 | `cddddca6dd9c5466b136e8cc85127522fd4ceab654cc22cd8adc731067b2aeda` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/g256_r256_final_task.csv` | 1143 | `f83a2c109363025d5575ac4eb1d535882e459b29fa9c04c710b014591ff15ff4` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/oracle_abcd_final_task.csv` | 4223 | `3df53cfd1309db0b2e1411c2b93384f6c8c10b9e8a0603c1447c4d23c205f7e2` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR5/prototype_tracking_by_age.csv` | 1660 | `038f72d3370f79fb69b2cc93337f12b8819b62de0587e4b21eae0236a88afd2c` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/error_concentration_by_task.csv` | 2478 | `4e3cf3f523dadfca86b2cc124e0511a0ab16641db73ca2e36c824fd688cf387e` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/g256_r256_final_task.csv` | 1257 | `a12c24140fabae608031cd61222bf2da9796d13333bc17706038a013afa7b28d` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/oracle_abcd_final_task.csv` | 4296 | `ed5ab2841ae17194766965907bcba9714225f9f245ea4425a9f8b8ace7c16f71` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/prototype_tracking_by_age.csv` | 2909 | `9e6b07984f891135bfe7c033cd01e7b138bec238ae205666410175224ba29d64` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/transition_summary.csv` | 1008 | `9de4526b91b036200fd1ef94217ac48d1be29135314e2bce7ffe612683da1a81` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/g256_r256_final_task.csv` | 1130 | `409983bf1f2fe12d98fa32bd89e774a258e3615550e16673e5f622a242a7ce6a` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/oracle_abcd_final_task.csv` | 4216 | `5782ef5b300b2d8b5b01d79b80ab34d5ccd660d832310b592d9066e13cdae5eb` |
| `structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA5/prototype_tracking_by_age.csv` | 1640 | `c7544a57f9bab64b19d1510c525800fb98bcdc88e06dca94f12f65877a1ce5ee` |
| `structure_rp_protoclad_task5/outputs/IA5/RP_C_P020_L030/seed_1993/summary.json` | 1005 | `0e8c6fe46bbd2a39f4cf714f25b35d862d51b006e01bb45e8cd2cc0daed09bf3` |
| `structure_rp_protoclad_task5/outputs/IA5/RP_C_P020_L100/seed_1993/summary.json` | 1003 | `cf817759b858e20e6cceb79cfce5c8425b67086a8ec143d7441cf5f93a765a8c` |
| `structure_rp_protoclad_task5/rp/losses.py` | 3538 | `74ca239c25e7759494796f22bb9b5bcd306f3db126d8a17ab022870da90120a0` |
| `structure_rp_protoclad_task5/rp/projection.py` | 1994 | `9ba07d29c6d61ad66850f3632e3de92ddf65bf926ee19a298678456527c01451` |
| `structure_selective_similarity_sweep/outputs/CAR5/A_P005/seed_1993/summary.json` | 996 | `ad3cbc754608ca8078882b898a66ac472a41189f04fbd335d054b1e06237e43d` |
| `structure_selective_similarity_sweep/outputs/CAR5/A_P005/seed_1993/task_summary.csv` | 1810 | `f5b54a9c10cc734f4540133c69cd3f0308b4faced43622dfd10b98072daf20e5` |
| `structure_selective_similarity_sweep/outputs/CAR5/B_P020_A100/seed_1993/summary.json` | 998 | `5e030bf7b039ad6f1888a7aedec389584cfbb01a06a4769b53478e60f1a57cd2` |
| `structure_selective_similarity_sweep/outputs/CAR5/B_P020_A100/seed_1993/task_summary.csv` | 1860 | `f3f5e70c4f657cfa9a9089b12ca1eadff9c188df2dac38c9518e6f46418eeb3b` |
| `structure_selective_similarity_sweep/outputs/CAR5/C_P020_L030/seed_1993/summary.json` | 999 | `1a8c006620a6e053cc14fed90a31e0591a21a53bf005ecc10c6aa079bc2f9716` |
| `structure_selective_similarity_sweep/outputs/CAR5/C_P020_L030/seed_1993/task_summary.csv` | 1861 | `095a7d91f82e114713b9b1813ccc432f594a869cb0a0544300694ee8406d1545` |
| `structure_selective_similarity_sweep/outputs/CAR5/C_P020_L100/seed_1993/summary.json` | 1000 | `47286cd68e9a32b974811746585a0fb60d0ae1d735a4585b5c4c4f7c2c66167a` |
| `structure_selective_similarity_sweep/outputs/CAR5/C_P020_L100/seed_1993/task_summary.csv` | 1859 | `cc788cbd59523e2d69424c26711a2d9608c33d6a0a458fea815bf7d37be2953d` |
| `structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/run_metadata.json` | 94527 | `cb07ae19e5bc4b0bc52ae4566564962120da5490bb86b525e7996f413e4b7599` |
| `structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/summary.json` | 985 | `41687784420be6174f7ed1394c82f7eeb030cd208f4fa390b5d037fa93047a28` |
| `structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/task_summary.csv` | 1839 | `08168fe89738bae8552aa6a890159128e7c9f40beabb33e0ff6a76b4f3792f7b` |
| `structure_selective_similarity_sweep/outputs/IA5/A_P005/seed_1993/summary.json` | 996 | `3503cbcbcea6a5fa19788adef58bec06eed6ab0207582cdc31a65755057f41c2` |
| `structure_selective_similarity_sweep/outputs/IA5/A_P005/seed_1993/task_summary.csv` | 1736 | `1ea905191c1f9fb9fee84d08af2752a592881e1c6ff76e4e921536961a25beb8` |
| `structure_selective_similarity_sweep/outputs/IA5/B_P020_A100/seed_1993/summary.json` | 1000 | `c9e0bc00d384253320550e48e17fee80eaf6a1b5917a56e975a6c69954e6232d` |
| `structure_selective_similarity_sweep/outputs/IA5/B_P020_A100/seed_1993/task_summary.csv` | 1826 | `ae0bdd8765e8df1ac51a79aa88c23ec04ec3d78adb80e2e52edcfe572b5a4b00` |
| `structure_selective_similarity_sweep/outputs/IA5/C_P020_L030/seed_1993/summary.json` | 1002 | `409878744e4438b5a60a95252bd36c729d55752fac0d4f72a96c7e772b690515` |
| `structure_selective_similarity_sweep/outputs/IA5/C_P020_L030/seed_1993/task_summary.csv` | 1793 | `c26525eabcc1be23df92fe356ccd985328dd689a3e24d54c8c3efa4cd7e713a8` |
| `structure_selective_similarity_sweep/outputs/IA5/C_P020_L100/seed_1993/summary.json` | 1003 | `838c110e92dc7ba055dd5a25a84b6f7c04ea5227da69d05f185a7a12d3d9c871` |
| `structure_selective_similarity_sweep/outputs/IA5/C_P020_L100/seed_1993/task_summary.csv` | 1824 | `c04fe1b1e246d31631ffc6d81c11a07187df59bc8c6bcbaadc107b485c95bdbf` |
| `structure_selective_similarity_sweep/selective/losses.py` | 7005 | `78f1d63a99fdb8a0a7be45dabe5e791e5d7d2377b439146588a93098a7c2393b` |
| `structure_similarity_aware_followup/outputs/imageneta/metrics_per_seed.csv` | 839 | `bc3f7e9603f47b29d6649a2ccf5f97df01a29abeeb5a913ed8c9853c9413db5e` |
| `structure_similarity_aware_followup/similarity/losses.py` | 14117 | `82c0c25a325ff63709379ecd8a1271749caed2dcd03fa646a4e01f552fe73317` |
| `structure_similarity_control_experiments/control/losses.py` | 2777 | `abb2d0d6c0181655fb1a8a2cfacee6a3f50269ed2a31f33ad84046cffc6723ef` |
| `structure_similarity_control_experiments/outputs/imageneta/metrics_per_seed.csv` | 618 | `cf3388fa71b98a133f782c3c65a47bf2f2d9d2e2f0adf8756dc01f5230f977bb` |
| `structure_stored_apr_rp/outputs/CAR5/StoredAPR_RP_C_P020_L030/seed_1993/summary.json` | 1012 | `1c28e2fc8f04ff205be49ad82954a54f6a7f6d8ae5d5f4f5c7820a2c0001fa03` |
| `structure_stored_apr_rp/outputs/IA5/StoredAPR_RP_C_P020_L030/seed_1993/summary.json` | 1012 | `f09f2fecb8a33d8f0d6de10c92ff2ff862fbf8d041c0d36e454c8c14144b3d12` |
| `structure_stored_apr_rp/reports/stored_apr_rp_final_20260923/report.md` | 8114 | `98eec940a1987065ef1179dba237c7f131e8075eb89e366b7d29b81c26a859fa` |
| `structure_stored_apr_rp/stored_apr_rp/prototypes.py` | 2791 | `75573c6526575f98f89cb93ef81824e6ccf99d06db406856e6adf8854eaae424` |
| `type_a_generative_augmentation/audit/CAR10/class_matched_support.csv` | 135585 | `c1d2eb35be0ff53f2758df36ea841af4cf61b9f70fcc09eac7a33c1c468df5fa` |
| `type_a_generative_augmentation/audit/IA10/class_matched_support.csv` | 115005 | `c2f8027970f78c5c99278c0c1767c2d13494df168aa42c30142e1abbe5489798` |
| `type_a_generative_augmentation/automatic_sanity_summary.json` | 15202 | `3b3bc7bd3698ed6aa88b94effa9ed68beab51c9ccde9110e8d452083ddb0d4d7` |
| `type_a_generative_augmentation/birth_support_audit.csv` | 4493772 | `95312061053b5993f53f83820a6657647c1961e69d7e44d5a9eb05c7199722d2` |
| `type_a_generative_augmentation/config.json` | 1804 | `e38339b883c51ee8a2f0e46a5c5307df8f46cda92593053fe48815d304c7bacf` |
| `type_a_generative_augmentation/exploratory_config.json` | 555 | `16596fb51f501de83d8c9e8c5b9c6b5eca170a2320261b4ec920c9a4d4dfb98a` |
| `type_a_generative_augmentation/learner.py` | 12631 | `edb0d01127c95f5bc21b7194bce43ce14bccbddefaeb8ba50d03ab3894b2cb8a` |
| `type_a_generative_augmentation/metrics.json` | 90694 | `9d574a33d70e0fac5d5280e313c337549258cc92c9d972919fd19678f94034a7` |
| `type_a_generative_augmentation/moments.py` | 735 | `1f7dd8c05c6c63dc8b6f63ff6b36cb2b57a30e46fd8c69656ec002d36d7d9807` |
| `type_a_generative_augmentation/per_sample_predictions.csv` | 11545805 | `2382432c3003a39a04b6a370d4c6ec91169d56fe9155173443b6369207816cac` |
| `type_a_generative_augmentation/reports/type_a_generative_augmentation_20261012/report.md` | 33020 | `d4a1b2c5d19cd1814ad2ddf18f9fa3a1878c22f84106715fd0757e94ee560611` |
| `type_a_generative_augmentation/run_manifest.json` | 326 | `6fe4086975e0c7c807d012f2651405773bfa22ca9556fdcb2a567424fe26b596` |
| `type_a_generative_augmentation/sanity_policy.json` | 1513 | `e990793c9376bfb31086bf75286fa294b9be5424c174c52e56adb03990da7e3c` |
| `variation_consistency_pilot/cross_protocol/summary.csv` | 1052 | `b94542a29ee7c85c716253028d45737ecf1ee95612705f5f2e8510a52402a086` |
| `variation_consistency_pilot/pilot_learner.py` | 9573 | `16c5ab6b2db0943cf00af1cc2e70ec5cd5374bf32529f988d331d7c1815dca9b` |
| `논문/0_Exemplar-FreeClassIncrementalLearningviaPreservingClass-DiscriminativeStructure.pdf` | 3103077 | `1c971f5071bbbf0ff418c32583236c3befa161062168b13ad478a007f6f80533` |
