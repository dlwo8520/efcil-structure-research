# Final Error Trajectory — consistency audit

이 문서는 read-only 진단 실행 중 갱신된다. 원본 checkpoint는 수정하지 않는다.

- Seed: 1993
- Physical evaluation GPU: 4 (process-local cuda:0)
- Evaluation: frozen task-end network, post-CA at T2–T10; T1 has no CA sidecar
- Candidate set: each task's mapped classes 0..seen-1
- Sample IDs: `representation_relation.pipeline._dataset_identity` on official test split
- Score source: direct checkpoint inference; older feature-cache logits are not used
- No training, new loss, replay, or test-label tuning

## IA10

- Status: complete
- Source training branch: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993`
- Original class order: `[168, 136, 51, 9, 183, 101, 171, 99, 42, 159, 191, 70, 16, 188, 27, 10, 175, 26, 68, 187, 98, 6, 85, 35, 112, 43, 100, 0, 103, 181, 88, 59, 4, 2, 116, 174, 94, 80, 106, 1, 147, 17, 141, 131, 72, 23, 173, 54, 197, 118, 87, 32, 79, 104, 91, 19, 135, 107, 178, 36, 11, 199, 142, 8, 122, 3, 28, 57, 153, 172, 190, 56, 49, 44, 97, 62, 151, 169, 194, 55, 192, 12, 189, 78, 66, 180, 15, 137, 109, 134, 92, 119, 126, 52, 170, 40, 148, 65, 144, 64, 138, 45, 77, 89, 154, 90, 71, 193, 74, 30, 113, 143, 96, 84, 67, 50, 186, 156, 69, 21, 18, 111, 108, 58, 125, 157, 150, 110, 182, 129, 166, 83, 81, 60, 13, 165, 14, 176, 63, 117, 5, 22, 145, 121, 38, 41, 82, 127, 114, 20, 31, 53, 37, 163, 196, 130, 152, 162, 86, 76, 24, 34, 184, 149, 33, 128, 198, 155, 146, 167, 139, 120, 140, 102, 47, 25, 158, 123, 46, 164, 61, 7, 115, 75, 133, 160, 105, 132, 179, 124, 48, 73, 93, 39, 95, 195, 29, 177, 185, 161]`
- Task partition: `[20, 20, 20, 20, 20, 20, 20, 20, 20, 20]`
- Final official test samples: 1519
- Original official test split SHA-256: `ef131815607d6399ccfe94223f80ebf5de85701144cc5e25c4d70e1ceebb5aa1`
- Test split hash mode: `sorted_relative_path_and_file_size`
- Final sample-ID/label SHA-256: `c3f500e5e6fa7964b56e575ee3072e4fd708b297fc44f8f98d9d4c9bdb148eb3`
- Test preprocessing and normalization: `[Resize(size=256, interpolation=bicubic, max_size=None, antialias=True), CenterCrop(size=(224, 224)), ToTensor()] | []`
- Training-recorded git commit: `None`
- Training-recorded source snapshot hash: `ea63875fe9a103c250b6d2e926372a15c919a82b5304390e31f0bda06b1c7c3c`
- Current baseline source git commit: `a0db2168663982a4cad24981231721a3518ddf29`

| Task | Checkpoint SHA-256 | State | Candidate classes | Test N | Correct | Historical correct Δ | ID-map SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| T1 | `5192a6b7e02fff988200c94bec35a156a4ebcc7bdb42cf6ce3ceee27a28630ed` | task_end_no_CA_at_T1 | 0..19 | 175 | 151 | 0 | `a33c477f4448cf35776e338dce64126650ebf10c50d766a0337e0e1d5ab915ab` |
| T2 | `9e82b1d19ca6354d71d26e188b15e41cb22d99c170091ebe171c1412e19c035f` | post_CA | 0..39 | 360 | 300 | 0 | `734f361de2f7a55a11ee70a53927909c33791f58229fe57877ec3231218a332f` |
| T3 | `47d36d3f4d1667637fbb62a898f08a61fde2f34e75d387e14aaf7020ed21288f` | post_CA | 0..59 | 476 | 369 | 0 | `25ed6e70f64b14d16136ad6946aa8874e4255766b54112d43415feee68c673cb` |
| T4 | `b73c95eb7837702c920edfb1658390fc4ea24b76b60cd4c7bcb5c1e61a86a3f5` | post_CA | 0..79 | 698 | 546 | 0 | `31f900679c337d442910468ce4ae9d0a998fe8eb82fc802fef08f53b67482f3e` |
| T5 | `4c0f5af6c9e3d653f88492ddeb349be0103396fbc8b4f29b903c4275abb6f5fc` | post_CA | 0..99 | 841 | 624 | 0 | `8cb71db34475e9f3f53cf0fafcf280e98884c6afd461a5b7c4d503da4f91dd44` |
| T6 | `8bba6680ffe2feb6b4fcb1f821c31ef9f5858a547804338c7da9ca76f96d7575` | post_CA | 0..119 | 975 | 708 | 0 | `daa1497a013b7ba009554b9eebabb121874b4f36358616894b8c9355fb7e1c17` |
| T7 | `1a24c9278de2079260ee37859be48b3ab5fba2e7743e2f4c2d6395f8ffbb139d` | post_CA | 0..139 | 1121 | 778 | 0 | `cc1ccd3ac5c690a88dac57c285373d9d763e613103eb4dc79a44aff029056f5c` |
| T8 | `366e84ae7cac3146bc3310eac465765ae9fed469bc15f5e6ebbbedfd940ddbf2` | post_CA | 0..159 | 1248 | 836 | 0 | `251bb288238366ec094ef21d0904f53f890a7bcff3e59fbf01ad0948cd5f0fec` |
| T9 | `c5be43fdda38e5f0b8012439baac984412b8fb3b6624ece4f9c6079f68489c83` | post_CA | 0..179 | 1395 | 910 | 2 | `d028f828daa5d8d8ec9773b14fdf244bfd891a8fe2755645e28f6832159e1058` |
| T10 | `fa0b5a20e3e07229e356cfcf576037c418abc5e8393d5d99a3dd47b6b347235c` | post_CA | 0..199 | 1519 | 983 | 1 | `c3f500e5e6fa7964b56e575ee3072e4fd708b297fc44f8f98d9d4c9bdb148eb3` |

Historical correct Δ compares task-end checkpoint inference with the original task-summary evaluation, which ran immediately before `after_task()`; nonzero Δ is recorded, not silently treated as exact parity.


Source-code SHA-256:

- `${RESEARCH_WORKSPACE}/error_trajectory_diagnostic/run.py`: `8928ed649210597fbedc168438bb37dd3429be40b8243766350b143688ee9f5a`
- `${RESEARCH_WORKSPACE}/error_trajectory_diagnostic/metrics.py`: `5fb91dc1a92635f72271630aa58d86d3f5f5a5578bc3bb2b53dddc2c1828c5d2`
- `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/oracle_audit/runtime.py`: `26d9e97182ea1ffb077d15d5d10508e4d075ea8b36dfd3de24ef8c616ae1a2a6`
- `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/representation_relation/pipeline.py`: `aade64832df7a3fe5431673f0b8349d5c2b1d003c0622a94c892442e0a56c110`
- `${RESEARCH_WORKSPACE}/structure_multicenter_separation/models/CDS_adapter.py`: `c4278e55fe29fb9a0f7b9d4516b8aea037089046605f2927fe45c3ca18c56f87`
- `${RESEARCH_WORKSPACE}/structure_multicenter_separation/data/data.py`: `ddb3f67d7c6f8d491544b31ff86b4e8a3cc1169b7f8c452fead6857689a5f7d0`
- `${RESEARCH_WORKSPACE}/structure_multicenter_separation/data/data_manager.py`: `e350d0b83615a8a34f33d0594d7ee4836b0335c11d79d1ba223b602008e804a1`

## CAR10

- Status: complete
- Source training branch: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993`
- Original class order: `[59, 134, 162, 9, 42, 72, 118, 99, 109, 79, 43, 100, 16, 144, 27, 10, 172, 26, 191, 80, 189, 6, 87, 153, 97, 70, 107, 0, 88, 178, 35, 4, 2, 170, 171, 141, 51, 166, 145, 188, 17, 167, 101, 116, 23, 157, 180, 129, 68, 85, 32, 54, 103, 104, 19, 113, 106, 175, 36, 11, 159, 169, 8, 98, 3, 28, 57, 147, 138, 187, 56, 49, 44, 125, 62, 151, 165, 94, 55, 92, 12, 122, 78, 66, 193, 15, 135, 168, 184, 91, 137, 186, 52, 126, 40, 112, 65, 142, 64, 136, 45, 77, 89, 154, 90, 71, 190, 74, 30, 119, 96, 84, 67, 50, 183, 150, 69, 21, 18, 111, 108, 58, 174, 131, 148, 110, 194, 1, 173, 83, 81, 60, 13, 163, 14, 156, 63, 117, 5, 22, 143, 121, 38, 41, 82, 127, 114, 20, 31, 53, 37, 195, 192, 130, 152, 176, 86, 76, 24, 34, 181, 149, 33, 128, 182, 155, 146, 139, 120, 140, 102, 47, 25, 158, 123, 46, 164, 61, 7, 115, 75, 133, 160, 105, 132, 179, 124, 48, 73, 93, 39, 95, 29, 177, 185, 161]`
- Task partition: `[20, 20, 20, 20, 20, 20, 19, 19, 19, 19]`
- Final official test samples: 8041
- Original official test split SHA-256: `2fb5cc31c9ccf42f6bfd24fbac94b83c64f10b9b4b9f0ff0883c107cc0b09207`
- Test split hash mode: `sorted_relative_path_and_file_size`
- Final sample-ID/label SHA-256: `d6197835970e91b1833b15336ca5fe51cab1eb0cde201d06f166b5451b27ea94`
- Test preprocessing and normalization: `[Resize(size=256, interpolation=bicubic, max_size=None, antialias=True), CenterCrop(size=(224, 224)), ToTensor()] | []`
- Training-recorded git commit: `None`
- Training-recorded source snapshot hash: `ea63875fe9a103c250b6d2e926372a15c919a82b5304390e31f0bda06b1c7c3c`
- Current baseline source git commit: `a0db2168663982a4cad24981231721a3518ddf29`

| Task | Checkpoint SHA-256 | State | Candidate classes | Test N | Correct | Historical correct Δ | ID-map SHA-256 |
|---|---|---|---:|---:|---:|---:|---|
| T1 | `ac6311ee2e869fdf55701169aabc4625830eb0aeb3f21f450a713d9b270940b3` | task_end_no_CA_at_T1 | 0..19 | 838 | 756 | 1 | `61892eee26d7f63fb5cbf1cec379d763a8b69142ab1cc0d47360489eedc7c6e2` |
| T2 | `902bf041c5623cd5f29515c1e668ac3f323bda950411883e0a32f6c4e1ce3def` | post_CA | 0..39 | 1676 | 1441 | 1 | `c81441c1c6e4f0614f403ccc8762c2e42da703b34a52ebfae7aed1771560ebbf` |
| T3 | `1d2afa6ef2bb802e2890db51f6a7d3cc93532fbe91c7d5bcd3f2a696ac22195a` | post_CA | 0..59 | 2487 | 2115 | 0 | `f28e19206b2216a17e8aa18d5c856cbcba746c87893082f2fdce24bcaab2bcad` |
| T4 | `6f3055da0092bcaee47dfe0996358e8daf88ab6cafd61d0c75342fa49efed2e9` | post_CA | 0..79 | 3304 | 2766 | 1 | `b1b746b22a1cfb27d3cb8e039efb23e3feafc4de51f8d136c2dcbaf9fb594b7d` |
| T5 | `03eb7f1ce42e1721460a384225880b98f1896765f0ac70c3264449e0bf3bfee8` | post_CA | 0..99 | 4116 | 3444 | 0 | `17410196302666fc85f95da1c0864fba451ff09c2fa2dfa1eba898a7d257b6cf` |
| T6 | `7056f6f048b5ff135612f1e422098c3eacd9fc89d3623906ec06800fd772a008` | post_CA | 0..119 | 4958 | 4093 | 0 | `90376afaa35ceb56318cfee3437d7dca1ae490a9204f9ed66c0cd314e5ee4a75` |
| T7 | `dfb1d2f291831a117e75a96c48e89e6e8692aa9a71ba144489d6468548a934b3` | post_CA | 0..138 | 5727 | 4701 | 1 | `ff6f703d08e0594e52b29081cbc02b0449ce5e998358e720f11e2a7873f8364f` |
| T8 | `fe146360af4b8c7fc4dc14cd58b3a0f32f8d0398db0053ace1033746531cfcd9` | post_CA | 0..157 | 6504 | 5250 | 0 | `47d761b5ed58e85ad4400b908bfaf66cf40a92ca1251f3a0e67664afe4585e7e` |
| T9 | `4b611001256ec6ac74eafc4f32aaa2042dabca940b884fa881c1b06192b89b3e` | post_CA | 0..176 | 7262 | 5759 | 1 | `f8019d5cc12f8088e243563f2fecd7d48f94171fe77b93dc9520910ae7a95ee0` |
| T10 | `623382eb9fb5a7f06c4512f57754e9bbf99e9814fd7136b2bf6463adbd37a142` | post_CA | 0..195 | 8041 | 6349 | 0 | `d6197835970e91b1833b15336ca5fe51cab1eb0cde201d06f166b5451b27ea94` |

Historical correct Δ compares task-end checkpoint inference with the original task-summary evaluation, which ran immediately before `after_task()`; nonzero Δ is recorded, not silently treated as exact parity.


Source-code SHA-256:

- `${RESEARCH_WORKSPACE}/error_trajectory_diagnostic/run.py`: `8928ed649210597fbedc168438bb37dd3429be40b8243766350b143688ee9f5a`
- `${RESEARCH_WORKSPACE}/error_trajectory_diagnostic/metrics.py`: `5fb91dc1a92635f72271630aa58d86d3f5f5a5578bc3bb2b53dddc2c1828c5d2`
- `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/oracle_audit/runtime.py`: `26d9e97182ea1ffb077d15d5d10508e4d075ea8b36dfd3de24ef8c616ae1a2a6`
- `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/representation_relation/pipeline.py`: `aade64832df7a3fe5431673f0b8349d5c2b1d003c0622a94c892442e0a56c110`
- `${RESEARCH_WORKSPACE}/structure_multicenter_separation/models/CDS_adapter.py`: `c4278e55fe29fb9a0f7b9d4516b8aea037089046605f2927fe45c3ca18c56f87`
- `${RESEARCH_WORKSPACE}/structure_multicenter_separation/data/data.py`: `ddb3f67d7c6f8d491544b31ff86b4e8a3cc1169b7f8c452fead6857689a5f7d0`
- `${RESEARCH_WORKSPACE}/structure_multicenter_separation/data/data_manager.py`: `e350d0b83615a8a34f33d0594d7ee4836b0335c11d79d1ba223b602008e804a1`
