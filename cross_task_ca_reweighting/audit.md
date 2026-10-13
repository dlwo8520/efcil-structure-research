# Cross-Task CA Reweighting Pilot — audit

- 새 backbone 학습, SCC/APR 변경, test 기반 coefficient tuning 없음.
- Control/Intervention은 동일 T10 pre-CA head, 동일 epoch별 Gaussian bank, 동일 label/permutation/minibatch 순서, 동일 SGD·scheduler·5 epochs·256/class를 사용.
- Gaussian bank는 원본 pre-CA RNG 상태에서 생성하며 epoch마다 다시 샘플링. 같은 bank를 두 head에 lockstep으로 공급하고 SHA-256만 저장한다.
- Inference는 모든 seen class에 대한 argmax; task ID와 후보 필터는 사용하지 않음.
- 실행 physical GPU: `4`

## IA10

- Source experiment: `Structure_AUDIT`; seed `1993`
- Training-recorded git commit: `None`; source snapshot hash: `ea63875fe9a103c250b6d2e926372a15c919a82b5304390e31f0bda06b1c7c3c`
- Task partition: `[20, 20, 20, 20, 20, 20, 20, 20, 20, 20]`
- Fixed B1 ID count: `126` (expected `126`)
- CE identity max abs: synthetic `4.7683716e-07`, first real batch `6.2584877e-07`
- Optimizer steps: `{'control': 1000, 'intervention': 1000}`
- Original CA task-size argument for Gaussian mean scaling: `20`; loss G(y) follows the true task partition
- Cached final-feature vs trajectory baseline prediction mismatches: `0`
- Matched Control − cached vanilla accuracy: `-0.1317 pp`
- Original training T10 top-1: `982` correct, `64.6478%`; cached vanilla − original: `+0.0658 pp`
- Matched Control vs original final head max abs weight diff: `0.011874177`

| Source | SHA-256 |
|---|---|
| `checkpoint`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/task_checkpoints/task_10.pt` | `fa0b5a20e3e07229e356cfcf576037c418abc5e8393d5d99a3dd47b6b347235c` |
| `sidecar`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/pre_ca_classifier/task_10.pt` | `6d660b9582be810d32e4fe241ce9cb620cd782179f42420041784f7053a881f9` |
| `training_config`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/resolved_config.json` | `000e052c4388185714a556c33e71c3bec6f125d572ca61f24e85b1f4764a8890` |
| `protocol_manifest`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/protocol_manifest.json` | `299bf492452001f91d237de6a7883eb03681de63758406a5fe11c9bbaf0a5f4e` |
| `class_order`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/class_order.json` | `6befb5846bc3f428875da06bb716ad366fca6e71c3c05f57d3477cdf953c2c3d` |
| `task_summary`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/task_summary.csv` | `0e9bc01be0e8583e0baa3942653d4848a4154bb3948c7bcc9b90b6d80dc2c6d3` |
| `snapshot_manifest`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/IA10/Structure_AUDIT/seed_1993/task_checkpoints/manifest.json` | `3f5080c50ca54eb82e47c31884c5f33fdee33284c96a675f886913a686978d7f` |
| `feature_cache`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/IA10/.cache/model_10_on_task_10.npz` | `583298bdbbd7375727f66e95b17098e679337a2c638b22638125b9c39d4d72ff` |
| `trajectory`: `${RESEARCH_WORKSPACE}/error_trajectory_diagnostic/IA10/sample_trajectory.csv` | `5840aa869d70941b697eed4ac7bf869d056e7b02b8ee959e2d6b9d2df3ad144a` |
| `trajectory_manifest`: `${RESEARCH_WORKSPACE}/error_trajectory_diagnostic/run_manifest.json` | `ccd3bed29fbba909ac4482d4cfee54b3040b5a47481b7dff3cc12ee8729d29dc` |

Epoch bank hashes are in `IA10/bank_manifest.json`.

## CAR10

- Source experiment: `Structure_AUDIT`; seed `1993`
- Training-recorded git commit: `None`; source snapshot hash: `ea63875fe9a103c250b6d2e926372a15c919a82b5304390e31f0bda06b1c7c3c`
- Task partition: `[20, 20, 20, 20, 20, 20, 19, 19, 19, 19]`
- Fixed B1 ID count: `473` (expected `473`)
- CE identity max abs: synthetic `0`, first real batch `6.2584877e-07`
- Optimizer steps: `{'control': 980, 'intervention': 980}`
- Original CA task-size argument for Gaussian mean scaling: `19`; loss G(y) follows the true task partition
- Cached final-feature vs trajectory baseline prediction mismatches: `0`
- Matched Control − cached vanilla accuracy: `-0.1990 pp`
- Original training T10 top-1: `6349` correct, `78.9578%`; cached vanilla − original: `+0.0000 pp`
- Matched Control vs original final head max abs weight diff: `0.0069486145`

| Source | SHA-256 |
|---|---|
| `checkpoint`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/task_checkpoints/task_10.pt` | `623382eb9fb5a7f06c4512f57754e9bbf99e9814fd7136b2bf6463adbd37a142` |
| `sidecar`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/pre_ca_classifier/task_10.pt` | `cd00b0acfd9f9fc558c2a1f199fd776340d285ecea25a9ab647b2c060b0cfbd5` |
| `training_config`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/resolved_config.json` | `27be7cbe033bdf5688ebe621b5c78ffb6ef75032c1fd66e16aada1887a52d4dd` |
| `protocol_manifest`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/protocol_manifest.json` | `c66e9e9b07aa84825cd193692395f3e5e58e3c150be35bbc58ba6a59d7c912c0` |
| `class_order`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/class_order.json` | `18c16ea49c25afe12204ee766c8a2f511df3e049040acb8b87db0931e9d164c0` |
| `task_summary`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/task_summary.csv` | `58eb990edacb3181cc6d8791ba3df96d73cd5f93f28269f53403080c393f41ea` |
| `snapshot_manifest`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/training_outputs/CAR10/Structure_AUDIT/seed_1993/task_checkpoints/manifest.json` | `909f46faa0302dbf65c8b248ac46bdc683698e82739f04b9d1a9f2e1be923647` |
| `feature_cache`: `${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit/ten_task_hypothesis_revalidation/CAR10/.cache/model_10_on_task_10.npz` | `58557fe308a67817eb66eeb01d39214d1dc50be83c07b06a397e5bba40636669` |
| `trajectory`: `${RESEARCH_WORKSPACE}/error_trajectory_diagnostic/CAR10/sample_trajectory.csv` | `d25f1a558007ab07c7838fdef9b789b84983bffe77b18064c61703a2bd03a184` |
| `trajectory_manifest`: `${RESEARCH_WORKSPACE}/error_trajectory_diagnostic/run_manifest.json` | `ccd3bed29fbba909ac4482d4cfee54b3040b5a47481b7dff3cc12ee8729d29dc` |

Epoch bank hashes are in `CAR10/bank_manifest.json`.
