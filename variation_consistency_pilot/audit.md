# Current-task variation consistency pilot — code audit

## 기존 경로

- 원본: `structure_multicenter_separation/models/CDS_adapter.py::Learner.incremental_train`, `_train`, `_init_train`; `models/base.py::_compute_class_mean`, `_stage2_compact_classifier`; `oracle_audit/snapshot_learner.py::AuditSnapshotLearner`.
- T8 task-end checkpoint: `structure_prototype_oracle_audit/training_outputs/{IA10,CAR10}/Structure_AUDIT/seed_1993/task_checkpoints/task_08.pt`. 둘 다 존재하며 `cur_task=7`, `stage=task_end`.
- checkpoint config: `scc=True`, `apr=True`, `ca=True`, `old_new_separation=False`, `inc_epochs=15`, `batch_size=48`, `ca_epochs=5`, `replay_samples_per_class=256`, `optimizer=sgd`.
- 원래 train dataset의 `DummyDataset.__getitem__`가 기존 train transform을 한 번 sampling한다. `PairedStructureViews`는 같은 index를 두 번 읽어 독립 stochastic view를 만든다. 원래 transform은 `RandomResizedCrop(224, scale=(0.05,1.0), ratio=(3/4,4/3))`, `RandomHorizontalFlip(p=0.5)`, `ToTensor()`이다. 새 augmentation을 추가하지 않는다.
- 원래 representation loss는 current-class CosFace classification + `scc_value * LobachevskyRelativeGeomLoss`. 새 두-view control은 두 view의 각 항을 평균하므로 gradient scale이 단순 두 배가 되지 않는다.
- 원래 `LobachevskyRelativeGeomLoss`는 한 forward의 student/teacher feature와 저장 old class mean을 사용한다. 각 view에 동일하게 적용하고 평균한다.
- 원래 APR은 representation training **후** `_compute_class_mean`으로 current 통계를 한 번 갱신하고 `correct_prototypes`를 한 번 호출한다. paired view로 통계/APR 호출 수를 늘리지 않는다.
- 원래 Gaussian CA는 `AuditSnapshotLearner`를 통해 기존 `_stage2_compact_classifier`를 호출한다. sampling/optimizer/LR/epoch/step 코드를 수정하지 않는다.
- T8 task-end snapshot의 `convnet.flag`는 state_dict에 저장되지 않으므로 로드 후 `cur_task+1`로 복원한다. 그렇지 않으면 student/teacher adapter가 뒤바뀐다.

## Phase A artifact 점검

- 기존 C0 paired test logits: `structure_prototype_oracle_audit/paired_source_prediction_diagnostic/{IA10,CAR10}/matched_scores.npz`의 `scores_C`.
- 이전 C1 결과에는 scalar metric만 저장되었고 head weight/logit은 없었다. 사용자의 명시적 승인에 따라 **동일 C1 한 번만** 재실행하여 logits을 복원했다. 저장된 C1 accuracy·train loss와 일치해야 분석을 진행하도록 assert했다.
- 신규 학습 조건이나 CEOS tuning은 없다. 비교는 동일 canonical New test sample ID에서만 수행한다.

## Branch 동등성 검사

- Control/JS는 같은 T8 hash, task sample order, 양쪽 view pixel SHA-256, epoch 수, batch/step 수, stats/APR/CA 호출 수, CA exposure를 비교한다.
- consistency candidate set은 원래 current-class CosFace loss와 동일한 `logits[:, known:total]` (`current-only`). T=1, lambda=0.1. 원래 unscaled cosine logits에 JS를 계산한다.
- Test label은 lambda/augmentation/epoch 선택에 사용하지 않는다. One-view 기존 Structure는 참고 자료이며 인과 비교는 two-view Control 대 JS이다.
- 기존 checkpoint/result는 읽기 전용이다. 새 산출물은 `variation_consistency_pilot/` 아래에만 기록한다.
- `structure_prototype_oracle_audit` 폴더는 Git repository가 아니므로 해당 경로의 commit은 manifest에 `null`로 남긴다. 원본 `structure_multicenter_separation` Git commit과 T8 checkpoint SHA-256을 별도로 기록해 실행 출처를 식별한다.
