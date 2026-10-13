# Vanilla Structure prototype/covariance oracle audit plan

## 1. 코드 audit 결론

vanilla Structure는 새 class에 대해서만 현재 training sample을 `mode='test'` transform으로 다시 forward하고 class mean/full covariance를 계산한다. 근거는 `models/base.py`의 `_compute_class_mean()` 327–361행이다. mean은 sample feature 평균이고 covariance는 unbiased full covariance에 `1e-3 I`를 더한 값이다.

incremental task 시작 시 기존 `_class_means`, `_class_covs`는 `_task_source_means`, `_task_source_covs`로 복사된다(`models/CDS_adapter.py` 155–163행). feature training 뒤 new-class before/after statistics를 입력으로 `LobachevskyPrototypeCorrector.correct_prototypes()`를 호출하고(176–192행), APR가 켜져 있으면 corrected old mean/covariance를 `_class_means`, `_class_covs`에 덮어쓴다(226–232행). classifier adaptation은 그 다음 240–241행에서 수행되므로 checkpoint의 `class_means/class_covs`는 APR-corrected 값이자 classifier adaptation의 sampling source다.

APR mean correction은 new-class drift `new_after - new_before`를 old/new cosine similarity softmax로 가중합해 old mean에 더한다(`utils/adjust_distribution.py` 64–86행). covariance는 old covariance에 weighted drift outer product와 regularizer를 더한다(88–137행).

task-end checkpoint는 trainer가 evaluation 뒤 `after_task()`를 호출하고 저장한다(`trainer.py` 393–396행). payload에는 network, APR-after `class_means/class_covs`, APR-before `task_source_means/task_source_covs`, new-class before statistics가 모두 포함된다(`models/CDS_adapter.py` 998–1029행). 하지만 기본 구현은 `checkpoint_latest.pt`를 원자적으로 교체하므로(1031–1046행) 완료된 기존 IA5/CAR5에는 최종 task만 남았다.

## 2. 비교 정의

- `mu_before_APR`: checkpoint의 `task_source_means[k]`
- `mu_after_APR = mu_structure`: checkpoint의 `class_means[k]`
- `Sigma_before_APR`: checkpoint의 `task_source_covs[k]`
- `Sigma_after_APR = Sigma_structure`: checkpoint의 `class_covs[k]`
- `mu_oracle(k,t)`: frozen `f_t`로 old train sample을 다시 forward한 평균
- `Sigma_oracle(k,t)`: 같은 feature의 full sample covariance + Structure와 동일한 jitter
- `delta_APR`: `mu_after_APR - mu_before_APR`
- `delta_oracle`: `mu_oracle(k,t) - mu_oracle(k,t-1)`

train split 통계는 `oracle_train_statistics`, test split 통계는 `oracle_test_statistics`로 명시한다. 둘 다 분석 전용이다.

## 3. 누수 방지 구조

학습 실행기 `run_structure.py`는 원래 `StructureLearner`의 training/APR/SCC/classifier adaptation을 override하지 않는다. `AuditSnapshotLearner`가 override하는 것은 checkpoint 저장뿐이며, 정상 task-end 파일을 hard-link/copy해 보존한다.

분석 실행기 `run_audit.py`는 별도 process다. network를 frozen `eval()`로 만들고 모든 parameter의 `requires_grad=False`를 확인하며, feature 추출은 `torch.no_grad()` assertion 안에서만 수행한다. optimizer/backward/model-selection 경로는 없다. 분석 전후 checkpoint size/mtime/edge hash가 같은지도 확인한다.

## 4. 기존 checkpoint 재사용 판정

- IA5: 기존 `B0_STRUCTURE/checkpoint_latest.pt`는 T5 단면 audit에 재사용 가능하다.
- CAR5: 기존 `CAR5/Structure/checkpoint_latest.pt`는 T5 단면 audit에 재사용 가능하다.
- 두 파일 모두 T1–T4를 포함하지 않으므로 temporal drift/age audit 전체를 복원할 수 없다.
- IA10/CAR10은 현재 exact matched vanilla Structure task snapshot series가 없다.

따라서 final-only 단면은 재학습 없이 audit하고, T1…T5/T10 전체 질문을 답할 때만 별도 `training_outputs`에 audit-ready vanilla Structure를 실행한다. 기존 결과는 수정하거나 삭제하지 않는다.

## 5. 산출물

필수 CSV 6개와 class accuracy/ranking 보조 CSV, task-local t-SNE, shared PCA, prototype trajectory, protocol별/통합 Markdown report를 생성한다. `audit_manifest.json`에 temporal coverage와 leakage guard를 기록한다.

## 6. 실행 순서

1. unit/synthetic smoke
2. 기존 IA5 final checkpoint를 class/sample 제한으로 real-model smoke
3. IA5, CAR5 audit-ready run + full audit
4. IA10, CAR10 audit-ready run + full audit
5. combined report에서 dataset 및 5/10-task 비교

full covariance Mahalanobis는 매우 비싸므로 기본 queue는 diagonal Mahalanobis를 사용한다. full covariance Frobenius audit은 항상 수행되며, D²까지 full로 확인할 때 `--covariance-mode both`를 명시한다.
