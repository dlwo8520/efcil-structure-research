# 공식 소스 및 기존 Structure 경로 점검

## 고정 입력

- IA10/CAR10, seed 1993, final task T10.
- 기존 `structure_prototype_oracle_audit/training_outputs/{IA10,CAR10}/Structure_AUDIT/seed_1993/task_checkpoints/task_10.pt`는 읽기 전용.
- final frozen train/test feature는 `ten_task_hypothesis_revalidation/{IA10,CAR10}/.cache/model_10_on_task_10.npz`에서 재사용한다. 재추출/재학습 없음.
- `oracle_audit.oracle_head.checkpoint_classifier_weights`와 `cosine_scores`가 F0와 boundary audit 고정 Structure cosine head를 제공한다.
- `oracle_audit.oracle_head.load_classifier_initialization`의 `exact_pre_ca_sidecar`가 CEOS C1 초기화를 제공한다.
- `classifier_diagnostic.pipeline._fixed_gaussian_bank`, `_assemble_bank`, `classifier_procedure.pipeline._make_epoch_permutations`, `_train_cosine`를 재사용한다. 원본 코드는 수정하지 않는다.

## FeCAM 공식 코드

- 공식 repo: <https://github.com/dipamgoswami/FeCAM>, 점검한 commit `e33f39d112ff2d2a2df2e68c490af579a50edd31`.
- 논문: <https://arxiv.org/abs/2309.14062>.
- `models/base.py`: `_mahalanobis`는 L2 정규화된 query와 mean의 차이에 class별 covariance의 pseudo-inverse를 적용한 이차 거리. `_maha_dist`는 `per_class` + `full_cov` + `norm_cov` 경로에서 class covariance 사용.
- `models/base.py`: `shrink_cov`는 `cov + alpha1*diag_mean*I + alpha2*off_diag_mean*(1-I)`. `normalize_cov`는 shrunk covariance를 축별 표준편차 곱으로 나눈다. `torch.linalg.pinv` 사용.
- `FeCAM_vit_imagenet-r.py`: pretrained ViT 예제의 `alpha1=alpha2=10`, feature를 L2 정규화한 뒤 class covariance를 계산, Tukey 사용 안 함. `np.linalg.pinv` 사용.
- `utils/maha_utils.py`: sample covariance는 `np.cov(vectors.T)`.
- `NCM_vit_imagenet-r.py`: sample별 L2 정규화 feature를 평균한 class mean(평균을 다시 정규화하지 않음)에 Euclidean distance.

### 이식상 불가피한 차이

Structure checkpoint의 old class moment는 정규화 *전* final feature에서 저장되었다. `mean(normalize(z))` 및 `cov(normalize(z))`는 이들 moment만으로 정확히 복원할 수 없다. F2/F3 비교의 통계 정의를 맞추기 위해 둘 다 원 feature의 mean/covariance를 사용하고, Oracle covariance에도 Structure의 `stats_cov_jitter`를 더한다. 공식 `models/base.py`처럼 query/mean을 score 단계에서 정규화한다. 따라서 공식 pretrained-ViT script 완전 재현으로 해석하면 안 된다. Tukey는 사용하지 않는다. Full-rank shrunk covariance에서는 Cholesky solve를 사용하고 실패 시 공식 pinv로 fallback한다.

## CEOS 공식 코드

- 공식 repo: <https://github.com/HXuSz11/ACB_CEOS_CVPR2026_Findings>, 점검한 commit `b1d7743b17729438ff8e81d98b6046fcb1f516f0`.
- 논문: <https://openaccess.thecvf.com/content/CVPR2026F/html/Xu_Revisiting_Prototype_Rehearsal_for_Exemplar-Free_Continual_Learning_Manifold-Aware_Boundary_Sampling_CVPRF_2026_paper.html>.
- `continual_learning/CEOS.py::expansive_oversampling`: `torch.cdist(proto_aug, batch_features,p=2)` 후 label이 다른 nearest enemy를 고른다. `k=1` 기본. `lambda=0.7+0.3*Uniform(0,1)`, `mixed=lambda*proto_aug+(1-lambda)*enemy`. 기본 `F.normalize(mixed,p=2,dim=1)`.
- `continual_learning/utils/proto.py::ProtoGenerator.perturbe`: `proto_aug`는 class mean 자체가 아니라 Gaussian으로 perturb한 old prototype sample이다. 따라서 C1에서도 C0와 동일한 stored Gaussian old sample을 CEOS 입력으로 사용한다.
- ACB/focal/class weighting은 CEOS 함수와 독립이므로 이번 실험에서 사용하지 않는다.
- 공식 구현은 current minibatch feature를 enemy 후보로 사용한다. 이번 offline classifier-only 실험은 final Structure feature cache의 **전체 current-task actual train pool**을 고정 후보로 쓴다. 이는 가장 가까운 enemy의 범위를 넓히는 변경이며 원 논문 그대로의 reproduction은 아니다.

## 안전성

기존 checkpoint/cache/result에 쓰지 않는다. 새 출력은 `fecam_boundary_followup/` 아래만 생성한다. Oracle old feature는 연구 진단 전용이며 deployable EF-CIL 방법이 아니다.
