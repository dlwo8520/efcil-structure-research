# Representation / Class-Relation Diagnostic 구현 계획

> 대상: IA5 T5, CAR5 T5 / seed 1993  
> 추가 checkpoint: 같은 protocol의 T4  
> 실행 정책: 코드를 작성하고 synthetic smoke test까지만 수행하며, 실제 GPU 실험은 사용자 실행  
> 범위: oracle diagnostic only; 새로운 CIL method나 loss를 만들지 않음

## 1. 기존 T4/T5 checkpoint 위치

- IA5 T4: `training_outputs/IA5/Structure_AUDIT/seed_1993/task_checkpoints/task_04.pt`
- IA5 T5: `training_outputs/IA5/Structure_AUDIT/seed_1993/task_checkpoints/task_05.pt`
- CAR5 T4: `training_outputs/CAR5/Structure_AUDIT/seed_1993/task_checkpoints/task_04.pt`
- CAR5 T5: `training_outputs/CAR5/Structure_AUDIT/seed_1993/task_checkpoints/task_05.pt`

실행 전후 네 checkpoint의 size와 edge SHA-256 fingerprint를 비교한다.

## 2. Train / validation / test split 위치

원본 train/test split은 protocol별 dataset root와 `SweepDataManager`가 제공한다.

- IA5: `DEFAULT_DATA_ROOTS["IA5"]`
- CAR5: `DEFAULT_DATA_ROOTS["CAR5"]`
- protocol/class order: `oracle_audit.config.protocol_config()`와 `selective.protocol.SweepDataManager`

T5 train/test feature cache:

- `classifier_bottleneck_diagnostic/{IA5,CAR5}/seed_1993/.cache/task_05_features.npz`

Head validation은 replacement sampling 전에 원본 train image index를 class별로 나눈다. 기존 procedure diagnostic과 같은 정책을 재사용한다.

- class sample 수가 5 이상: deterministic 20% validation, 최소 2개 train 유지
- 5 미만: validation에 넣지 않고 모두 train 유지
- seed: `1993 + 5300`

validation image는 diagnostic head training과 k-NN reference에서 제외한다. 다만 해당 image는 과거 Structure backbone/adapter의 CIL train split에 포함됐으므로 representation 자체에 대해 완전히 unseen인 데이터는 아니다. 이 사실을 report에 명시한다.

## 3. Frozen feature 추출 경로

`oracle_audit.runtime.load_frozen_learner()`로 checkpoint를 strict load하고 eval/frozen 상태를 확인한다. `torch.no_grad()`에서 network output의 raw `sfeatures`를 수집하고 diagnostic 입력에서는 L2-normalize한다.

- T5: 검증된 기존 full train/test cache 재사용
- T4: T5 전체 class set을 T4 frozen checkpoint에 forward하여 새 diagnostic cache 생성

기존 T4 cache는 T1–T4 class만 포함하므로 Diagnostic C에는 사용할 수 없다. T5 신규 class image를 T4에 forward하는 것은 diagnostic-only이며 T4 training/update에는 사용하지 않는다.

## 4. 동일 sample 및 ID 검증

`DataManager.get_dataset(..., ret_data=True)`가 반환하는 class-concatenated 순서를 사용한다. sample ID는 다음으로 만든다.

- dataset-local index
- source split(train/test)
- 원본 path 문자열 또는 in-memory image bytes의 SHA-256 prefix

T4 추출 label sequence와 T5 cache label sequence가 완전히 같은지 assert한다. 같은 source sample hash/order를 T4/T5에 공통으로 사용한다.

## 5. 기존 Structure prediction/logit 재사용

T5 Structure post-CA weight는 T5 checkpoint에서 직접 읽는다. T5 cached test feature와 동일 scale-20 cosine 식으로 logits/prediction을 재구성한다. 기존 classifier diagnostic과 같은 score path이므로 별도 model forward 없이 pair confusion을 계산할 수 있다.

T4 original classifier는 T5 신규 class head가 없어 후보 class set이 다르므로 T4/T5 비교에 사용하지 않는다. Diagnostic C에서는 양 representation에 동일한 200/196-class diagnostic cosine probe를 새로 학습한다.

## 6. 기존 linear probe 구현과 설정

기존 코드는 `classifier_procedure/pipeline.py::_train_linear()`이다.

- objective: inverse-class-frequency weighted mean CE + `0.5 * lambda * sum(W^2)`
- current lambda: `5e-4`
- optimizer: PyTorch full-batch L-BFGS
- lr 1.0, strong-Wolfe line search
- max_iter 100, max_eval 200
- tolerance_grad `1e-5`, tolerance_change `1e-9`, history_size 20
- feature: L2-normalized actual feature
- split: 원본 image 단위 deterministic train/validation split
- initialization: 기존에는 raw pre-CA cosine weight, bias는 0

이 objective에서 scale-20 converted cosine weight는 큰 L2 penalty를 받을 수 있으므로 기존 linear 결과는 regularization artifact 가능성을 배제하지 못한다.

## 7. Cosine → linear exact conversion

T5 Structure classifier의 class weight `w_k`를 다음과 같이 변환한다.

`v_k = 20 * w_k / ||w_k||`

normalized feature `u`에 대해:

`20 * cosine(u, w_k) = v_k^T u`

전체 T5 test feature에서 max/mean absolute logit difference와 prediction mismatch를 계산한다. mismatch가 0이 아니거나 max difference가 `1e-5`를 넘으면 FAIL하고 이후 linear-vs-cosine 해석을 중단한다.

추가로 재구성한 Structure test accuracy를 기존 `gaussian_vs_actual_classifier_accuracy.csv`의 T5 `STRUCTURE_CA` summary와 비교한다. 차이가 `1e-8`을 넘으면 feature/cache/head mapping regression으로 FAIL한다.

## 8. Linear regularization sanity check

대규모 sweep 없이 사전 고정한 세 설정만 사용한다.

- current: `lambda=5e-4`
- intermediate: `lambda=5e-5`
- weak positive: `lambda=5e-6`

모두 동일 normalized actual train feature, inverse-frequency class weighting, bias 없음, converted cosine initialization을 사용한다. L-BFGS max_iter 200/max_eval 400, tolerance_grad `1e-7`로 solver 문제를 줄인다.

선택 규칙은 validation class-macro accuracy 우선, micro accuracy 차순, 더 강한 regularization 차순이다. Test는 선택에 사용하지 않는다. 각 lambda에서 converted cosine solution의 동일 objective도 계산한다.

## 9. Directional pair confusion

T5 Structure prediction에서 `true i -> predicted j`를 방향성 있게 집계한다.

- `P(pred=j | true=i) = count(i->j) / test_count(i)`
- class-i error share = `count(i->j) / all_error_count(i)`
- global error share = `count(i->j) / all_test_errors`
- pair type: old-old / old-new / new-old / new-new
- task ID: partition 누적 경계로 계산

Focused set은 global contribution top 20과 conditional confusion top 20의 union이다. 동률은 confusion count, class ID 순으로 고정한다. Test에서 고른 pair는 hypothesis-generation 용도로만 사용한다.

## 10. Pair margin

Focused pair `i -> j`에서 `m_ij(x)=s_i(x)-s_j(x)`를 계산한다. Structure T5 score로 원본 train, held-out validation, test를 각각 요약한다.

저장 통계:

- mean, median, std
- negative fraction
- q01, q05, q10, q25, q50

Confusion count는 final argmax가 j인 경우만 의미하지만 negative margin은 j가 i보다 높은 모든 경우이므로 두 metric을 분리한다.

## 11. k-NN reference와 k 선택

Reference bank는 validation을 제외한 normalized original train feature다. Replacement sample은 사용하지 않는다.

- distance: cosine distance `1 - cosine_similarity`
- 후보 k: `{1, 3, 5, 9}`
- 선택: validation class-macro accuracy 우선, micro accuracy 차순, 작은 k 차순
- test는 k 선택에 사용하지 않음
- voting: class count, tie 시 neighbor cosine 합, 다시 tie면 작은 class ID

Sample-count group은 reference train count의 class-level quartile(q25/q50/q75)로 very-low/low/medium/high를 정의한다. Micro와 macro accuracy, old/new, count-group accuracy를 모두 저장한다.

## 12. Neighborhood purity

Query의 true label과 같은 neighbor 비율을 purity로 정의한다.

`purity(x) = count(neighbor_label == y) / k`

sample-level group:

- B1: cosine wrong, k-NN correct, purity ≥ 0.8
- B2: cosine wrong, k-NN도 같은 wrong class, purity ≤ 0.4
- B3: 해당 class의 leave-one-out train purity ≥ 0.8, held-out purity ≤ 0.4
- 그 외: OTHER

Train-query 분석에서는 동일 `original_sample_id`인 모든 reference entry를 similarity `-inf`로 mask한다. 이 동작을 unit test로 고정한다.

## 13. T4/T5 동일 조건 비교

T4/T5 모두 동일 T5 class set, 동일 original train/validation/test index를 사용한다.

각 representation에서:

1. train split class mean으로 200/196-class cosine head 초기화
2. class당 256 replacement-balanced actual feature
3. scale 20, SGD lr 0.01/momentum 0.9/weight decay 0.0005
4. 5 epoch, 동일 label permutation 및 optimizer-step
5. 동일 selected k(T5 validation에서 선택)를 적용한 k-NN

Focused pair마다 diagnostic probe confusion/margin과 k-NN purity/error를 T4/T5에서 계산한다. Original T4 head와 T5 head는 후보 class 수가 달라 직접 비교하지 않는다.

## 14. Leakage 방지

- backbone/adapter는 frozen eval이며 gradient를 만들지 않음
- actual old/new feature는 diagnostic classifier와 k-NN reference에서만 사용
- APR/SCC/main checkpoint/next task에 전달하지 않음
- T4가 보지 못한 T5 class image는 forward-only oracle diagnostic
- validation은 diagnostic head/k 선택에만 사용
- test는 pair hypothesis 생성 및 최종 기술 통계에만 사용
- test-selected pair 결과를 method selection 성능으로 표현하지 않음
- 실행 전후 T4/T5 checkpoint fingerprint 확인

## 15. 산출물

Output root: `representation_relation_diagnostic/`

Protocol directory: `IA5_T5/`, `CAR5_T5/`

- `cosine_linear_equivalence.json`
- `linear_probe_training_audit.md`
- `linear_regularization_check.csv`
- `validation_split_audit.csv`
- `directional_pair_confusion.csv`
- `pair_margin_distribution.csv`
- `top_pair_analysis.csv`
- `knn_neighborhood_diagnostic.csv`
- `knn_summary.csv`
- `pair_knn_recovery.csv`
- `pair_task_transition_analysis.csv`
- `run_manifest.json`

Plots:

- `pair_error_contribution/`
- `pair_margin/`
- `neighborhood_purity/`
- `knn_vs_cosine/`
- `task_transition/`

Reports:

- `{protocol}_representation_relation_report.md`
- `representation_relation_diagnostic_final_report.md`

기존 variant logits는 exact sample-ID mapping이 검증된 파일이 있을 때만 optional 분석한다. 없으면 `variant_pair_error_transfer.csv`를 억지로 생성하지 않는다.

## 16. 예상 runtime

- T5 cache 재사용: 즉시
- T4 full T5-class frozen extraction: protocol당 약 1–4분
- linear 3개 설정: protocol당 약 1–5분
- k-NN 및 pair 분석: protocol당 약 1–3분
- T4/T5 diagnostic cosine: protocol당 약 1–3분

GPU 2 기준 전체 예상 runtime은 약 8–30분이다. 실제 실행은 사용자 명령으로만 시작한다.
