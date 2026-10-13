# 실험 기록 항목 설명서

모든 run은 `outputs/imageneta/E{0..7}/seed_<seed>/`에 분리된다. CSV accuracy는
반올림하지 않은 0–100 단위이며, 비활성·측정 불가 값은 빈 값으로 저장한다. CSV의
`task_id`, `trained_task`, `evaluated_task`, `epoch`은 사람이 읽는 1-based 번호다.

## 재현성과 상태

- `run_metadata.json`: 전체 resolved config, git commit, tracked diff와 untracked 파일을
  포함해 계산한 `working_tree_sha256`, seed, class order/task 구성, split hash, UTC
  시작·갱신·종료 시각, 완료 task 수, 실행 상태와 실패 traceback을 저장한다. epoch와
  task 종료 때 갱신된다.
- `checkpoint_latest.pt`: 최신 model/class 및 local statistics/optimizer/scheduler/RNG/APR
  source state이다. 원자적으로 교체되며 과거 epoch checkpoint를 뜻하지 않는다.
- `checkpoint_manifest.json`: 현재 latest checkpoint의 `checkpoint_state_id`, task,
  stage, next epoch을 저장한다. `epoch_metrics.csv`와 `task_summary.csv`가 같은 ID로
  학습 상태를 연결한다.
- `protocol_manifest.json`, `class_order.json`: 5-task class order/remapping, class별
  표본 수, split 내용 hash와 교차 중복 수이다.

## K-means와 분포

- `cluster_summary.csv`: K-means 실행/class별 requested K, actual K, 전체 표본 수,
  cluster count/ratio 목록, K 축소/fallback 이유, feature-space와 통계 checkpoint key.
- `cluster_details.csv`: 각 cluster의 sample 수/비율, covariance rank/fallback,
  covariance view별 birth tau와 `local_stats[class][cluster]` key.
- `cluster_assignments.csv`: train root 기준 상대 `sample_id`와 remapped/original class,
  cluster ID의 대응. 실제 feature vector는 저장하지 않으며 이후 학습에서 읽지 않는다.
- `region_statistics_task_<task>.json`: mean/covariance 자체를 제외한 region 요약,
  K-means 시도 정보, mixture와 class-level moment 차이.
- `apr_diagnostics.csv`: task-final APR의 old class별 mean shift L2와 covariance 변화
  Frobenius norm. APR drift 크기 및 이상치를 비교하기 위해 task 2 이후 기록한다.

Cluster ID는 `clustering_run_id` 내부에서만 유효하다. 이 구현은 K를 최적화하지 않고
config의 K에서 시작해 표본/unique feature/최소 cluster 크기 제약이나 invalid cluster가
있을 때 K를 줄여 처음부터 다시 실행한다. 병합하거나 표본을 제거하지 않는다.

## Loss와 separation

- `epoch_metrics.csv`: `(experiment_id, seed, task_id, stage, epoch)` key로 upsert한다.
  resume가 같은 epoch를 다시 기록하면 기존 행을 교체한다.
- `stage=feature_training`: classification/SCC/feature 또는 logit distillation/separation의
  active 여부, raw 값, 계수, weighted 값, original Structure loss와 total loss를 기록한다.
- `stage=classifier_retraining`: synthetic replay classifier CE만 활성 loss로 기록한다.
- epoch loss와 feature norm은 각 batch mean에 실제 batch sample 수를 곱한 후 전체 처리
  sample 수로 나눈 값이다. 마지막 작은 batch를 동일 가중치로 평균하지 않는다.
- `learning_rate(s)`는 해당 epoch optimizer step에 사용된 시작 LR, 시간은 cache 갱신을
  포함하고 epoch 평가를 제외한 초, `optimizer_steps`는 실제 update 수다.
- separation의 `intrusion_ratio`는 `min_j rho_j(z)<1`인 sample 비율이며
  `nearest_old_rho_*`는 normalized squared Mahalanobis distance의 표본 통계다.
- `nan_or_inf_detected`와 factorization jitter/eigen clipping 횟수인
  `numerical_fallback_count`는 수치 실패 원인 추적용이다. non-finite total loss는 즉시
  실패시키고 `run_metadata.json`에 원인을 남긴다.
- `training_diagnostics.jsonl`, `separation_cache.jsonl`은 상세 호환 로그다. 공식 비교
  집계는 중복 방지 key가 있는 `epoch_metrics.csv`를 사용한다.

## CIL 평가

- `task_accuracy.csv`: trained/evaluated task별 correct/sample/accuracy long format.
  test sample을 task별로 나누지만 예측은 현재까지 학습한 모든 class의 single head다.
- `task_accuracy_matrix.csv|json`: 위 long-format과 같은 값을 5×5 lower-triangular 형태로
  저장하며 미래 task cell은 빈 값이다.
- `task_summary.csv`: task-final seen/old/new 및 top-5의 correct/sample/accuracy, old→new와
  new→old error, 시간과 checkpoint state.
- `class_accuracy.csv`: task-final remapped/original class와 이름별 correct/sample/accuracy.
- `task_metrics.csv|json`, `summary.json`: A1–A5, Inc-Acc, Last-Acc, forgetting, 시간,
  peak memory, 영구 statistics 및 factor-cache 용량. `Inc-Acc=mean(A1..A5)`이고
  forgetting은 final 이전 task에 대해
  `mean_i(max_{l=i..T-1} a[l,i] - a[T,i])`이다.

추가 평가 전후 Python/NumPy/Torch/CUDA RNG와 model train/eval 상태를 복구한다. Test
결과는 학습 cache나 hyperparameter 선택에 사용하지 않는다.

## E0–E7 재집계

`scripts/aggregate_results.py`는 모델/데이터를 로드하지 않고 저장된 CSV/JSON만 읽는다.
모든 E0–E7에서 완료되고 동일 protocol signature인 공통 seed만 비교 평균에 포함한다.

- `aggregate/comparison_per_seed.csv`: 공통 seed의 experiment별 A1–A5와 원자료.
- `aggregate/comparison_mean_std.csv`: experiment별 평균/표준편차/유효 seed 수.
- `aggregate/loss_per_epoch.csv`, `loss_mean_std.csv`: stage·loss 이름을 섞지 않은 loss 자료.
- `aggregate/task_summary_per_seed.csv`, `task_summary_mean_std.csv`: task별
  seen/old/new 정확도의 seed 원자료와 평균/표준편차.
- `aggregate/excluded_runs.csv`: incomplete/failed/protocol mismatch와 이유.
- `aggregate/comparison_report.md`: 결과표, task accuracy matrices, 누락/실패 상태.
- `aggregate/figures/`: task accuracy 및 stage별 loss 곡선 PNG.

새 기록 형식 도입 전에 생성된 loss·correct count는 소급 추정하지 않는다. 기존
checkpoint로 나중에 평가한 값은 별도 `eval_stage`의 재평가 값으로만 기록해야 한다.
