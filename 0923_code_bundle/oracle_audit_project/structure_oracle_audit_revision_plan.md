# Structure Oracle Audit revision plan

## 목적과 불변 조건

기존 prototype/drift/covariance/Mahalanobis/forgetting/t-SNE/shared-PCA/trajectory audit은 유지한다. 이번 revision은 저장된 task checkpoint를 읽는 post-hoc 분석에만 APR ablation, oracle classifier replacement, score-competition 진단을 추가한다. Structure feature training, APR, SCC, classifier adaptation의 본 실행 경로와 old-data 접근 조건은 바꾸지 않는다.

old train sample로 만든 oracle statistics는 `run_audit.py` process 안에서만 존재한다. diagnostic head 학습 입력은 원본 이미지가 아니라 class mean/covariance로 생성한 synthetic feature이며, backbone/adapter object를 optimizer에 전달하지 않는다. diagnostic head와 oracle statistics는 task가 끝나면 폐기되고 어떤 checkpoint에도 training state로 저장하지 않는다.

## 수정 파일

| 파일 | 수정 내용 |
|---|---|
| `oracle_audit/snapshot_learner.py` | 원래 CA 호출 직전 classifier state와 RNG를 compact sidecar로 보존. 정상 CA 호출은 그대로 `super()`에 위임 |
| `oracle_audit/oracle_head.py` | A/B/C/D statistics 조합, 동일 초기 head/RNG의 synthetic-feature CA, frozen-feature 평가 |
| `oracle_audit/diagnostics.py` | new-only/old-only, margin, boundary-direction, Gaussian moment 진단 |
| `oracle_audit/pipeline.py` | 기존 audit과 새 진단 통합, all/final head task routing, trace delta 및 새 CSV 출력 |
| `oracle_audit/metrics.py` | covariance trace/effective-rank, skew/kurtosis 보조 metric |
| `oracle_audit/visualize.py` | pre-APR triangle, APR X, oracle star와 APR improved/worsened focused PCA |
| `oracle_audit/report.py` | oracle replacement 기반 Case A–F 및 추가 Q1–Q10 자동 판정 |
| `run_audit.py`, `run_queue.py` | oracle-head 범위/비용 옵션 전달 |
| `tests/` | 공정성, leakage, competition, 기존 metric regression test |

## classifier replacement 구현 경로

1. task-end checkpoint에서 frozen feature extractor와 post-CA classifier를 읽는다.
2. future audit-ready run은 `pre_ca_classifier/task_XX.pt` sidecar의 CA 직전 head와 RNG를 사용한다. 이 sidecar가 있으면 Condition A가 원래 Structure CA를 재현하는지 weight/accuracy 차이도 기록한다.
3. legacy checkpoint는 pre-CA state가 없으므로 모든 조건이 동일한 post-CA checkpoint head에서 시작한다. 결과에 `shared_post_ca_legacy_init`라고 표시하며 원래 CA의 exact replay로 주장하지 않는다.
4. current class mean/covariance는 네 조건에서 항상 Structure 값으로 고정한다.
5. old statistics만 다음처럼 교체한다.

   - A Structure: APR mean + Structure covariance
   - B Oracle Mean: oracle mean + Structure covariance
   - C Oracle Covariance: APR mean + oracle covariance
   - D Full Oracle: oracle mean + oracle covariance

6. 네 조건은 같은 initial weights, RNG state, sample count, epoch, SGD/momentum/weight decay, cosine scheduler, scale, mean scaling, batch order를 사용한다.
7. test feature는 한 번만 frozen extractor로 계산하고 각 diagnostic head로 평가한다. head는 평가 후 폐기한다.

## 추가 CSV

- `apr_ablation_audit.csv`: pre-APR/APR/oracle error, improved/unchanged/worsened, one-step/cumulative error
- `oracle_classifier_replacement.csv`: checkpoint reference와 A/B/C/D의 seen/old/new accuracy 및 confusion, A 대비 delta
- `new_class_competition_diagnostics.csv`: single-head/new-only/old-only와 error decomposition
- `boundary_direction_error.csv`: actual checkpoint head의 strongest competitor 방향 prototype error
- `margin_diagnostics.csv`: new sample의 true-new score, strongest-old score, margin, outcome
- `gaussian_assumption_diagnostics.csv`: oracle feature skew/kurtosis 및 Mahalanobis moment summary
- 보강된 `covariance_oracle_audit.csv`: trace ratio, effective rank, Structure/oracle trace delta

## leakage 보장

- oracle sample forward는 frozen `eval()` + `torch.no_grad()` assertion 안에서만 수행한다.
- diagnostic CA 함수는 raw sample/data loader/backbone/adapter를 인자로 받지 않는다.
- optimizer parameter ID가 diagnostic head parameter ID 집합과 정확히 같은지 assertion한다.
- 입력 mean/covariance/initial weight의 checksum을 전후 비교한다.
- 원본 checkpoint fingerprint를 audit 전후 비교한다.
- sidecar 저장은 old sample을 사용하지 않고 정상 CA 직전 state/RNG를 관찰만 한다.
- diagnostic 결과는 `outputs/`에만 기록되고 `training_outputs/.../checkpoint*.pt`에 쓰지 않는다.

## 실행 범위와 예상 비용

- prototype/covariance/competition/margin audit: 모든 task.
- oracle classifier replacement: 기본 final task only, `--oracle-head-eval-tasks all`에서 모든 task.
- 기본 production 값은 원 checkpoint의 `ca_epochs=5`, `replay_samples_per_class=256`을 그대로 쓴다.
- final-only A–D는 원 CA 한 번의 약 4배 classifier-only 연산이다. backbone forward는 추가 반복하지 않는다.
- all-task A–D는 task 수에 비례해 증가한다. 10-task에서는 final-only를 기본으로 유지한다.
- full covariance 보관/비교가 CPU RAM과 disk의 주 비용이다. classifier-only CA는 768-d synthetic feature와 작은 cosine head만 사용하므로 feature training보다 훨씬 가볍다.
- sanity run은 `--oracle-head-epochs 1 --oracle-head-samples-per-class 2`와 class/sample cap을 사용한다.

## 검증 순서

1. covariance code audit 문서 확정
2. synthetic unit test와 optimizer leakage test
3. IA5 legacy final checkpoint small subset GPU smoke
4. 기존 CSV/figure regression 확인
5. audit-ready IA5/CAR5 5-task 실행
6. IA10/CAR10 실행
7. final combined report
