# Cross-Task CA Reweighting Pilot — 사전 구현 명세

상태: **코드 작성만 완료, 실험 미실행**. 기존 `Structure`/oracle audit 코드는 수정하지 않는다.

## 개입 및 대조

- T10 task-end checkpoint의 frozen `sfeatures`와 T10 pre-CA classifier sidecar를 사용한다. Backbone, SCC, APR, Gaussian class mean/covariance, inference 후보 class는 변경하지 않는다.
- 선택된 데이터 구성은 **기존 Structure CA와 일치하는 전 class Gaussian**이다. 명세의 “current/new actual feature bank” 문구는 실제 원본 CA 구현과 충돌했으며, 사용자가 원본 방식 선택을 확인했다. 따라서 current/new class도 old class와 같이 Gaussian에서 샘플링한다. New 실제 train feature는 학습에 투입하지 않는다.
- 원본 CA 코드(`structure_multicenter_separation/models/base.py::_stage2_compact_classifier`)대로 매 epoch 모든 class에서 256개씩 Gaussian을 다시 뽑고 CPU `torch.randperm`으로 섞는다. T10 pre-CA sidecar의 RNG를 복원한다. 각 epoch의 동일 bank와 label sequence를 Control·Intervention head에 lockstep으로 공급하므로 두 branch 간 샘플·순서 차이가 없다. Bank의 SHA-256은 저장하지만 대용량 bank tensor와 old raw exemplar는 저장하지 않는다.
- 원본 CA는 각 epoch 뒤 test DataLoader를 평가하면서 다음 epoch의 CPU `randperm` 전에 worker base-seed용 CPU torch RNG를 1회 소비한다. 이 pilot은 test label을 학습에 넣지 않고 그 **RNG 소비만** 재현한다. 원본 head와의 weight/accuracy parity를 별도 감사하여 실제 재현 정도를 판정한다.
- 원본 Gaussian mean scaling에는 `class_id // task_size_argument`를 사용한다. CAR10은 마지막 task 크기가 19로 앞 task의 20과 달라도 **원본의 scaling을 그대로 보존**한다. 이는 loss의 `G(y)` 정의와 다르며, `G(y)`는 실제 task partition `[20×6, 19×4]`을 따른다.
- Original all-class CE를 Control에 적용한다. 안정적인 log-sum-exp로 계산한 `L_within+L_cross`가 `torch.nn.functional.cross_entropy`와 `atol=2e-5` 이내로 일치해야 학습을 시작한다. Intervention은 동일 logits에서 `L_within+2L_cross`다. coefficient는 정수 2로 고정되며 sweep이나 test 기반 선택은 없다.
- 두 head는 `DiagnosticCosineHead`의 task별 weight Parameter로 같은 pre-CA weight를 복사한다. SGD(momentum 0.9, sidecar learning rate/weight decay), cosine scheduler, 5 epochs, 256/class/minibatch, optimizer step count가 동일하다. Representation은 T10 final feature cache에서 고정되고 어떤 backbone parameter도 optimizer에 들어가지 않는다.
- 실제 평가에는 T10 모든 seen class에 대한 cosine head argmax를 사용한다. Task ID나 birth 후보 제한은 inference에 제공하지 않는다.

## B1 및 source 정합성

- 이전 `error_trajectory_diagnostic`의 sample ID를 고정한다. IA10=126, CAR10=473이어야 하고 T10 real test feature cache의 ID/label과 1:1 결합되어야 한다.
- 이전 trajectory baseline과 final feature cache+원본 final head의 예측 차이는 기록한다. 2개 초과면 진단이 중단된다. Control과 기존 vanilla 정확도·head weight 차이는 별도 audit에 보고한다. 핵심 인과 비교는 동일 bank를 공유한 matched Control vs Intervention이다.
- 기존 T10 snapshot, pre-CA sidecar, feature cache, trajectory 파일은 읽기만 하고 수정하지 않는다.

## 실행 전 고정한 판정 기준

Control 재현성 S0과 아래 S1–S6이 **IA10·CAR10 모두**에서 참일 때에만 `CROSS-TASK CA PROTOTYPE PROMISING = YES`다. 이 기준은 단일 seed pilot에서 사용할 운영 판정이며 통계적 유의성 기준이 아니다.

- S0: matched Control의 원본 final head 대비 최대 절댓값 차이 ≤0.001이고 cached vanilla 대비 정확도 차이 ≤0.5 pp. 이 기준이 실패하면 Control이 원본 CA를 충분히 재현하지 못한 것으로 표시한다.
- S1: 전체 real test에서 Repair > Harm.
- S2: matched Control 대비 Seen accuracy 증가.
- S3: 고정 B1 subset에서 net repair가 5개 이상이고 정확도 개선이 5 percentage points 이상.
- S4: class-macro accuracy 증가.
- S5: cross-task error 감소량이 same-task error 증가량(증가하지 않으면 0)보다 큼.
- S6: T10 old 또는 new의 accuracy가 어느 쪽도 1 percentage point를 초과하여 악화하지 않음.

한 조건이라도 실패하면 이 **계수 2의 T10 CA prototype은 유망하지 않다**고 결론 내리고 계수 튜닝으로 이어가지 않는다. `FULL METHOD READY`는 결과와 무관하게 항상 `NO`다. 성공하더라도 full incremental training, 독립 seed/class order, 기존 intra/inter-task method와의 novelty 비교가 선행되어야 한다.

## 산출물

실행 시 `run_manifest.json`, `audit.md`, IA10/CAR10별 branch 예측 CSV·paired CSV·B1 CSV·오류 유형 요약·head weight·epoch bank hash, `cross_protocol` 요약·판정 표·5개 plot, `final_report.md`가 생성된다.
