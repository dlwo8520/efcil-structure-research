# Implementation Audit

## 격리

- 기존 `structure_multicenter_separation`, `structure_e3_followup`,
  `structure_similarity_aware_followup` 소스는 수정하지 않는다.
- 각 run에 네 소스 계층의 SHA-256과 control 소스 사본을
  `control_source_snapshot/`으로 저장한다.
- GPU 인자는 physical ID이며 `run.py`가 `CUDA_VISIBLE_DEVICES`를 설정한 뒤 내부
  logical device 0으로 변환한다. 기본 physical GPU는 2다.

## 비교 공정성

- C0의 all-one pair weight는 기존 E3 `max_relative_intrusion`과 정확히 같은 식이다.
- C1/C2/C3은 old class 수의 25%를 올림한 동일 개수를 new class별로 선택한다.
- similarity 동률은 old class ID로 결정한다.
- C3은 seed/task/new-class로 만든 전용 CPU generator를 사용해 학습 RNG를 바꾸지 않는다.
- similarity는 이전 task teacher feature에서 optimizer step 전에 동결한다.
- relative-margin beta는 손실에 쓰이지 않지만, 상속 초기화가 외부 S0 파일을 요구하지
  않도록 현재 task similarity quantile로 독립 계산한다.

## 로그

- `control_pair_selection.csv`: 선택, 순위, quantile, 적용 weight
- `control_pair_separation.csv`: epoch별 pair 평균 D2/rho/loss/weight/intrusion
- `control_confusion.csv`: old-new 방향, similarity/group이 포함된 non-zero confusion
- `control_group_metrics.csv`: high/low/random accuracy, confusion, forgetting
- `control_intrusion.csv`: raw/effective high/low/random intrusion
- 상속 로그: class accuracy/forgetting, feature norm, task/epoch/gradient metrics

## 검증

- `scripts/smoke_test.py`가 C0 parity, exact-count selection, deterministic random,
  global RNG 보존, finite gradient, epoch/evaluation CSV 생성을 검사한다.
- `configs/smoke_C3.json`은 실제 dataset/GPU 경로를 5-task × 1 epoch로 검사한다.
- smoke 출력은 `SMOKE_C3`로 분리되어 논문용 C0--C4 집계에 포함되지 않는다.
