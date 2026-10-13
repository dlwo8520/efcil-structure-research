# StoredAPR RP-ProtoCLAD 구현 계획

## 범위와 보존 원칙

- 새 프로젝트: `${RESEARCH_WORKSPACE}/structure_stored_apr_rp`.
- 새 experiment ID: `StoredAPR_RP_C_P020_L030`; 새 출력은 이 프로젝트의 `outputs/`만 사용한다.
- IA5와 CAR5, seed 1993, P=0.20, λ=0.30, R=[768,2048]/seed 1993만 실행한다.
- 기존 Structure, C 및 `RP_C_P020_L030` 코드·설정·출력·checkpoint는 읽기만 한다.
- 기존 RP와 다른 학습 연산은 old RP prototype의 생성/갱신뿐이다. Top-P, APR, SCC, classifier adaptation 및 모든 원공간 통계는 유지한다.

## 수식 및 생명주기

1. 각 class의 최초 task에서 feature learning 종료 후 현재 train sample `sfeatures`로 `mu_birth=mean(z)`와 `q_birth=mean(ReLU(zR))`를 만든다. sample별 L2 정규화는 하지 않는다. Task 1의 class도 포함한다.
2. 원공간 APR가 매 epoch 제공하는 `mu_APR`에 대해 `q_current=q_birth+ReLU(mu_APR R)-ReLU(mu_birth R)`를 계산한다. Cosine 직전에 current sample과 `q_current`를 각각 L2 정규화한다.
3. `q_birth`와 `mu_birth`는 class-level statistic으로만 보존한다. raw exemplar/feature는 저장하지 않는다.
4. Birth statistic과 class metadata를 task-end 및 epoch checkpoint에 넣고 resume에서 class 수, 차원, R checksum을 검증한다. 이전 RP checkpoint에서는 정확한 birth statistic을 복원할 수 없으므로 새 run은 Task 1부터 시작한다.

## 변경 파일 목록

| 새 파일 | 역할 |
|---|---|
| `stored_apr_rp/prototypes.py` | birth 평균과 APR displacement의 순수 계산·검증 |
| `stored_apr_rp/losses.py` | 기존 Proto-CLAD selection/scale을 유지하며 old RP vector만 교체 |
| `stored_apr_rp/learner.py` | task-final birth statistic 수집, epoch loss/diagnostic, checkpoint 저장·복원 |
| `configs/IA5.json`, `configs/CAR5.json` | 기존 RP λ=0.30 설정을 상속하고 ID/variant만 변경 |
| `run.py`, `run_all.py`, `run_stored_apr_rp.sh`, `scripts/wait_for_gpu_then_run.sh` | 검증, 단일 run, IA5→CAR5 queue, 물리 GPU 지정/중복 방지 및 GPU 2 대기 |
| `stored_apr_rp/snapshot.py` | 새 variant 소스의 독립 snapshot/checksum |
| `tests/`, `scripts/run_tests.py` | 식, R checksum, Top-P/gradient, 기존 RP 회귀, checkpoint round-trip |
| `analysis/build_report.py`, `stored_apr_rp_report.md` | Structure/old RP 대비 성능·geometry·prototype 진단 보고 |
| `README.md` | 환경, GPU, 백그라운드 실행과 진척도 확인 |

## 실행 전 검증

1. `q_birth`와 sample-wise 투영 평균 일치, `ReLU(mean(z)R)`와 일반적으로 불일치.
2. APR drift 0에서 `q_current=q_birth`; drift 있는 경우 식 일치.
3. 기존 RP의 R checksum 및 기존 RP loss 회귀 검사.
4. 새 checkpoint의 birth statistic round-trip 및 불일치 거부.
5. IA5/CAR5 config validate-only, 소형 CPU smoke 후 사용자가 GPU 2 대기 명령으로 training을 예약·실행한다.

## 결과 및 해석

- `stored_rp_prototype_diagnostics.csv`에 birth A/B 차이와 실제 old `q_current` 대 기존 old-style prototype 차이를 기록한다.
- 기존 RP geometry/gradient/confusion 및 task/class별 성능 diagnostic을 유지한다.
- 보고서에서 Structure 대비 `ΔStruct_*`, 기존 RP 대비 `ΔOldRP_*`를 계산한다. 완료 전에는 질문 Q1–Q7의 답을 `PENDING`으로 둔다.
