# 코드 검증 기록 — 2026-10-08

사용자 실행으로 IA10 R의 T1 학습이 완료됐다. 이어지는 S T1은 strong transform 순서 오류로 중단됐다. 해당 오류를 새 모듈에서 수정하고 CPU 검증했다. 이번 수정에서 본 GPU 학습을 재시작하지 않았다. 원 Structure/B1 실험 결과와 코드, 기존 R 결과는 보존했다.

검증 환경: `${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python`, Python 3.9, PyTorch 2.4.1+cu121. 합성 테스트는 `CUDA_VISIBLE_DEVICES=''`로 실행했다.

| 검사 | 결과 | 범위 |
|---|---|---|
| Python compileall | PASS | 새 모듈의 문법 |
| bash -n | PASS | 환경 설치/feasibility/학습 wrapper |
| CPU unittest 및 회귀 검사 15개 | PASS | 기존 검증 + strong transform 실제 IA/CAR layout/Solarize + completed-branch skip |
| Baseline preflight | PASS | IA10/CAR10 source config, 20개 checkpoint 파일, 18개 birth test cache schema |
| 실제 data manager | PASS | 두 protocol class-order 일치, 실제 train roots, class 수 IA=200/CAR=196 |
| 원 CA/stats 설정 | PASS | 256 Gaussian samples/class, stats batch 64 |
| Local generator | 사용자 생성 완료 | 고정 revision에서 1,192개 artifact 생성 |
| Generator pipeline import | PASS | 설치된 .venv_generator에서 Diffusers pipeline class import, CUDA 비활성 |
| 자동 sanity GPU 4 | READY_EXPLORATORY | 1,192개 검사, IA10=560/568 · CAR10=622/624 채택, 10개 safety-blocked/단색 붕괴 제외 |
| 새 quality-check | READY_EXPLORATORY | Frozen policy·manifest·채택 이미지 hash 검증, human 검수 불필요 |

CPU 합성 테스트:

1. Pooled covariance = concatenation covariance, between-source scatter 포함, generated n=1.
2. R/S/G 동일 real anchor 및 auxiliary parent hash, update/exposure count, real-only SCC 호출.
3. G/GM 같은 CA 난수 draw, G state 복원 및 별도 GM head 유지.
4. 미검수 표본의 HOLD, 검수 완료 gate, 이미지 hash 변경 차단.
5. Fixed A paired 비교에서도 harm이 생길 수 있음.
6. T9 correct→wrong/wrong→correct와 CE/오류 방향 집계.
7. Train support의 leave-one-out self exclusion.
8. 전체 synthetic 결과 집계 → JSON/CSV/Markdown 생성 및 STOP decision serialization.
9. 실제 실행 폴더의 fresh Python process에서 원 `data.data`/`data.data_manager`/learner import. 이 검사로 local `data.py` namespace 충돌의 재발을 확인한다.
10. 동일 이미지 near-duplicate 탐지 및 fixed-rule survivor 처리.
11. 단색 붕괴/gross-deviation/structure-loss proxy 규칙.
12. 깨진 이미지 자동 제외, human review 파일 무시, 모든 survivor 채택, cached bank tamper 차단. 기존 report 합성 테스트에는 exploratory limitation/label-unverified metadata 확인도 추가했다.
13. `_train_trsf` 안의 ToTensor와 이후 Normalize가 있는 layout에서 RandAugment가 PIL 단계에 위치하는지 검사. Solarize를 강제 호출해 이전 오류가 재발하지 않음을 확인했다.
14. 실제 IA10/CAR10 data manager의 transform 구성으로 강제 Solarize 및 protocol별 24개 seed의 random strong transform 실행. 출력 shape/dtype/finite 확인.
15. 완료 checkpoint와 predictions가 존재하는 branch는 새 learner를 만들지 않고 skip하는지 확인.

이는 전체 GPU 실험이나 label fidelity 검증을 대신하지 않는다. Audit와 생성은 사용자 실행으로 완료됐고 생성 환경은 `.venv_generator`다. 자동 sanity는 기존 generator의 frozen CLIP component를 재사용했다. 약 121초, peak allocated 1,505,283,584 bytes, 추가 weights 다운로드 없음. Near-duplicate/gross-deviation/structure-loss 추가 flag는 0개지만 객체 보존이 인증됐다는 뜻이 아니다. CAR 차종/연식 포함 모든 generated label은 미검증이다. IA10 R T1의 checkpoint/predictions/resource 기록은 완료됐고, 최종 downstream 비교는 아직 완료되지 않았다.

사용자의 첫 실행은 audit 첫 단계에서 `data.py`가 원 Structure namespace package를 가린 import 오류로 종료됐다. 이 폴더의 `data.py`를 `matched_views.py`로 이름 변경하고 import 및 preflight를 수정했다. 원 Structure 코드, 기존 로그, 실패 manifest는 보존했다. GPU job은 재시작하지 않았으며 아래 재실행은 사용자 작업이다.

실행 방법은 [README](README.md)에 있다. 사용자 실행 후에만 `final_report.md`가 작성된다.
