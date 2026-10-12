# T9 Local/All Paired Transition Audit — 실행·검증 정의

이 폴더는 기존 prediction CSV만 읽는 analysis-only diagnostic이다. Model 학습·inference, 이미지 generation·augmentation, threshold tuning은 하지 않는다. Original Structure 및 generation pilot 산출물은 read-only로 취급한다.

## 입력

- `../type_a_generative_augmentation/training/{IA10,CAR10}/T9_T10/{S,G,R}/predictions_T{09,10}.csv`
- 같은 pilot의 T1 S/G/R prediction CSV와 `metrics.json`, task partition용 `resolved_config.json`
- `../error_trajectory_diagnostic/{IA10,CAR10}/sample_trajectory.csv`
- `../type_a_generative_augmentation/audit/{IA10,CAR10}/birth_support_audit.csv`

Local prediction은 기존 `training.evaluate_task()`가 sample의 birth task class로 제한해 저장한 값이다. T9 cohort의 T10 prediction에도 local 후보는 G9다. Code는 label/local candidate 범위를 task partition과 대조한다. 필요한 필드가 없으면 실패하며 자동 재학습·inference fallback은 없다. Optional true-logit은 저장되지 않았으므로 임의 복원하지 않는다.

## State 및 분모

W: local 오답. C: local 정답·all 오답. K: all 정답. All 정답/local 오답은 허용하지 않는다.

Primary direct cohort는 birth==T9 (IA10 147 / CAR10 758)다. 고정 vanilla A-within/A-cross와 non-A로 정확히 분할한다. A subtype은 **기존 vanilla audit**에서 읽으며 반복 support seed의 subtype/예측이 동일한지 확인한다. Historical birth≤T8는 별도 all-seen collateral table이다. T10 whole seen은 historical+T9 direct+T10 new로 분리한다. T1은 local=all의 별도 표이며 continuation을 주장하지 않는다.

G−S는 S→G, G−R는 R→G다. Matrix 행=reference state, 열=G state, order=W/C/K.

Local repair=W→C+W→K, harm=C→W+K→W. All repair=W→K+C→K, harm=K→W+K→C. 모든 net은 repair−harm이며 accuracy 차이는 100×net/N이다. Class macro는 해당 cohort에 sample이 있는 class의 정답률을 동일 가중치로 평균한다. Class selector를 만들지 않는다.

## Consistency checks

- A1: T9 direct N=147/758; T9와 T10의 direct ID 집합 동일.
- A2: 저장 metrics와 T9 new all/local accuracy 일치.
- A3: 저장 metrics와 T10 T9 cohort all accuracy 일치.
- A4/A5: W/C/K 유일 상태, matrix 9-cell 합=N.
- 동일 ID·true label·birth task·class mapping·branch·평가 시점·local/all candidate range 검증.
- 기존 G−S/G−R paired raw repair/harm, T1 결과와 whole seen 결과 일치.
- Scope별 N/repair/harm/net 합=whole seen; 서로 다른 분모의 %p는 합산하지 않음.
- 모든 원본 입력의 실행 전후 SHA-256 동일. 해시는 `run_manifest.json`에 저장.

## Decision 의미

Protocol별 `LOCAL_SIGNAL=YES`는 T9 direct local net>0이라는 관측만 의미하며 유의성 기준이 아니다. 공통 persistence는 명세 Case 1에 따라 **양 protocol에서 T9 local net>0이고 T10 local net≥0**여야 YES다. T10 net=0은 비음수 조건을 만족하지만 strictly-positive advantage는 아니다. 이 차이를 별도 decision column으로 보존한다. Generation direction은 모든 경우 STOP이다. Next method는 검토 방향이며 새 method 구현/학습을 승인하거나 수행하지 않는다.

## 실행

```bash
cd ${RESEARCH_WORKSPACE}/generation_local_all_transition_audit
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
CUDA_VISIBLE_DEVICES= python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES= python run.py
```

GPU 번호는 필요 없다. `run.py`는 torch나 학습 모듈을 import하지 않는다. Matplotlib Agg로 PNG만 생성한다. Source와 겹치는 output directory는 거절한다. 재실행은 이 diagnostic 폴더의 분석 CSV/PNG/report/manifest만 재작성한다.

## 산출물

각 protocol의 wide sample CSV에 필수 S/G T9/T10 local/all prediction·state, 고정 vanilla subtype, transition을 저장한다. R과 기존 margin 및 vanilla birth correctness도 추가한다. Summary CSV는 G−S/G−R 각각 4 direct cohort의 raw counts·micro/macro accuracy·%p를 포함한다. Class별 summary와 historical sample-level collateral도 제공한다. 모든 matrix는 `transition_matrices.json`에 저장한다.

그림은 T9/T10 × 4 cohort × 2 protocol heatmap 16개, protocol별 subtype bar plot 2개, 통합 local/all net bar plot 1개다. 최종 보고서에는 direct 4개 heatmap, 2개 subtype plot, 통합 bar plot을 embed한다. 한계는 단일 model seed 및 원 generation pilot의 미검증 label fidelity다.
