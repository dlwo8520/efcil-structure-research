# 구현·실행 노트

## 명세 대비 최소 변경

`pilot_learner.py`의 `PairedStructureViews`는 원본 train transform을 같은 current-task 이미지에 두 번 적용한다. `VariationPilotLearner._init_train`에서 CosFace와 SCC를 view 평균하고, JS branch에만 `0.1 * JS(softmax(logits1),softmax(logits2))`를 더한다. Both branch에서 동일한 계산을 수행하며 Control의 계수만 0이다. JS 후보는 current-only이다.

`incremental_train`의 원래 통계 계산→APR→Gaussian CA 경로는 상속 그대로 둔다. `stats_update_count=APR correction count=CA call count=1`을 task별로 assert한다. T8 snapshot load 후 adapter flag를 복원한다. 원본의 매 epoch 1.1GB checkpoint 저장은 디스크 사용을 줄이기 위해 양 branch 모두 끄고 task-end T9/T10 snapshot만 남긴다. 이 설정은 training math를 바꾸지 않는다.

## 실행

`cds` 환경, physical GPU 4에서 순차 실행:

```bash
cd ${RESEARCH_WORKSPACE}/variation_consistency_pilot
conda activate cds
python -u run_pilot.py --gpu 4 --protocol all --branch all
python analyze_pilot.py
python report.py
```

각 branch는 T8 checkpoint에서 시작하고 T9→T10을 연속 진행한다. T9 task-end snapshot이 있으면 해당 branch의 T10부터 재개한다. 기존 T10 snapshot이 있으면 보존하고 건너뛴다.

백그라운드 실행 시:

```bash
cd ${RESEARCH_WORKSPACE}/variation_consistency_pilot
conda activate cds
nohup python -u run_pilot.py --gpu 4 --protocol all --branch all > pilot_gpu4.log 2>&1 &
echo $! > pilot_gpu4.pid
tail -f pilot_gpu4.log
```

## Phase A

기존 C1 logit이 저장되지 않아 사용자가 승인한 동일 C1 재실행으로 복원했다. 재실행 script는 `phaseA_ceos_error_destination/run.py`이며 현재 산출물은 `*_paired_new_logits.npz`, `*_new_error_breakdown.csv`, `*_summary.json`, `summary.md`이다. 추가 CEOS variant는 없다.

## 분석 출력

- `IA10/after_T9.csv`, `after_T10.csv`, `retention.csv`, `plots/`
- `CAR10/after_T9.csv`, `after_T10.csv`, `retention.csv`, `plots/`
- `cross_protocol/summary.csv`, `delta_table.csv`, `decision_table.csv`
- `run_manifest.json`, `final_report.md`

`analyze_pilot.py`는 paired view hash/steps/APR/CA 불일치 시 결과표 작성 전에 실패한다. 이 경우 수치를 인과 비교에 쓰지 않는다.

탐색용 prototype 통과 기준은 사전에 고정한다: 각 protocol에서 T9 New delta>0, T10 T9-cohort delta>0이며 immediate delta의 최소 25% 유지, prior-Old 손실이 T9-cohort 이득보다 크지 않음, T10 New 손실이 2pp 미만, T10 Seen delta>0. 두 protocol 모두 통과해야 `CONSISTENCY PROTOTYPE PROMISING=YES`이다. 이 기준은 단일 seed의 통계적 유의성을 뜻하지 않는다.
