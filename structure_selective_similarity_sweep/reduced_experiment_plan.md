# Reduced Stage 1 실행 계획

Filesystem audit: 2026-09-21 13:00:53 KST  
Seed: `1993`  
운영 범위: **IA5 + CAR5 only**  
Primary baseline: **vanilla Structure**

## 결론

- 이미 완료된 선택 run: **6개**
- 새로 실행할 run: **2개**
- checkpoint에서 resume할 run: **2개**
- 기존 108-grid 중 미실행 상태로 SKIP할 run: **39개**
- 전체 audit row action: KEEP_COMPLETE 6, RUN_NEW 2, RESUME 2, SKIP 102

실제 새 GPU 작업은 CAR5의 `Structure`, `B_P020_A100`, `C_P020_L030`, `C_P020_L100` 네 개다. IA5 선택 run은 모두 완료되어 다시 실행하지 않는다.

## 기존 대규모 queue 중단 확인

다음 기존 sweep 프로세스가 아직 발견되므로 reduced queue를 실행하면 안 된다:
- `3354429 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`
- `3360705 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`
- `3360753 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`
- `3360785 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`
- `3360817 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`
- `3360849 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`
- `3360884 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`
- `3360917 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`
- `3360951 ${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/run.py --protocol CAR5 --run-id B_P020_A100 --gpu 1 --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993`

중단 시 실행 중이던 비선택 `CAR5/B_P010_A100`은 삭제하지 않았다. `checkpoint_latest.pt`와 partial metadata를 그대로 보존했으며 새 queue에서는 resume하지 않는다.

## Audit 판정 기준

COMPLETE는 `summary.json`, `run_metadata.json`, `resolved_config.json`, `task_summary.csv`, `checkpoint_latest.pt`가 모두 완결되고 예상 task 수와 모든 `task_final` 행이 일치해야 한다. `metrics_per_seed.csv`도 교차 확인하되, 학습 직후 aggregate export만 실패한 경우 완료 run을 재학습하지 않도록 보조 증거로 사용한다. IA5 Structure는 외부의 검증된 `B0_STRUCTURE` 결과다.

기존 108-grid 상태: COMPLETE 61, PARTIAL 2, FAILED 4, NOT_STARTED 41.

## 선택 matrix 상태

| protocol | run_id | method | P | alpha | lambda | current_status | action | resume_from | reason | command | output_dir |
|---|---|---|---:|---:|---:|---|---|---|---|---|---|
| IA5 | Structure | Structure | — | — | — | COMPLETE | KEEP_COMPLETE | — | selected and already complete; summary+metadata complete; task_final 5/5; checkpoint present; metrics external-not-required | — | ${RESEARCH_WORKSPACE}/structure_e3_followup/outputs/imageneta/B0_STRUCTURE/seed_1993 |
| IA5 | A_P005 | A | 0.05 | — | — | COMPLETE | KEEP_COMPLETE | — | selected and already complete; summary+metadata complete; task_final 5/5; checkpoint present; metrics present | — | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/IA5/A_P005/seed_1993 |
| IA5 | B_P020_A100 | B | 0.2 | 1.0 | — | COMPLETE | KEEP_COMPLETE | — | selected and already complete; summary+metadata complete; task_final 5/5; checkpoint present; metrics present | — | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/IA5/B_P020_A100/seed_1993 |
| IA5 | C_P020_L030 | C | 0.2 | — | 0.3 | COMPLETE | KEEP_COMPLETE | — | selected and already complete; summary+metadata complete; task_final 5/5; checkpoint present; metrics present | — | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/IA5/C_P020_L030/seed_1993 |
| IA5 | C_P020_L100 | C | 0.2 | — | 1.0 | COMPLETE | KEEP_COMPLETE | — | selected and already complete; summary+metadata complete; task_final 5/5; checkpoint present; metrics present | — | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/IA5/C_P020_L100/seed_1993 |
| CAR5 | Structure | Structure | — | — | — | FAILED | RESUME | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/checkpoint_latest.pt | selected failure has a preserved checkpoint; root cause was diagnosed and fixed before resume; failure metadata/marker; last task=1, epoch=30 | `python run.py --protocol CAR5 --run-id Structure --gpu ${GPU_ID} --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993 --resume ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/checkpoint_latest.pt` | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993 |
| CAR5 | A_P005 | A | 0.05 | — | — | COMPLETE | KEEP_COMPLETE | — | selected and already complete; summary+metadata complete; task_final 5/5; checkpoint present; metrics present | — | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/A_P005/seed_1993 |
| CAR5 | B_P020_A100 | B | 0.2 | 1.0 | — | PARTIAL | RESUME | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/B_P020_A100/seed_1993/checkpoint_latest.pt | selected partial run; resume latest validated checkpoint; artifacts preserved; latest checkpoint state=train_epoch:task_1:next_epoch_4 | `python run.py --protocol CAR5 --run-id B_P020_A100 --gpu ${GPU_ID} --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993 --resume ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/B_P020_A100/seed_1993/checkpoint_latest.pt` | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/B_P020_A100/seed_1993 |
| CAR5 | C_P020_L030 | C | 0.2 | — | 0.3 | NOT_STARTED | RUN_NEW | — | selected run has no completed result; no run artifacts found | `python run.py --protocol CAR5 --run-id C_P020_L030 --gpu ${GPU_ID} --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993` | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/C_P020_L030/seed_1993 |
| CAR5 | C_P020_L100 | C | 0.2 | — | 1.0 | NOT_STARTED | RUN_NEW | — | selected run has no completed result; no run artifacts found | `python run.py --protocol CAR5 --run-id C_P020_L100 --gpu ${GPU_ID} --data-root ${RESEARCH_WORKSPACE}/cds-main/data/cars196 --output-root ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs --seed 1993` | ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep/outputs/CAR5/C_P020_L100/seed_1993 |

## 실행 순서

1. `CAR5/Structure` — `RESUME`
2. `CAR5/B_P020_A100` — `RESUME`
3. `CAR5/C_P020_L030` — `RUN_NEW`
4. `CAR5/C_P020_L100` — `RUN_NEW`

IA10은 완료/실패/partial 결과를 모두 참고용으로 보존하지만 추가 실행, 실패 복구, matched Structure 생성을 하지 않는다. CAR10은 전체 HOLD다.

## 진단 수집 유지

실험 개수만 줄이고 각 선택 run은 기존 `run.py`와 동일한 recorder/export 경로를 사용한다. 따라서 performance/task/class accuracy, similarity selection/rank/cutoff, APR drift/tau, separation D2/intrusion/loss, Proto-CLAD cosine/loss, gradient cosine/norm, feature norm, old-new 및 full confusion matrix, wall-time/GPU/RAM 진단을 계속 저장한다. Structure에서는 비활성 방법의 loss 항목만 해당 없음으로 남는다.

## 실행 방법

GPU 번호를 첫 번째 인자로 지정한다.

```bash
cd ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep
bash run_reduced_experiments.sh 1
```

백그라운드 실행:

```bash
cd ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep
nohup bash run_reduced_experiments.sh 1 > logs/reduced_stage1_gpu1.log 2>&1 &
echo $! > logs/reduced_stage1_gpu1.pid
```

이 plan 생성 단계에서는 새 학습을 시작하지 않았다. PositiveDefinite fallback도 아직 적용하지 않았으며, 선택 run에서 실제 오류가 발생할 때만 적용하고 횟수를 기록한다.

## SKIP 보존 현황

- COMPLETE 보존: 56
- PARTIAL 보존: 1
- FAILED 보존: 4
- NOT_STARTED/HOLD: 41
