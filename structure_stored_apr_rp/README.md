# StoredAPR RP-ProtoCLAD (독립 ablation)

기존 `structure_rp_protoclad_task5`를 수정하지 않고, IA5/CAR5의 새
`StoredAPR_RP_C_P020_L030` 두 run을 별도 `outputs/`에 기록한다.
학습의 유일한 방법 차이는 old RP prototype이다.

```text
q_birth = mean_i(ReLU(z_i R))
q_old(t,e) = q_birth + ReLU(mu_APR(t,e) R) - ReLU(mu_birth R)
```

Task-final current train sample의 `sfeatures`에서 q_birth를 만들고 class-level
statistic만 checkpoint에 보존한다. 원공간 APR/SCC/classifier adaptation과
original-space Top-P 선택은 기존 RP와 같다.

## 검증

```bash
cd ${RESEARCH_WORKSPACE}/structure_stored_apr_rp
${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python scripts/run_tests.py
bash run_stored_apr_rp.sh 2 --dry-run
```

위 명령은 GPU training을 시작하지 않는다.

## GPU 2가 비면 자동 실행 예약

```bash
cd ${RESEARCH_WORKSPACE}/structure_stored_apr_rp
mkdir -p logs
nohup bash scripts/wait_for_gpu_then_run.sh 2 > logs/stored_apr_gpu2_queue.log 2>&1 &
echo $! > logs/stored_apr_gpu2_queue.pid
tail -f logs/stored_apr_gpu2_queue.log
```

대기 스크립트는 GPU 2의 compute process가 사라질 때까지 60초 간격으로 확인하고,
60초 후에도 비어 있으면 IA5→CAR5를 순차 실행한다. 기존 GPU 1/2 프로세스를 중단하지 않는다.
이는 GPU 자원을 실제로 예약하는 스케줄러가 아니라 유휴 상태를 확인하는 best-effort
대기다. 다른 대기열도 같은 GPU를 노린다면 먼저 시작한 작업과 경합할 수 있다.
GPU 2가 비어 있는 시점에 수동으로 시작하려면 `nohup bash
run_stored_apr_rp.sh 2 > logs/stored_apr_gpu2_queue.log 2>&1 &`를 사용한다.
개별 run 로그는 `logs/{IA5,CAR5}__StoredAPR_RP_C_P020_L030__seed_1993.log`다.
완료된 새 run은 건너뛰고, 미완료 checkpoint는 같은 run에서 resume한다.

`stored_apr_rp_report.md`는 완료 전에는 질문을 PENDING으로 두며, run 완료 후
자동 갱신된다. 기존 RP 결과는 읽기만 한다.
