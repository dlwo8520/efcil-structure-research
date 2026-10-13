# Similarity-selective Structure sweep

기존 `structure_multicenter_separation`, `structure_e3_followup`,
`structure_similarity_aware_followup`를 읽기 전용 의존성으로 사용하고 새 폴더에서만 구현한 실험입니다.

원래 Stage 1은 IA5, IA10, CAR5, CAR10의 108-run grid였으나 2026-09-21에 중단했습니다.
현재 reduced Stage 1은 **IA5/CAR5 5-task만** 사용하며 다음 다섯 조건만 비교합니다.

- vanilla `Structure`
- `A_P005`
- `B_P020_A100`
- `C_P020_L030`
- `C_P020_L100`

IA10의 기존 결과는 참고용으로 보존하고 추가 실행·복구하지 않습니다. CAR10도 전체 HOLD입니다.
기존 결과 폴더는 삭제하지 않으며, 실제 filesystem audit는 `reduced_experiment_queue.md`에 있습니다.

## 환경 및 reduced Stage 1 실행

```bash
cd ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep
nohup bash run_reduced_experiments.sh 1 \
  > logs/reduced_stage1_gpu1.log 2>&1 &
echo $! > logs/reduced_stage1_gpu1.pid
```

GPU 번호는 첫 번째 인자로 바꿀 수 있습니다. 실행기는 시작 직전 filesystem을 다시 audit하고
완료 run은 skip하며, 선택된 partial run만 checkpoint에서 resume합니다. 현재 audit에서는 IA5
5개와 CAR5 A_P005가 완료되어 새 GPU 작업은 CAR5 4개뿐입니다.

## 진행 확인

```bash
cd ${RESEARCH_WORKSPACE}/structure_selective_similarity_sweep
conda activate cds

python scripts/build_reduced_queue.py
tail -f logs/reduced_stage1_gpu1.log
tail -f logs/reduced_stage1/CAR5__Structure__seed_1993.log
nvidia-smi -i 1
```

각 run의 종료 코드는 `logs/reduced_stage1/exit_codes.csv`에 누적됩니다. legacy
`run_all_similarity_sweeps.sh`는 실수로 108-run queue를 재시작하지 않도록 비활성화했습니다.

## HOLD 항목

IA10/CAR10, seed 1996/1997, 추가 P/alpha/lambda sweep, 실패한 비선택 run 복구는 지금
수행하지 않습니다. 5-task에서 후보 1–2개를 확정한 뒤 10-task 검증을 별도로 계획합니다.

## 검증과 보고서

```bash
python scripts/run_tests.py
python scripts/build_reduced_queue.py
bash run_reduced_experiments.sh 1 --dry-run
python analysis/build_report.py
```

검증 명령은 설정, protocol partition, reduced action을 검사하며 학습은 수행하지 않습니다.
큰 진단 테이블은 현재 `cds` 환경에 Parquet
엔진이 없으므로 동일 schema의 CSV로 정직하게 저장하고 `diagnostic_storage.json`에 이
사실을 기록합니다.

`analysis/build_report.py`는 실행 중 자동 요약을 `report_auto.md`에 쓰며, 검산된 상세
중간/최종 보고서인 `report.md`는 기본 동작으로 덮어쓰지 않습니다.
