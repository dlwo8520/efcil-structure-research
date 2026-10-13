# Final Error Trajectory Diagnostic

IA10/CAR10, seed 1993의 vanilla Structure `task_end` checkpoint T1~T10으로 official test sample을 birth task부터 T10까지 추적한다. 학습은 하지 않는다. 원본 checkpoint와 원본 연구 폴더는 읽기 전용으로 취급한다.

## GPU 3에서 실행

```bash
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd ${RESEARCH_WORKSPACE}/error_trajectory_diagnostic
mkdir -p logs
nohup python -u run.py --gpu 3 --resume > logs/error_trajectory_gpu3.log 2>&1 &
echo $! > logs/error_trajectory_gpu3.pid
tail -f logs/error_trajectory_gpu3.log
```

`--resume`는 이 진단의 `.inference_cache`만 체크포인트 SHA-256, 샘플 ID, test transform 및 코드 해시를 확인한 뒤 재사용한다. 예전 feature cache는 사용하지 않는다. 기존 캐시가 현재 입력과 달라졌다면 자동 무시하지 않고 오류로 중단한다. 변경 사유를 확인한 뒤 해당 cache 파일을 수동으로 치워야 한다.

`run.py`는 단일 프로세스에서 IA10 다음 CAR10을 순차 평가한다. 물리 GPU 3을 `CUDA_VISIBLE_DEVICES`로 선택하고, 프로세스 안에서는 `cuda:0`으로 접근한다. IA10이 끝난 뒤 중단되더라도 같은 명령으로 다시 시작할 수 있다.

출력은 `IA10/`, `CAR10/`, `cross_protocol/`, `run_manifest.json`, `audit.md`, `final_report.md`에 저장된다. 두 protocol이 모두 완료되어야 cross-protocol 표·3개 그래프·최종 보고서가 생성된다. 샘플별 원자료에는 `sample_trajectory.csv`, 집계에는 `final_error_composition.csv`, `loss_recovery.csv`, `typeB_candidate_restriction.csv`가 들어간다.

실행 전 이 폴더의 `audit.md`는 준비 상태만 표시한다. 실제 checkpoint hash, canonical sample-ID digest, 각 task 후보 class 및 평가 수치는 실행 중에 채워진다.
