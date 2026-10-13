# B1 ranking-preserving task correction pilot

기존 IA10/CAR10 seed-1993 Structure T10 **post-CA checkpoint를 그대로 고정**하고 C0(원본), C1(task별 상수), C2(입력별 task-common 선형 보정)를 비교한다. 원본 연구 폴더나 checkpoint는 수정하지 않는다. 실제 GPU 실험은 사용자가 실행한다.

```bash
cd ${RESEARCH_WORKSPACE}/b1_ranking_preserving_correction
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
read -rp '사용할 GPU 번호: ' PILOT_GPU
python -u run.py --gpu "$PILOT_GPU"
```

백그라운드 실행은 아래 명령을 사용한다. 프롬프트에서 물리 GPU 번호를 입력한다.

```bash
cd ${RESEARCH_WORKSPACE}/b1_ranking_preserving_correction
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
mkdir -p logs
read -rp '사용할 GPU 번호: ' PILOT_GPU
nohup python -u run.py --gpu "$PILOT_GPU" > "logs/pilot_gpu${PILOT_GPU}.log" 2>&1 &
echo $! > "logs/pilot_gpu${PILOT_GPU}.pid"
tail -f "logs/pilot_gpu${PILOT_GPU}.log"
```

진행상태는 `run_manifest.json`, 결과는 `IA10/metrics.json`, `CAR10/metrics.json`, `final_report.md`에서 확인한다. `tail -f`는 Ctrl+C로 빠져나와도 백그라운드 실험을 중단하지 않는다.

`run.py`는 IA10 다음 CAR10을 순차 실행한다. 각각의 baseline prediction parity, fixed B1 수, correction zero identity, within-task ranking 불변성, paired repair/harm identity를 검사한다. 불일치가 있으면 해당 protocol에서 실패 상태를 기록하고 뒤 protocol은 시작하지 않는다. C1/C2는 같은 fit Gaussian bank를 공유하며 check bank와 test는 학습에 쓰지 않는다. 원본 CA는 다시 실행하지 않는다.

CPU 코드 테스트만 하려면:

```bash
python -m unittest discover -s tests -v
```
