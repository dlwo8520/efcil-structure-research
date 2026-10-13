# Cross-Task CA Reweighting Pilot

새 폴더에 독립 구현한 T10 head-only, 2-branch diagnostic. 원본 Structure 코드와 checkpoint를 수정하지 않는다. 선택된 source는 전 class Gaussian이며 Control은 원본 all-class CE, Intervention은 within-task CE와 cross-task term의 고정 가중치 2를 사용한다.

## 환경 활성화 및 실행

아래를 터미널에 입력하면 GPU 번호를 물은 뒤 IA10→CAR10을 순차적으로 **백그라운드**에서 실행한다.

```bash
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
cd ${RESEARCH_WORKSPACE}/cross_task_ca_reweighting
read -rp "사용할 물리 GPU 번호: " GPU_ID
mkdir -p logs
nohup python -u run.py --gpu "$GPU_ID" > "logs/pilot_gpu${GPU_ID}.log" 2>&1 &
echo $! > "logs/pilot_gpu${GPU_ID}.pid"
tail -f "logs/pilot_gpu${GPU_ID}.log"
```

`tail -f`에서 Ctrl+C를 눌러도 백그라운드 실험은 계속된다. 추출 완료 후 `final_report.md`와 `cross_protocol/plots/`의 그래프 5개가 생성된다. 소스 checkpoint 해시, B1 join, CE identity 또는 feature-cache baseline 정합성이 실패하면 안전하게 중단된다. 실패 시 원인을 확인하기 전에는 기존 출력에 다른 실행을 겹치지 말 것.

사전 고정된 판단 기준 및 원본 CA와의 차이/일치 여부는 [implementation_notes.md](implementation_notes.md)에 있다.
