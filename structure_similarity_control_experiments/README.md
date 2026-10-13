# Similarity-Selective AM-Drift Controls (C0--C4)

이 폴더는 기존 Structure, E3 후속 실험, similarity-aware 후속 실험을 수정하지 않고
읽기 전용 의존성으로 사용하는 추가 실험 레이어다.

## 실험

| ID | pair 제어 |
|---|---|
| C0 | 모든 pair에 AM-Drift 적용 |
| C1 | similarity 상위 25% separation off |
| C2 | similarity 하위 25% separation off |
| C3 | 같은 개수의 seed 고정 random pair off |
| C4 | similarity 상위 25%에 0.3 배 separation |

C1/C2/C3은 각 new class마다 `ceil(0.25 * old_class_count)`개의 pair를 정확히
선택한다. 동률은 old class ID로 결정하며 C3은 학습 RNG와 분리된 generator를 쓴다.

## GPU 2 실행

```bash
conda activate cds
cd ${RESEARCH_WORKSPACE}/structure_similarity_control_experiments
python scripts/run_experiments.py \
  --stage pilot \
  --gpu 2 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
```

우선순위는 `C0 -> C1 -> C3 -> C4 -> C2`이고 pilot은 seed 1993 하나다.
전체 3-seed 실행은 `--stage final`을 사용한다.

백그라운드 실행:

```bash
mkdir -p logs
nohup python -u scripts/run_experiments.py \
  --stage pilot \
  --gpu 2 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
  > logs/control_pilot_gpu2.log 2>&1 &
echo $! > logs/control_pilot_gpu2.pid
```

진척도와 로그 확인:

```bash
python scripts/status.py
tail -f logs/control_pilot_gpu2.log
nvidia-smi -i 2
```

중단 후 같은 명령을 다시 실행하면 `checkpoint_latest.pt`가 있는 미완료 run은 자동
resume하고 완료된 run은 건너뛴다.

## 검증

```bash
python run.py --config configs/C0.json configs/C1.json configs/C2.json \
  configs/C3.json configs/C4.json --validate-only
python scripts/smoke_test.py
```

`configs/smoke_C3.json`은 5-task 각 1 epoch의 실제 데이터/GPU 통합 확인용이며 논문
결과에 포함하지 않는다.
