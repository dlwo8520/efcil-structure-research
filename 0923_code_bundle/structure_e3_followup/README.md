# Structure E3 follow-up experiments

기존 `${RESEARCH_WORKSPACE}/structure_multicenter_separation`은 수정하지 않고 읽기 전용
base로 import한다. 후속 손실, CEOS replay, 설정, 진단과 실행기는 모두 이 폴더에
있다. 실제 ImageNet-A 학습은 자동으로 시작하지 않는다.

## 구현 범위

- E3 lambda sweep: `0.1, 0.3, 0.5, 1.0`
- Structure + FCS PIC-inspired separation
- E3 + LUCIR-inspired Angular Hard Negative
- E3-Norm 근사 ablation
- Structure + CEOS50 / CEOS100 (ACB 제외, class당 총 replay 수 고정)
- variance / age / drift adaptive threshold를 각각 독립 실행
- feature norm, geometry, intrusion, loss, interval gradient, CEOS sample 진단
- run별 source snapshot/config/git 상태/diff/environment/command 저장

`REFERENCE_AUDIT.md`에 원본 공식 구현과 Structure용 변형을 구분해 기록했다.

## 실행

GPU 번호는 실행할 때 지정한다.

```bash
cd ${RESEARCH_WORKSPACE}/structure_e3_followup

# 설정만 검사 (학습하지 않음)
python run.py --validate-only --config configs/L03.json configs/FCS_pic.json

# 한 방법, 한 seed
python run.py \
  --config configs/L03.json \
  --gpu 1 \
  --seed 1993 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a

# 공통 3 seeds. 쉼표 또는 --seed 반복 사용 가능
python run.py \
  --config configs/L03.json \
  --gpu 1 \
  --seed 1993,1996,1997 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
```

여러 설정도 한 명령에서 순차 실행할 수 있다.

```bash
python run.py \
  --config configs/L01.json configs/L03.json configs/L05.json configs/L10.json \
  --gpu 1 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
```

## 백그라운드 실행과 진척도

```bash
mkdir -p logs
nohup python -u run.py \
  --config configs/FCS_pic.json \
  --gpu 1 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
  > logs/fcs_pic.log 2>&1 & echo $! > logs/fcs_pic.pid

tail -f logs/fcs_pic.log
ps -fp "$(cat logs/fcs_pic.pid)"
nvidia-smi
```

완료 여부는 log의 `--- Finished ... ---`와 각 run의 `run_metadata.json`에서
`"status": "complete"`를 함께 확인한다.

## 작은 작동 검사

데이터셋이나 pretrained ViT를 읽지 않는 CPU synthetic smoke test다. 각 loss의
forward/backward/optimizer step, base E3 수치 일치, adaptive tau, CEOS50/100 총량과
constraint, source snapshot을 확인한다.

```bash
${RESEARCH_WORKSPACE}/miniconda3/envs/cds/bin/python scripts/smoke_test.py
```

결과는 `reports/smoke/latest.json`에 기록된다. 이는 성능 실험이 아니며 accuracy
주장의 근거로 사용하면 안 된다.

## 설정 주의

- `FCS_pic.json`: E3 diagonal loss를 끈 PIC 단독 비교다.
- `E3_ahn.json`: 현재 기본 `sep_lambda=0.3`; sweep 후 best lambda로 명시적으로 수정한다.
- `CEOS50/100.json`: E3/PIC/AHN을 모두 끄며 ACB를 사용하지 않는다.
- adaptive 세 설정은 서로 결합하지 않는다.
- `E3_norm.json`은 old raw data 없이 delta-method diagonal covariance를 사용하는 근사
  ablation이며 원본 Structure 방법이 아니다.
- 최종 결론은 세 공통 seed가 모두 끝난 뒤에만 작성한다.
