# 실행 방법

코드는 작성 및 작은 CPU 검증까지만 수행했다. ImageNet-A 본 학습은 아래 명령으로
사용자가 실행한다. GPU 번호는 환경 변수 수정 없이 `--gpu`에 물리 번호를 넣는다.
예를 들어 `--gpu 4`는 process 내부에서 logical `cuda:0`으로 remap된다.

## 환경과 단위 검증

```bash
cd ${RESEARCH_WORKSPACE}/structure_multicenter_separation
conda activate cds
python -m unittest discover -s tests -v
```

작성 완료 시 `cds` conda 환경에서 위 검증을 실제 실행해 8개 테스트가 모두 통과했다.
검증 범위는 E0--E7 config 행렬, single-center 동치, K-means 결정성/K 축소, replay
배분, iso/diag/full 거리, separation loss와 gradient, feature-space mismatch 차단,
APR cache 비누적이다. 실제 ImageNet-A protocol 로딩도 CPU로 확인했으며 200 classes,
`[40,40,40,40,40]`, train/test 5,981/1,519장, 내용 중복 0건이었다. 이 검증은 전체
ImageNet-A 학습이나 정확도 결과를 의미하지 않는다.

## 단일 run

먼저 seed 1993의 E0를 5 tasks 끝까지 실행한다.

```bash
python main.py \
  --config exps_jsons/imageneta_5task/E0.json \
  --gpu 4 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
  --seed 1993
```

E1 예시는 다음과 같다. E2--E7도 config 이름만 바꾼다.

```bash
python main.py \
  --config exps_jsons/imageneta_5task/E1.json \
  --gpu 4 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
  --seed 1993
```

## 권장 순서 자동 실행

pilot은 seed 1993을 E0, E1, ..., E7 순서로 한 GPU에서 순차 실행한다.

```bash
python scripts/run_e0_e7.py \
  --stage pilot \
  --gpu 4 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
```

pilot 검토 후 남은 두 seed를 실행한다.

```bash
python scripts/run_e0_e7.py \
  --stage remaining \
  --gpu 4 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
```

스크립트는 여러 GPU에 run을 자동 분산하지 않는다. `--gpu 2,3`처럼 입력하면 해당
단일 run 안에서 DataParallel을 사용한다.

## 재개

config/seed/output root를 원래 run과 동일하게 지정한다.

```bash
python main.py \
  --config exps_jsons/imageneta_5task/E7.json \
  --gpu 4 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
  --seed 1993 \
  --resume outputs/imageneta/E7/seed_1993/checkpoint_latest.pt
```

pilot 전체를 이어가면서 완료 run은 건너뛰려면 다음을 사용한다.

```bash
python scripts/run_e0_e7.py \
  --stage pilot \
  --gpu 4 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
  --resume-incomplete
```

## 집계

완료되지 않은 run은 제외 사유와 함께 별도 기록되고 평균에 포함되지 않는다.

```bash
python scripts/aggregate_results.py --output-root outputs --dataset imageneta
```

결과는 `outputs/imageneta/aggregate/comparison_report.md`,
`comparison_per_seed.csv`, `comparison_mean_std.csv`와 `figures/`에 생성된다. 모든
E0--E7에서 동일 protocol로 완료된 공통 seed만 비교 평균에 포함하며, 제외 run은
`excluded_runs.csv`에 이유를 남긴다. 기록 파일별 정의는
[`RECORDING_SCHEMA_KO.md`](RECORDING_SCHEMA_KO.md)를 참고한다.

현재 새 기록 형식 도입 전 실행된 E0/seed 1993의 과거 epoch loss는 소급 생성되지
않는다. 해당 checkpoint에서 resume하면 이후 epoch부터 `epoch_metrics.csv`가 기록된다.
다만 과거 구간까지 완전한 loss/correct-count 비교가 필요하면 기존 결과를 보존하고 새
output root에서 처음부터 실행한다.

```bash
python scripts/run_e0_e7.py \
  --stage pilot \
  --gpu 4 \
  --data-root ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a \
  --output-root ${RESEARCH_WORKSPACE}/structure_multicenter_separation/outputs_recording_v2
```

기존 checkpoint를 resume한 run은 학습 완료 결과를 남기지만 `recording_status`가
`partial_legacy`가 되어 엄격한 E0--E7 공통 기록 집계에서는 제외된다.

## 주의

- E4/E7 full covariance는 CPU 통계 메모리와 GPU factor cache가 크다. 기본 config는
  반복적인 PCIe 전송을 피하려고 full factor를 GPU에 둔다.
- 첫 실행은 timm의 ViT-B/16-IN21K pretrained weight가 로컬 cache에 없으면 다운로드가
  필요하다.
- 현재 local ImageNet-A split의 provenance는 논문/공식 저장소에서 확인되지 않았으므로
  결과를 논문 표의 엄밀한 재현으로 표시하지 않는다.
