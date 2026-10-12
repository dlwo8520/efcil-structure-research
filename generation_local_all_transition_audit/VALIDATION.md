# 검증 기록

검증일: 2026-10-12 KST.

## 실행 범위

`CUDA_VISIBLE_DEVICES=`로 GPU를 숨긴 cds 환경에서 analysis-only 코드를 실행했다. 기존 S/G/R prediction CSV와 고정 vanilla subtype만 읽었다. Training, generation, augmentation 및 inference는 실행하지 않았다. `run.py`는 torch를 import하지 않는다. 최종 분석 실행의 기록 시간은 약 4.47초이며 새 학습 완료 시간이 아니다.

## 확인 결과

- Unit test 10개 통과: W/C/K 상태, 모순 prediction 거부, T9/T10 class 범위, 모든 9개 전이와 net identity, W→C repair와 K→W harm, class macro/micro 차이, 빈 cohort, 고정 vanilla subtype의 seed 간 일치, historical binary net, 한 dataset의 이득을 공통 증거로 잘못 판정하지 않는 규칙.
- IA10 147행 + CAR10 758행, 총 905 direct sample의 unique ID와 필수 wide CSV column 확인.
- 32개 summary 행렬(2 protocol × 2 시점 × 4 cohort × G−S/G−R)을 wide sample CSV에서 독립 재집계해 모든 cell, local/all net 및 local %p와 일치 확인.
- 원본 입력 35개 파일의 실행 전후 SHA-256 일치. 해시는 `run_manifest.json`에 저장되어 있다.
- 원본 metrics와 T9/T10 direct all accuracy, paired repair/harm 및 whole-seen 결과 일치. T1 local=all 검증.
- PNG 19개 정상 decode. 통합 bar 및 CAR10 T9 heatmap도 눈으로 확인.
- 보고서의 embed 그림 7개와 전체 local link 19개 존재. Markdown table column 수 일치.

## 핵심 재집계 값: G−S direct T9 cohort

| Protocol | 시점 | Local repair | Local harm | Local net | All repair | All harm | All net |
|---|---|---:|---:|---:|---:|---:|---:|
| IA10 | T9 | 1 | 0 | +1 | 2 | 2 | 0 |
| IA10 | T10 | 1 | 1 | 0 | 1 | 1 | 0 |
| CAR10 | T9 | 2 | 7 | −5 | 8 | 12 | −4 |
| CAR10 | T10 | 1 | 9 | −8 | 8 | 14 | −6 |

W→C는 IA10 T9/T10에서 0/1개, CAR10 T9/T10에서 2/1개다. IA10 T9 local +1은 W→K이며 A-within이 아니라 non-A에서 발생했다. 새로운 subtype·selector·threshold를 만들지 않았다.
