# 구현 및 실행 노트

## 순서

1. `fecam/run.py`: F0 기존 final Structure CA head; F1 normalized mean/query Euclidean NCM; F2 old stored mean/cov + new actual mean/cov; F3 모든 class actual mean/cov. F3는 diagnostic-only. FeCAM covariance에는 공식 alpha=10 shrinkage와 correlation normalization을 적용한다.
2. `boundary_audit/run.py`: 재학습 없이 final Structure head 하나로 old 실제/Oracle Gaussian/Stored Gaussian feature의 class별 margin을 평가한다. class별 point 수는 실제 train 수와 같고 synthetic은 3 independent draws다. q05/q10의 class macro gap 및 양의 gap class 비율을 출력한다.
3. `ceos/run.py`: 2가 **IA10/CAR10 양쪽에서**, 두 Gaussian source 모두 macro q05/q10 gap>0이고 과반 class가 양의 gap일 때만 C1 실행. Stored Gaussian old bank를 CEOS로 변환하고 C0와 같은 new bank, pre-CA weights, label order, batch boundaries, SGD, 5 epoch, 256/class exposure를 사용한다. C0는 기존 matched `scores_C`를 재사용한다.
4. `report.py`: 산출물을 `final_report.md`로 집계한다.

이 gate는 실험 전에 정한 보수적 운영 기준이지 통계적 유의성 검정은 아니다. 하나라도 미충족이면 CEOS 성능 실험은 하지 않는다. Test labels는 gate나 CEOS hyperparameter 결정에 사용하지 않는다.

## 실행

기존 `cds` 환경에서, 예를 들어 physical GPU 4:

```bash
cd ${RESEARCH_WORKSPACE}/fecam_boundary_followup
conda activate cds
python -u run.py --gpu 4 --stage all
```

백그라운드:

```bash
cd ${RESEARCH_WORKSPACE}/fecam_boundary_followup
conda activate cds
nohup python -u run.py --gpu 4 --stage all > run_gpu4.log 2>&1 &
echo $! > run_gpu4.pid
tail -f run_gpu4.log
```

GPU 번호는 `--gpu`로 지정하며 process-local 장치는 `cuda:0`이다. 이미 완료된 새 output은 재실행 시 보존된다. 하위 단계만 다시 수행하려면 `--stage fecam|boundary|ceos|report`를 쓴다. `--stage ceos`도 양 protocol gate를 강제한다.

## 결과 위치

- `fecam/{IA10,CAR10}/metrics.json`, `covariance_status.csv`
- `boundary_audit/{IA10,CAR10}/per_class_margins.csv`, `summary.json`
- 조건부 `ceos/{IA10,CAR10}/metrics.json`
- `final_report.md`

covariance 수치 진단은 train count, empirical rank upper bound, pre-shrink diagonal mean, post-shrink diagonal 최소값, inversion fallback으로 제한한다. 768차원 covariance의 class별 정밀 condition number는 비용이 커 기본 실행에서는 계산하지 않는다. 
