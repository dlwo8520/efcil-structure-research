# RP-ProtoCLAD Task-5

이 폴더는 RP-ProtoCLAD 네 run의 코드와 output을 기존 `structure_selective_similarity_sweep`와 분리한 extension이다. Parent 쪽에는 CAR5 vanilla Structure에서 비활성 separation diagnostic이 `None` cache를 읽던 guard와 해당 checkpoint resume 판정만 최소 수정했다.

실행 범위는 다음으로 고정되어 있다.

```text
IA5  × RP_C_P020_L030, RP_C_P020_L100
CAR5 × RP_C_P020_L030, RP_C_P020_L100
seed=1993, P=0.20, RP dim=2048, RP seed=1993
```

IA10/CAR10, 추가 seed, RP dimension/P/lambda/alpha sweep은 지원하지 않는다.

## 구현 요약

기존 frozen original-space similarity table로 Top-P conflict class를 선택한다. 선택이 끝난 뒤 Proto-CLAD auxiliary cosine만 아래 공간에서 계산한다.

```text
original feature z
  → fixed R ~ N(0, 1/768), R shape [768, 2048]
  → ReLU
  → L2 normalize
  → selected old APR prototype와 RP-ProtoCLAD cosine loss
```

classifier, SCC, APR, covariance/tau, classifier adaptation은 원 공간과 parent 설정을 그대로 사용한다. RP matrix는 학습되지 않으며 모든 task와 두 lambda run에서 동일한 생성 규칙과 checksum을 사용한다.

## 코드 검증

```bash
cd ${RESEARCH_WORKSPACE}/structure_rp_protoclad_task5
conda activate cds
python scripts/smoke_test.py
```

이 검증은 CPU에서 projection/loss unit test와 네 config의 parent-equivalence validation만 수행하며 학습을 시작하지 않는다.

실행 계획과 현재 parent 상태 갱신:

```bash
python scripts/build_rp_plan.py
```

## GPU 2 백그라운드 실행

기존 reduced Stage 1의 CAR5 parent가 완료된 뒤 실행하는 것이 가장 단순하다. 실행기 자체도 각 run 직전에 parent 상태를 다시 검사한다.

```bash
cd ${RESEARCH_WORKSPACE}/structure_rp_protoclad_task5

nohup bash run_rp_protoclad_task5.sh 2 \
  > logs/rp_task5_gpu2.log 2>&1 &

echo $! > logs/rp_task5_gpu2.pid
```

진행 확인:

```bash
tail -f ${RESEARCH_WORKSPACE}/structure_rp_protoclad_task5/logs/rp_task5_gpu2.log
```

개별 run 로그는 `logs/rp_task5/`에 저장된다. 재실행하면 COMPLETE는 건너뛰고, checkpoint가 있는 PARTIAL RP run만 resume한다. parent가 미완료된 run은 `DEFER`하며 종료 코드 2를 반환하므로 parent 완료 후 같은 명령을 다시 실행하면 된다.

명령만 확인:

```bash
bash run_rp_protoclad_task5.sh 2 --dry-run
```

## 결과

- 실행 계획: `rp_experiment_plan.md`
- live 결과 보고서: `rp_protoclad_report.md`
- 통합 성능표: `outputs/metrics_per_seed.csv`
- run output: `outputs/{IA5,CAR5}/RP_C_*/seed_1993/`

각 run에는 original/RP pair geometry, correct-new gap, nonlinear prototype approximation, RP norm/sparsity, gradient interaction, full/selected-pair confusion 진단이 저장된다.
