# Structure prototype oracle audit

기존 vanilla Structure의 학습 조건은 유지하고, task-end checkpoint를 frozen read-only 상태에서 분석하는 프로젝트다. 원본 Structure 및 기존 결과 폴더는 수정하지 않는다. 기존 prototype/drift/covariance/Mahalanobis/시각화에 더해 pre-APR 대조, A–D oracle classifier replacement, new-only/old-only, margin, boundary-direction, Gaussian 보조 진단을 포함한다.

## 환경 및 GPU 1 백그라운드 실행

```bash
conda activate cds
cd ${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit

nohup bash run_gpu1.sh \
  > logs/oracle_audit_gpu1.log 2>&1 &
echo $! > logs/oracle_audit_gpu1.pid
```

기본 queue는 `IA5 → IA10 → CAR5 → CAR10` 순서로 exact vanilla Structure task snapshots를 준비한 뒤 audit한다. 특정 protocol만 실행하려면:

```bash
nohup bash run_gpu1.sh --protocol IA5 \
  > logs/ia5_gpu1.log 2>&1 &
echo $! > logs/ia5_gpu1.pid
```

## 진척도

```bash
cd ${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit
tail -f logs/oracle_audit_gpu1.log
python status.py
nvidia-smi -i 1
```

## 기존 완료 checkpoint의 final-task 재사용

IA5 T5를 재학습 없이 audit:

```bash
python run_audit.py \
  --protocol IA5 \
  --gpu 1 \
  --checkpoint ${RESEARCH_WORKSPACE}/structure_e3_followup/outputs/imageneta/B0_STRUCTURE/seed_1993/checkpoint_latest.pt \
  --output-root outputs_legacy_final
```

CAR5 T5도 동일하게 `structure_selective_similarity_sweep/outputs/CAR5/Structure/seed_1993/checkpoint_latest.pt`를 지정한다. 이 방식은 final-task 단면만 제공하므로 APR true drift와 T1→T5 trajectory는 `INCONCLUSIVE`다.

## 단계별 실행

```bash
# 학습만
python run_queue.py --gpu 1 --phase train --protocol IA5

# 이미 보존된 task checkpoint 분석만
python run_queue.py --gpu 1 --phase audit --protocol IA5

# A–D diagnostic head를 모든 task에서 실행(기본은 final task만)
python run_queue.py --gpu 1 --phase audit --protocol IA5 \
  --oracle-head-eval-tasks all

# full-covariance Mahalanobis까지 포함(느림)
python run_audit.py --protocol IA5 --gpu 1 --covariance-mode both
```

`--oracle-head-eval-tasks final|all|none`으로 heavy diagnostic 범위를 정한다. `all`도 원 Structure가 classifier adaptation을 수행하는 T2 이후만 대상으로 한다. production에서는 checkpoint의 `ca_epochs`와 `replay_samples_per_class`를 그대로 쓰며, `--oracle-head-epochs`와 `--oracle-head-samples-per-class`는 sanity/debug용이다.

과거 checkpoint에 `pre_ca_classifier/task_XX.pt`가 없으면 A–D는 동일한 post-CA head에서 시작하며 결과에 `shared_post_ca_legacy_init`로 명시된다. 새 audit-ready 학습은 CA 직전 head/RNG sidecar를 자동 저장하므로 동일 초기화·동일 RNG의 재현 비교가 가능하다.

결과는 `outputs/{IA5,IA10,CAR5,CAR10}/seed_1993/` 아래에 저장되고, 통합 보고서는 `outputs/structure_prototype_oracle_audit.md`다. covariance 구현의 코드 감사 문서는 [structure_covariance_code_audit.md](structure_covariance_code_audit.md), revision 설계는 [structure_oracle_audit_revision_plan.md](structure_oracle_audit_revision_plan.md)다.
