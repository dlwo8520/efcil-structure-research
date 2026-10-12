# 0923 Structure Prototype Oracle Audit 코드 번들

이 압축 파일은 2026-09-23 GPU 1에서 수행한 IA5/CAR5 5-task Structure 학습 및 oracle audit의 구현, 실제 명령, 설정, 로그, 결과를 검토하기 위한 코드 묶음이다.

## 먼저 확인할 점

- 정식 IA5와 CAR5 학습은 모두 T1–T5까지 완료됐다.
- audit도 두 protocol 모두 T1–T5 전체 checkpoint를 읽어 완료됐다.
- A–D classifier replacement는 final task에서만 실행됐다.
- 첫 sanity 보고서의 `4 classes / class당 2 samples / 1 epoch` 결과와 정식 결과는 분리되어 있다.
- 원 dataset과 약 9.3GB의 `.pt` task checkpoint는 zip에 포함하지 않았다.
- 대신 resolved config, class order, task manifest, checkpoint fingerprint, pre-CA classifier/RNG sidecar, 전체 CSV, 그림, 로그를 포함했다.

## 실제 정식 실행

```bash
conda activate cds
cd ${RESEARCH_WORKSPACE}/structure_prototype_oracle_audit

nohup bash run_gpu1.sh \
  --phase all \
  --protocol IA5 \
  --protocol CAR5 \
  --oracle-head-eval-tasks final \
  > logs/oracle_audit_5task_gpu1.log 2>&1 &
```

queue가 내부에서 수행한 순서는 다음과 같다.

```text
IA5 Structure 학습 T1–T5
IA5 audit T1–T5, A–D final task
CAR5 Structure 학습 T1–T5
CAR5 audit T1–T5, A–D final task
```

실제 audit 명령은 protocol별로 다음 형태였다.

```bash
python run_audit.py \
  --protocol IA5 \
  --gpu 1 \
  --covariance-mode diagonal \
  --oracle-head-eval-tasks final
```

CAR5도 `--protocol CAR5`만 다르다. 정식 실행에는 class/sample/head override가 없었다.

## 폴더 설명

| 경로 | 내용 |
|---|---|
| `oracle_audit_project/` | 작성한 oracle audit 프로젝트 코드, 테스트, 문서 |
| `structure_multicenter_separation/` | 실행 시 import된 base Structure 소스 |
| `structure_e3_followup/` | follow-up 설정/의존 소스 |
| `structure_similarity_aware_followup/` | similarity 설정/의존 소스 |
| `structure_selective_similarity_sweep/` | IA5/CAR5 protocol 및 data manager 소스 |
| `run_artifacts/full_outputs/` | 정식 IA5/CAR5 CSV, report, PNG, manifest |
| `run_artifacts/sanity_outputs/` | 최초 축소 sanity 출력 |
| `run_artifacts/training_metadata/` | resolved config, class order, summary, task manifest, pre-CA sidecar |
| `run_artifacts/logs/` | 실제 background queue 로그 |
| `docs/EXPERIMENT_FLOW.md` | 데이터부터 보고서까지 코드 호출 경로 |
| `docs/KNOWN_CONCERNS.md` | 결과가 이상해 보일 수 있는 구현/재현 이슈 |
| `docs/ENVIRONMENT.md` | 실행 환경과 제외 항목 |

## 재현성상 가장 중요한 파일

1. `oracle_audit_project/oracle_audit/snapshot_learner.py`
2. `oracle_audit_project/oracle_audit/pipeline.py`
3. `oracle_audit_project/oracle_audit/oracle_head.py`
4. `structure_multicenter_separation/models/base.py`
5. `structure_multicenter_separation/models/CDS_adapter.py`
6. `structure_multicenter_separation/utils/adjust_distribution.py`
7. `run_artifacts/training_metadata/*/resolved_config.json`
8. `run_artifacts/full_outputs/*/seed_1993/audit_manifest.json`
9. `run_artifacts/full_outputs/*/seed_1993/oracle_classifier_replacement.csv`

## 중요 결론

현재 A–D 코드는 같은 statistics를 쓰는 Condition A가 원 checkpoint classifier를 **완전히 동일하게 재현하지 못했다**. 차이는 작지만 exact replay라는 주장에는 맞지 않는다. 자세한 수치와 의심 지점은 `docs/KNOWN_CONCERNS.md`에 적었다. 이 문제를 먼저 해소하기 전에는 A–D의 작은 차이를 확정적인 연구 결론으로 사용하면 안 된다.
