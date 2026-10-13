# CIL/EFCIL 연구 관리 안내

기존 파일은 원본 위치에 그대로 보존하고, 여기에 정리한 복사본만 commit합니다. 상위 Structure 구현과 자체 제안/진단을 `experiments/registry.json`과 카드에서 구분했습니다. 다른 실험의 sibling import와 config 상속을 유지했으며 계산 로직을 임의로 통합하지 않았습니다.

현재 상태는 `docs/STATUS.md`입니다. generic generation pilot은 해당 recipe STOP이며, 최신 BOB 비교는 GPU smoke 실패가 기록된 미완료 상태입니다. 보고서의 real-only SupCon은 제안 단계입니다. oracle/old-data 진단을 exemplar-free 학습 성과로 표시하지 않았습니다.

새 실험은 `python tools/new_experiment.py --id EFCIL-YYYYMMDD-001 --name '이름' --parent EFCIL-기준ID`로 등록합니다. 코드는 새 실험 폴더 또는 명시한 sibling extension, 설정/분석/선택된 결과는 `experiments/ID/`, 대용량 실행 출력은 `new_runs/ID/`에 저장합니다. 날짜별 실제 기록은 `research_log/`, 방법 변경은 `CHANGELOG.md`에 남깁니다. 기록 없는 날짜는 채우지 않습니다.

큰 원자료는 `python tools/restore_tables.py`로 복원하고, `python tools/rebuild_indexes.py`, `python tools/validate_export.py`를 실행한 뒤 변경 파일을 검토해 commit/push합니다. 자세한 흐름은 `docs/WORKFLOW.md`, 환경/실행 명령은 `docs/REPRODUCIBILITY.md`입니다.

2026-10-13 KST에 [GitHub private 저장소](https://github.com/dlwo8520/efcil-structure-research)로 업로드하고 main 및 연구 이력 태그 일치를 확인했습니다. 후속 변경은 검토·commit 후 `git push origin main`으로 반영합니다. `docs/UPLOAD.md`에 업로드 기록과 확인 명령이 있습니다. 기존 source Git 이력은 보존하며 개인 경로가 들어 있는 초기 진단 Git bundle은 로컬 보관만 합니다. 실제 GPU 전체 재현 성공을 주장하지 않습니다.
