# ImageNet-A 10-task: BOB-full vs S-full

실행은 사용자가 한다. 기존 프로젝트는 읽기 전용으로 재사용하고 여기서만 새 파일을 만든다. CAR/CIFAR/GM/vanilla 재학습은 없다.

## 처음 한 번: 생성용 환경 설치

```bash
cd ${RESEARCH_WORKSPACE}/ia10_bob_vs_strong_full
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
bash setup_bob_env.sh
```

새 `.bob_env`에만 설치한다. Qwen2.5-VL-7B weights를 다운로드하므로 네트워크·disk 공간이 필요하다. Existing SD1.5 local snapshot을 재사용한다. 이 단계는 학습/이미지 생성을 실행하지 않는다.

## GPU 번호를 입력하고 백그라운드 실행

```bash
cd ${RESEARCH_WORKSPACE}/ia10_bob_vs_strong_full
source ${RESEARCH_WORKSPACE}/miniconda3/etc/profile.d/conda.sh
conda activate cds
bash run_background.sh
```

입력한 물리 GPU 하나만 사용한다. 내부 logical GPU는 `cuda:0`이므로 `--gpu 4`도 4번 물리 GPU에서 실행된다. 실행 순서: preflight/CPU smoke → 1-class BOB GPU smoke(2 real/LoRA 1 update/생성 1장) → 실제 ViT adapter 짧은 GPU backward·checkpoint smoke → S-full T1..T10 → BOB-full 각 task의 caption/LoRA/generation/acquisition → 결과 집계·보고서·그림. Smoke는 `feasibility/`에 격리하고 test 정확도를 계산하지 않는다. 같은 GPU에서 순차 실행하며 다른 experiment process를 중지하지 않는다.

```bash
# GPU를 직접 인자로 주려면
bash run_background.sh 4
# foreground
bash run_full.sh 4
```

Default generator는 task마다 LoRA 400 epoch, 총 약 300,800 generator updates 및 20,000 candidate images다. **이전 img2img pilot보다 상당히 오래 걸릴 수 있다.** Full task 학습은 baseline T1 30 epoch/T2..T10 15 epoch로 유지한다. GPU 실행 feasibility는 사용자 실행에서 확인해야 하며 code 작성 단계의 CPU smoke가 actual diffusion generation 검증은 아니다.

## 진척도 / 중단 후 재개

```bash
python run.py --stage status
tail -f logs/full_gpu4_실제출력된시간.log
```

Background launcher가 정확한 LOG/PID 경로를 출력한다. S/BOB 각 완료 task 수, 현재 caption/generated count, Structure epoch/CA를 `progress.json`과 log에 기록한다. 외부 LoRA는 stdout/tqdm step과 TensorBoard log를 남긴다.

재실행은 같은 `bash run_background.sh`다. 완료 task·caption·generation을 보존하고 epoch checkpoint에서 이어간다. 기존 lock을 지우거나 다른 GPU job을 kill하지 않는다. 단일 queue만 허용한다. GPU 번호는 재개 시 바꿀 수 있지만 code/spec/model recipe는 바꿀 수 없다.

## GPU 없이 검증 / 결과 다시 집계

```bash
python run.py --stage preflight
python run.py --stage smoke
# 생성 환경의 실제 Accelerate/TensorBoard 회귀 검증 (GPU/학습 없이)
.bob_env/bin/python -m fulltraj.bob_worker logging-check
python run.py --stage analyze
```

Preflight는 class order/split/vanilla trajectory를 검사하고 generator update 예상량을 기록한다. Smoke는 toy network 2-step backward, original checkpoint save/load, same-anchor RNG, SCC source, technical QC 등을 CPU에서 검증한다. Analyze는 저장 prediction만 읽고 incomplete이면 PARTIAL report로 명시한다.

## 저장 결과

- `checkpoints/{S_full,BOB_full}/task_checkpoints/task_01.pt` … `task_10.pt`: 자기 branch 누적 checkpoint, 원 snapshot을 덮어쓰지 않음.
- `matched_training_Txx.json`: real/aux count, steps, anchor order/view hashes, LR schedule.
- `bob_tasks/Txx/{real_train.csv,captions.csv,generation_manifest.csv,bank_complete.json}`: train-only BOB provenance 및 기술 QC.
- `per_task_metrics.csv`, `per_sample_predictions.csv`, `repair_harm_by_task.csv`, `cohort_retention.csv`.
- `generation_manifest.csv`, `resource_usage.json`, `process_attempts.json`: 전체 generation/caption/LoRA 비용 및 failed attempt 포함.
- `figures/`: 요구한 trajectory/new delta/seen delta/final repair-harm/cohort heatmap 5개.
- `final_report.md`: full completion 때 final/average incremental/New wins/old trade-off/repair-harm 판정.

Synthetic data는 current acquisition 전용이고 moments/APR/statistics에 들어가지 않는다. Future/test-driven selection 또는 human QC는 하지 않으며 label fidelity를 보장하지 않는다. 자세한 공식 구현 차이는 [BOB notes](bob_method_notes.md)에 기록했다.

## 구현 검증 (2026-10-12)

CPU contract tests 15개, Python compile 검사 및 shell 문법 검사를 통과했다. 기존 IA10 class order/split, 10개 vanilla checkpoint 및 저장 reference의 sample ID/label/10-task prediction coverage를 preflight로 검증했다. T1 real-train metadata/symlink만 준비했으며 diffusion generation·실제 ViT GPU 학습·full trajectory는 실행하지 않았다. GPU feasibility는 사용자가 위 실행 명령을 실행할 때 가장 먼저 수행한다.

## TensorBoard 초기화 오류 수정 (2026-10-12)

GPU 5의 두 초기 실행은 BOB smoke의 tracker 초기화에서 실패했고, LoRA optimizer update 및 S/BOB 본 학습은 시작하지 못했다. Official args의 `classnames` 목록과 `None` 값을 TensorBoard hparams가 지원하지 않는 것이 원인이었다. 원 official 코드/훈련 args를 변경하지 않고 tracker로 전달하는 복사본만 JSON 문자열로 변환한다. 원 설정은 `lora/logs/full_tracker_config.json`에 별도 저장한다. Full launch는 모델을 GPU에 올리기 전에 생성 환경의 실제 logger를 CPU에서 검증한다. 실패 로그·caption을 보존하고 동일 실행 명령으로 재개한다.

수정 후 CPU 테스트 18개 및 `.bob_env`의 실제 Accelerator/TensorBoard logger 검증을 통과했다. 후자는 원 오류를 재현한 뒤 호환 wrapper의 성공·원 args 보존·CUDA 미초기화를 확인했다. 수정본으로 GPU 학습은 아직 실행하지 않았다. 학습 artifact가 전혀 없고 model/recipe/core 코드가 동일한 것을 audit한 뒤, 이전 실패 manifest와 hash를 `repairs/tensorboard_config_20261012/`에 보존하여 새 코드 hash로 갱신했다. 다른 변경에 대한 일반적인 hash 불일치 차단은 그대로다.
