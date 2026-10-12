# BOB 공식 구현 확인 및 IA10 adaptation

확인일: 2026-10-12. 기존 Structure/생성 pilot을 수정하지 않는다.

## 출처와 고정 버전

- [공식 repository](https://github.com/princetonvisualai/BeyondObjects), commit `8bc17e0537991f6bb5c0399339bddb3d1418d220`, `vendor/BeyondObjects/` 원본을 그대로 보존.
- [논문](https://arxiv.org/abs/2510.24078), context preservation/marginalization과 Qwen2.5-VL-7B captioning 절차 확인.
- 공식 핵심 함수: `finetune/finetune_t2i.py::Dataset/main`, `generation/generate.py::get_pipe/GenerateImage.run_pipe`.
- Caption 모델은 설치 시 resolved HF revision을 `assets/model_lock.json`으로 고정하고 이후 local snapshot만 사용한다.
- Diffusion: 논문에 사용된 SD1.5 variant, 기존 local snapshot `runwayml/stable-diffusion-v1-5@451f4fe16113bff5a5d2269ed5ad43b0592e9a14`. 공식 YAML의 기본 SD2.1-base와 다르지만 논문에는 두 variant가 있다. SD2.1 remote 접근은 이 환경에서 401이므로 SD1.5를 사전 고정했다. img2img 대체가 아니다.
- Semantic WNID→name mapping은 [timm의 공개 label metadata](https://github.com/huggingface/pytorch-image-models/blob/main/timm/data/_info/imagenet_synset_to_lemma.txt)를 로컬 고정하고 SHA-256으로 추적한다. WNID 문자열 자체를 class prompt로 사용하지 않는다.

## 공식 핵심 algorithm을 그대로 연결

1. 현재 20 class의 real train image 각각에서 frozen Qwen2.5-VL-7B로 background와 pose를 별도 추출한다. Query에는 class 이름을 넣지 않고 generic object라고 요청한다.
2. Class 이름 + 해당 real image의 background/pose caption을 official Dataset의 prompt에 넣는다.
3. 원 official main의 diffusion objective로 **U-Net attention 및 CLIP text encoder attention LoRA 모두** 학습한다. VAE와 base weights는 frozen이다.
4. Current-task 전체 context bank에서 background/pose **pair**를 target class와 독립적으로 샘플링하여 official text-to-image inference를 호출한다. Background/pose를 서로 독립적으로 섞지 않는다.
5. Synthetic bank의 수와 무관하게 downstream aux는 real anchor당 하나이며 학습 step을 늘리지 않는다.

공식 caption extraction 실행 코드는 공개 repo에 없고 기존 dataset별 TSV만 있다. 따라서 논문의 Qwen captioning을 ImageNet-A adapter로 구현했다. Dataset descriptor가 여러 object 종류를 포함하므로 `object`를 사전 고정했다. Caption이 실제 class 정보를 완전히 제거했다는 주장은 하지 않는다.

## Hyperparameters / resource 의미

Official `finetune.sh`의 400 epoch, batch 8, AdamW lr 1e-4, cosine LR, warmup 100, grad accumulation 1; config default rank 16을 따른다. FP32(`mixed_precision=no`)이며 single GPU의 메모리 완화를 위해 gradient checkpointing을 활성화한다. 생성은 official default guidance=2, steps=50, LoRA scale=1, 100 images/class, 512×512, batch=1이다. Result-based sweep은 없다.

원 실험은 5/10-shot dataset이다. 이번에는 **현재 task의 모든 real train**을 diffusion fine-tuning에 쓴다. IA10 adaptation이지 공식 few-shot 숫자의 재현이 아니다. 5,981 real train/10 tasks에서 default LoRA updates는 합계 300,800이며 generation candidate는 20,000장이다. Captioning/LoRA 시간이 이전 SD img2img pilot보다 크게 증가한다. 시간을 줄이려고 epoch/shot 수를 조용히 낮추지 않았다.

## 필요한 최소 integration/compatibility 변경

- 공식 ImageNet-A loader가 없어 current-only ImageFolder symlink/TSV/YAML adapter를 새 프로젝트에 추가했다. Label order/partition/split은 기존 IA10 source manifest와 검증한다.
- Upstream `config.py`에는 `train_text_encoder`가 없지만 finetune main 및 save/load hook이 참조한다. Wrapper에서 `args.train_text_encoder=True`만 보완한다. 원 code는 이미 두 LoRA를 만들고 optimizer에 모두 넣는다.
- 외부 W&B 대신 local TensorBoard logging을 사용한다. 학습 수식 변경은 없다.
- TensorBoard가 `None`/class name list를 hparams로 받지 못하는 오류는 integration-local Accelerator subclass의 tracker config 복사본만 scalar/JSON string으로 변환하여 해결한다. 원 args 및 공식 파일은 그대로이며 full config JSON을 별도로 보존한다.
- 원 generation loop 대신 재실행 가능한 per-record driver를 쓰되 official `get_pipe` 및 `run_pipe`를 호출한다. Seed는 task/class/sample ID로 고정하며 prompt/context/model/config를 모두 기록한다.
- Generator는 task마다 pretrained base와 새 LoRA에서 시작한다. Old LoRA/old real/old generated image를 다음 task의 generation이나 learner input으로 쓰지 않는다. Artifact는 삭제하지 않고 learner access만 해제한다.
- Upstream LoRA resume는 partial epoch를 재실행할 수 있다. 이를 그대로 유지하고 자원 기록에 명시한다. Downstream Structure는 epoch checkpoint/RNG stream으로 이어간다.

## Downstream 공정성

각 branch는 T1 초기화부터 자기 checkpoint만 이어 T10까지 진행한다. Vanilla T8 fork/GM은 없다. Both: ordinary real anchor 0.5 CosFace + aux 0.5 CosFace + real-only SCC 5. APR 및 all-class Gaussian CA는 기존 함수로 위임하고 저장 current statistics는 real only다.

S auxiliary는 같은 anchor image의 두 번째 strong view다. 기존 pilot의 RandAugment-before-ToTensor / RandomErasing recipe를 재사용하지만, 이전의 작은 accepted parent subset으로 S를 제한하지 않는다. BOB는 source parent가 필수인 img2img가 아니므로 anchor label의 current synthetic bank를 uniformly sampling한다. 양쪽 real/aux exposure·batch·update 수는 같고 transformed real anchor까지 hash로 검사한다.

Both CA의 standard-normal stream은 이전 matched pilot처럼 `1993+task*100000`으로 맞춘다. 같은 seed/stream이지 서로 다른 representation에서 만들어진 mean/cov/head까지 같다는 뜻은 아니다. Vanilla와 exact RNG parity 재학습을 주장하지 않는다.

## QC / 정보 경계

Decode 실패, 64px 미만, std≤1인 blank, exact pixel duplicate 및 diffusion safety failure만 사전 고정된 기술 규칙으로 제외한다. 나머지 bank를 모두 동일하게 사용한다. Human QC, confidence filtering, test-derived subtype/score/accuracy를 이용한 generator 선택은 없다. Label fidelity는 검증되지 않았다.

Protocol loader는 기존 split 검증을 위해 train/test path metadata를 읽지만 generator에 넘기는 CSV/TSV는 current task real train만이다. Future/test image를 captioning·LoRA에 전달하거나 future label을 generation target으로 넣지 않는다. 생성 bank·context caption·LoRA는 체크포인트 learner state에 포함하지 않는다.

## 평가 정의와 재시작

각 task의 `after_task` 이후 task-end model에서 평가하고 그 checkpoint를 보존한다. Vanilla는 기존 저장 task-end per-sample trajectory에서 재집계한다. 원 training summary는 before-after_task라 IA10 마지막 1 sample 차이가 있으므로 별도 reference로만 남기며 숫자를 강제로 맞추지 않는다.

실행 도중 code/spec/model lock/label map을 변경하면 provenance hash가 달라져 재개를 거부한다. 다른 recipe는 별도 폴더에서 수행해야 한다. 이번 default success 기준 중 'large old sacrifice'의 수치 threshold는 명세에 없으므로 결과를 보고 새로 설정하지 않는다. 모든 사전 숫자 기준이 양수이고 final/mean old가 감소하지 않을 때만 conservative BETTER를 자동 표시한다.
