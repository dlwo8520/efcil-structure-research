# Type A Current-Task Generative Augmentation — 탐색적 pilot 결과 보고서

작성일: 2026-10-12 (KST)  
대상: IA10 / CAR10, seed 1993, GPU 4  
범위: T1 acquisition 및 vanilla T8에서 분기한 T9→T10 suffix  
상태: 예정된 모든 분기 완료. 사람 검수 없이 고정된 자동 sanity 규칙을 적용한 exploratory pilot.

## 1. 핵심 결론

**현재 generation recipe는 전체 trajectory 실험으로 확장하지 않는 것이 사전 기준에 맞다. IA10과 CAR10 모두 `STOP_CURRENT_RECIPE`다.**

T1에서는 생성 증강 G가 strong conventional augmentation S보다 IA10 **+1.143%p**, CAR10 **+1.074%p** 높았다. 그러나 T9의 신규 class accuracy 차이는 각각 **0.000%p / −0.528%p**, T10 전체 seen accuracy 차이는 **−0.263%p / −0.274%p**였다. 초기 acquisition의 작은 개선이 late-task acquisition 및 다음 task 이후 성능 개선으로 이어지지 않았다.

| 주 비교: G−S | IA10 | CAR10 |
|---|---:|---:|
| T1 신규 class accuracy | +1.143%p | +1.074%p |
| T9 신규 class all-seen accuracy | 0.000%p | −0.528%p |
| T1·T9 birth 차이의 산술 평균 | +0.571%p | +0.273%p |
| T9 전체 seen accuracy | −0.143%p | −0.207%p |
| T10에서 평가한 T9 cohort accuracy | 0.000%p | −0.792%p |
| T10 전체 seen accuracy | −0.263%p | −0.274%p |

G의 representation을 공유하고 current stored statistics만 real+generated로 바꾼 GM도 일관된 이득을 주지 않았다. T9 분기 직후 GM−G는 IA10 전체 seen **−0.287%p**, CAR10 전체 seen **−0.110%p**였다. 특히 CAR10 T9 신규 class는 **−1.583%p**였다.

이 결론은 **선택한 generator·편집 종류·생성량·자동 필터·학습 recipe**에 한정된다. 모든 생성 증강이나 raw-image coverage 가설을 반박하는 결과가 아니다. CAR fine-grained label fidelity는 검증되지 않았으며, 이 보고서는 이를 검증된 데이터 증강의 효능 실험으로 취급하지 않는다.

## 2. 실제 완료 상태 및 결과 검증

[run_manifest.json](../../run_manifest.json)의 audit / generate / auto-sanity / train은 모두 `complete`다. [최종 실행 로그](../../logs/matched_exploratory_gpu4_fix_s.log)는 `Matched pilot and report complete: final_report.md`로 끝나며, 보고서 작성 시 해당 학습 프로세스는 남아 있지 않았다. `tail -f` 프로세스는 로그 조회용이지 학습 프로세스가 아니다.

| 확인 항목 | 확인 결과 |
|---|---|
| IA10 | T1 및 T9→T10의 R/S/G/GM, 8개 분기 최종 checkpoint·예측 파일 존재 |
| CAR10 | T1 및 T9→T10의 R/S/G/GM, 8개 분기 최종 checkpoint·예측 파일 존재 |
| 최종 task 상태 | T1 분기는 task 1, suffix 분기는 task 10의 `task_end` |
| 예측 결과 | V를 포함한 30개 평가 조건, 총 96,150행 |
| 재집계 검증 | CSV의 sample ID 중복·정답률 및 300개 paired subset의 N/repair/harm을 JSON과 대조, 모두 일치 |
| Matched 학습 검증 | R/S/G의 초기 network hash, 공통 parent manifest hash, anchor hash, parent schedule hash, update 수·LR·이미지 exposure 일치 |
| Fork 검증 | 자동 보고서 생성 과정에서 G/GM checkpoint representation 및 old APR moments 일치를 검사한 뒤 정상 완료 |

마지막 task-end checkpoint 시간은 IA10 **2026-10-08 17:16:02 KST**, CAR10 **2026-10-08 19:37:21 KST**다. 기존 자동 보고서는 **2026-10-08 19:37:24 KST**에 생성되어 있었다. 본 문서는 그 결과를 2026-10-12에 재검증·해석한 한국어 보고서다. 추가 학습이나 재생성은 실행하지 않았다.

## 3. 실험 구성 — 무엇을 비교했는가

### 3.1 Protocol 및 범위

| 항목 | IA10 | CAR10 |
|---|---|---|
| Class 수 / task 수 | 200 / 10 | 196 / 10 |
| Task partition | 20 × 10 | 20 × 6 + 19 × 4 |
| Model seed | 1993 | 1993 |
| Backbone | pretrained ViT-B/16 + adapters | 동일 |
| 원 Structure 구성 | SCC / APR / CA 활성, single prototype | 동일 |
| T1 실험 | 동일 pretrained initialization부터 T1만 학습 | 동일 |
| Late 실험 | 동일 vanilla T8 checkpoint → T9 intervention → T10 real-only | 동일 |

Class order는 기존 vanilla Structure의 order와 일치시켰다. **T1의 augmented checkpoint를 T9까지 이어서 학습한 실험이 아니다.** T1과 late suffix는 서로 별개의 실험 위치다. 따라서 아래 T10 숫자를 “생성 증강을 T1부터 T10까지 사용한 최종 accuracy”로 해석하면 안 된다.

### 3.2 조건 정의

| 조건 | T1 / T9의 current representation·head 학습 | Current stored mean/covariance | 역할 |
|---|---|---|---|
| V | 기존 vanilla Structure trajectory 재사용 | Real train | 참고 baseline; 추가 compute가 matched가 아님 |
| R | Real anchor + 같은 class parent의 ordinary view | Real train | 추가 exposure / 반복 학습 대조 |
| S | Real anchor + parent의 strong view | Real train | 주 비교 기준 |
| G | Real anchor + parent의 generated view | Real train | 생성 이미지 source intervention |
| GM | G의 학습 완료 representation 공유 | Current real + generated | Statistics-only fork |

주 비교는 **G−S**, 보조 비교는 **G−R**, memory-source 비교는 **GM−G**다. G−V는 추가 image exposure와 계산량 차이가 포함되므로 생성 자체의 효과를 분리하는 주 근거로 쓰지 않는다. G도 representation뿐 아니라 acquisition 중 classifier head가 함께 학습되므로, G−S를 representation만의 효과라고 부르지 않는다.

R/S/G는 real anchor B와 auxiliary B에 대해 `0.5 × CE(real) + 0.5 × CE(aux)`를 사용한다. 기존 SCC는 real anchor에서만 계산한다. Batch B=48, effective image batch=96이며 순차 backward 후 한 번의 optimizer update를 수행한다. T10에는 auxiliary bank를 제거하고 모든 suffix 분기가 real-only로 진행한다.

| Protocol / task | Epoch | Real anchor / epoch | Forward·backward 이미지 / epoch | Optimizer update / epoch |
|---|---:|---:|---:|---:|
| IA10 T1 | 30 | 705 | 1,410 | 15 |
| IA10 T9 | 15 | 571 | 1,142 | 12 |
| IA10 T10 | 15 | 527 | 527 | 11 |
| CAR10 T1 | 30 | 847 | 1,694 | 18 |
| CAR10 T9 | 15 | 766 | 1,532 | 16 |
| CAR10 T10 | 15 | 792 | 792 | 17 |

이 exposure와 update budget, 실제 anchor transform 결과, parent 선택 순서 및 LR은 R/S/G trace에서 동일했다. 벽시계 실행 시간까지 같다는 뜻은 아니다.

S의 최종 transform 순서는 `RandomResizedCrop → RandomHorizontalFlip → RandAugment(2, 9) → ToTensor → RandomErasing(p=0.25)`다. 이전 Solarize 오류는 RandAugment가 float tensor 뒤에 적용되어 발생했으며, 최종 결과는 **RandAugment를 tensor 변환 앞으로 이동한 뒤 완주한 결과**다. 이미 완료된 IA10 T1 R은 보존했고, 실패한 S의 불완전 결과를 최종 결과로 집계하지 않았다.

근거: [matched_views.py](../../matched_views.py), [learner.py](../../learner.py), [training.py](../../training.py), [IA10 T1 S 실제 trace](../../training/IA10/T1/S/matched_training_T01.json).

## 4. Generator 및 자동 sanity 검사

### 4.1 고정 generation recipe

Frozen `runwayml/stable-diffusion-v1-5`, revision `451f4fe16113bff5a5d2269ed5ad43b0592e9a14`를 사용했다. Img2img strength **0.2**, requested steps **30**, guidance **7.5**, FP16, 512×512이며 scheduler는 checkpoint 설정이다. Strength에 따라 실제 img2img denoising 구간은 requested steps 전체와 다를 수 있다.

T1/T9의 current real train에서 class당 최대 16개의 parent를 seed `20261008`로 선택하고, parent당 1장을 생성했다. Prompt는 동일 객체의 identity·shape·markings·pose 보존을 요청하면서 조명/배경/context variation을 주는 고정 recipe다. **요청 문구가 label 보존의 보장은 아니다.** 별도의 객체 mask/inpainting 보존 보장이나 fine-grained identity verifier는 없다. 미래 class나 test image를 generation source로 사용하지 않았고, Type A나 test confidence로 parent를 골라내지 않았다.

### 4.2 채택·제외 및 공통 parent bank

| Protocol | 생성 후보 | 자동 채택 | 자동 제외 | 채택 T1 / T9 | 채택 parent 수 / class |
|---|---:|---:|---:|---:|---|
| IA10 | 568 | 560 | 8 | 286 / 274 | T1: 9–16, T9: 4–16 |
| CAR10 | 624 | 622 | 2 | 320 / 302 | T1: 16, T9: 15–16 |
| 합계 | 1,192 | 1,182 | 10 | 606 / 576 | 모든 대상 class에 최소 1개 존재 |

제외된 10장은 `generation_failed_or_safety_blocked`와 `collapsed_image`에 동시에 해당한다. 이유는 비배타적이므로 20장 제외로 합산하지 않는다. 그 외 near-duplicate / gross semantic mismatch / extreme structure-loss 규칙에 추가로 걸린 후보는 없었다. 이는 “실제 복제·객체 손상·label 변경이 전혀 없다”는 뜻이 아니라 **해당 자동 threshold를 넘지 않았다**는 뜻이다.

실패·깨진 이미지 검사, SHA/크기 검사, near-duplicate의 block-SSIM·MAE·dHash 결합, image collapse의 pixel std·edge 결합, 전역/중앙 CLIP similarity 및 중앙 edge collapse proxy를 사용했다. 규칙은 downstream 결과 전에 고정했으며, test accuracy·class confidence·사람의 개별 판단으로 선택하지 않았다. 검사에는 기존 generator의 frozen CLIP 구성요소를 재사용했고 추가 모델 다운로드는 없었다.

R/S/G는 **동일한 자동 채택 parent 목록**을 사용한다. Excluded image는 학습 자격만 제외되었고 원 파일은 삭제되지 않았다. Contact sheet의 고정 표본 200쌍은 결과 확인용 artifact일 뿐 선택 gate가 아니며, 전체 생성 후보를 사람 검수한 자료도 아니다.

| Protocol / subset | N | Global CLIP cosine 평균 | Cosine 5% quantile | Center cosine 평균 | Block SSIM 평균 | Pixel MAE 평균 |
|---|---:|---:|---:|---:|---:|---:|
| IA10 전체 decodable | 568 | 0.81683 | 0.65578 | 0.81884 | 0.68967 | 0.05205 |
| IA10 채택 | 560 | 0.82159 | 0.67428 | 0.82377 | 0.69950 | 0.04622 |
| CAR10 전체 decodable | 624 | 0.88877 | 0.79639 | 0.90587 | 0.78079 | 0.03951 |
| CAR10 채택 | 622 | 0.88981 | 0.79704 | 0.90678 | 0.78313 | 0.03817 |

Block SSIM은 비중첩 8×8 RGB block의 자체 지표이며 표준 Gaussian-window SSIM과 동일하지 않다. 높은 CLIP similarity는 CAR 차종·연식의 보존을 인증하지 않는다. 고정 자동 규칙이라도 source distribution을 바꿀 수 있으므로 “selection bias가 완전히 없다”고 주장하지 않는다.

기존 `config.json`의 human-review-required 및 `generation_quality.md`의 HOLD는 이전 gate의 기록이다. **이번 학습의 실제 mode는 [exploratory_config.json](../../exploratory_config.json)의 `human_review_required=false`, `label_fidelity_verified=false`**이며, 자동 gate는 `READY_EXPLORATORY`였다. 사람 QC를 완료한 실험으로 보고하지 않는다.

근거: [자동 sanity 보고서](../../automatic_sanity_report.md), [고정 규칙](../../sanity_policy.json), [개별 이미지 측정](../../automatic_sanity.csv), [통계 JSON](../../automatic_sanity_summary.json), [class별 수](../../automatic_parent_counts.csv), [generation manifest](../../generation_manifest.csv), [contact sheets](../../contact_sheets/).

## 5. Accuracy 결과

모든 accuracy는 백분율(%), 차이는 percentage point(%p)다. Seen은 해당 시점까지의 모든 후보 class를 사용한 micro accuracy다. New all-seen은 현재 task sample을 전체 seen 후보로 평가한 accuracy, New local은 후보를 현재 task class로 제한한 **진단 지표**이지 최종 inference protocol이 아니다. T1은 old class가 없으므로 seen=new=local이다. V에는 저장된 local logits가 없어 `—`로 표시한다.

### 5.1 T1 acquisition

| Protocol | 조건 | 정답 / N | Seen = New (%) | Macro (%) |
|---|---|---:|---:|---:|
| IA10 | V | 151 / 175 | 86.286 | 73.589 |
| IA10 | R | 149 / 175 | 85.143 | 73.034 |
| IA10 | S | 151 / 175 | 86.286 | 73.936 |
| IA10 | G | 153 / 175 | 87.429 | 77.611 |
| IA10 | GM | 153 / 175 | 87.429 | 77.611 |
| CAR10 | V | 756 / 838 | 90.215 | 90.235 |
| CAR10 | R | 755 / 838 | 90.095 | 90.138 |
| CAR10 | S | 750 / 838 | 89.499 | 89.491 |
| CAR10 | G | 759 / 838 | 90.573 | 90.516 |
| CAR10 | GM | 759 / 838 | 90.573 | 90.516 |

G−S의 순증 정답은 IA10 **2개**, CAR10 **9개**다. 단일 model seed의 이 차이를 안정적인 우위나 통계적 유의성으로 주장하지 않는다.

**T1 G=GM은 정상 동작이다.** 원 Structure는 T1에서 CA를 수행하지 않는다. GM은 저장할 moments만 바꾸므로 T1 evaluation의 representation과 head가 G와 같으며, 이 실험은 그 이후 T2를 진행하지 않았다. T1 GM의 동일 성능은 mixed memory가 무해하다는 검증이 아니다.

### 5.2 T9 intervention 직후

| Protocol | 조건 | Seen (%) | Macro (%) | Old (%) | New all-seen (%) | New local (%) |
|---|---|---:|---:|---:|---:|---:|
| IA10 | V | 65.233 | 54.856 | 66.026 | 58.503 | — |
| IA10 | R | 65.018 | 54.680 | 65.946 | 57.143 | 82.993 |
| IA10 | S | 65.233 | 54.750 | 66.186 | 57.143 | 82.313 |
| IA10 | G | 65.090 | 55.626 | 66.026 | 57.143 | 82.993 |
| IA10 | GM | 64.803 | 55.410 | 65.705 | 57.143 | 83.673 |
| CAR10 | V | 79.303 | 79.374 | 79.074 | 81.266 | — |
| CAR10 | R | 79.358 | 79.441 | 79.090 | 81.662 | 94.327 |
| CAR10 | S | 79.510 | 79.594 | 79.228 | 81.926 | 94.063 |
| CAR10 | G | 79.303 | 79.392 | 79.059 | 81.398 | 93.404 |
| CAR10 | GM | 79.193 | 79.258 | 79.121 | 79.815 | 94.063 |

IA10: N=1,395, old=1,248, new=147. CAR10: N=7,262, old=6,504, new=758.

IA10 G는 S보다 macro가 높지만 micro seen은 낮고 new all-seen은 같았다. CAR10 G는 S보다 new/local/seen 모두 낮았다. IA10 GM의 local 개선이나 CAR10 GM의 local 회복도 전체 seen class와의 경쟁을 포함한 accuracy 개선으로 이어지지 않았다.

### 5.3 Real-only T10終了後

| Protocol | 조건 | Seen (%) | Macro (%) | Old (%) | T10 new (%) | T9 cohort (%) |
|---|---|---:|---:|---:|---:|---:|
| IA10 | V | 64.714 | 53.586 | 64.516 | 66.935 | 57.143 |
| IA10 | R | 64.582 | 53.427 | 64.444 | 66.129 | 57.823 |
| IA10 | S | 64.845 | 53.798 | 64.659 | 66.935 | 57.143 |
| IA10 | G | 64.582 | 53.777 | 64.444 | 66.129 | 57.143 |
| IA10 | GM | 64.845 | 53.649 | 64.731 | 66.129 | 55.782 |
| CAR10 | V | 78.958 | 79.059 | 78.684 | 81.515 | 82.190 |
| CAR10 | R | 78.784 | 78.858 | 78.560 | 80.873 | 81.003 |
| CAR10 | S | 78.883 | 78.957 | 78.670 | 80.873 | 81.003 |
| CAR10 | G | 78.610 | 78.694 | 78.381 | 80.745 | 80.211 |
| CAR10 | GM | 78.510 | 78.585 | 78.270 | 80.745 | 78.628 |

IA10: N=1,519, old=1,395, T10 new=124. CAR10: N=8,041, old=7,262, T10 new=779. T9 cohort의 분모는 T9 시점과 동일한 147 / 758이다.

G−R의 T10 seen 차이는 IA10 **0.000%p**, CAR10 **−0.174%p**다. G는 repetition에도 지속적인 우위를 보이지 않았다. G−V의 T10 seen 차이도 IA10 −0.132%p, CAR10 −0.348%p지만, 이는 non-matched 참고 비교다.

## 6. 동일 sample pair의 repair / harm

`X−Y`에서 Y 오답→X 정답을 repair, Y 정답→X 오답을 harm으로 정의한다. `net = repair − harm`, `ΔAcc(%p) = 100 × net / N`이다. 총 accuracy뿐 아니라 동일 sample ID에서 예측이 바뀐 방향을 측정했다.

| Protocol | 평가 집합 | 비교 | N | Repair | Harm | Net | ΔAcc (%p) |
|---|---|---|---:|---:|---:|---:|---:|
| IA10 | T1全体 | G−S | 175 | 5 | 3 | +2 | +1.143 |
| IA10 | T1全体 | G−R | 175 | 6 | 2 | +4 | +2.286 |
| IA10 | T9全体 | G−S | 1,395 | 16 | 18 | −2 | −0.143 |
| IA10 | T9全体 | G−R | 1,395 | 15 | 14 | +1 | +0.072 |
| IA10 | T10全体 | G−S | 1,519 | 7 | 11 | −4 | −0.263 |
| IA10 | T10全体 | G−R | 1,519 | 11 | 11 | 0 | 0.000 |
| CAR10 | T1全体 | G−S | 838 | 16 | 7 | +9 | +1.074 |
| CAR10 | T1全体 | G−R | 838 | 15 | 11 | +4 | +0.477 |
| CAR10 | T9全体 | G−S | 7,262 | 32 | 47 | −15 | −0.207 |
| CAR10 | T9全体 | G−R | 7,262 | 42 | 46 | −4 | −0.055 |
| CAR10 | T10全体 | G−S | 8,041 | 29 | 51 | −22 | −0.274 |
| CAR10 | T10全体 | G−R | 8,041 | 25 | 39 | −14 | −0.174 |

T9 신규 sample만의 G−S repair/harm은 IA10 **2/2**, CAR10 **8/12**다. T10 시점의 T9 cohort에서는 IA10 **1/1**, CAR10 **8/14**로, late acquisition 및 유지의 정미 개선은 확인되지 않았다.

### 6.1 고정 vanilla error subset

Error subset은 기존 vanilla trajectory로 고정하고 이번 prediction으로 재분류하지 않았다. A는 관측한 task-end에서 지속 오답, B는 birth 정답→final 오답, C는 그 밖의 비단조적인 final-error trajectory, final_correct는 vanilla 최종 정답이다. B1/B2는 vanilla 최종 head의 후보를 birth 시점까지의 seen class로 제한했을 때 정답이 복원되는 / 복원되지 않는 B다. 아래 T10 표는 **birth≤T9**만 포함하므로 T10 신규 class는 들어가지 않는다.

| Protocol | 고정 subset | N | G−S repair / harm / net | G−S Δ (%p) | GM−G repair / harm / net | GM−G Δ (%p) |
|---|---|---:|---|---:|---|---:|
| IA10 | A | 327 | 1 / 1 / 0 | 0.000 | 2 / 0 / +2 | +0.612 |
| IA10 | B1 | 126 | 1 / 0 / +1 | +0.794 | 2 / 0 / +2 | +1.587 |
| IA10 | B2 | 29 | 0 / 1 / −1 | −3.448 | 0 / 0 / 0 | 0.000 |
| IA10 | C | 13 | 2 / 1 / +1 | +7.692 | 0 / 1 / −1 | −7.692 |
| IA10 | final_correct | 900 | 3 / 7 / −4 | −0.444 | 5 / 4 / +1 | +0.111 |
| CAR10 | A | 907 | 2 / 6 / −4 | −0.441 | 3 / 4 / −1 | −0.110 |
| CAR10 | B1 | 473 | 7 / 8 / −1 | −0.211 | 16 / 6 / +10 | +2.114 |
| CAR10 | B2 | 87 | 7 / 1 / +6 | +6.897 | 0 / 1 / −1 | −1.149 |
| CAR10 | C | 81 | 0 / 3 / −3 | −3.704 | 0 / 1 / −1 | −1.235 |
| CAR10 | final_correct | 5,714 | 9 / 28 / −19 | −0.333 | 15 / 30 / −15 | −0.263 |

CAR10 G의 B2 개선만 보면 좋아 보이지만 A·C·final_correct의 정미 손실을 합치면 전체는 악화된다. GM도 CAR10 B1을 10개 정미 복구하는 한편 vanilla final_correct에서 15개 정미 악화되어 전체 개선으로 이어지지 않았다. Subset에 소수 사례만 있는 경우 큰 %p만으로 우열을 판단하지 않는다.

T1의 고정 A에 대한 G−S는 IA10 **2/0/net +2 (N=19)**, CAR10 **7/0/net +7 (N=67)**였다. 다만 T1의 A와 T9→T10의 A는 범위가 다르고 T1 branch를 T10까지 이어서 학습하지 않았다. T1 복구가 최종적으로 유지되었다고 말할 수 없다.

## 7. Mixed statistics — GM−G는 무엇을 보여주는가

G/GM 분기에서 representation, pre-CA head, old APR mean/covariance, CA의 Gaussian epsilon·permutation RNG와 original CA recipe를 공유했다. GM만 current real+generated의 mean/covariance를 사용하며 APR displacement의 anchor source는 real로 유지했다. CA는 256 samples / seen class, 5 epochs다. Pooled covariance에 between-source scatter를 포함하고 직접 concatenate한 unbiased covariance와의 최대 차이도 확인했다.

| Protocol / 분기 | Current class 수 | Generated 비율 class 평균 | Mean shift L2 평균 | Covariance shift Fro 평균 | Covariance식 검증 최대 절대차 |
|---|---:|---:|---:|---:|---:|
| IA10 T1 | 20 | 0.37190 | 6.17239 | 295.88384 | 3.55e−14 |
| IA10 T9 | 20 | 0.38254 | 6.01922 | 268.88660 | 2.13e−14 |
| CAR10 T1 | 20 | 0.27749 | 6.49824 | 261.39305 | 1.07e−14 |
| CAR10 T9 | 19 | 0.28415 | 6.49155 | 254.26882 | 8.88e−15 |

Mean/covariance shift는 raw feature 단위이며 정규화된 효과량이 아니다. 생성 비율은 class마다 다르며 모든 class를 일률적으로 50% 혼합한 실험도 아니다.

| Protocol | 평가 | Subset | N | GM−G repair / harm / net | Δ (%p) |
|---|---|---:|---:|---|---:|
| IA10 | T9直後 | Seen | 1,395 | 4 / 8 / −4 | −0.287 |
| IA10 | T9直後 | New | 147 | 1 / 1 / 0 | 0.000 |
| IA10 | T10後 | Seen | 1,519 | 9 / 5 / +4 | +0.263 |
| IA10 | T10後 | T9 cohort | 147 | 2 / 4 / −2 | −1.361 |
| CAR10 | T9直後 | Seen | 7,262 | 30 / 38 / −8 | −0.110 |
| CAR10 | T9直後 | New | 758 | 13 / 25 / −12 | −1.583 |
| CAR10 | T10後 | Seen | 8,041 | 36 / 44 / −8 | −0.099 |
| CAR10 | T10後 | T9 cohort | 758 | 11 / 23 / −12 | −1.583 |

T9 직후에는 동일 representation의 current memory-source/CA 분기이므로 mixed moments intervention의 영향을 비교하기 쉽다. 반면 **T10에서는 memory 차이가 SCC/APR 및 다음 representation 학습에 영향을 줄 수 있다.** 따라서 T10 GM−G를 “동일 representation에서 memory만 바꾼 효과”라고 표현하지 않는다.

IA10의 T10 전체가 +0.263%p 개선되어도 T9 cohort는 −1.361%p여서 current class 유지의 일관된 이점은 없었다. CAR10의 T9에서 발생한 −1.583%p cohort 차이는 T10에도 남았다. 이를 synthetic contamination의 증명이라고 단정하지 않고, **이 혼합 통계와 기존 CA/다음 task 학습의 조합이 유리하지 않았다**고 결론 내린다.

근거: [statistics_comparison.json](../../statistics_comparison.json), [moments.py](../../moments.py), [learner.py](../../learner.py), [results.py](../../results.py).

## 8. T9→T10 유지와 head 진단

### 8.1 동일 T9 sample의 시점 간 변화

이 표는 조건 간 repair/harm이 아니라 **각 branch 자체의 T9 직후→T10 후** 변화다.

| Protocol | 조건 | N | T9 정답→T10 오답 | T9 오답→T10 정답 | 정미 변화 | Accuracy 차이 (%p) |
|---|---|---:|---:|---:|---:|---:|
| IA10 | V | 147 | 2 | 0 | −2 | −1.361 |
| IA10 | R | 147 | 1 | 2 | +1 | +0.680 |
| IA10 | S | 147 | 1 | 1 | 0 | 0.000 |
| IA10 | G | 147 | 2 | 2 | 0 | 0.000 |
| IA10 | GM | 147 | 4 | 2 | −2 | −1.361 |
| CAR10 | V | 758 | 16 | 23 | +7 | +0.923 |
| CAR10 | R | 758 | 23 | 18 | −5 | −0.660 |
| CAR10 | S | 758 | 21 | 14 | −7 | −0.923 |
| CAR10 | G | 758 | 23 | 14 | −9 | −1.187 |
| CAR10 | GM | 758 | 23 | 14 | −9 | −1.187 |

CAR10 G/GM의 net은 동일한 −9지만 birth accuracy가 달라 T10 cohort accuracy도 다르다. 동일 transition net을 동일 최종 accuracy로 혼동하지 않는다.

### 8.2 T9 current sample의 경쟁·CE·margin

| Protocol | 조건 | New all-seen (%) | New local (%) | New→old 오답 | New→new 오답 | New CE 평균 | New all margin 평균 | New local margin 평균 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| IA10 | S | 57.143 | 82.313 | 56 | 7 | 1.5653 | 0.6238 | 3.0756 |
| IA10 | G | 57.143 | 82.993 | 57 | 6 | 1.6263 | 0.4893 | 2.8664 |
| IA10 | GM | 57.143 | 83.673 | 57 | 6 | 1.6412 | 0.4170 | 2.7429 |
| CAR10 | S | 81.926 | 94.063 | 118 | 19 | 1.0361 | 1.4768 | 3.2816 |
| CAR10 | G | 81.398 | 93.404 | 119 | 22 | 1.0426 | 1.4621 | 3.2627 |
| CAR10 | GM | 79.815 | 94.063 | 134 | 19 | 1.0924 | 1.3643 | 3.2024 |

Margin은 scale=20을 곱한 cosine logits로 계산하며 평가 시 training margin을 더하지 않았다. Audit의 raw cosine margin과 척도가 다르다. V의 CE/margin은 미저장으로 알 수 없는 값이지 0이 아니다.

CAR10 GM은 G보다 local accuracy가 높지만 new→old 오답이 119→134로 늘고 new all-seen은 악화되었다. **Task 내부 식별과 전체 class 경쟁을 포함한 accuracy는 다르며 local 개선만으로 채택할 수 없다.** CE·margin도 S→G→GM에서 일관되게 좋아지지 않았다. 이는 관측한 head 동작으로, representation 문제와 classifier 문제를 유일하게 분리하는 인과 증명은 아니다.

## 9. Birth support audit 결과와 해석

Intervention 전 vanilla birth checkpoint에서 고정 A (birth≤T9)를 local 오답인 A-within과 local 정답/all 오답인 A-cross로 분해했다. Current-task real train feature bank만 사용하고 같은 class의 birth-correct와의 support 차이를 macro 집계했다. Subsample seeds 1993/1994/1995는 **bank sampling seed**이며 독립 model training seed가 3개 있는 것은 아니다.

| Protocol | 고정 A | A-within | A-cross | 동일 class 비교 가능성 | A−birth-correct own support macro 차이: 3 seeds |
|---|---:|---:|---:|---|---|
| IA10 | 327 | 169 | 158 | A는 있으나 birth-correct가 없는 class: 23 | −0.10980 / −0.10201 / −0.11105 |
| CAR10 | 907 | 411 | 496 | 위 비교군 부재 class: 0 | −0.11801 / −0.11909 / −0.11959 |

Balanced bank는 task별 `n=min(64, 최소 class train 수)`, `k=min(5,n−1)`이다. IA10은 n=2–10, k=1–5로 특히 작은 bank의 support probe에 불확실성이 있다. CAR10은 n=24–38, k=5다. Train reference는 leave-one-out이며 모든 task에서 birth prediction과 기존 trajectory의 불일치는 0이었다.

A-within 중 support margin이 양수인데 local head가 오답인 사례는 IA10 **34/169**, CAR10 **86/411**이다. A-cross는 IA10 **158/327=48.32%**, CAR10 **496/907=54.69%**다. 모든 Type A가 task 내부 variation 부족에 직접 대응하지는 않는다.

낮은 support는 current feature에서의 association이며 원본 이미지 coverage 부족·feature 불변성 부족·annotation/class ambiguity 등을 분리하지 못한다. 이번 generation으로 부족한 특정 pose/background를 채웠다는 증거도 아니다. Audit을 parent/prompt/class 선택에 사용하지 않았다.

Parent의 train-only spherical clustering 분포는 [parent 분포 CSV](../../parent_distribution.csv)에 저장되어 있다. 추출 parent와 전체 real train의 cluster fraction 차이는 완전히 0이 아니며 class 내 균일 random 추출이 feature cluster를 완전히 대표할 보장은 없다. 이 표는 원래 추출 parent의 audit이며 자동 제외 후 bank를 재클러스터링한 표가 아니다.

상세: [IA10 birth audit](../../audit/IA10/report.md), [CAR10 birth audit](../../audit/CAR10/report.md), [기존 고정 error trajectory 보고서](../../../error_trajectory_diagnostic/final_report.md).

## 10. 사전 정의한 확장 기준 판정

Birth 평균은 T1과 T9의 new all-seen 차이를 **동일 가중치로 산술 평균**한 값이며 sample 수 가중 평균이 아니다. 아래 판정은 반올림 전 수치로 수행했다.

| 기준: G−S를 기본으로 사용 | IA10 실측 | 판정 | CAR10 실측 | 판정 |
|---|---:|---|---:|---|
| Birth 평균 ≥ +1.0%p | +0.571%p | 실패 | +0.273%p | 실패 |
| 각 birth의 최소 차이 ≥ −0.5%p | 0.000%p | 통과 | −0.528%p | 실패 |
| T9 seen ≥ −0.5%p | −0.143%p | 통과 | −0.207%p | 통과 |
| T10 후 T9 cohort ≥ +0.5%p | 0.000%p | 실패 | −0.792%p | 실패 |
| T10 seen ≥ 0.0%p | −0.263%p | 실패 | −0.274%p | 실패 |
| T9/T10 old·new 모든 차이 ≥ −1.0%p | 최소 −0.806%p | 통과 | 최소 −0.528%p | 통과 |
| G−R birth 평균 > 0이고 T10 seen > 0 | +1.143 / 0.000%p | 실패 | +0.107 / −0.174%p | 실패 |
| 모든 기준·양 protocol 통과 | `STOP_CURRENT_RECIPE` | 불채택 | `STOP_CURRENT_RECIPE` | 불채택 |

CAR10의 −0.528%p는 −0.5%p를 조금 밑돌지만 결과를 보고 threshold를 완화하지 않는다. 이 항목만 완화해도 birth 평균·retention·T10 seen 조건을 충족하지 못한다.

이 판정은 탐색적인 수치 screen이다. Human QC나 엄격한 label 보전 검증을 통과했다는 뜻이 아니다. 이번 recipe를 그대로 full trajectory로 확장할 근거는 얻지 못했다.

## 11. 실행 시간과 추가 리소스

[resource_usage.json](../../resource_usage.json)에 기록된 성공 분기 시간을 집계했다. 초기화 전 시간·audit·보고서 생성·중간 실패·수동 대기는 포함하지 않으며 재개 전후를 포함한 완전한 wall-clock이 아니다. G 시간에는 shared GM fork의 moments 계산·CA가 포함되어 있고 T1 GM representation은 공유하므로 이중 합산하지 않는다.

| Protocol | T1 R | T1 S | T1 G + GM fork | T9→T10 R | T9→T10 S | T9→T10 G + GM fork | GM 독립 T10 | 합계 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IA10 (分) | 6.06 | 11.40 | 10.99 | 6.57 | 6.41 | 8.28 | 2.46 | 52.18 |
| CAR10 (分) | 14.37 | 19.73 | 15.85 | 21.76 | 21.68 | 37.59 | 9.93 | 140.90 |

성공 branch의 기록 합계는 **11,584.40초, 약 3시간 13분 04초**다. 생성 invocation은 364.64초(약 6분 05초), 자동 sanity는 120.95초(약 2분 01초)다. 이 값만 더하면 약 **3시간 21분 10초**지만 전체 workflow의 총 경과 시간은 아니다.

Training peak GPU allocated는 최대 약 **5.85 GiB**, generator는 약 **3.19 GiB**, sanity는 약 **1.40 GiB**다. 모두 PyTorch allocated peak이며 GPU 전체 reserved/다른 process 사용량과는 다르다. 생성 PNG는 420,170,585 bytes(약 400.71 MiB), generator weights는 5,482,652,200 bytes(약 5.11 GiB)다. Generator는 추가 외부 pretrained model 자원이므로 baseline과 동일 model 자원만의 비교가 아니다.

Pilot의 실제 GPU는 run manifest / 실행 로그의 **4**를 채택한다. Branch `resolved_config.json`에 남아 있는 `physical_gpus: ["1"]` 및 `launch_command`는 원본 vanilla audit에서 상속한 metadata이며 이번에 GPU 1에서 실행했다는 근거가 아니다. Device는 CUDA_VISIBLE_DEVICES 적용 후 논리 `cuda:0`이다.

## 12. 한계와 연구 판단

1. **Label fidelity 미검증.** Parent label을 상속했지만 CAR 차종·연식의 세부나 대상 객체의 완전 보전은 자동 CLIP/edge 규칙으로 보장하지 못한다. Failure proxy는 명백한 손상도 놓칠 수 있다.
2. **단일 model seed.** 작은 accuracy 차이나 소수 repair/harm에 유의성·재현성 주장을 두지 않는다. Support의 3 seeds는 학습 replicate가 아니다.
3. **제한된 intervention.** T1 단독 및 vanilla T8에서 시작한 T9 intervention만 평가했다. 모든 task의 누적 augmentation, T1 GM memory의 다음 task 영향, 긴 suffix는 미평가다.
4. **제한된 variation family.** Strength 0.2의 generic image-conditioned lighting/background 편집이 필요한 pose/viewpoint·discriminative 구조 variation을 충분히 보완했다고 확인하지 않았다.
5. **고정 자동 선택도 distribution shift를 만들 수 있다.** 사람의 선택·confidence filtering은 없었지만 filter 및 최대 16개 parent 추출이 모집단 분포를 완전히 보전하지는 않는다.
6. **외부 model 자원.** Frozen diffusion/CLIP의 추가 지식·계산을 포함한다. Generated current image는 새로운 actual exemplar memory로 다음 task 학습에 읽지 않지만 실험 artifact는 디스크에 보존한다.
7. **원인을 단정하지 않는다.** G 실패는 coverage 가설 전체의 부정이 아니며 GM 실패는 synthetic moment contamination만의 증명도 아니다. 추가 intervention 없이 representation/head/CA/다음 task interaction을 분리할 수 없다.

다음 판단은 **현재 recipe를 그대로 full T1→T10·대량 seed로 확장하지 않는 것**이다. T1의 작은 개선은 관측으로 보존하되 late-task 및 전체 repair/harm의 손실을 우선한다. 새 recipe를 시도한다면 별도 실험으로 variation family·label 보전 검증·채택 기준을 사전에 정의하고 동일 matched S/R 대조를 유지해야 한다. 이번에는 그 구현·실행을 하지 않았다.

## 13. 재현·참조 파일

| 내용 | 파일 |
|---|---|
| 기존 자동 집계 보고서, 전체 paired·head 표 | [final_report.md](../../final_report.md) |
| Accuracy / comparisons / retention / decision | [metrics.json](../../metrics.json) |
| Sample ID를 포함한 전체 예측 | [per_sample_predictions.csv](../../per_sample_predictions.csv.gz) |
| Fork equality / class別moment shift | [statistics_comparison.json](../../statistics_comparison.json) |
| Generation / training / sanity 시간·memory | [resource_usage.json](../../resource_usage.json) |
| Stage 완료 상태 | [run_manifest.json](../../run_manifest.json) |
| 고정 recipe / 실제 탐색 mode | [config.json](../../config.json), [exploratory_config.json](../../exploratory_config.json) |
| 개별 parent / generated hash·prompt·seed | [generation_manifest.csv](../../generation_manifest.csv), [accepted_parent_manifest.csv](../../accepted_parent_manifest.csv) |
| Automatic sanity / 규칙 / 유사도 | [automatic_sanity_report.md](../../automatic_sanity_report.md), [sanity_policy.json](../../sanity_policy.json), [automatic_sanity_summary.json](../../automatic_sanity_summary.json) |
| Birth support 진단 | [IA10](../../audit/IA10/report.md), [CAR10](../../audit/CAR10/report.md) |
| 구현 / checkpoint 저장·평가 경로 | [training.py](../../training.py), [learner.py](../../learner.py), [matched_views.py](../../matched_views.py), [results.py](../../results.py) |
| 정상 완료 로그 | [matched_exploratory_gpu4_fix_s.log](../../logs/matched_exploratory_gpu4_fix_s.log) |

본 보고서는 저장된 실측 결과와 코드 경로를 근거로 작성했다. 원본 Structure, B1, 생성 이미지, 기존 checkpoint, 원본 JSON/CSV, 자동 보고서는 변경하지 않았다.
