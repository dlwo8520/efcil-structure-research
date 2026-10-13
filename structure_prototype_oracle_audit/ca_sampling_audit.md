# Structure Classifier Adaptation Sampling Audit

> 확인 대상: vanilla Structure, IA5/CAR5 seed 1993 T5  
> 근거 코드: `${RESEARCH_WORKSPACE}/structure_multicenter_separation/models/base.py`  
> classifier 코드: `${RESEARCH_WORKSPACE}/structure_multicenter_separation/network/classifier.py`

## 실제 코드 경로

`BaseLearner._stage2_compact_classifier(task_size, ca_epochs=5)`가 classifier adaptation을 수행한다.

| 항목 | 실제 동작 | 코드 위치 |
|---|---|---|
| 학습 parameter | `model.fc.parameters()` 중 `requires_grad=True` | `base.py:59–69` |
| optimizer | SGD, lr 0.01, momentum 0.9, weight decay 0.0005 | `base.py:66–69`, saved sidecar |
| scheduler | `CosineAnnealingLR(T_max=run_epochs)` | `base.py:71` |
| scheduler step | epoch 종료마다 1회 | `base.py:166` |
| class sample 수 | class당 256/epoch | `base.py:89–116` |
| Gaussian 생성 | 모든 seen class에서 매 epoch 새로 생성 | `base.py:80–116` |
| mean scaling | `0.9 + (class_id // task_size + 1)/(task_index+1)*0.1` | `base.py:91–94` |
| shuffle | 전체 tensor에 `torch.randperm` 1회 | `base.py:118–124` |
| sampler | DataLoader/sampler 없음 | `base.py:80–166` |
| replacement | finite bank 없음; Gaussian에서 iid draw | `base.py:113–114` |
| batch size | 256 | `base.py:126–128` |
| batch composition | global shuffle 후 contiguous 256개; class별 고정 quota batch 아님 | `base.py:122–128` |
| loss | scale 20 cosine logits의 cross entropy | `base.py:131–150` |
| classifier normalization | feature/weight 모두 L2 normalization | `classifier.py:52–58` |
| classifier bias | 없음 | `classifier.py:21`, `classifier.py:41` |

## Vanilla/Full Oracle dynamic CA exposure

### IA5 T5

- seen class: 200 = old 160 + new 40
- bank size: finite bank 없음
- samples drawn/class/epoch: 256
- samples drawn/epoch: 51,200
- batches/optimizer steps/epoch: 200
- epoch: 5
- total samples drawn: 256,000
- total optimizer steps: 1,000
- old total draw: 204,800
- new total draw: 51,200
- per-class total draw min/mean/max: 1,280/1,280/1,280

### CAR5 T5

- seen class: 196 = old 157 + new 39
- bank size: finite bank 없음
- samples drawn/class/epoch: 256
- samples drawn/epoch: 50,176
- batches/optimizer steps/epoch: 196
- epoch: 5
- total samples drawn: 250,880
- total optimizer steps: 980
- old total draw: 200,960
- new total draw: 49,920
- per-class total draw min/mean/max: 1,280/1,280/1,280

## 기존 G_FIXED/R_FIXED exposure

기존 classifier bottleneck diagnostic의 G_FIXED/R_FIXED는 서로 같은 label layout과 step을 사용했지만, old class에는 해당 class의 actual count만 사용하고 new class에는 256개를 사용했다.

| Protocol | Old class bank min/max | Old draw/epoch | New class draw | New draw/epoch | Total/epoch | Steps/epoch | Total steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| IA5 | 2/86 | 4,883 | 256 | 10,240 | 15,123 | 60 | 300 |
| CAR5 | 24/68 | 6,545 | 256 | 9,984 | 16,529 | 65 | 325 |

G_FIXED/R_FIXED의 fixed bank는 5 epoch 동안 반복된다. epoch마다 deterministic `torch.randperm`으로 섞지만 Gaussian을 다시 생성하지 않는다. scheduler는 vanilla와 마찬가지로 epoch마다 step하므로 LR의 epoch trajectory는 같지만 optimizer step당 LR exposure는 다르다.

## Audit 결론

1. vanilla 및 Full Oracle dynamic CA는 모든 seen class에 동일한 256 exposure를 주며 epoch마다 Gaussian을 regenerate한다.
2. 기존 G_FIXED/R_FIXED는 old class의 unique actual count를 exposure로 사용해 new class보다 훨씬 적게 노출된다.
3. 기존 G/R끼리는 paired comparison이지만 Full Oracle dynamic CA와 비교하면 bank lifecycle, class exposure, 총 sample 수 및 optimizer step이 다르다.
4. 다음 실험의 G256_FIXED/R256_FIXED는 class당 256 exposure, global label order, batch boundary, 총 optimizer step 및 LR schedule을 완전히 동일하게 맞춰야 한다.

