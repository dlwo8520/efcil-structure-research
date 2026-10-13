# Reference code audit

감사일: 2026-09-17. 공식 저장소만 확인했으며 외부 저장소 코드는 복사하지 않았다.

| Reference | 확인한 공식 코드/commit | 이번 구현과의 관계 |
|---|---|---|
| Structure | <https://github.com/lambor9973/cds>, local base commit `a0db2168663982a4cad24981231721a3518ddf29` | backbone, SCC, APR, statistics, classifier retraining을 **직접 상속**한다. 로컬 base에는 이전 기록 기능이 추가되어 있으므로 run snapshot에 working tree를 함께 저장한다. |
| LUCIR | <https://github.com/hshustc/CVPR19_Incremental_Learning>, commit `e5a90aed7640f3b00e5b1a8dfb5376c1628bfe6a`, `utils_incremental/incremental_train_and_eval_MR_LF.py:108-122` | old exemplar의 GT score와 top-K novel score를 비교하는 `MarginRankingLoss`를 확인했다. 이번 S2는 방향을 뒤집어 **new sample vs similar old prototype**에 쓰는 아이디어 차용이며 LUCIR 재현이 아니다. |
| REMINDER / CSW-KD | <https://github.com/HieuPhan33/REMINDER>, commit `f8d225eb9566dd5d6ea97133ac00178b59ba64f6`, `utils/loss.py:265,326-365`, `train.py:437-449` | new prototype-old prototype cosine으로 KD weight를 구성하는 방식을 확인했다. 이번 S1/S3의 scalar weight 아이디어만 차용하며 semantic-segmentation KD 코드는 사용하지 않는다. |
| STAR-TPAMI | <https://github.com/jinpeng0528/STAR-TPAMI>, commit `6c9203aa0c91e9a2d4e40664c97754ae02226675`, `models/loss.py:118-133`, `compute_prototype.py:141-165` | new center와 previous prototypes의 nearest distance 및 prototype 간 nearest distance를 확인했다. 이번 구현은 **similar old selection** 아이디어만 사용한다. |
| CLAD | <https://ojs.aaai.org/index.php/AAAI/article/view/29555> | similar old-new conflict라는 연구 질문의 근거로만 사용. 확인된 공식 코드를 직접 사용하지 않는다. |
| DiAPR | <https://ojs.aaai.org/index.php/AAAI/article/view/39263> | prototype consistency 관련 해석 참고만 한다. 확인된 공식 코드를 직접 사용하지 않는다. |

## 구현상 구분

- **직접 사용:** 로컬 Structure와 이전 E3 확장의 상속 가능한 코드.
- **아이디어만 차용:** LUCIR hard margin, REMINDER class-similarity weighting, STAR nearest prototype selection.
- **독립 구현:** S0 pair logging, S1 pair tau, S2 relative route, S3 per-sample SCC weighting, S4 sigmoid hybrid.

