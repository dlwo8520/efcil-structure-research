# Pair Boundary-Risk Diagnostic 구현 계획

> IA5/CAR5 · T4→T5 · seed 1993 · frozen diagnostic only

## 1. 입력 checkpoint와 cache

- checkpoint: `training_outputs/{IA5,CAR5}/Structure_AUDIT/seed_1993/task_checkpoints/task_{04,05}.pt`
- T5 feature: `classifier_bottleneck_diagnostic/{protocol}/seed_1993/.cache/task_05_features.npz`
- T4 full-T5-class feature: `representation_relation_diagnostic/{protocol}_T5/.cache/t4_full_t5_class_features.npz`
- T5 classifier weight: T5 checkpoint `network_state_dict`의 task-wise head를 class 순으로 concatenate
- stored statistics: T5 checkpoint의 `class_means`, `class_covs`

입력 checkpoint는 실행 전후 size와 edge SHA-256 fingerprint를 비교하며 쓰지 않는다.

## 2. Sample identity와 동일 sample 추적

T4 cache 생성 시 저장한 `train_sample_ids`, `test_sample_ids`를 canonical ID로 사용한다. T4/T5 train/test label sequence, sample 수, class set을 assert한다. T4/T5 feature row `r`은 동일 ID의 동일 원본 sample이다. ID 중복이 있으면 중단한다.

## 3. 공통 class set과 score 정의

T4 original head에는 T5 신규 class row가 없으므로 original T4/T5 head를 직접 비교하지 않는다. T5의 frozen Structure cosine head를 **고정 공통 readout**으로 사용해 같은 T5 class set에서 T4/T5 representation을 비교한다.

`s_t,k(x)=20 * normalize(w_k)^T normalize(f_t(x))`

따라서 변화는 fixed current boundary에 대한 representation 이동을 뜻하며, historical T4 classifier 성능 변화로 표현하지 않는다.

## 4. Boundary direction과 normalization

`a_ij = normalize(w_i)-normalize(w_j)`를 raw Structure feature coordinate에서 정의한다. `||z||>0`이면 `s_j(z)>s_i(z)`와 `a_ij^T z<0`의 sign이 동일하다. 전체 test sample/경쟁 class에서 이를 regression test한다.

Stored/Oracle mean과 covariance는 모두 T5 raw `sfeatures` coordinate를 사용한다. Oracle covariance에는 checkpoint와 같은 `stats_cov_jitter`를 더한다.

## 5. Oracle / Stored statistics

- Oracle: T5 actual train feature의 class mean/covariance. Old actual sample은 diagnostic 계산에만 사용한다.
- Stored: T5 checkpoint `class_means`, `class_covs`; main learner 또는 다음 task에 전달하지 않는다.
- score: `gamma=(a^T mu)/sqrt(a^T Sigma a + 1e-12)`, `q_hat=Phi(-gamma)`.
- prototype baseline: deployable baseline 비교를 위해 stored mean cosine similarity를 사용한다. Oracle mean similarity도 감사용 별도 컬럼에 저장한다.

## 6. Pair candidate와 actual target

모든 ordered seen-class pair `i→j, i!=j`를 후보로 사용한다.

- Target A: `B_ij(T5)` 및 violation count
- Target B: positive `delta_B` 및 positive violation-count 증가량
- Target C: `correct(T4)→wrong-j(T5)` rate/count

Current-hard는 `n_i>=5`, T5 error count `>=2` 중 error count/C/global share 순이다. Newly-degraded는 `n_i>=5`, `delta_C>0`, `delta_B>0`, margin q05 악화, correct→wrong-j `>=1`을 모두 만족한 pair를 lexicographic하게 정렬한다. 분모와 count를 항상 함께 저장한다.

## 7. Ranking 평가

Ranking:

1. random: seed 1993의 1,000개 random ordering 평균
2. stored prototype cosine: descending
3. oracle gamma: ascending
4. stored gamma: ascending

K는 `{10,20,50,100}`을 pair 수에 맞게 축소한다. 각 target에서 target top-K recall, captured non-negative mass, Spearman/Kendall을 기록한다. Ties와 zero-mass target은 report에서 INCONCLUSIVE로 둔다.

## 8. Optional region 설명

Current-hard/newly-degraded 상위 pair union(최대 20)에만 normalized train feature의 shared within-class covariance를 사용한다. shrinkage `lambda=0.1`, epsilon `1e-5`; Cholesky solve로 test sample의 continuous `D_i^2`, `D_j^2`, gap을 요약한다. IN/OUT 판정과 main decision에는 사용하지 않는다.

## 9. Leakage 방지

- frozen cache/checkpoint read-only
- backbone/adapter/APR/SCC/classifier update 없음
- test target은 diagnostic ranking 평가에만 사용
- Stage D pair selection이나 method hyperparameter로 test target을 사용하지 않음
- output은 별도 `pair_boundary_risk_diagnostic/` 아래 생성

## 10. 산출물과 runtime

Protocol별 요청 CSV/JSON/plot/report를 모두 만들고, 통합 `pair_boundary_risk_diagnostic_final_report.md`를 생성한다. Existing cache 재사용 기준 GPU 4 예상 runtime은 protocol당 약 1–5분, 합계 약 2–10분이다. 실제 실행은 사용자 명령으로만 시작한다.
