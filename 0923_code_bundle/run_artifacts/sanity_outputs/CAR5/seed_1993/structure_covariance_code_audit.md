# Vanilla Structure covariance code audit

## 결론

vanilla single-center Structure가 저장하는 class covariance는 **centered unbiased sample covariance**에 diagonal jitter를 더한 값이다. second moment가 아니다. old-class covariance는 APR에서 new-class mean displacement의 weighted outer product를 누적하고 task마다 `1e-6 I`를 추가한다. classifier adaptation은 이 full covariance를 float32 `MultivariateNormal`에 직접 전달하며, vanilla 경로에는 별도의 Cholesky retry/eigenvalue clipping fallback이 없다.

## 1. 최초/current-class covariance

`models/base.py::_compute_class_mean()`은 class feature `vectors`에 대해 다음을 저장한다.

```python
class_cov = torch.cov(torch.tensor(vectors, dtype=torch.float64).T)
class_cov += torch.eye(feature_dim) * 1e-3
```

`torch.cov`는 mean-centered unbiased covariance를 계산한다. follow-up의 동등 경로 `models/CDS_adapter.py::_statistics_from_features()`도 `centered.T @ centered / (n-1)`을 명시적으로 계산한 뒤 대칭화하고 `stats_cov_jitter`를 diagonal에 더한다. 현재 config의 jitter는 `1e-3`이다.

따라서 저장 통계는 `E[(z-mu)(z-mu)^T]`의 sample estimator이고 `E[zz^T]` second moment가 아니다.

## 2. APR covariance update

`utils/adjust_distribution.py::LobachevskyPrototypeCorrector.correct_prototypes()`에서 new-class mean displacement는

```text
Delta_j = mu_new_after_j - mu_new_before_j
```

이고 old class k의 cosine-softmax weight는 `w_kj`다. 실제 update는 다음과 같다.

```text
Sigma_k,new
  = Sigma_k,old
  + sum_j cov * (w_kj Delta_j)(w_kj Delta_j)^T
  + 1e-6 I
```

즉 단일 공통 `rho * Delta Delta^T`라기보다 class-pair별 `cov * w_kj^2` 계수를 가진 PSD rank-1 항들의 합이다. 코드가 계산하는 `relative_stretch_factor_j`는 최종 coefficient에 사용되지 않고, nonzero displacement이면 config의 `cov` 값(현재 5.0)을 그대로 사용한다.

## 3. epsilon 누적

APR corrected covariance는 `_class_covs`에 다시 저장되고 다음 task 시작 시 `_task_source_covs`로 복사된다. 따라서 old class가 매 transition을 지날 때마다 APR의 `1e-6 I`가 누적된다. birth task에서 들어간 empirical jitter `1e-3 I`도 이후 covariance 안에 계속 남는다.

이론적으로 trace increment는 transition마다 최소 `D * 1e-6`이고, 여기에 모든 weighted outer-product trace가 더해진다. D=768이면 epsilon만으로 transition당 trace `0.000768`이 증가한다. 실제 팽창의 주된 크기는 displacement outer product가 결정한다.

## 4. APR mean displacement와 covariance

APR mean은

```text
mu_k,new = mu_k,old + sum_j w_kj Delta_j
```

로 이동한다. covariance에는 이 합의 outer product를 한 번 넣는 것이 아니라 각 `w_kj Delta_j`의 outer product를 각각 합한다. 서로 다른 new-class displacement 사이의 cross term은 포함되지 않는다.

## 5. classifier adaptation에서 쓰는 covariance

`models/CDS_adapter.py::incremental_train()`은 APR update 뒤 `_stage2_compact_classifier()`를 호출한다. `models/base.py::_stage2_compact_classifier()`의 single-center 경로는 모든 seen class에 대해 `_class_means[c_id]`와 `_class_covs[c_id]`를 읽는다. 따라서 old class는 APR-corrected covariance, current class는 현재 feature로 계산한 empirical covariance를 사용한다.

mean에는 task-age 기반 `mean_scale = 0.9 + decay`가 곱해지고 covariance에는 scale을 곱하지 않는다. 매 epoch class당 `replay_samples_per_class`개를 full Gaussian에서 새로 sampling한다.

## 6. full-covariance sampling 안정화

vanilla single-center CA는 다음을 직접 실행한다.

```python
MultivariateNormal(cls_mean.float(), cls_cov.float())
```

즉 저장된 float64 covariance를 float32로 변환한 뒤 사용한다. 이 경로에는 adaptive jitter, Cholesky retry, eigenvalue clipping 또는 diagonal fallback이 없다. `utils/distribution_regions.py`의 stabilization 함수는 multi-center/separation region용이며 현재 `multi_center=false`인 vanilla CA에는 적용되지 않는다.

## 7. positive-definite 보장 수준

- empirical covariance: PSD이며 `1e-3 I`가 더해져 exact arithmetic에서는 PD다.
- APR update: PSD outer products와 `1e-6 I`만 더하므로 PD를 유지한다.
- 따라서 이론적 PD 보장은 있다.
- 다만 CA 직전 float32 cast 후의 수치적 PD를 별도로 검사하거나 복구하지 않는다. condition number가 매우 크면 `MultivariateNormal` construction이 실패할 가능성은 남아 있다.

oracle classifier replacement는 원 Structure condition에 추가 안정화를 몰래 적용하지 않는다. oracle covariance도 Structure와 동일한 `1e-3 I`를 포함하고, 추가 numerical fallback이 필요하면 그 횟수와 jitter를 결과에 명시한다.

## audit에서 확인할 항목

보강된 `covariance_oracle_audit.csv`는 class/task별로 trace, trace ratio, diagonal Pearson/Spearman, relative Frobenius error, participation effective rank, Structure/oracle delta trace를 저장한다. report는 Structure trace가 계속 증가하는데 oracle trace가 유지 또는 감소하는 class 비율을 별도로 집계한다.
