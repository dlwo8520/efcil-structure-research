# 검토가 필요한 문제점

이 문서는 “결과가 이상해 보인다”는 판단과 직접 관련된 구현상 우려를 우선순위대로 적는다. 코드는 zip 생성 시점 그대로 보존했으며 여기서 수정하지 않았다.

## 1. Condition A가 원 classifier adaptation을 exact replay하지 못함

`exact_pre_ca_sidecar`와 동일 RNG/설정을 사용했지만 A 결과가 실제 checkpoint classifier와 완전히 같지 않다.

| Protocol | Main checkpoint T5 | Condition A | A−Main | A weight relative L2 |
|---|---:|---:|---:|---:|
| IA5 | 66.4253 | 66.5569 | +0.1317 pp | 0.003522 |
| CAR5 | 81.1964 | 81.2585 | +0.0622 pp | 0.003121 |

따라서 CSV의 `exact_original_CA_replay_candidate=True`는 **후보 조건을 충족했다는 뜻이지 실제 exact equality를 통과했다는 뜻이 아니다**.

주요 의심 지점:

- 원 Structure는 `SimpleContinualLinear.heads`의 여러 `nn.Linear` parameter를 optimizer에 전달한다.
- diagnostic은 모든 head weight를 concatenate해 하나의 `DiagnosticCosineHead.weight` parameter로 만든다.
- 수학적으로 유사하지만 parameter layout, optimizer update, kernel 호출 순서가 달라 bitwise replay가 아닐 수 있다.
- 원 CA 자체를 동일 network/head 구조로 복제하지 않고 별도 compact head를 사용한다.

A가 exact하지 않으므로 B/C/D와 A 사이의 수십 분의 1~수십 분의 몇 pp 차이는 이 구현 오차와 분리해 검증해야 한다.

## 2. CAR5의 task-age mean scaling

원 `models/base.py::_stage2_compact_classifier()`는 class birth task를 다음처럼 계산한다.

```python
t_id = c_id // task_size
```

CAR5 partition은 `[40,39,39,39,39]`인데 final `task_size` 인수는 39다. 따라서 실제 partition boundary와 `c_id // 39`가 일부 class에서 일치하지 않는다. diagnostic도 원 코드 재현을 위해 같은 계산을 사용한다.

이것은 A–D 조건 간 공정성에는 공통으로 적용되지만, CAR5의 mean scaling 자체가 실제 birth task와 정확히 대응하지 않는다는 의미다.

## 3. 상속된 extension config의 혼동 가능성

정식 log에는 다음이 포함된다.

```text
followup_extension = True
similarity_aware_extension = True
```

동시에 실제 method 설정은 다음처럼 vanilla로 검증됐다.

```text
old_new_separation = false
selective_method = none
adaptive_margin_mode = none
similarity_relation_mode = none
similarity_scc_enabled = false
multi_center = false
APR = true
```

즉 extension flag 이름만 보면 다른 방법이 켜진 것처럼 보일 수 있다. 실제 learner factory와 false/no-op 설정이 vanilla 동작을 유지하는지는 bundled source 기준으로 다시 확인할 필요가 있다.

## 4. Oracle replacement의 의미

- old class mean/covariance만 oracle로 교체한다.
- current/new statistics는 Structure statistics로 유지한다.
- raw old feature를 classifier에 직접 학습시키지 않고 Gaussian synthetic feature를 sampling한다.
- 따라서 D Full Oracle도 “실제 old feature로 학습한 완전 oracle classifier”가 아니라 “old oracle 1·2차 moment를 쓴 Gaussian CA”다.

D가 개선되지 않는다고 해서 oracle old feature가 도움이 없다는 결론은 낼 수 없다. Gaussian moment approximation과 CA optimizer의 영향이 남는다.

## 5. 단일 seed와 자동 Case 판정

두 정식 실행 모두 seed 1993 하나다. report의 `SCIENTIFIC` 표기는 class/sample/head override가 없다는 뜻이며, 통계적 반복이 완료됐다는 뜻은 아니다. Case A–F threshold도 heuristic이다.

## 6. 우선 확인할 수정 전 검증

1. 원 `SimpleContinualLinear` 구조를 그대로 복제해 Condition A weight와 prediction을 exact equality로 맞춘다.
2. A reproduction assertion을 통과하지 않으면 B/C/D를 실행하지 않게 한다.
3. CAR5 mean-scale birth task를 실제 partition 기반과 원 `// task_size` 기반으로 각각 비교한다.
4. D와 별도로 real old feature로 head만 학습하는 upper-bound diagnostic을 추가해 Gaussian approximation 효과를 분리한다.
5. 최소 3 seeds로 반복하기 전 자동 병목 판정을 확정 결론으로 사용하지 않는다.
