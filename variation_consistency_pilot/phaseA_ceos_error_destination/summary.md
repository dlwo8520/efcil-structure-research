# Phase A — CEOS New-error destination

저장되지 않았던 C1 logits는 사용자가 승인한 동일 설정 결정론적 재실행으로 복원했다. 기존 C1 accuracy·train loss와 정확히 일치함을 확인한 뒤 paired New test sample에만 적용했다.

| Protocol | N New | Gaussian correct→CEOS wrong | Gaussian wrong→CEOS correct | Type 1 Old competition | Type 2 New internal | Type 3 Both |
|---|---:|---:|---:|---:|---:|---:|
| IA10 | 124 | 5 (4.03%) | 0 (0.00%) | 5 | 0 | 0 |
| CAR10 | 779 | 45 (5.78%) | 0 (0.00%) | 45 | 0 | 0 |

손실 사례는 모두 true New class가 New 내부에서는 1위지만 Old class에 밀린 Type 1이었다. Recovery는 없었다. 즉 CEOS의 New 손실은 이 실행에서 Old/New 경쟁 악화로 국소화되며 New-New 단독 판별 악화 근거는 없다.

`delta_g`/`delta_w`의 전체·손실·회복 부분집합별 mean/median/q10/q25/q75/q90은 각 protocol의 `*_summary.json`에, sample별 값은 CSV에 있다.
