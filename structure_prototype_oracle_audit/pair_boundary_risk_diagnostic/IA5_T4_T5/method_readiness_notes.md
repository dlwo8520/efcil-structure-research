# Method Readiness Notes

> METHOD READY: **NO**

## Evidence gate

- R1 충분한 net newly-degraded mass: **NO** — eligible=9/39800 (0.0%); new errors=14 < recovered=26; fixed-head accuracy Δ=0.790 pp; boundary violations Δ=-151
- R2 Oracle gamma ranking: **NO** — new-risk B/C: Oracle beats random 3/4, prototype 3/4; correlations remain small
- R3 Stored gamma vs prototype: **NO** — Stored gamma beats prototype on 3/4 B/C recall/mass comparisons; consistent superiority requires 4/4
- Stored↔Oracle reproduction: **YES** — Spearman(gamma_stored, gamma_oracle)=0.980

## Optional Stage D 설계 제안

METHOD READY가 YES일 때만 random/prototype/stored-risk/oracle-development selection을 동일 pair 수, 동일 total coefficient, 동일 optimizer step으로 비교한다. Test target으로 pair 또는 hyperparameter를 선택하지 않는다.

첫 intervention은 separation이 아니라 risk-weighted relation preservation 후보로 둔다. Existing SCC의 learnable current feature 경로에 selected old-class boundary 방향을 연결해야 하며, detached stored prototype끼리만 계산하는 loss는 gradient가 adapter/backbone으로 흐르지 않으므로 사용하지 않는다.

이 문서는 구현 readiness만 기록하며 새 method의 성능 또는 인과효과를 주장하지 않는다.
