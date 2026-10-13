# Structure prototype oracle audit — IA5

> 상태: **PARTIAL** · audited tasks: `[5]` · primary oracle: `oracle_train_statistics`

이 보고서는 학습 완료 체크포인트를 `eval()`/`torch.no_grad()`로 읽은 사후 분석이다. old sample은 loss, optimizer, APR, SCC, classifier adaptation, model selection에 전달되지 않았다.

## 핵심 수치

| 항목 | 값 |
|---|---:|
| old-class Structure↔oracle cosine 평균 | 0.6167 |
| old-class prototype relative L2 평균 | 0.7884 |
| class age↔prototype error Pearson | INCONCLUSIVE |
| APR error reduction 평균 | 0.4457 |
| APR drift↔oracle drift cosine 평균 | INCONCLUSIVE |
| covariance Frobenius relative error 평균 | 0.9838 |
| diagonal Mahalanobis D² Pearson 평균 | INCONCLUSIVE |
| nearest-old ranking changed rate 평균 | INCONCLUSIVE |
| final oracle prototype classifier gain (pp) | 0.0000 |
| prototype error↔forgetting Spearman | INCONCLUSIVE |

APR class 집계: `improved`=2.

## 요청 질문에 대한 답

1. **Q1 prototype 근접도:** cosine 평균 `0.6167`, relative L2 평균 `0.7884`다.
2. **Q2 task/age 누적:** class age와 relative L2의 Pearson이 `INCONCLUSIVE`다. 전체 task coverage가 없으면 이 값은 확정 결론으로 쓰면 안 된다.
3. **Q3 APR 개선:** 평균 error reduction은 `0.4457`이고 improved/unchanged/worsened는 `{'improved': 2}`다.
4. **Q4 drift 방향:** APR drift와 true drift cosine 평균은 `INCONCLUSIVE`다.
5. **Q5 covariance:** diagonal relative L2 `0.9245`, full Frobenius relative error `0.9838`다.
6. **Q6 Mahalanobis boundary:** D² correlation `INCONCLUSIVE`, nearest-old 변경률 `INCONCLUSIVE`다.
7. **Q7 oracle prototype classifier:** final gap은 `0.0000` percentage point다.
8. **Q8 forgetting/confusion 연관:** prototype error와 forgetting Spearman은 `INCONCLUSIVE`다.
9. **Q9 5-task vs 10-task:** 이 protocol 하나만 완료되어 5-task/10-task 차이는 INCONCLUSIVE다.
10. **Q10 dataset 반복성:** 이 dataset 하나만 완료되어 ImageNet-A/Cars-196 반복 여부는 INCONCLUSIVE다.
11. **Q11 원인 판정:** INCONCLUSIVE

## 해석 제한

- t-SNE 좌표는 task마다 별도로 학습되므로 task 간 절대 이동으로 해석하지 않았다. task 간 비교는 shared PCA만 사용한다.
- `checkpoint_latest.pt` 하나만 audit한 경우 최종 task 단면 분석이며 true drift와 시간 누적 질문은 INCONCLUSIVE다.
- correlation은 인과성을 뜻하지 않는다. single-seed 결과는 후속 seed 반복이 필요하다.
