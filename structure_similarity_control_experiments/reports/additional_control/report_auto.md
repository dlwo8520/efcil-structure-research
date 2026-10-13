# Additional Control: Does Similarity Matter?

공통 완료 seed: 1993. 모든 차이는 동일 seed의 C0와 paired 비교한다.

## Main Table

| Method | n | Inc-Acc | Last-Acc | Forgetting | Old | New | High-sim confusion |
|---|---|---|---|---|---|---|---|
| C0 AM-Drift | 1 | 74.77 (SD N/A) | 66.69 (SD N/A) | 6.97 (SD N/A) | 67.55 (SD N/A) | 62.73 (SD N/A) | 0.26 ± 0.07 |
| C1 High-Off | 1 | 74.53 (SD N/A) | 66.36 (SD N/A) | 7.10 (SD N/A) | 67.39 (SD N/A) | 61.62 (SD N/A) | 0.26 ± 0.07 |
| C2 Low-Off | 1 | 74.61 (SD N/A) | 66.69 (SD N/A) | 6.89 (SD N/A) | 67.47 (SD N/A) | 63.10 (SD N/A) | 0.26 ± 0.07 |
| C3 Random-Off | 1 | 74.43 (SD N/A) | 66.36 (SD N/A) | 7.17 (SD N/A) | 67.15 (SD N/A) | 62.73 (SD N/A) | 0.26 ± 0.07 |
| C4 High-Weak | 1 | 74.59 (SD N/A) | 66.43 (SD N/A) | 7.24 (SD N/A) | 67.23 (SD N/A) | 62.73 (SD N/A) | 0.26 ± 0.07 |

![group delta](figures/group_accuracy_delta.png)

![high confusion](figures/high_similarity_confusion.png)

![high old/new](figures/high_similarity_old_new_accuracy.png)

![feature norm](figures/feature_norm.png)

![intrusion](figures/intrusion.png)

![similarity delta](figures/similarity_pair_accuracy_delta.png)

## Paired contrasts

- C1−C0: high-similarity separation 제거 효과
- C1−C2: similarity 위치 선택 효과
- C1−C3: similarity selection 대 random selection
- C4−C1: weak constraint 대 complete-off

## 해석 질문

1. High-similarity separation 제거가 new learning을 개선하는가?
2. 단순 constraint 수 감소가 아니라 similarity selection의 효과인가?
3. 완전 제거와 weak constraint 중 어느 쪽이 나은가?
4. 제거 시 similar-pair confusion이 증가하는가?
5. 결과가 similarity-aware tau 또는 relative margin을 지지하는가?

위 질문의 확정 결론은 표와 그래프의 paired delta 및 seed 간 변동을 함께 보고 작성해야 한다.
