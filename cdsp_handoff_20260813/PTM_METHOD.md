# PTM-Guided Adapter 방법론

이 문서는 `PTM Place Only`와 `PTM Place+Move`의 구현 목적, 수식, 학습 순서 및 비교 기준을 독립적으로 설명한다. 상세 실험값은 `RESULTS_PTM_GUIDED_VS_STRUCTURE.md`, 실행 방법은 `RUN_PTM_GUIDED.md`를 참고한다.

## 1. 문제 설정

- Exemplar-Free Class-Incremental Learning(EFCIL)
- Backbone: ImageNet-21K로 사전학습한 ViT-B/16
- Backbone parameter는 고정하고 bottleneck adapter와 classifier만 학습
- 이전 task의 raw image replay는 사용하지 않음
- old-class 정보는 클래스/layer 단위 평균 anchor와 adapter response만 저장

핵심 가설은 frozen pretrained model(PTM)의 feature geometry를 새 클래스의 안정적인 배치 기준으로 사용하고, 이전 클래스 위치에서는 adapter 함수의 급격한 변화를 제한하면 representation drift를 줄일 수 있다는 것이다.

## 2. 공통 Adapter와 분류 손실

layer $\ell$의 residual bottleneck adapter는 다음과 같다.

$$
A_{\ell,\phi}(h)=s_\ell W^{\ell}_{\mathrm{up}}
\operatorname{ReLU}(W^{\ell}_{\mathrm{down}}h+b^{\ell}_{\mathrm{down}})
+b^{\ell}_{\mathrm{up}}.
$$

분류에는 CosFace objective를 사용한다.

$$
\mathcal L_{\mathrm{cls}}=-\frac1B\sum_{i=1}^{B}
\log\frac{\exp(s(\cos\theta_{y_i}-m))}
{\exp(s(\cos\theta_{y_i}-m))+\sum_{j\ne y_i}\exp(s\cos\theta_j)}.
$$

## 3. Frozen-PTM Placement

클래스 $c$가 처음 등장할 때 adapter를 끈 frozen backbone으로 클래스 prototype을 계산한다.

$$
p_c=\operatorname{Norm}\left(
\frac1{|D_c|}\sum_{x\in D_c}f_{\theta_0}(x;\text{adapter off})
\right).
$$

현재 batch에서 adapter feature의 클래스 평균을 $\bar z_c$라고 하면 placement loss는 다음과 같다.

$$
\bar z_c=\operatorname{Norm}\left(
\frac1{|B_c|}\sum_{i:y_i=c}f_{\theta_0,\phi}(x_i)
\right),
$$

$$
\mathcal L_{\mathrm{place}}=
\frac1{|C_B|}\sum_{c\in C_B}\left(1-\cos(\bar z_c,p_c)\right).
$$

학습 후반에는 classification refinement를 방해하지 않도록 cosine schedule로 placement weight를 0까지 감소시킨다.

$$
\lambda_{\mathrm{place}}(e)=\lambda_f+
\frac{\lambda_0-\lambda_f}{2}
\left(1+\cos\frac{\pi e}{E-1}\right),\qquad \lambda_f=0.
$$

## 4. Old-Anchor Functional Movement

클래스가 학습될 때 각 adapter 입력의 클래스 평균을 저장한다.

$$
a_{\ell,c}=\frac1{|D_c|}\sum_{x\in D_c}h_\ell(x).
$$

task $t-1$ 종료 시 해당 anchor에서의 adapter response도 함께 저장한다.

$$
r^{t-1}_{\ell,c}=A_{\ell,\phi^{t-1}}(a_{\ell,c}).
$$

다음 task에서는 같은 anchor에 대한 현재 response가 직전 response에서 과도하게 이동하지 않도록 제한한다.

$$
\mathcal L_{\mathrm{move}}=
\frac1{|\mathcal A_{\mathrm{old}}|}
\sum_{(\ell,c)\in\mathcal A_{\mathrm{old}}}
\operatorname{SmoothL1}\left(
A_{\ell,\phi^t}(a_{\ell,c}),r^{t-1}_{\ell,c}
\right).
$$

첫 task에는 old anchor가 없으므로 $\mathcal L_{\mathrm{move}}=0$이다.

## 5. 비교 Method

$$
\begin{aligned}
\text{Adapter Baseline}:\quad
&\mathcal L=\mathcal L_{\mathrm{cls}},\\
\text{PTM Place Only}:\quad
&\mathcal L=\mathcal L_{\mathrm{cls}}
+\lambda_{\mathrm{place}}(e)\mathcal L_{\mathrm{place}},\\
\text{PTM Place+Move}:\quad
&\mathcal L=\mathcal L_{\mathrm{cls}}
+\lambda_{\mathrm{place}}(e)\mathcal L_{\mathrm{place}}
+\lambda_{\mathrm{move}}\mathcal L_{\mathrm{move}}.
\end{aligned}
$$

세 조건은 backbone, adapter 크기, optimizer, dataset split, class order seed를 동일하게 유지한다. 기본 config는 `ptm_place_weight=1.0`, `old_move_weight=1.0`, cosine placement schedule을 사용한다.

## 6. Task 학습 순서

1. 현재 task의 frozen-PTM class prototype을 수집한다.
2. old anchor가 있으면 직전 adapter response를 movement target으로 고정한다.
3. classification, placement, movement loss로 adapter와 classifier를 학습한다.
4. 현재까지 본 클래스에 대해 평가하고 task accuracy matrix를 갱신한다.
5. 새 클래스의 layer anchor와 adapter response를 저장한다.
6. JSON/CSV metric과 선택적 resume checkpoint를 저장한다.

## 7. Structure 및 CDSP와의 차이

Structure는 SCC로 현재 sample과 old prototype의 angular relation을 보존하고, APR로 old-class 분포를 보정한 뒤 synthetic feature classifier adaptation을 수행한다. PTM-Guided는 SCC/APR/synthetic CA 없이 frozen-PTM placement와 old-anchor adapter response를 직접 제약한다.

기존 CDSP는 old discriminative subspace에 대한 gradient hard projection을 사용한다. PTM-Guided movement는 gradient 방향을 hard projection하지 않고, 저장된 anchor 위치에서 nonlinear adapter 함수 변화에 soft penalty를 부여한다.

## 8. 구현 파일

| 파일 | 역할 |
|---|---|
| `models/PTM_guided_adapter.py` | learner, loss 계산, task lifecycle, checkpoint |
| `utils/ptm_guidance.py` | PTM prototype 및 anchor/response memory |
| `utils/ptm_metrics.py` | JSON/CSV metric과 상태 기록 |
| `exps_jsons/ptm_guided_base.json` | 공통 hyperparameter |
| `exps_jsons/ptm_guided_*.json` | dataset/task/method별 config |
| `scripts/summarize_ptm_guided_results.py` | 결과 MD와 그래프 자동 생성 |
| `scripts/plot_ptm_guided_results.py` | Structure 막대그래프 및 task heatmap |

## 9. 그래프 해석 기준

- 최종 성능 막대: Structure 논문 Last-Acc/Inc-Acc보다 높으면 초록, 낮으면 빨강
- 10-task method heatmap: 대각선은 PTM 절대 정확도, 아래 삼각형은 PTM과 Adapter Baseline의 동일 셀 차이
- 5/20-task retention heatmap: 대각선은 task를 처음 학습했을 때의 정확도, 아래 삼각형은 이후 정확도에서 대각선 값을 뺀 유지·망각량
- Structure의 task accuracy matrix는 공개되어 있지 않으므로 task heatmap은 Structure와 직접 비교하지 않음

## 10. Reference

1. Xin Zhang, Liang Bai, Guanchao Wang, and Xian Yang. “Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure.” *CVPR 2026*. [CVF paper page](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_Exemplar-Free_Class_Incremental_Learning_via_Preserving_Class-Discriminative_Structure_CVPR_2026_paper.html)
2. Alexey Dosovitskiy et al. “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.” *ICLR 2021*. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

