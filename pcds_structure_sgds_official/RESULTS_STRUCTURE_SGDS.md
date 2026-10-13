# Cars-196: official Structure (PCDS) vs Structure + SGDS Activation Guidance

## What was run

Both rows use the official `models/CDS_adapter.py` Structure learner from upstream commit `a0db2168663982a4cad24981231721a3518ddf29`: ViT-B/16 ImageNet-21K, attention adapters (bottleneck 64), SCC, APR, cosine classifier, and Gaussian classifier adaptation. The SGDS row adds only Eq. (4)-(10) semantic exploration/activation compaction to the final ViT block's adapter input. It does not use the SGDS adapter bank, retrieval, fusion, entropy selection, or classifier.

Cars-196 uses the prepared official train/test split (8,144/8,041 images), a single expanding head without Task-ID, no raw replay, and task sizes `20 x 9 + 16`. Training seed is 0 and the shared class-order seed is 1993. The official Cars hyperparameters are retained: 70 epochs on Task 0, 40 thereafter, SCC weight 5, APR gamma/rho 10, and CA 5 epochs.

`k=0.6` is explicitly interpreted as Top-K retain ratio (up to 60% retained, about 40% zero); the Bernoulli mask may make the realized retained ratio lower. The mask is stochastic only during training. Its seed (100003) is isolated from the ordinary training RNG so it does not shift baseline data-order/dropout draws. Counters are updated only from dimensions surviving both Bernoulli and Top-K.

## Methodology and equation-level difference

### 1. Shared official Structure (PCDS) method

Let $h^l$ be the normalized hidden tokens entering the attention module at ViT block $l$. The official attention adapter is

$$
A(h)=0.1\,W_{up}\operatorname{Dropout}_{0.1}
\left(\operatorname{ReLU}(W_{down}h)\right),
$$

and the key/value branches used by **both methods** are

$$
k^l=W_k^l h^l+A_k^l(h^l),\qquad
v^l=W_v^l h^l+A_v^l(h^l).
$$

The classifier uses normalized features and normalized weights (cosine logits), followed by the official CosFace loss. From Task 1 onward, Structure-preserving consistency (SCC) compares the current student $z_i^s$ and frozen preceding adapter branch $z_i^t$ against every old prototype $\mu_c$:

$$
\mathcal L_{SCC}
=\frac1B\sum_i\sum_{c\in\mathcal Y_{old}}
\left[
\cos(z_i^s,\mu_c)-\cos(z_i^t,\mu_c)
\right]^2.
$$

The actual official Cars objective used here is

$$
\mathcal L_{Structure}
=\mathcal L_{CosFace}+5\mathcal L_{SCC},
$$

with SCC omitted at Task 0 because no old prototypes exist.

For APR, let the representation drift of current class $j$ be

$$
\Delta\mu_j=\mu_j^{after}-\mu_j^{before},
$$

and let its relevance to old class $k$ be

$$
w_{kj}=
\operatorname{softmax}_j
\left(10\cos(\mu_k,\mu_j^{before})\right).
$$

The old Gaussian statistics are rectified as implemented in the official code:

$$
\mu'_k=\mu_k+\sum_j w_{kj}\Delta\mu_j,
$$

$$
\Sigma'_k=\Sigma_k+
10\sum_j
(w_{kj}\Delta\mu_j)(w_{kj}\Delta\mu_j)^\top
+\epsilon I.
$$

Classifier adaptation (CA) then samples

$$
\tilde z_c\sim\mathcal N(\alpha_{c,t}\mu'_c,\Sigma'_c),\qquad
\alpha_{c,t}=0.9+0.1\frac{t_c+1}{t+1},
$$

and optimizes **only the cosine classifier** for five epochs. Here $t_c$ is the task-age index used by the official implementation. No raw image from an old task is replayed.

### 2. SGDS activation guidance added in the second condition

Frozen pretrained-backbone class prototypes are

$$
\bar\mu_y=\frac1{|\mathcal D_y|}\sum_{x\in\mathcal D_y}\phi_{frozen}(x).
$$

For each new class $c$, Eq. (4)-(6) are

$$
P(y\mid c)=
\frac{\exp(\cos(\bar\mu_c,\bar\mu_y))}
{\sum_{y'\in\mathcal Y_{old}\cup\mathcal C_t}
\exp(\cos(\bar\mu_c,\bar\mu_{y'}))},
$$

$$
S_{old}(c)=\sum_{y\in\mathcal Y_{old}}P(y\mid c),\qquad
S_{new}(c)=\sum_{y\in\mathcal C_t}P(y\mid c).
$$

The current-class sum includes $c$ itself. If $S_{old}>S_{new}$, the class uses Knowledge Reuse; otherwise it uses New Subspace Allocation.

During the first half of training, the per-dimension Bernoulli probability is

$$
p_j(c)=
\begin{cases}
1-\exp\left[-\sum_{y\in\mathcal Y_{old}}P(y\mid c)
\frac{F_c[y,j]}{\max_iF_c[y,i]+\epsilon}\right],
& S_{old}>S_{new},\\
\exp\left[-0.5\frac{F[j]}{\max_iF[i]+\epsilon}\right],
& \text{otherwise}.
\end{cases}
$$

During the second half, activation compaction is

$$
p_j(c)=1-\exp\left[-1.0\frac{F_c[c,j]}{\max_iF_c[c,i]+\epsilon}\right].
$$

For each sample, $m_j\sim\operatorname{Bernoulli}(p_j)$, followed by deterministic Top-K retention:

$$
\hat h^{11}
=\operatorname{TopK}_{\lfloor0.6\times768\rfloor}
(h^{11}\odot m).
$$

Only dimensions surviving both operations update the global counter $F[j]$ and class counter $F_c[c,j]$. One class-conditioned dimension mask is shared across all tokens of a sample; Top-K magnitude is the token-mean absolute activation. Stochastic masking is disabled at evaluation.

The **only forward-path difference** is therefore at the final ViT block ($l=11$), and only in the actively trained student adapter branch; the SCC teacher branch stays dense:

$$
\begin{aligned}
\text{Structure:}\quad & k^{11}=W_kh+A_k(h),\quad v^{11}=W_vh+A_v(h),\\
\text{Structure+SGDS:}\quad & k^{11}=W_kh+A_k(\hat h),\quad v^{11}=W_vh+A_v(\hat h).
\end{aligned}
$$

The Structure loss, SCC, APR, classifier, and CA are unchanged; SGDS contributes no additional loss. Full SGDS adapter retrieval/fusion, entropy selection, MOS alignment, and SGDS classifier are not included.

| Component | Official Structure | Structure + SGDS |
| --- | --- | --- |
| Backbone/adapters | Official ViT-B/16 + attention adapters | Identical |
| Objective | CosFace + 5 SCC | Identical |
| APR / Gaussian CA | Enabled | Identical |
| Final-block adapter input | Dense $h$ | Bernoulli + Top-K $\hat h$ |
| Semantic Eq. (4)-(10) | None | Enabled |
| Inference mask | None | None (training-only stochastic mask) |
| Raw replay / Task-ID | None / none | None / none |

### 3. Metric definitions

At the end of task $t$, Seen Accuracy is accuracy over every class observed through $t$; Old/New Accuracy splits that set at the current task boundary. Average Incremental Accuracy is

$$
\operatorname{AvgInc}_t=\frac1{t+1}\sum_{r=0}^t\operatorname{SeenAcc}_r.
$$

Using task-accuracy matrix entry $A_{t,j}$, average forgetting is

$$
F_t=\frac1t\sum_{j<t}
\left(\max_{r\in\{j,\ldots,t\}}A_{r,j}-A_{t,j}\right).
$$

## Main results

| Method | Avg Inc. | Final | Old | New | Forgetting |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official Structure (PCDS) | 84.86 | 79.62 | 79.12 | 85.15 | 8.79 |
| Structure + SGDS | 67.83 | 26.89 | 21.19 | 90.61 | 69.07 |
| Delta (SGDS - Structure) | -17.03 | -52.73 | -57.93 | +5.45 | +60.28 |

The official Structure baseline Final Accuracy is **79.62%**, which is **-0.61pp** relative to the paper's Cars-196 10-task reference of 80.23%. It lies inside the paper's reported `80.23 +/- 0.68` interval, although this single run is not a statistical reproduction.

## Complete task-wise results

### Official Structure (PCDS)

| Task | Seen cls | Seen Acc | Old Acc | New Acc | Forgetting | Avg Inc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20 | 93.08 | N/A | 93.08 | 0.00 | 93.08 |
| 1 | 40 | 88.42 | 87.71 | 89.14 | 5.37 | 90.75 |
| 2 | 60 | 87.58 | 86.46 | 89.89 | 4.89 | 89.69 |
| 3 | 80 | 86.65 | 86.33 | 87.64 | 4.50 | 88.93 |
| 4 | 100 | 85.57 | 84.78 | 88.79 | 5.24 | 88.26 |
| 5 | 120 | 83.76 | 83.53 | 84.92 | 6.22 | 87.51 |
| 6 | 140 | 82.59 | 81.93 | 86.63 | 7.00 | 86.81 |
| 7 | 160 | 81.15 | 80.89 | 83.01 | 7.70 | 86.10 |
| 8 | 180 | 80.17 | 79.25 | 87.70 | 8.63 | 85.44 |
| 9 | 196 | 79.62 | 79.12 | 85.15 | 8.79 | 84.86 |

### Structure + SGDS

| Task | Seen cls | Seen Acc | Old Acc | New Acc | Forgetting | Avg Inc. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 20 | 93.32 | N/A | 93.32 | 0.00 | 93.32 |
| 1 | 40 | 88.19 | 87.95 | 88.42 | 5.37 | 90.75 |
| 2 | 60 | 87.41 | 86.16 | 90.01 | 4.77 | 89.64 |
| 3 | 80 | 85.68 | 84.44 | 89.47 | 6.13 | 88.65 |
| 4 | 100 | 82.39 | 80.12 | 91.63 | 10.15 | 87.40 |
| 5 | 120 | 75.94 | 73.13 | 89.67 | 17.36 | 85.49 |
| 6 | 140 | 62.17 | 57.24 | 92.45 | 33.17 | 82.16 |
| 7 | 160 | 43.58 | 36.82 | 91.20 | 53.79 | 77.33 |
| 8 | 180 | 32.76 | 25.99 | 88.71 | 64.66 | 72.38 |
| 9 | 196 | 26.89 | 21.19 | 90.61 | 69.07 | 67.83 |

### Task-wise delta (SGDS - Structure)

| Task | Delta Seen | Delta Old | Delta New | Delta Forgetting | Delta Avg Inc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.24 | N/A | +0.24 | +0.00 | +0.24 |
| 1 | -0.24 | +0.24 | -0.72 | -0.00 | +0.00 |
| 2 | -0.16 | -0.30 | +0.12 | -0.12 | -0.05 |
| 3 | -0.97 | -1.89 | +1.84 | +1.63 | -0.28 |
| 4 | -3.18 | -4.66 | +2.83 | +4.91 | -0.86 |
| 5 | -7.83 | -10.40 | +4.75 | +11.15 | -2.02 |
| 6 | -20.41 | -24.69 | +5.82 | +26.18 | -4.65 |
| 7 | -37.58 | -44.07 | +8.19 | +46.09 | -8.77 |
| 8 | -47.41 | -53.27 | +1.00 | +56.02 | -13.06 |
| 9 | -52.73 | -57.93 | +5.45 | +60.28 | -17.03 |

Compact Seen-Accuracy comparison:

| Task | Structure | +SGDS | Delta |
| --- | ---: | ---: | ---: |
| 0 | 93.08 | 93.32 | +0.24 |
| 1 | 88.42 | 88.19 | -0.24 |
| 2 | 87.58 | 87.41 | -0.16 |
| 3 | 86.65 | 85.68 | -0.97 |
| 4 | 85.57 | 82.39 | -3.18 |
| 5 | 83.76 | 75.94 | -7.83 |
| 6 | 82.59 | 62.17 | -20.41 |
| 7 | 81.15 | 43.58 | -37.58 |
| 8 | 80.17 | 32.76 | -47.41 |
| 9 | 79.62 | 26.89 | -52.73 |

## Complete task-accuracy matrices

Each cell is accuracy on that task's class partition after the row's training task. The last column is accuracy over all seen test samples, not the unweighted mean of task cells.

### Official Structure matrix

| After task | Eval T0 | Eval T1 | Eval T2 | Eval T3 | Eval T4 | Eval T5 | Eval T6 | Eval T7 | Eval T8 | Eval T9 | Seen avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 93.08 | — | — | — | — | — | — | — | — | — | 93.08 |
| 1 | 87.71 | 89.14 | — | — | — | — | — | — | — | — | 88.42 |
| 2 | 83.29 | 89.62 | 89.89 | — | — | — | — | — | — | — | 87.58 |
| 3 | 82.22 | 87.23 | 89.64 | 87.64 | — | — | — | — | — | — | 86.65 |
| 4 | 78.28 | 86.63 | 87.67 | 86.66 | 88.79 | — | — | — | — | — | 85.57 |
| 5 | 73.87 | 83.77 | 85.08 | 87.15 | 88.05 | 84.92 | — | — | — | — | 83.76 |
| 6 | 69.57 | 79.36 | 83.35 | 87.15 | 88.55 | 83.97 | 86.63 | — | — | — | 82.59 |
| 7 | 65.75 | 78.16 | 80.89 | 84.82 | 87.81 | 83.85 | 85.40 | 83.01 | — | — | 81.15 |
| 8 | 62.77 | 75.89 | 76.45 | 83.72 | 85.34 | 82.90 | 85.15 | 82.27 | 87.70 | — | 80.17 |
| 9 | 63.37 | 70.53 | 74.48 | 81.88 | 85.22 | 81.59 | 85.64 | 81.78 | 88.46 | 85.15 | 79.62 |

### Structure + SGDS matrix

| After task | Eval T0 | Eval T1 | Eval T2 | Eval T3 | Eval T4 | Eval T5 | Eval T6 | Eval T7 | Eval T8 | Eval T9 | Seen avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 93.32 | — | — | — | — | — | — | — | — | — | 93.32 |
| 1 | 87.95 | 88.42 | — | — | — | — | — | — | — | — | 88.19 |
| 2 | 83.77 | 88.54 | 90.01 | — | — | — | — | — | — | — | 87.41 |
| 3 | 79.59 | 84.49 | 89.40 | 89.47 | — | — | — | — | — | — | 85.68 |
| 4 | 69.33 | 80.67 | 82.61 | 88.13 | 91.63 | — | — | — | — | — | 82.39 |
| 5 | 57.04 | 71.96 | 69.05 | 80.42 | 87.68 | 89.67 | — | — | — | — | 75.94 |
| 6 | 34.49 | 47.85 | 44.27 | 61.20 | 75.74 | 80.05 | 92.45 | — | — | — | 62.17 |
| 7 | 8.23 | 18.97 | 15.54 | 22.64 | 51.97 | 62.23 | 78.96 | 91.20 | — | — | 43.58 |
| 8 | 1.43 | 2.63 | 6.66 | 4.77 | 22.41 | 33.61 | 59.78 | 77.75 | 88.71 | — | 32.76 |
| 9 | 0.12 | 0.24 | 2.59 | 0.12 | 4.31 | 10.33 | 34.16 | 62.96 | 78.54 | 90.61 | 26.89 |

## Effective rank (current-task deterministic train features)

| Task | Structure rank | +SGDS rank | Delta |
| --- | ---: | ---: | ---: |
| 0 | 453 | 454 | +1 |
| 1 | 435 | 433 | -2 |
| 2 | 440 | 434 | -6 |
| 3 | 438 | 426 | -12 |
| 4 | 424 | 397 | -27 |
| 5 | 421 | 373 | -48 |
| 6 | 411 | 337 | -74 |
| 7 | 410 | 294 | -116 |
| 8 | 395 | 252 | -143 |
| 9 | 366 | 220 | -146 |

Mean rank delta (SGDS - Structure): **-57.30**. Negative means the measured representation rank became more compact.

## Classifier adaptation

| Task | Structure pre-CA | Structure post-CA | Structure gain | +SGDS pre-CA | +SGDS post-CA | +SGDS gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 93.08 | 93.08 | +0.00 | 93.32 | 93.32 | +0.00 |
| 1 | 85.86 | 88.42 | +2.57 | 85.86 | 88.19 | +2.33 |
| 2 | 85.40 | 87.58 | +2.17 | 84.20 | 87.41 | +3.22 |
| 3 | 84.11 | 86.65 | +2.54 | 81.39 | 85.68 | +4.30 |
| 4 | 80.47 | 85.57 | +5.10 | 69.41 | 82.39 | +12.97 |
| 5 | 78.80 | 83.76 | +4.96 | 59.42 | 75.94 | +16.52 |
| 6 | 79.36 | 82.59 | +3.23 | 51.25 | 62.17 | +10.93 |
| 7 | 78.30 | 81.15 | +2.86 | 41.48 | 43.58 | +2.10 |
| 8 | 77.31 | 80.17 | +2.86 | 28.79 | 32.76 | +3.97 |
| 9 | 75.82 | 79.62 | +3.79 | 24.61 | 26.89 | +2.28 |

## Resources

| Method | Peak alloc. GB | Peak reserved GB | Final trainable | Total parameters |
| --- | ---: | ---: | ---: | ---: |
| Official Structure (PCDS) | 6.288 | 14.381 | 4909056 | 90707712 |
| Structure + SGDS | 6.289 | 13.189 | 4909056 | 90707712 |

## SGDS semantic behavior

| Task | Old cls | New cls | Reuse | Allocate | Reuse % | Mean S_old | Mean S_new |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 20 | 0 | 20 | 0.00 | 0.0000 | 1.0000 |
| 1 | 20 | 20 | 11 | 9 | 55.00 | 0.5019 | 0.4981 |
| 2 | 40 | 20 | 20 | 0 | 100.00 | 0.6645 | 0.3355 |
| 3 | 60 | 20 | 20 | 0 | 100.00 | 0.7465 | 0.2535 |
| 4 | 80 | 20 | 20 | 0 | 100.00 | 0.7940 | 0.2060 |
| 5 | 100 | 20 | 20 | 0 | 100.00 | 0.8319 | 0.1681 |
| 6 | 120 | 20 | 20 | 0 | 100.00 | 0.8561 | 0.1439 |
| 7 | 140 | 20 | 20 | 0 | 100.00 | 0.8739 | 0.1261 |
| 8 | 160 | 20 | 20 | 0 | 100.00 | 0.8881 | 0.1119 |
| 9 | 180 | 16 | 16 | 0 | 100.00 | 0.9169 | 0.0831 |

## SGDS activation behavior

| Task | Layer | Actual nonzero % | Actual zero % | Mean active dims | Mean p | Global max | Global mean | Class max | Class mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 11 | 59.49 | 40.51 | 456.9 | 0.6105 | 36637 | 35270.0 | 3050 | 1763.5 |
| 1 | 11 | 52.74 | 47.26 | 405.1 | 0.5331 | 54895 | 53096.8 | 1205 | 891.3 |
| 2 | 11 | 51.75 | 48.25 | 397.5 | 0.5189 | 72451 | 70092.1 | 1061 | 849.8 |
| 3 | 11 | 53.73 | 46.27 | 412.6 | 0.5390 | 90677 | 87843.8 | 1127 | 887.6 |
| 4 | 11 | 54.83 | 45.17 | 421.1 | 0.5504 | 109368 | 105958.5 | 1171 | 905.7 |
| 5 | 11 | 55.74 | 44.26 | 428.1 | 0.5601 | 128794 | 125000.0 | 1128 | 952.1 |
| 6 | 11 | 56.23 | 43.77 | 431.8 | 0.5647 | 147706 | 143443.4 | 1129 | 922.2 |
| 7 | 11 | 56.65 | 43.35 | 435.1 | 0.5690 | 166967 | 162207.2 | 1172 | 938.2 |
| 8 | 11 | 56.94 | 43.06 | 437.3 | 0.5720 | 185608 | 180542.3 | 1149 | 916.8 |
| 9 | 11 | 57.61 | 42.39 | 442.5 | 0.5791 | 201316 | 196029.2 | 1213 | 967.9 |

- Correlation between old-class count and mean S_old: **0.8165**.
- Tasks with both strategies present: **1 / 10**.
- Tasks saturated at 100% Knowledge Reuse: **2, 3, 4, 5, 6, 7, 8, 9**.
- Correlation of maximum old-class cosine similarity with Knowledge Reuse: **0.3357**.
- Correlation of maximum old-class cosine similarity with final class accuracy: **0.3030**.
- Mean maximum old cosine, Reuse vs Allocation: **0.9431 vs 0.8943**.

## Interpretation of the task-wise behavior

- Through Task 2 the methods are nearly tied (Seen delta between +0.24pp and -0.24pp). The failure becomes visible at Task 4 (-3.18pp), accelerates at Task 6 (-20.41pp), and reaches -52.73pp at Task 9.
- This is not failure to learn the newest task: final New Accuracy is 90.61% with SGDS versus 85.15% for Structure. The loss comes from old classes (21.19% versus 79.12%), producing 69.07% forgetting.
- The accuracy matrix shows direct recency bias: under SGDS, Task-0 accuracy falls from 93.32% when learned to 0.12% after Task 9, while the newest Task-9 partition remains at 90.61%.
- Semantic selection becomes fully saturated: every new class is assigned Knowledge Reuse from Task 2 onward. The correlation between old-class count and mean $S_{old}$ is 0.8165, supporting the concern that probability mass is strongly affected by the growing number of old classes.
- SGDS does compact the measured representation: the final current-task rank drops from 366 to 220. In this run, however, lower rank accompanies severe old-class collapse, so rank reduction is not evidence of beneficial structure preservation.
- Realized zero ratios remain approximately 40.5-48.2%, so the failure cannot be explained by the sparsification code silently doing nothing. CA sometimes recovers substantial Seen Accuracy, but it cannot repair the progressively destroyed old-task geometry.
- These statements describe one controlled seed. They establish that this particular training-only final-block SGDS adaptation is incompatible with official Structure under the tested settings; they do not establish that the complete SGDS method fails.

## Direct answers

1. Final accuracy improvement: **no** (-52.73 percentage points).
2. Larger than the prior CIFAR-100 +0.07pp screening delta: **no**.
3. Old accuracy delta: **-57.93pp**.
4. Forgetting delta: **+60.28pp** (negative is better).
5. New-class accuracy delta: **+5.45pp**.
6. Mean effective-rank delta: **-57.30**.
7. Class-dependent semantic variation occurred in **1** tasks.
8. Full-reuse saturation occurred in tasks **2, 3, 4, 5, 6, 7, 8, 9**.
9. Similarity/reuse correlation: **0.3357**.
10. Geometry-related conclusions should be treated as descriptive because this is one seed.

The paper's Cars-196 10-task PCDS value (80.23 +/- 0.68 Last-Acc) is a multi-run reference, not a baseline hardcoded into this comparison. This report compares the two runs produced here under the requested single-seed protocol.
