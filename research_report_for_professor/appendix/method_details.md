# 수식·실험 구현 보충

본문의 핵심 비교를 보충한다. 아래는 기존 구현을 설명하며 새 method를 구현한 문서가 아니다. 본문에서 쓰는 mean/prototype의 위치, 후보 집합, normalization 순서는 코드 정의를 따른다.

## 1. 초기 multi-center / separation

Single class의 region을 하나 또는 K-means component로 저장한다. Multi-center CA의 source는

$$
z_c\sim\sum_{r=1}^{K_c}\pi_{cr}\mathcal N(\beta\mu_{cr},\Sigma_{cr}),\qquad \pi_{cr}=N_{cr}/N_c.
$$

총 replay는 class당 256으로 유지하며 component당 256이 아니다. Isotropic/diagonal/full 거리에서 birth train-distance quantile을 threshold로 저장하고 dimensionless max-intrusion을 사용한다. Full-covariance E4/E7은 실제 loss/intrusion/gradient가 0이므로 active intervention 실패의 근거에서 제외한다.

고정: protocol·class order·학습량. 변경: component representation 또는 separation 거리. 질문: richer distribution과 old-region 회피가 이득을 주는가?

근거: [distribution_regions.py](../../structure_multicenter_separation/utils/distribution_regions.py)의 `build_class_regions`, `SeparationCache`; [raw metrics](../../structure_multicenter_separation/reports/e0_comparison_20260916/tables/metrics_per_seed.csv). 당시 소스 snapshot 부족과 full covariance의 factor device 차이가 있어 완전한 구현 동등성까지 주장하지 않는다.

## 2. Similarity relaxation, relative routing, SCC 완화

Frozen old-model 공간에서 new mean과 old mean의 cosine을 $S_{yk}$라 하자. AM-Drift의 sample penalty를 $A_i=[1-\min_kD_{ik}^2/\tau_k]_+$로 쓴다.

S1은 similarity가 클수록 threshold를 축소한다. Selective B의 threshold **확대**와 방향이 다르다.

$$
\tau_{yk}^{S1}=\tau_k\max\left(1-\alpha\operatorname{clip}((S_{yk}+1)/2,0,1),\epsilon\right).
$$

Relative route는 current sample의 가장 높은 cosine old가 아니라 **class-level frozen similarity**에서 $k_y^*=\operatorname{argmax}_k S_{yk}$를 선택한다.

$$
R_i=[m-\cos(z_i,\mu_{y_i}^{\mathrm{current}})+\cos(z_i,\mu_{k_{y_i}^*}^{\mathrm{APR}})]_+.
$$

S2/S3는 $h_y=\mathbf1[S_{y,k_y^*}\ge\beta]$로

$$
L_{S2}=\frac1B\sum_i[(1-h_{y_i})A_i+h_{y_i}R_i].
$$

S4는 $g_y=\operatorname{sigmoid}((S_{y,k_y^*}-\beta)/T_g)$로 같은 두 항을 부드럽게 혼합한다.

$$
L_{S4}=\frac1B\sum_i[(1-g_{y_i})A_i+g_{y_i}R_i].
$$

S3는 relative routing에 sample별 SCC 완화도 추가한다. $E_i=\sum_k(r_{ik}^{\mathrm{new}}-r_{ik}^{\mathrm{old}})^2$라 하면 다음과 같다.

$$
u_i=\max\left(1-\alpha\operatorname{clip}((S_{y_i,k_{y_i}^*}+1)/2,0,1),u_{\min}\right),\qquad L_{\mathrm{SCC},S3}=\frac1B\sum_i u_iE_i.
$$

S3를 separation-only 비교로 세지 않는다. S2의 threshold는 전체 pair quantile에서 설정되어 max-similarity routing이 약 99.91% relative로 치우쳤다. 의도한 선택성과 실제 route가 다를 수 있다는 구현상 관찰이다.

고정: old-model similarity 및 기존 SCC/APR/CA(S3의 SCC 변경 제외). 변경: threshold/routing 및 S3의 SCC weight. 질문: 비슷한 class를 덜 밀어내면 plasticity 비용을 줄이는가?

근거: [similarity/losses.py](../../structure_similarity_aware_followup/similarity/losses.py)의 `similarity_tau_relation`, `_relative_samples`, `hard_relative_relation`, `soft_hybrid_relation`, `similarity_weighted_scc`.

## 3. Pair off/weak control

동일 수의 high/low/random pair를 off하거나 high pair만 0.3으로 줄인다. 선택 weight를 $u_{yk}\in[0,1]$로 쓰면

$$
L_{\mathrm{control}}=\frac1B\sum_i\max_k\left(u_{y_i k}[1-D_{ik}^2/\tau_k]_+\right).
$$

Max reduction이며 selective A/B의 selected-pair 평균과 같지 않다. High-Off/Low-Off/Random-Off는 각 4,000 pair를 끄고 High-Weak는 같은 high 4,000 pair를 0.3으로 가중한다.

고정: off pair 수·budget·AM-Drift baseline. 변경: off 위치/weight만. 질문: constraint 수가 아니라 similarity 위치가 문제인가?

근거: [control/losses.py](../../structure_similarity_control_experiments/control/losses.py)의 `selective_control_relation`.

## 4. FeCAM 이식의 정확한 변환

Input covariance를 symmetrize하여 $\Sigma$라 하자. Diagonal 평균을 $\bar d$, 0이 아닌 off-diagonal 평균을 $\bar o$로 계산하고, 없으면 0이다. 구현의 shrinkage/correlation은

$$
\Sigma'=\Sigma+10\bar dI+10\bar o(\mathbf1\mathbf1^\top-I),\qquad C_{ab}=\Sigma'_{ab}/\sqrt{\Sigma'_{aa}\Sigma'_{bb}}.
$$

Query/mean을 L2 정규화하고 Mahalanobis quadratic score를 쓴다. Positive definite면 Cholesky solve, 그렇지 않으면 pseudo-inverse fallback이다. Tukey transform은 Off다.

고정: final feature와 평가 후보. 변경: score/readout 및 stored/oracle statistics. 질문: covariance-sensitive readout으로 head/source 문제가 복구되는가?

Normalized-sample covariance를 저장하지 않았기 때문에 이전 raw mean/covariance만으로 official normalized-feature moments를 복원한 것처럼 설명하지 않는다. F2/F3는 동일 raw-space moment 정의를 사용한다.

근거: [fecam/run.py](../../fecam_boundary_followup/fecam/run.py)의 `_fecam_covariance`, `_mahalanobis_scores`.

## 5. 서로 다른 CEOS 구현을 혼동하지 않기

초기 E3 follow-up CEOS는 old prototype mean을 시작점으로 nearest current enemy와 보간했다. Later matched CEOS-only는 old Gaussian **sample**을 시작점으로 쓴다. 두 경로 모두 L2 정규화하지만 동일 sampling intervention이 아니다.

고정: 각 실험 내 class당 replay budget. 변경: old replay source. 질문: boundary-aware source가 old/new 경쟁을 개선하는가?

근거: [초기 sample_ceos](../../structure_e3_followup/followup/ceos.py), [later ceos_expand](../../fecam_boundary_followup/ceos/run.py). 초기 CEOS50/100의 New 증가를 later CEOS-only의 New 감소와 모순된 동일 실험으로 부르지 않는다.

## 6. 결과의 비교 규칙

- Oracle A/B/C/D는 re-generated-epoch CA, matched G256/R256은 fixed source bank일 수 있다. 공통의 absolute head baseline으로 합치지 않는다.
- Pretrained/final probe는 representation contrast이며 정식 EF-CIL method 성능이 아니다.
- A와 birth-correct support는 같은 class를 match하고 class별 차이를 macro 평균한다. All-query micro support와 같지 않다.
- Birth 시점 후보 제한 $C_{\le b}$, birth task block $G_b$, latest T9 local $G_9$를 구분한다.
- T9 acquisition 생성 개입은 vanilla T8에서 시작했다. T1 생성 개입을 이어받지 않았고 GM/G가 같은 표현이라는 보장은 statistics fork 시점에 한정된다.
- Original task summary와 post-after_task task-end inference의 불일치를 새로운 수치로 덮어쓰지 않는다.

세부값의 정확한 파일/조건/단위는 [전체 수치](full_numeric_tables.md)와 [source map](source_map.md)에 있다.

## 7. 초기 PIC / angular hard negative의 변경점

PIC-inspired 구현의 normalized query는 $q_i$이고 candidate는 detached same-batch key, correct current-class epoch prototype, old APR prototype이다. Self key는 제외하고 같은 label의 batch key와 current prototype을 positive $P_i$로, 나머지를 negative $N_i$로 둔다. Old prototype은 negative label -1이다.

$$
L_{\mathrm{PIC}}=\frac1B\sum_i\frac1{|P_i|}\sum_{p\in P_i}\log\left(1+\sum_{n\in N_i}\exp((q_i^\top k_n-q_i^\top k_p)/T)\right).
$$

Prototype을 guaranteed positive로 쓰는 one-view adaptation이며 original two-view FCS 전체 재현은 아니다. 같은 label의 current prototypes가 candidate 목록에서 반복되면 그 multiplicity도 구현에 포함된다.

Angular hard negative(AHN)는 sample cosine에서 top-K old APR prototype을 골라 current-class epoch prototype보다 margin 낮도록 한다.

$$
L_{\mathrm{AHN}}=\frac1{BK}\sum_i\sum_{k\in\operatorname{TopK}_{\mathrm{old}}(q_i)}[m-\cos(z_i,\mu_{y_i}^{\mathrm{current}})+\cos(z_i,\mu_k^{\mathrm{APR}})]_+.
$$

고정: current sample 및 기존 SCC/APR/CA. 변경: PIC 또는 E3+AHN의 별도 loss 항. 질문: point-prototype contrastive/angular repulsion으로 plasticity/stability를 함께 개선할 수 있는가?

근거: [followup/losses.py](../../structure_e3_followup/followup/losses.py)의 `pic_loss`, `angular_hard_negative_loss`; run별 계수는 원 config를 따른다. AHN은 LUCIR의 old-sample→new-negative 방향과 반대로 current-new→old-prototype이므로 동일 loss의 충실한 재현으로 부르지 않는다.
