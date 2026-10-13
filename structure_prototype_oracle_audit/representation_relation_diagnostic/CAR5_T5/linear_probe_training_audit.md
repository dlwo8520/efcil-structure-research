# Linear Probe Training Audit

## 기존 procedure diagnostic 설정

- 구현: `classifier_procedure/pipeline.py::_train_linear()`
- objective: inverse-class-frequency weighted mean CE + `0.5 * lambda * sum(W^2)`
- class weighting: sample weight `N / (C * n_y)`
- regularization: L2, lambda `5e-4`
- solver: PyTorch full-batch L-BFGS, strong-Wolfe line search
- max_iter/max_eval: 100/200
- tolerance_grad/tolerance_change: `1e-5` / `1e-9`
- initialization: raw pre-CA cosine weight
- bias: condition에 따라 없음/있음
- feature: L2-normalized frozen `sfeatures`
- split: replacement 이전 original image 단위 deterministic stratified split
- sklearn `C`, `n_iter_`, convergence warning: 해당 없음(PyTorch LBFGS 사용)

## 이번 sanity 설정

- lambda: `5e-4`, `5e-5`, `5e-6` 세 개만 사용
- max_iter/max_eval: 200/400
- tolerance_grad/tolerance_change: `1e-7` / `1e-10`
- initialization: exact converted Structure cosine solution
- 선택: validation macro accuracy, micro accuracy, stronger L2 순
- 선택된 lambda: `5e-06`
- test는 regularization 선택에 사용하지 않음

기존 lambda의 penalty는 scale-20 converted cosine weight에 대해 클 수 있으므로, 이전 linear 저성능을 representation 근거로 사용하기 전에 이번 sanity 결과를 확인한다.
