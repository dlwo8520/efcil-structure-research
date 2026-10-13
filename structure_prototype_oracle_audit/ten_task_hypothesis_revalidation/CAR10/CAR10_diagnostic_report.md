# CAR10 10-Task Hypothesis Revalidation

## First-page answers

- Final old-class tracking similarity: **0.9885**; relative L2: **0.1328**.
- Final old-class covariance relative error: **0.4034**; trace ratio: **1.1327**.
- Final Oracle D-A effect: **1.107 pp**.
- Matched R256-G256 effect: **0.895 pp**.
- Final top-20 directional error share: **0.1330**.
- Across-transition degradation/improvement mass: **1493/2397**.
- Test neighborhood purity, correct/wrong: **0.5457/0.1690**.

## Controls and scope

- Frozen task-end Structure checkpoints only; no backbone, adapter, APR, or CIL training was performed.
- Same-sample transitions use canonical sample IDs and the current-task classifier as a common readout on both feature spaces.
- Gamma is staged: final transition plus the Phase-1 transition with the largest positive degradation mass.
- Validation is split from training data before diagnostic selection; test labels never select a hyperparameter.
- This is seed 1993 only. Different task sizes/class counts confound a purely causal accumulation claim.
