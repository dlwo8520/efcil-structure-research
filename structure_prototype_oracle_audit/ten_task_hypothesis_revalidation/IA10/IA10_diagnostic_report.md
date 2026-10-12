# IA10 10-Task Hypothesis Revalidation

## First-page answers

- Final old-class tracking similarity: **0.9927**; relative L2: **0.1100**.
- Final old-class covariance relative error: **0.3529**; trace ratio: **1.1538**.
- Final Oracle D-A effect: **-0.066 pp**.
- Matched R256-G256 effect: **0.066 pp**.
- Final top-20 directional error share: **0.0840**.
- Across-transition degradation/improvement mass: **931/1214**.
- Test neighborhood purity, correct/wrong: **0.6208/0.1190**.

## Controls and scope

- Frozen task-end Structure checkpoints only; no backbone, adapter, APR, or CIL training was performed.
- Same-sample transitions use canonical sample IDs and the current-task classifier as a common readout on both feature spaces.
- Gamma is staged: final transition plus the Phase-1 transition with the largest positive degradation mass.
- Validation is split from training data before diagnostic selection; test labels never select a hyperparameter.
- This is seed 1993 only. Different task sizes/class counts confound a purely causal accumulation claim.
