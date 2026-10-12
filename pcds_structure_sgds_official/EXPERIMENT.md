# Official Structure (PCDS) + SGDS Activation Guidance

This directory is a clean clone of the PCDS/Structure authors' repository at commit `a0db2168663982a4cad24981231721a3518ddf29`. It replaces the previous Cars experiment that imported a separate reimplementation from `efcil_direction_diagnostic`.

The two Cars-196 methods here share the official `models/CDS_adapter.py` learner and all PCDS components: ViT-B/16 ImageNet-21K, alternating attention-adapter banks, SCC, APR, cosine classifier, and Gaussian classifier adaptation. The SGDS condition adds only Eq. (4)-(10) activation guidance at the final block's adapter input.

Not included from full SGDS: task adapter bank/retrieval, entropy selection, universal fusion, its prototype classifier, MOS alignment, or SGDS's own task protocol.

## Protocol

- Dataset: prepared official Cars-196 train/test split at `${RESEARCH_WORKSPACE}/cds-main/data/cars196`
- Images: 8,144 train / 8,041 test
- Tasks: `20 x 9 + 16 = 196` classes
- Training seed: 0
- Shared official CDS class-order seed: 1993
- Task-ID at inference: no
- Raw exemplar replay: no
- Official Cars epochs: 70 for Task 0, 40 for Tasks 1-9
- SCC weight: 5.0; APR gamma/rho: 10.0; CA: 5 epochs
- SGDS: final block only, beta 0.5, gamma 1.0, first-half exploration
- `k=0.6`: Top-K retain ratio, so at most 60% remain after Bernoulli and Top-K
- SGDS mask RNG seed: 100003, isolated so it does not shift baseline data-order/dropout RNG

## Run on physical GPU 4

```bash
cd ${RESEARCH_WORKSPACE}/pcds_structure_sgds_official
bash scripts/run_cars196_structure_sgds_gpu4.sh
```

`CUDA_VISIBLE_DEVICES=4` is set inside the script; `cuda:0` inside PyTorch therefore means physical GPU 4. The two methods run sequentially, completed methods are skipped safely, and analysis runs only after both have a completion marker.

Progress can be watched with:

```bash
tail -f ${RESEARCH_WORKSPACE}/pcds_structure_sgds_official/logs/cars196_structure_pcds_seed0.log
```

or, during the second method:

```bash
tail -f ${RESEARCH_WORKSPACE}/pcds_structure_sgds_official/logs/cars196_structure_pcds_sgds_seed0.log
```

Final comparison is written to `RESULTS_STRUCTURE_SGDS.md`. A completion marker is written only after all 10 tasks for a method finish.
