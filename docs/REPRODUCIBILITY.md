# Environment, data and execution

## Environments

The inspected legacy Structure environment is Python 3.9.19, torch 2.4.1, torchvision .19.1, timm .6.12, NumPy 1.23.5, SciPy 1.13.1. `environments/structure.json` records inspected versions. `environments/requirements-structure.txt` consolidates the core imports; original `cds.yaml` files are preserved but may contain obsolete/unavailable package pins. CUDA build selection is hardware-dependent; install a compatible torch/torchvision pair separately as needed.

The earlier CIFAR diagnostic package has its own newer pinned requirements and uses timm 1.0.26/NumPy 2.2.6. Do not install it into the legacy Structure environment blindly. Generator setup is separate (`setup_generator_env.sh` / `setup_bob_env.sh`, original asset locks and source notes). None of these environments or weights are committed.

## Available CPU evidence workflow

```bash
python tools/restore_tables.py
python tools/rebuild_indexes.py
python tools/validate_export.py
# In the legacy Structure environment, after table restoration:
python research_report_for_professor/build_report.py --stage all
```

The last command reconstructs figures/tables from stored evidence without training; it rewrites that report's generated artifacts in your clone. Use a clean clone or a new branch and review the diff. CPU functional checks, static compilation and report verification are detailed in [validation](VALIDATION.md). These are not full accuracy reruns.

## Training commands and explicit paths

Base Structure:

```bash
cd structure_multicenter_separation
python main.py --config exps_jsons/imageneta_5task/E0.json   --gpu 0 --data-root /path/to/imagenet-a --seed 1993   --output-root /path/to/new-run
```

E3 configuration validation and optional new training:

```bash
cd ../structure_e3_followup
python run.py --config configs/AM_drift.json --cpu --validate-only
python run.py --config configs/AM_drift.json --gpu 0   --data-root /path/to/imagenet-a --seed 1993 --output-root /path/to/new-run
```

The sibling dependency roots remain in place. Selective/RP runners have their own inherited protocol configs; inspect `--help` and the recorded queue spec before executing. Do not automatically resume a historical held grid. Create an unused output path and preserve the exact protocol/content/class-order hashes, seed, resolved config, source SHA and evaluation timing.

## External artifacts and frozen path markers

Checkpoints, old raw training images, cached feature arrays, dataset banks and diffusion/Qwen model assets are excluded. Diagnostic reruns require the originally recorded checkpoint state and split; summaries alone are insufficient. Restore external artifacts to the path stated in the record or explicitly override CLI/config paths. The original artifact locations remain in the source workspace; `provenance/excluded_files.csv` records relative origins. Dataset images are not shipped.

The default sibling data locations are `datasets/ptm_guided/imagenet-a` and `cds-main/data/cars196`; mount/link your matching local split there or pass a supported `--data-root`. Former home prefixes are represented by `${RESEARCH_WORKSPACE}` in frozen JSON/CSV. `tools/materialize_config.py` creates a separate local resolved-path copy. Never overwrite historical `resolved_config.json` to make it runnable.

BOB preflight checks `git rev-parse HEAD` and a clean official vendor working tree. The archive retains source plus safe Git history tags, but not a nested `.git`. Restore a separate exact upstream checkout with `tools/restore_bob_vendor.py`; it refuses to replace a modified source copy. Legacy generator paths/environment activation and model locks require local setup. Rewriting archived asset paths does not prove original hash parity. The copied interrupted spec/locks remain evidence.

Relocation may change file-path-based sample IDs and source-state fingerprints. Historical prediction comparisons remain valid within their original matching groups; a fresh relocated run must emit its own consistent IDs and manifests. Model/loss computations were not changed during packaging.

Full regeneration of the preserved report requires the external diagnostic CSVs listed in `provenance/external_evidence.csv` as well as tracked evidence. A fresh GitHub clone can regenerate the main numeric/run indexes from tracked files, but cannot reconstruct every original diagnostic table without transferring the sibling local-evidence directory.
