# 실행 환경 및 번들 범위

## Environment

```text
OS timezone: Asia/Seoul
Conda env: ${RESEARCH_WORKSPACE}/miniconda3/envs/cds
Python: 3.9.19
PyTorch: 2.4.1+cu121
torchvision: 0.19.1+cu121
NumPy: 1.23.5
matplotlib: 3.9.2
timm: 0.6.12
Pillow: 10.4.0
Training GPU: physical GPU 1, NVIDIA RTX A6000
Seed: 1993
```

## Dataset paths used

```text
IA5:  ${RESEARCH_WORKSPACE}/datasets/ptm_guided/imagenet-a
CAR5: ${RESEARCH_WORKSPACE}/cds-main/data/cars196
```

dataset 자체는 이 zip에 포함하지 않았다.

## 제외된 대용량 파일

- `training_outputs/**/checkpoint_latest.pt`
- `training_outputs/**/task_checkpoints/task_*.pt`
- dataset 이미지
- cached feature NPZ (`outputs/**/features/task_*.npz`)
- Python `__pycache__`
- 다른 연구 프로젝트의 output/checkpoint

checkpoint 경로, 크기, mtime, edge SHA-256은 각 `audit_manifest.json`에 남아 있다. task 순서와 state ID는 `training_metadata/*/task_checkpoints/manifest.json`에 남아 있다.

## 핵심 source SHA-256

```text
a46bbed0e66054ba33be0c0f3feed4b36f8e1f771502c26878f8c75d43935223  run_audit.py
7886896514be1b461501d0eae0e4178c9dc76d9bae8816c7d2ecfdabb848170a  oracle_audit/pipeline.py
e510ed387ddd47629e8d694dc5c0ffc4a561ce6934462ca73532aeb3d287c4d1  oracle_audit/oracle_head.py
```
