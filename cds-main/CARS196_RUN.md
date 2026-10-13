# Cars-196 PCDS vs PCDS+SGDS

The training code expects the official Stanford Cars train/test split in the
same ImageFolder layout used by the released CDS loader:

```text
/data/datasets/cars196/
├── train/
│   ├── <class-000>/...
│   └── ... 196 class directories
└── test/
    ├── <class-000>/...
    └── ... the same 196 class directories
```

The loader rejects missing classes or different train/test class mappings.
The server did not contain this dataset when the experiment code was prepared.

Run both methods sequentially on physical GPU 3:

```bash
cd ${RESEARCH_WORKSPACE}/cds-main
bash scripts/run_cars196_pcds_sgds_gpu3.sh
```

For a detached run:

```bash
cd ${RESEARCH_WORKSPACE}/cds-main
mkdir -p results/cars196_pcds_comparison
nohup bash scripts/run_cars196_pcds_sgds_gpu3.sh \
  > results/cars196_pcds_comparison/launcher.log 2>&1 &
echo $!
```

Monitor progress:

```bash
tail -f ${RESEARCH_WORKSPACE}/cds-main/results/cars196_pcds_comparison/launcher.log
```

Each task writes a resumable checkpoint. Re-running the same launcher resumes
an incomplete method from its latest completed task. After both methods finish,
the analyzer appends `## Cars-196: PCDS vs PCDS + SGDS` to `RESULTS_PCMD.md`.
