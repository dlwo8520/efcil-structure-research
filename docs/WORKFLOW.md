# Adding an experiment and updating GitHub

1. Create a unique dated ID and metadata skeleton:

   ```bash
   python tools/new_experiment.py --id EFCIL-20261013-001 --name "Current-only SupCon pilot" --parent EFCIL-019
   ```

   This creates `experiments/ID/experiment.json`, a config placeholder and analysis template, plus `research_log/DATE.md` only if absent. It does not implement or run the method. Replace the proposed name/parent with the actual study and baseline.

2. Add code under `experiments/ID/code/` or a named sibling extension. Import the intended base explicitly and record the base/source commit; do not copy outputs recursively into snapshots. Link `code`, `config`, `result_path` and `analysis_document` in the experiment record. Resolve all hyperparameters and stop/success criteria before running. Use `new_runs/ID/` for local heavy artifacts; select useful metrics/manifests into `experiments/ID/results/` after review.
3. Save dataset/split/class-order hashes, task partition, seed, software versions, actual command, source SHA, evaluation timing and completion state. Report only measured metrics. Keep old/new/forgetting/separation or repair/harm when measured, with definitions/units. Leave missing fields null rather than inventing values.
4. Write the hypothesis, method difference, results, negative findings, limitations and next decision in the analysis document. Record daily observed work in `research_log/DATE.md`. Add method/experiment changes to `CHANGELOG.md`; do not infer events from file modification times.
5. Regenerate indexes and verify before a normal commit:

   ```bash
   python tools/rebuild_indexes.py
   python tools/validate_export.py
   git status --short
   git diff --check
   git add experiments research_log docs results CHANGELOG.md
   git diff --cached --stat
   git commit -m "Record EFCIL experiment and evidence"
   git push origin main
   ```

Initial private publishing/authentication is in [UPLOAD.md](UPLOAD.md). The normal workflow never uses force push. Checkpoint/dataset/cache files stay local. If changed source belongs to a preserved sibling project, stage that explicit directory too and update its metadata. Frozen historical tables remain unchanged; new evidence gets a new ID.

If you intentionally modify an already preserved source/config file, its historical export hash will no longer match. After reviewing the change, record its lineage explicitly:

```bash
python tools/record_source_update.py --path structure_e3_followup/followup/losses.py \
  --experiment EFCIL-YYYYMMDD-001 --reason "Describe the actual method change"
```

Use your registered ID. The tool retains the original hash, appends a dated source-update record, and updates only the reviewed source/config hash. It refuses frozen result metadata. Run method-specific tests and stage `provenance/file_manifest.csv` and `provenance/code_updates.jsonl` with the code and CHANGELOG. Historical packaging validation is a dated record; new scientific results require new evidence.
