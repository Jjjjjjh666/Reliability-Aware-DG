# Leave-one-domain-out validation ablation

This package reproduces the paper's LODO comparison. Inner models exclude an
unordered pair of domains and are scored on the two excluded domains. For an
outer target domain `T`, only the other excluded domain is used for
checkpoint selection; the target is evaluated after the selection is fixed.

LODO requires additional training and is therefore separate from ordinary AC.
The hard-metric path uses deterministic transforms, 15-bin ECE, and
class-frequency-weighted CwECE. Do not mix those values with the main soft-bin
metrics.

Generate and validate an inner-run manifest:

```bash
python -m selector.ablations.lodo_experiment.pipeline plan \
  --full-root outputs/paper_sweep \
  --inner-root outputs/lodo_inner \
  --output outputs/lodo_jobs.jsonl \
  --protocol-output outputs/lodo_protocol.json

python -m selector.ablations.lodo_experiment.run_jobs \
  --manifest outputs/lodo_jobs.jsonl \
  --data-dir data \
  --gpus 0 1 \
  --dry-run
```

Remove `--dry-run` to launch. The runner writes to a partial directory and
renames it atomically only after all expected metric rows are present, so the
same command can safely resume incomplete batches.

Select checkpoints from completed inner runs:

```bash
python -m selector.ablations.lodo_experiment.pipeline select \
  --full-root outputs/paper_sweep \
  --inner-root outputs/lodo_inner \
  --output outputs/lodo_selected.csv
```

See `python -m selector.ablations.lodo_experiment.pipeline --help` for dataset, algorithm, seed,
and metric-profile filters.
