# Source-only checkpoint selection

This package applies Source-Acc and accuracy-constrained reliability selection
(AC) to existing DomainBed `results.jsonl` trajectories. It does not retrain
models. Target-domain metrics may be present for evaluation, but they are never
used to choose a checkpoint.

## Reference configuration

The paper's reference rule is `AC-NC` with `delta=0.5` percentage points and
`D_infinity` aggregation:

1. Keep checkpoints whose mean source-validation accuracy is within 0.5
   percentage points of the best checkpoint in the same trajectory.
2. Min-max normalize source-validation NLL and CwECE inside that feasible set.
3. Minimize the maximum normalized error.
4. Break ties by higher source-validation accuracy, then earlier step.

The implementation also supports the single-objective sets `ECE`, `NLL`, and
`CwECE`; the joint sets `NE` and `NEC`; and `l1`, `l2`, and `linf`
aggregation.

## Run

```bash
python -m selector.select_checkpoints \
  --input-root outputs/paper_sweep \
  --objective-sets NC \
  --distances linf \
  --delta-pp 0.5 \
  --require-done \
  --write-report
```

Repeat `--input-root` to combine sweep roots. If two roots contain variants of
the same DomainBed algorithm, use `LABEL=PATH` to keep their trajectory keys
distinct:

```bash
python -m selector.select_checkpoints \
  --input-root ERM-control=outputs/erm_control \
  --input-root ERM-variant=outputs/erm_variant
```

Generated files are written to `outputs/selection/` by default:

- `checkpoint_metrics.csv`
- `selected_checkpoints.csv`
- `summary_by_dataset_method.csv`
- `pairwise_vs_source_acc.csv`
- `oracle_regret.csv`
- `report.md` when `--write-report` is set

Run the selector tests with:

```bash
python -m unittest discover -s selector/tests
```
