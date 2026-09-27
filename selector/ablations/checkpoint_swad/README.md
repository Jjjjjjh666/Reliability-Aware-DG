# Checkpoint-SWAD ablation

This package implements the paper's sparse-checkpoint approximation to SWAD
`LossValley`. It replays the official valley rules over saved DomainBed
endpoints, averages the selected endpoint parameters, and recalibrates
BatchNorm statistics on source-domain training data before target evaluation.

Because the original runs save one endpoint every 100 optimizer steps rather
than dense segment averages, this must be described as **checkpoint-SWAD**, not
an exact SWAD reproduction.

Preview the 360-run OfficeHome/TerraIncognita batch:

```bash
python -m selector.ablations.checkpoint_swad.run_all \
  --full-root outputs/paper_sweep \
  --data-dir data \
  --output-root outputs/checkpoint_swad \
  --gpus 0 1 \
  --dry-run
```

Remove `--dry-run` to launch. Completed runs are validated and skipped when
the command is resumed.

After all runs finish:

```bash
python -m selector.ablations.checkpoint_swad.analyze \
  --full-root outputs/paper_sweep \
  --swad-root outputs/checkpoint_swad \
  --output-dir outputs/checkpoint_swad_analysis
```

The implementation follows the official SWAD queue settings used in the paper:
`n_converge=3`, `n_tolerance=6`, and `tolerance_ratio=0.3`.
