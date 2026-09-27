# Reproducibility guide

## Environment

The released dependency set targets Python 3.8-3.10 and the original
DomainBed-era PyTorch stack.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r domainbed/requirements.txt
```

Run the source-only selector tests before launching experiments:

```bash
python -m unittest discover -s selector/tests
```

## Data

DomainBed expects `data/<dataset>/<domain>/<class>/<image>`. Use the upstream
download helper where its dataset license and host still permit automated
download:

```bash
python -m domainbed.scripts.download --data_dir data
```

PACS, OfficeHome, and TerraIncognita are third-party datasets and are not
redistributed in this repository.

## One trajectory

This command trains one ERM trajectory, logs source and target evaluation
metrics every 100 updates, and materializes the paper's reference AC checkpoint
as `best_model.pkl`:

```bash
python -m domainbed.scripts.train \
  --data_dir data \
  --dataset PACS \
  --algorithm ERM \
  --test_envs 0 \
  --steps 5001 \
  --checkpoint_freq 100 \
  --best_model_selection_rule ac \
  --ac_objectives NC \
  --ac_distance linf \
  --ac_delta_pp 0.5 \
  --output_dir outputs/pacs_erm_env0
```

AC saves temporary step checkpoints during the run, copies the selected one to
`best_model.pkl`, and removes the unselected step files. Add
`--save_model_every_checkpoint` only when later analyses need every weight
file; storage grows quickly.

## Main sweep

The paper uses five training algorithms, three datasets, four held-out domains,
three hyperparameter seeds, three trial seeds, 5,001 updates, and 51 logged
checkpoints per trajectory:

```bash
python -m domainbed.scripts.sweep launch \
  --data_dir data \
  --output_dir outputs/paper_sweep \
  --command_launcher multi_gpu \
  --datasets PACS OfficeHome TerraIncognita \
  --algorithms CORAL ERM GroupDRO IRM VREx \
  --single_test_envs \
  --n_hparams 3 \
  --n_trials 3 \
  --steps 5001 \
  --checkpoint_freq 100 \
  --skip_model_save \
  --skip_confirmation
```

This reproduces trajectory metrics while avoiding hundreds of gigabytes of
weights. Do not add `--skip_model_save` if deployment checkpoints or
weight-based baselines are required.

Apply Source-Acc and AC to the completed logs:

```bash
python -m selector.select_checkpoints \
  --input-root outputs/paper_sweep \
  --objective-sets ECE,NLL,CwECE,NC,NE,NEC \
  --distances l1,l2,linf \
  --delta-pp 0.5 \
  --require-done \
  --out-dir outputs/selection \
  --write-report
```

Selection operates independently inside every fixed trajectory. Target fields
are read only for the generated evaluation tables.

## Optional prediction-level diagnostics

For a small number of runs, `--save_selection_predictions` retains aligned
target predictions for Source-Acc and the configured AC rule. These files are
for post-selection diagnostics and may include local sample paths; do not
publish them without reviewing their contents.

## Ablation studies

- `selector/ablations/checkpoint_swad/` contains the sparse checkpoint-SWAD comparison. It
  requires saved step weights.
- `selector/ablations/lodo_experiment/` contains the additional-training LODO validation
  comparison.

Each directory has a dedicated README and a dry-run command. These cohorts use
their own validation and metric protocols and must not be pooled with the main
540-trajectory table.
