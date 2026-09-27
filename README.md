# Reliability-Aware Checkpoint Selection for Domain Generalization

This repository implements **Reliability-Aware Checkpoint Selection for Domain
Generalization**. It extends
[DomainBed](https://github.com/facebookresearch/DomainBed) with source-domain
probability-quality metrics and accuracy-constrained reliability selection
(AC).

AC addresses a deployment decision that remains after a domain-generalization
model has been trained: which saved checkpoint should be used when no target
validation data are available? It keeps checkpoints with near-best
source-validation accuracy and ranks only those candidates by source-side
reliability. It returns an existing checkpoint without target data, additional
training, weight averaging, ensembling, or post-hoc calibration.

## Reference rule

The paper's reference configuration, **AC-NC**, is:

1. retain checkpoints within `0.5` percentage points of the best mean
   source-validation accuracy;
2. min-max normalize source-validation NLL and class-wise ECE (CwECE) inside
   that feasible set;
3. minimize the `L-infinity` norm of the normalized errors;
4. break ties by higher source-validation accuracy, then earlier step.

The code also supports ECE-, NLL-, and CwECE-only ranking, the `NE` and
`NEC` objective sets, and `L1`/`L2` aggregation. See
[docs/METHOD.md](docs/METHOD.md) for the exact rule and metric definitions.

## Main result

The post-development evaluation contains 360 OfficeHome and TerraIncognita
trajectories across CORAL, ERM, GroupDRO, IRM, and VREx. Relative to
Source-Acc, AC-NC (`L-infinity`, `delta=0.5 pp`) changed mean target
accuracy by `+0.2133 pp`, ECE by `-0.2395`, CwECE by `-0.1821`, and NLL
by `-0.0295`. ECE and CwECE are reported after multiplication by 100.

| Dataset | Selector | Target accuracy (%) | ECE x100 | CwECE x100 | NLL |
| --- | --- | ---: | ---: | ---: | ---: |
| OfficeHome | Source-Acc | 60.87 | 2.77 | 3.32 | 4.2724 |
| OfficeHome | AC-NC | 61.06 | 2.48 | 3.05 | 4.2362 |
| TerraIncognita | Source-Acc | 42.74 | 9.10 | 11.35 | 2.3438 |
| TerraIncognita | AC-NC | 42.97 | 8.90 | 11.25 | 2.3210 |

These results support improved mean probability quality in the evaluated
trajectories. They do not establish target-accuracy preservation or universal
superiority over every single-objective ranking rule.

## Repository layout

```text
.
├── domainbed/          DomainBed training code and calibration logging
├── selector/           source-only Source-Acc and AC checkpoint selection
│   └── ablations/      checkpoint-SWAD and LODO ablation studies
└── docs/               method and reproducibility documentation
```

Generated datasets, checkpoints, predictions, and analysis tables belong under
`data/` or `outputs/`; both are ignored by Git.

## Installation

The released environment targets Python 3.8-3.10.

```bash
git clone https://github.com/Jjjjjjh666/ECE_DomainBed.git
cd ECE_DomainBed
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r domainbed/requirements.txt
```

Dataset licenses do not permit redistribution here. Prepare PACS, OfficeHome,
and TerraIncognita using the original DomainBed directory layout.

## Quick start

Train one trajectory and materialize the reference AC checkpoint:

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

The selected weights are written to `best_model.pkl`, with the decision
recorded in `best_model_selection.json`.

To reselect from existing DomainBed trajectories without retraining:

```bash
python -m selector.select_checkpoints \
  --input-root outputs/paper_sweep \
  --objective-sets NC \
  --distances linf \
  --delta-pp 0.5 \
  --require-done \
  --write-report
```

Outputs are written to `outputs/selection/`. See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the full 540-trajectory
design, ablation studies, and prediction diagnostics.

## Tests

```bash
python -m unittest discover -s selector/tests
python -m unittest selector.ablations.checkpoint_swad.test_selection
```

The original DomainBed test suite remains available through
`python -m unittest discover`.

## License and upstream attribution

The code is released under the [MIT License](LICENSE). The `domainbed/`
package is derived from Facebook Research's DomainBed; its original copyright
notices are retained. Please also cite DomainBed when using this repository.
