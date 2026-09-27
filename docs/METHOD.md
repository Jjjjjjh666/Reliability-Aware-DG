# Accuracy-constrained reliability selection

AC selects one existing checkpoint from a fixed training trajectory using only
held-out source-domain validation data.

Let `A(theta)` be the mean source-validation accuracy across source domains.
For a tolerance `delta` in percentage points, AC first constructs

```text
Theta_delta = {theta : A(theta) >= max(A) - delta}.
```

The code stores accuracy on a `[0, 1]` scale, so a command-line value of
`--ac_delta_pp 0.5` is converted internally to `0.005`.

For each reliability error `m` in the configured objective set, values are
min-max normalized inside `Theta_delta`:

```text
m_tilde(theta) = (m(theta) - min(m)) / (max(m) - min(m) + 1e-12).
```

A constant objective maps to zero. AC then minimizes an `L1`, `L2`, or
`L-infinity` norm over the normalized reliability vector. Ties are resolved
by higher source accuracy and then by the earlier training step.

## Objective sets

| Name | Source-validation objectives |
| --- | --- |
| `ECE` | ECE |
| `NLL` | negative log-likelihood |
| `CwECE` | class-wise ECE |
| `NC` | NLL and CwECE |
| `NE` | NLL and ECE |
| `NEC` | NLL, ECE, and CwECE |

The paper's reference rule is `NC + L-infinity + 0.5 pp`.

## Metric estimators

The main trajectory logs use Gaussian soft binning with squared calibration
gaps. ECE uses top-label confidence. CwECE computes a one-vs-rest calibration
error for each class and averages classes uniformly. NLL is the ordinary
cross-entropy of the predictive distribution. Source domains are averaged
equally rather than by sample count.

## What AC does not do

- It does not use target-domain observations for selection.
- It does not retrain the model, average weights, ensemble predictions, or fit
  a post-hoc calibrator.
- The tolerance controls empirical source-validation accuracy; it is not a
  guarantee on unseen target accuracy.
- A source-only selector cannot guarantee target-optimal selection for
  unrestricted target distributions.
