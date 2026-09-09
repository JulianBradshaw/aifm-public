# Detection elicitation

Experts' best-estimate **P(detected | n near-identical sites of this size, GPUs acquired
undetected, no silver bullet)** at the 3-year and 10-year horizons, per candidate per-site
power (MW).

Source: `detection-elicitation-full.xlsx` (tab 1), via the monorepo `apps/covert-detection-model`.
Mirrors the `detection_grids` block in [`covert_scenario_branch.yaml`](covert_scenario_branch.yaml).

The opaque gamma-frailty params (`pi`, `thetaC`, `eta`, `k`) are fit to these points per size
(Nelder–Mead on detection-space SSE; see `detection.fit_frailty_from_grids`), then interpolated
across `log10(size)`. Edit the probabilities here, not the fit params. The covert sim runs a
single facility (n=1); the larger-n rows pin the curve shape.

| Per-site power (MW) | n sites | P(detected by 3y) | P(detected by 10y) |
|--------------------:|--------:|------------------:|-------------------:|
| 10                  | 1       | 0.20              | 0.26               |
| 10                  | 3       | 0.25              | 0.30               |
| 10                  | 10      | 0.32              | 0.37               |
| 10                  | 100     | 0.55              | 0.75               |
| 100                 | 1       | 0.25              | 0.35               |
| 100                 | 3       | 0.35              | 0.40               |
| 100                 | 10      | 0.45              | 0.67               |
| 100                 | 100     | 0.65              | 0.80               |
| 1000                | 1       | 0.40              | 0.50               |
| 1000                | 3       | 0.45              | 0.70               |
| 1000                | 10      | 0.70              | 0.85               |
| 1000                | 100     | 0.85              | 0.90               |
