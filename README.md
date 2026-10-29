# Gaussian Random Fields & Gauss–Newton Groundwater Inversion

A focused teaching repository on **Gaussian random-field statistics, covariance structures, field generation, groundwater inverse modeling, and especially Gauss–Newton optimization**, with experiments using [pyRGA (`pyrga-ht`)](https://pypi.org/project/pyrga-ht/).

## Start here

- **[Study notes, derivations, worked examples, and answers](STUDY_NOTES.md)** — main teaching resource
- **[Standalone covariance and GRF fundamentals demo](examples/covariance_grf_fundamentals.py)** — two-coordinate covariance, matrix construction, dense Cholesky sampling and length-scale comparisons (no pyRGA needed)
- **[Introductory pyRGA visualization demo](examples/pyrga_groundwater_demo.py)** — field → variogram/kriging recap → hydraulic tomography → GN fit
- **[Optimizer benchmarking experiment](examples/optimizer_benchmark.py)** — configurable grid, retained modes, noise, initialization and iterations; writes comparable metrics

The fundamentals section now develops spatial statistics, variograms, kriging, and covariance kernels in detail, including pairwise covariance pseudocode and worked examples; these support the primary focus on GRF generation and Gauss–Newton optimization. Adjoint-state methods are mentioned only in overview.

## Setup

Requires Python ≥3.10. A fresh Python 3.11 environment is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

The package distribution is called `pyrga-ht` and is imported as `pyrga` (the install command in the PyPI description differs from its published distribution name).

## Run

```bash
python examples/covariance_grf_fundamentals.py
python examples/pyrga_groundwater_demo.py
python examples/optimizer_benchmark.py --grid 64 --modes 20 --noise 0.05 --iterations 8 --seed 17
```

Outputs:

- `fundamentals_outputs/grf_covariance.png`: standalone covariance/GRF visual experiment.

- `pyrga_demo_outputs/`: images and synthetic data from the instructional demo.
- `benchmark_outputs/summary.csv`: cumulative benchmark metrics (append one row per run).
- `benchmark_outputs/*.json` / `*.npz`: per-run settings, metrics, synthetic heads and fitted coefficients.

## Suggested class experiments

Run controlled tests with varied `--noise`, `--modes`, `--iterations`, `--grid`, and `--init-scale`. Use a common seed for matched comparisons, then replicate across seeds. See **Part IV of the notes** for suggested experimental matrix, reconstruction caveats, reporting template and critical questions.

**Accuracy:** compare fitted-head RMSE against observations **and noise-free synthetic head truth**. Head agreement is *not* sufficient evidence that log conductivity was recovered. Reconstruct `K` from optimized basis coefficients only after verifying the package basis/transform conventions.

**Scalability:** plot inversion time against grid-cell count and number of retained modes; record machine, implementation version, number of observations and timing scope. Do not assume that dense matrix solves dominate without measuring forward and Jacobian costs.

## Provenance and caveats

The code is based on the published pyRGA v0.1.4 API example. Python syntax can be validated locally, but **full inversion execution has not been validated here**. Check solver input dimensions, package version, field orientation, covariance length-scale convention and pumping/boundary-condition definitions before research use. The demonstration uses synthetic, not field data.

Reference: Zhao, Y. & Luo, J. (2020), *Reformulation of Bayesian Geostatistical Approach on Principal Components*, *Water Resources Research*, https://doi.org/10.1029/2019WR026732.
