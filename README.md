# Groundwater Geostatistics & Inverse Modeling — pyRGA Teaching Demo

An educational GitHub-ready example covering **variograms, Gaussian random fields, kriging, co-kriging theory, groundwater hydraulic tomography, inverse modeling, and Gauss–Newton optimization**.

- [Full study guide, mathematics, worked problems and answers](STUDY_NOTES.md)
- [Executable Python demonstration](examples/pyrga_groundwater_demo.py)
- [pyRGA package on PyPI](https://pypi.org/project/pyrga-ht/)

## Setup

Use **Python 3.10+** (a fresh Python 3.11 environment is a reasonable choice).

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the end-to-end example from the repository root:

```bash
python examples/pyrga_groundwater_demo.py
```

Expected files are generated in `pyrga_demo_outputs/`:

```text
01_true_fields.png
02_variogram.png
03_kriging.png
04_observed_vs_predicted.png
results.npz
```

The script first simulates a heterogeneous conductivity field, samples a variogram, performs an *illustrative independent simple-kriging step*, solves a hydraulic tomography forward problem, and then optimizes reduced-dimensional parameters using pyRGA's published Gauss–Newton routine.

## Important implementation details

- Install **`pyrga-ht`**, but import the module as **`pyrga`**. The PyPI project description has an inconsistent install command; this repository uses the published distribution name.
- Demonstration parameters and API calls follow the PyPI v0.1.4 quick-start/optimization example.
- The original teaching demo's covariance used in kriging is an **illustrative assumption** and is not claimed to match the package's field generator exactly.
- This repo **does not implement co-kriging**; the notes derive its principles and explain a possible extension.
- The sample conductivity field and heads are **synthetic**, not field measurements; check the package's spatial array ordering, boundary conditions, and unit conventions before scientific use.
- The end-to-end solver may take a while on a laptop, particularly the inverse step. Try changing `max_iter` only after confirming the initial example runs.
- The repository code was statically checked, but the pyRGA execution has **not** been verified in this environment. Please report any package-version-specific incompatibilities in the repository you create.

## Suggested learning path

| Week | Topics | Experiment |
|---|---|---|
| 1 | Stationarity, covariance, variogram, GRF | Compare fields and sample variograms |
| 2 | Kriging, uncertainty, co-kriging theory | Change borehole locations and sample size |
| 3 | Groundwater PDE, tomography, inverse problems | Generate synthetic pumping head observations |
| 4 | Optimization, priors, regularization, RGA | Fit reduced coefficients and examine residuals |

## Attribution

The laboratory uses `pyrga-ht` (pyRGA), with calls based on its published public example. For the underlying methodology, see Zhao & Luo (2020), [*Reformulation of Bayesian Geostatistical Approach on Principal Components*](https://doi.org/10.1029/2019WR026732).

## Reuse

Educational materials authored for this demo are provided under the [MIT License](LICENSE). The separately installed `pyrga-ht` package maintains its own licensing and attribution.
