"""Standalone fundamentals: pairwise covariance, matrix construction and GRF simulation.

Run: python examples/covariance_grf_fundamentals.py
No pyRGA required. Matplotlib output: fundamentals_outputs/grf_covariance.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def gaussian_covariance(p, q, variance, ax, ay):
    """Anisotropic, axis-aligned squared-exponential covariance at two 2D points."""
    if variance < 0 or ax <= 0 or ay <= 0:
        raise ValueError('variance must be >= 0; correlation lengths must be > 0')
    dx, dy = np.asarray(p, dtype=float) - np.asarray(q, dtype=float)
    return float(variance * np.exp(-((dx / ax) ** 2 + (dy / ay) ** 2)))


def covariance_matrix(coordinates, variance, ax, ay):
    """Vectorized equivalent of the nested pairwise pseudocode in STUDY_NOTES.md."""
    points = np.asarray(coordinates, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('coordinates must have shape (n_points, 2)')
    if variance < 0 or ax <= 0 or ay <= 0:
        raise ValueError('invalid structural parameters')
    delta = points[:, None, :] - points[None, :, :]
    d2 = (delta[..., 0] / ax) ** 2 + (delta[..., 1] / ay) ** 2
    return variance * np.exp(-d2)


def simulate_grf(coordinates, mean, variance, ax, ay, seed=2026):
    C = covariance_matrix(coordinates, variance, ax, ay)
    # Jitter is a tiny numerical regularization, not part of the exact kernel.
    jitter = max(variance, 1.0) * 1e-10
    L = np.linalg.cholesky(C + jitter * np.eye(len(C)))
    rng = np.random.default_rng(seed)
    logK = mean + L @ rng.standard_normal(len(C))
    return logK, np.exp(logK), C


def main():
    p, q = (10.0, 20.0), (30.0, 30.0)
    value = gaussian_covariance(p, q, variance=1.5, ax=40, ay=10)
    print(f'Pairwise covariance C(p,q) = {value:.6f} (expected ~0.429757)')
    assert np.isclose(value, 1.5 * np.exp(-1.25))
    coords2 = np.array([p, q])
    C2 = covariance_matrix(coords2, variance=1.5, ax=40, ay=10)
    assert np.isclose(C2[0, 1], value)
    assert np.allclose(C2, C2.T)
    assert np.linalg.eigvalsh(C2).min() >= -1e-10
    print('Two-point covariance matrix:\n', C2)

    # 18 x 18 cells are small enough for a fast, dense Cholesky demonstration.
    nx = ny = 18
    x = np.linspace(0, 100, nx)
    y = np.linspace(0, 100, ny)
    xx, yy = np.meshgrid(x, y, indexing='xy')
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    mean = np.log(1e-4)  # log of conductivity expressed numerically in m/s
    cases = [(15., 15., 1.), (50., 10., 1.), (50., 10., 0.25)]

    fig, axes = plt.subplots(1, len(cases), figsize=(12, 4), constrained_layout=True)
    for ax, (lx, ly, variance) in zip(axes, cases):
        logK, _, C = simulate_grf(coords, mean, variance, lx, ly, seed=17)
        assert np.allclose(np.diag(C), variance)
        img = ax.imshow(logK.reshape(ny, nx), origin='lower',
                        extent=(x.min(), x.max(), y.min(), y.max()),
                        vmin=mean - 2.5, vmax=mean + 2.5, cmap='viridis')
        ax.set(title=f'ax={lx:g}m, ay={ly:g}m, var={variance:g}', xlabel='x (m)', ylabel='y (m)')
    fig.colorbar(img, ax=list(axes), label='ln(K in m/s)', shrink=0.78)
    out = Path('fundamentals_outputs')
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'grf_covariance.png', dpi=160)
    plt.close(fig)
    print(f'Saved {out / "grf_covariance.png"}')


if __name__ == '__main__':
    main()
