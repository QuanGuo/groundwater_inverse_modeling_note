"""Teaching demo for pyRGA (pyrga-ht 0.1.4).

Steps: simulate log conductivity; inspect a sample variogram; simple kriging
with an illustrative covariance; generate pumping-test heads; invert with RGA.

Install: python -m pip install pyrga-ht numpy scipy matplotlib
Run: python pyrga_groundwater_demo.py

The code uses the API from the published PyPI example. The package itself was
not available to execute in the authoring environment; users should verify
shape/unit conventions against their installed release.
"""
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
from scipy.linalg import cho_factor, cho_solve

from pyrga import hydraulic_tomography
from pyrga.RGA import (
    generate_synthetic_field, prepare_physical_domain,
    observation_operator, forward_model, gauss_newton_dynamic_lambda, add_noise,
)

OUT = Path('pyrga_demo_outputs')
OUT.mkdir(exist_ok=True)
np.random.seed(17)

# The dimensions and API arguments follow the published PyPI example.
nx = ny = 64
n_modes = 50
mu = -4.0
Lx = Ly = 320.0

print('Step 1: generating a 2D random conductivity field...')
K, alpha, V, beta = generate_synthetic_field(
    nx, ny, k=n_modes, mu=mu, NR=400, cov_type='gaussian',
    variance=1.0, lx=0.15, ly=0.2,
)
K = np.asarray(K).reshape(-1)
if len(K) != nx*ny:
    raise ValueError(f'Expected {nx*ny} cells, got {len(K)}')
if np.any(K <= 0):
    raise ValueError('Conductivity must be positive')
logK = np.log(K)

# Flattening/plot orientation may vary by model convention; the 2-D image
# below is for teaching and is not assigned physical x/y axes.
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
a = axes[0].imshow(logK.reshape(ny, nx), origin='lower', cmap='viridis')
axes[0].set_title('True log conductivity')
plt.colorbar(a, ax=axes[0])
b = axes[1].imshow(K.reshape(ny, nx), origin='lower', cmap='viridis')
axes[1].set_title('True conductivity')
plt.colorbar(b, ax=axes[1])
fig.tight_layout()
fig.savefig(OUT/'01_true_fields.png', dpi=160)
plt.close(fig)

# Tutorial variogram: randomly sample cell pairs to avoid O(N^2) at 4096 cells.
# This empirical variogram comes from one simulated field, not an ensemble fit.
print('Step 2: empirical variogram...')
xx, yy = np.meshgrid(np.arange(nx)*Lx/nx, np.arange(ny)*Ly/ny)
xy = np.column_stack([xx.ravel(), yy.ravel()])
idx = np.random.choice(nx*ny, size=450, replace=False)
sep = pdist(xy[idx])
sqdiff = pdist(logK[idx, None], metric='sqeuclidean')
edges = np.linspace(0, 130, 18)
centers = 0.5*(edges[:-1]+edges[1:])
gamma = np.full(len(centers), np.nan)
for i in range(len(centers)):
    mask = (sep >= edges[i]) & (sep < edges[i+1])
    if np.any(mask):
        gamma[i] = 0.5*np.mean(sqdiff[mask])
plt.figure(figsize=(7, 4))
plt.plot(centers, gamma, 'o-')
plt.xlabel('Separation (m)')
plt.ylabel('Empirical semivariance of log K')
plt.title('Sample variogram (isotropic binning)')
plt.tight_layout()
plt.savefig(OUT/'02_variogram.png', dpi=160)
plt.close()

# Teaching-only simple kriging from hypothetical *direct* borehole logK data.
# Gaussian covariance chosen here is illustrative and not asserted to match
# the internal random-field construction of pyRGA.
print('Step 3: simple kriging from synthetic direct measurements...')
wells = np.random.choice(nx*ny, size=35, replace=False)
xy_w = xy[wells]
z_w = logK[wells]
length_m = 45.0
sigma2 = float(np.var(logK))
def cov(a, b):
    d2 = cdist(a, b, 'sqeuclidean')
    return sigma2*np.exp(-d2/(length_m**2))
Cww = cov(xy_w, xy_w) + 1e-6*np.eye(len(wells))
Cxw = cov(xy, xy_w)
fac = cho_factor(Cww, lower=True)
krig = mu + Cxw@cho_solve(fac, z_w-mu)
variance = np.maximum(sigma2-np.einsum('ij,ij->i', Cxw, cho_solve(fac, Cxw.T).T), 0)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
a=axes[0].imshow(krig.reshape(ny,nx), origin='lower', cmap='viridis')
axes[0].set_title('Simple kriging mean (log K)')
plt.colorbar(a, ax=axes[0])
b=axes[1].imshow(np.sqrt(variance).reshape(ny,nx), origin='lower', cmap='magma')
axes[1].set_title('Conditional std (assumed covariance)')
plt.colorbar(b, ax=axes[1])
fig.tight_layout()
fig.savefig(OUT/'03_kriging.png', dpi=160)
plt.close(fig)

print('Step 4: preparing well configuration and simulating tomography...')
well_nodes, Q, dx, dy = prepare_physical_domain(
    nx, ny, Lox=Lx, Loy=Ly, q_original=-0.02,
    well_relative_locs=None,
)
t0 = time.perf_counter()
heads = hydraulic_tomography(K, well_nodes, Q)
y_true = np.asarray(observation_operator(heads, well_nodes)).reshape(-1)
y_obs, obv_error = add_noise(y_true, noise_level=0.05)
y_obs = np.asarray(y_obs).reshape(-1)
print(f'Forward simulation: {time.perf_counter()-t0:.1f} seconds')
print('Observations:', len(y_obs))

print('Step 5: optimizing RGA coefficients with Gauss–Newton...')
history = {
    'true_alpha': alpha, 'true_mu': mu, 'true_y': y_true,
    'alpha': [], 'mu': [], 'loss': [], 'lambda': [],
    'step_norm': [], 'time': [], 'yp': []
}
b0 = np.concatenate((np.zeros(n_modes), np.array([beta])))
b_est, opt_history = gauss_newton_dynamic_lambda(
    lambda b: forward_model(b, V.T, Q, well_nodes),
    b0=b0, y_obs=y_obs, lam_init=1e-3, max_iter=10,
    tol=1e-5, history=history, adaptive_lambda=False,
    anneal_lambda=True, min_lambda=1e-5,
)
b_est = np.asarray(b_est).reshape(-1)
y_initial = np.asarray(forward_model(b0, V.T, Q, well_nodes)).reshape(-1)
y_fit = np.asarray(forward_model(b_est, V.T, Q, well_nodes)).reshape(-1)
if y_fit.size != y_obs.size:
    raise ValueError('Unexpected observation shape; inspect forward_model API')

rmse0 = np.sqrt(np.mean((y_initial-y_obs)**2))
rmse1 = np.sqrt(np.mean((y_fit-y_obs)**2))
print(f'RMSE: initial={rmse0:.5g}, fitted={rmse1:.5g}')
print('First 5 estimated reduced coefficients:', b_est[:5])

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(y_obs, y_initial, alpha=.7, label='Initial')
ax.scatter(y_obs, y_fit, alpha=.7, label='Estimated')
lo=min(y_obs.min(),y_initial.min(),y_fit.min())
hi=max(y_obs.max(),y_initial.max(),y_fit.max())
ax.plot([lo,hi],[lo,hi],'k--',label='1:1')
ax.set(xlabel='Observed head', ylabel='Predicted head',
       title='Hydraulic tomography fit')
ax.legend()
fig.tight_layout()
fig.savefig(OUT/'04_observed_vs_predicted.png', dpi=160)
plt.close(fig)

np.savez(OUT/'results.npz', K=K, logK=logK,
         x=xy, kriging_mean=krig, kriging_variance=variance,
         y_true=y_true, y_obs=y_obs, y_initial=y_initial,
         y_fit=y_fit, b0=b0, b_est=b_est)
print(f'Finished. Outputs in {OUT.resolve()}')
