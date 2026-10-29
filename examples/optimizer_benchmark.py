"""One controlled pyRGA Gauss–Newton benchmark case.

Usage example:
 python examples/optimizer_benchmark.py --grid 64 --modes 20 --noise 0.05 --iterations 8

Run multiple cases by invoking this script separately, keeping seeds fixed when
comparing hyperparameters. API based on pyrga-ht 0.1.4 published example.
This script does not assert an unverified mapping from coefficients to log K.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from pyrga import hydraulic_tomography
from pyrga.RGA import (generate_synthetic_field, prepare_physical_domain,
                       observation_operator, forward_model,
                       gauss_newton_dynamic_lambda, add_noise)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=int, default=64)
    ap.add_argument('--modes', type=int, default=20)
    ap.add_argument('--realizations', type=int, default=400)
    ap.add_argument('--noise', type=float, default=0.05)
    ap.add_argument('--iterations', type=int, default=8)
    ap.add_argument('--seed', type=int, default=17)
    ap.add_argument('--init-scale', type=float, default=0.0)
    ap.add_argument('--lx', type=float, default=0.15)
    ap.add_argument('--ly', type=float, default=0.2)
    ap.add_argument('--out', type=Path, default=Path('benchmark_outputs'))
    args = ap.parse_args()
    if args.grid < 4 or args.modes < 1 or args.noise < 0 or args.iterations < 1:
        ap.error('Require grid>=4, modes>=1, noise>=0, iterations>=1')
    args.out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    t_total = time.perf_counter()
    t0 = time.perf_counter()
    K, alpha, V, beta = generate_synthetic_field(
        args.grid, args.grid, k=args.modes, mu=-4, NR=args.realizations,
        cov_type='gaussian', variance=1.0, lx=args.lx, ly=args.ly)
    t_field = time.perf_counter() - t0
    K = np.asarray(K).reshape(-1)
    if K.size != args.grid**2 or np.any(K <= 0):
        raise ValueError('Unexpected conductivity field shape or nonpositive values')
    well_nodes, Q, _, _ = prepare_physical_domain(
        args.grid, args.grid, Lox=320, Loy=320,
        q_original=-0.02, well_relative_locs=None)
    t0 = time.perf_counter()
    h = hydraulic_tomography(K, well_nodes, Q)
    true_y = np.asarray(observation_operator(h, well_nodes)).reshape(-1)
    obs, _ = add_noise(true_y, noise_level=args.noise)
    obs = np.asarray(obs).reshape(-1)
    t_forward = time.perf_counter() - t0
    initial_coeff = np.random.default_rng(args.seed + 23).normal(
        size=args.modes) * args.init_scale
    b0 = np.r_[initial_coeff, float(beta)]
    model = lambda b: forward_model(b, V.T, Q, well_nodes)
    y0 = np.asarray(model(b0)).reshape(-1)
    history = {'true_alpha': alpha, 'true_mu': -4, 'true_y': true_y,
               'alpha': [], 'mu': [], 'loss': [], 'lambda': [],
               'step_norm': [], 'time': [], 'yp': []}
    t0 = time.perf_counter()
    b_est, out_hist = gauss_newton_dynamic_lambda(
        model, b0=b0, y_obs=obs, lam_init=1e-3,
        max_iter=args.iterations, tol=1e-5,
        history=history, adaptive_lambda=False,
        anneal_lambda=True, min_lambda=1e-5)
    t_inverse = time.perf_counter() - t0
    yf = np.asarray(model(b_est)).reshape(-1)
    if not (len(y0) == len(yf) == len(obs) == len(true_y)):
        raise ValueError('Predicted and observed head vectors have different sizes')
    def rmse(a, b):
        return float(np.sqrt(np.mean((a-b)**2)))
    metrics = {
        'grid': args.grid, 'cells': int(K.size), 'modes': args.modes,
        'realizations': args.realizations, 'noise': args.noise,
        'iterations_budget': args.iterations, 'seed': args.seed,
        'init_scale': args.init_scale, 'lx': args.lx, 'ly': args.ly,
        'observations': int(obs.size), 'parameters': int(np.asarray(b_est).size),
        'rmse_obs_initial': rmse(y0, obs), 'rmse_obs_final': rmse(yf, obs),
        'rmse_truth_initial': rmse(y0, true_y),
        'rmse_truth_final': rmse(yf, true_y),
        'time_field_s': round(t_field, 4),
        'time_forward_s': round(t_forward, 4),
        'time_inverse_s': round(t_inverse, 4),
        'time_total_s': round(time.perf_counter()-t_total, 4),
        'conductivity_reconstruction_validated': False,
    }
    case = (f'g{args.grid}_k{args.modes}_noise{args.noise:g}'
            f'_init{args.init_scale:g}_seed{args.seed}')
    (args.out / f'{case}.json').write_text(json.dumps(metrics, indent=2))
    with (args.out/'summary.csv').open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(metrics)
    # Save data + raw history; some history fields are package-specific.
    np.savez(args.out/f'{case}.npz', true_K=K, true_head=true_y,
             observed_head=obs, initial_head=y0, optimized_head=yf,
             initial_parameters=b0, optimized_parameters=np.asarray(b_est))
    print(json.dumps(metrics, indent=2))
    print('Optimizer history keys:', list(out_hist) if isinstance(out_hist, dict) else type(out_hist))


if __name__ == '__main__':
    main()
