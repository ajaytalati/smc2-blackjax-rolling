#!/usr/bin/env python3
"""
fsa_real_obs_5yr_rolling.py — Scenario driver for the FSA real-obs model.
==========================================================================

Thin wrapper around the ``smc2bj`` framework that:
  1. Generates a macrocycle training schedule (C0/C2/C3 — three excitation
     conditions studied in the v4.1 robustness experiments).
  2. Forward-simulates the 3-state FSA SDE (``models.fsa_real_obs.simulation``).
  3. Generates the 6 observation channels + 2 exogenous-input channels.
  4. Applies realistic missing-data corruption (rest days, dropout, broken-
     watch gap).
  5. Runs rolling-window SMC² estimation.
  6. Writes diagnostic plots + JSON checkpoint to ``outputs/<condition>_N<N>_s<seed>/``.

Replaces the monolithic 1538-line ``version_4_1/fsa_real_obs_5yr_rolling_smc.py``
by delegating the generic SMC²/pipeline/IO to ``smc2bj.*`` and the per-model
PF hooks to ``models.fsa_real_obs.estimation.FSA_REAL_OBS_ESTIMATION``.

Usage
-----
    python drivers/fsa_real_obs_5yr_rolling.py --seed 42 --condition C0
    python drivers/fsa_real_obs_5yr_rolling.py --seed 42 --condition C0 --windows 1
    python drivers/fsa_real_obs_5yr_rolling.py --sim-only --seed 42 --condition C0
    python drivers/fsa_real_obs_5yr_rolling.py --show-checkpoint \
        --seed 42 --condition C0
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

os.environ.setdefault('JAX_ENABLE_X64', 'True')
os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', '/tmp/jax_cache')

import jax

from smc2bj.estimation.config import (
    SMCConfig, RollingConfig, MissingDataConfig,
)
from smc2bj.pipeline.missing_data import apply_missing_data
from smc2bj.pipeline.rolling import rolling_window_smc
from smc2bj.io.checkpoint import show_checkpoint as _show_checkpoint
from smc2bj.plotting.rolling import (
    plot_parameter_tracking,
    plot_coverage_and_timing,
)

from models.fsa_real_obs.simulation import FSA_REAL_OBS_MODEL
from models.fsa_real_obs.estimation import (
    FSA_REAL_OBS_ESTIMATION, COLD_START_INIT,
)


# ═════════════════════════════════════════════════════════════════════════
# Scenario constants (match version_4_1 defaults)
# ═════════════════════════════════════════════════════════════════════════

N_DAYS_TOTAL = 365
DT = 1.0
N_SUBSTEPS = 10

# Frozen SDE noise (matches the model spec)
SIGMA_B = 0.01
SIGMA_F = 0.005
SIGMA_A = 0.02
EPS_A = 1e-4
EPS_B = 1e-4

# Observation-channel conventions for the FSA model
OBS_CHANNEL_NAMES = (
    'obs_RHR', 'obs_intensity', 'obs_duration',
    'obs_stress', 'obs_sleep', 'obs_timing',
)
ACTIVE_CHANNELS = ('obs_intensity', 'obs_duration', 'obs_timing')
PASSIVE_CHANNELS = ('obs_RHR', 'obs_stress', 'obs_sleep')


# ═════════════════════════════════════════════════════════════════════════
# 1. Macrocycle generators (C0/C2/C3 — three excitation conditions)
# ═════════════════════════════════════════════════════════════════════════

def generate_macrocycle_C0(n_days: int, seed: int = 42):
    """Baseline (C0): 28-day mesocycles + overreach (every 90d) + taper (every 180d).

    The base layer provides load/deload cycles; overlay overreach spikes
    and off-season tapers give enough B-F decoupling for kappa_vagal /
    kappa_chronic identifiability.
    """
    rng = np.random.default_rng(seed)
    T_B = np.zeros(n_days, dtype=np.float32)
    Phi = np.zeros(n_days, dtype=np.float32)

    # Base 28-day mesocycles
    for block_start in range(0, n_days, 28):
        block_end = min(block_start + 28, n_days)
        load_T_B = rng.uniform(0.6, 0.85)
        load_Phi = rng.uniform(0.08, 0.15)
        deload_T_B = rng.uniform(0.3, 0.5)
        deload_Phi = rng.uniform(0.01, 0.04)
        for d in range(block_start, block_end):
            day_in_block = d - block_start
            if day_in_block < 21:
                base_T_B, base_Phi = load_T_B, load_Phi
            else:
                base_T_B, base_Phi = deload_T_B, deload_Phi
            T_B[d] = base_T_B * (1.0 + 0.1 * rng.standard_normal())
            Phi[d] = base_Phi * (1.0 + 0.1 * rng.standard_normal())

    # Off-season tapers (every 180d, 21d duration)
    taper_period, taper_duration = 180, 21
    for taper_start in range(taper_period - taper_duration, n_days,
                             taper_period):
        for d in range(taper_start, min(taper_start + taper_duration, n_days)):
            T_B[d] = rng.uniform(0.15, 0.30)
            Phi[d] = rng.uniform(0.01, 0.02)

    # Overreach spikes (every 90d, 14d duration; skip if overlapping taper)
    overreach_period, overreach_duration = 90, 14
    for or_start in range(overreach_period - overreach_duration, n_days,
                          overreach_period):
        overlaps_taper = False
        for taper_start in range(taper_period - taper_duration, n_days,
                                 taper_period):
            taper_end = taper_start + taper_duration
            if (or_start < taper_end
                    and (or_start + overreach_duration) > taper_start):
                overlaps_taper = True
                break
        if overlaps_taper:
            continue
        for d in range(or_start,
                       min(or_start + overreach_duration, n_days)):
            T_B[d] = rng.uniform(0.80, 0.95)
            Phi[d] = rng.uniform(0.20, 0.25)

    T_B = np.clip(T_B, 0.05, 0.95)
    Phi = np.clip(Phi, 0.005, 0.25)
    return T_B, Phi


def generate_macrocycle_C2(n_days: int, seed: int = 42):
    """Strong (C2): 28d base + 35-day deep tapers every 90d + 21d overreach."""
    rng = np.random.default_rng(seed)
    T_B = np.zeros(n_days, dtype=np.float32)
    Phi = np.zeros(n_days, dtype=np.float32)

    for block_start in range(0, n_days, 28):
        block_end = min(block_start + 28, n_days)
        load_T_B = rng.uniform(0.6, 0.85)
        load_Phi = rng.uniform(0.08, 0.15)
        deload_T_B = rng.uniform(0.3, 0.5)
        deload_Phi = rng.uniform(0.01, 0.04)
        for d in range(block_start, block_end):
            day_in_block = d - block_start
            if day_in_block < 21:
                base_T_B, base_Phi = load_T_B, load_Phi
            else:
                base_T_B, base_Phi = deload_T_B, deload_Phi
            T_B[d] = base_T_B * (1.0 + 0.1 * rng.standard_normal())
            Phi[d] = base_Phi * (1.0 + 0.1 * rng.standard_normal())

    taper_duration, taper_period, overreach_duration = 35, 90, 21
    for cycle_start in range(55, n_days, taper_period):
        taper_end = min(cycle_start + taper_duration, n_days)
        for d in range(cycle_start, taper_end):
            T_B[d] = rng.uniform(0.10, 0.20)
            Phi[d] = rng.uniform(0.005, 0.015)
        or_start = cycle_start + taper_duration
        or_end = min(or_start + overreach_duration, n_days)
        for d in range(or_start, or_end):
            T_B[d] = rng.uniform(0.85, 0.95)
            Phi[d] = rng.uniform(0.22, 0.28)

    T_B = np.clip(T_B, 0.05, 0.95)
    Phi = np.clip(Phi, 0.005, 0.28)
    return T_B, Phi


def generate_macrocycle_C3(n_days: int, seed: int = 42):
    """Maximal (C3): 75-day cycles, 30d moderate + 30d taper + 15d overreach."""
    rng = np.random.default_rng(seed)
    T_B = np.zeros(n_days, dtype=np.float32)
    Phi = np.zeros(n_days, dtype=np.float32)

    cycle_length = 75
    for cycle_start in range(0, n_days, cycle_length):
        for d in range(cycle_start, min(cycle_start + cycle_length, n_days)):
            day_in_cycle = d - cycle_start
            if day_in_cycle < 30:
                base_T_B = rng.uniform(0.55, 0.75)
                base_Phi = rng.uniform(0.06, 0.12)
            elif day_in_cycle < 60:
                base_T_B = rng.uniform(0.10, 0.20)
                base_Phi = rng.uniform(0.005, 0.015)
            else:
                base_T_B = rng.uniform(0.85, 0.95)
                base_Phi = rng.uniform(0.22, 0.28)
            T_B[d] = base_T_B * (1.0 + 0.1 * rng.standard_normal())
            Phi[d] = base_Phi * (1.0 + 0.1 * rng.standard_normal())

    T_B = np.clip(T_B, 0.05, 0.95)
    Phi = np.clip(Phi, 0.005, 0.28)
    return T_B, Phi


_MACROCYCLE_GENERATORS = {
    'C0': generate_macrocycle_C0,
    'C2': generate_macrocycle_C2,
    'C3': generate_macrocycle_C3,
}


# ═════════════════════════════════════════════════════════════════════════
# 2. SDE forward simulation + obs generation (numpy Euler-Maruyama)
# ═════════════════════════════════════════════════════════════════════════

def simulate_sde(T_B_daily, Phi_daily, params, init_state,
                 n_substeps: int = 10, seed: int = 42,
                 deterministic: bool = False):
    """Euler-Maruyama forward integration of the FSA 3-state SDE."""
    rng = np.random.default_rng(seed)
    n_days = len(T_B_daily)
    sub_dt = 1.0 / n_substeps
    sqrt_sub_dt = np.sqrt(sub_dt)

    tau_B = params['tau_B']
    alpha_A = params['alpha_A']
    tau_F = params['tau_F']
    lambda_B_p = params['lambda_B']
    lambda_A_p = params['lambda_A']
    mu_0 = params['mu_0']
    mu_B_p = params['mu_B']
    mu_F_p = params['mu_F']
    mu_FF = params['mu_FF']
    eta_p = params['eta']

    trajectory = np.zeros((n_days, 3))
    B = float(init_state['B_0'])
    F = float(init_state['F_0'])
    A = float(init_state['A_0'])

    for d in range(n_days):
        T_B_k = float(T_B_daily[d])
        Phi_k = float(Phi_daily[d])

        for _ in range(n_substeps):
            mu_bif = mu_0 + mu_B_p * B - mu_F_p * F - mu_FF * F * F
            dB = (1.0 + alpha_A * A) / tau_B * (T_B_k - B)
            dF = Phi_k - (1.0 + lambda_B_p * B + lambda_A_p * A) / tau_F * F
            dA = mu_bif * A - eta_p * A * A * A

            B_cl = np.clip(B, EPS_B, 1.0 - EPS_B)
            F_cl = max(F, 0.0)
            A_cl = max(A, 0.0)

            nz = np.zeros(3) if deterministic else rng.standard_normal(3)

            B += (sub_dt * dB
                  + SIGMA_B * np.sqrt(B_cl * (1 - B_cl)) * sqrt_sub_dt * nz[0])
            F += (sub_dt * dF
                  + SIGMA_F * np.sqrt(F_cl + 1e-10) * sqrt_sub_dt * nz[1])
            A += (sub_dt * dA
                  + SIGMA_A * np.sqrt(A_cl + EPS_A) * sqrt_sub_dt * nz[2])

            B = np.clip(B, EPS_B, 1.0 - EPS_B)
            F = max(F, 0.0)
            A = max(A, 0.0)

        trajectory[d] = [B, F, A]

    return trajectory


def generate_observations(trajectory, T_B_daily, Phi_daily, params,
                          seed: int = 42):
    """Generate 6 observation channels + 2 exogenous channels."""
    rng = np.random.default_rng(seed)
    n = len(trajectory)
    B, F, A = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    t_idx = np.arange(n, dtype=np.int32)

    channels = {}

    RHR_true = (params['R_base'] - params['kappa_vagal'] * B
                + params['kappa_chronic'] * F)
    channels['obs_RHR'] = {
        't_idx': t_idx.copy(),
        'obs_value': (RHR_true + rng.normal(0, params['sigma_obs_R'], n)
                      ).astype(np.float32),
    }

    I_true = params['I_base'] + params['c_B'] * B - params['c_F'] * F
    channels['obs_intensity'] = {
        't_idx': t_idx.copy(),
        'obs_value': (I_true + rng.normal(0, params['sigma_obs_I'], n)
                      ).astype(np.float32),
    }

    D_true = params['D_base'] + params['d_B'] * B - params['d_F'] * F
    channels['obs_duration'] = {
        't_idx': t_idx.copy(),
        'obs_value': (D_true + rng.normal(0, params['sigma_obs_D'], n)
                      ).astype(np.float32),
    }

    S_true = params['S_base'] - params['s_A'] * A + params['s_F'] * F
    channels['obs_stress'] = {
        't_idx': t_idx.copy(),
        'obs_value': (S_true + rng.normal(0, params['sigma_obs_S'], n)
                      ).astype(np.float32),
    }

    Sleep_true = (params['Sleep_base'] + params['sl_A'] * A
                  + params['sl_B'] * B - params['sl_F'] * F)
    channels['obs_sleep'] = {
        't_idx': t_idx.copy(),
        'obs_value': (Sleep_true + rng.normal(0, params['sigma_obs_Sleep'], n)
                      ).astype(np.float32),
    }

    Time_true = params['Time_base'] + params['t_A'] * A - params['t_F'] * F
    channels['obs_timing'] = {
        't_idx': t_idx.copy(),
        'obs_value': (Time_true + rng.normal(0, params['sigma_obs_Time'], n)
                      ).astype(np.float32),
    }

    channels['T_B'] = {
        't_idx': t_idx.copy(),
        'T_B_value': T_B_daily.astype(np.float32),
    }
    channels['Phi'] = {
        't_idx': t_idx.copy(),
        'Phi_value': Phi_daily.astype(np.float32),
    }

    return channels


# ═════════════════════════════════════════════════════════════════════════
# 3. Truth + prior utility
# ═════════════════════════════════════════════════════════════════════════

def _truth_dict(true_params, true_init):
    d = {k: v for k, v in true_params.items()}
    d['mu_0_abs'] = abs(true_params['mu_0'])
    d.pop('kappa_chronic', None)  # frozen, not estimated
    d.pop('R_base', None)         # frozen, not estimated
    d.update(true_init)
    return d


# ═════════════════════════════════════════════════════════════════════════
# 4. Plotting (FSA-scenario-specific)
# ═════════════════════════════════════════════════════════════════════════

def _use_agg():
    import matplotlib
    matplotlib.use('Agg')


def plot_macrocycle_schedule(T_B_daily, Phi_daily, trajectory, out_dir):
    _use_agg()
    import matplotlib.pyplot as plt
    n_days = len(T_B_daily)
    t = np.arange(n_days)
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    axes[0].plot(t, T_B_daily, color='navy', lw=0.5, alpha=0.7)
    axes[0].set_ylabel(r'$T_B(t)$'); axes[0].set_title('Training Load Schedule')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, Phi_daily, color='darkorange', lw=0.5, alpha=0.7)
    axes[1].set_ylabel(r'$\Phi(t)$'); axes[1].set_title('Strain Input Schedule')
    axes[1].grid(True, alpha=0.3)
    for s, (col, lbl) in enumerate([('steelblue', 'B (fitness)'),
                                    ('firebrick', 'F (strain)'),
                                    ('darkgreen', 'A (amplitude)')]):
        axes[2 + s].plot(t, trajectory[:, s], color=col, lw=0.8)
        axes[2 + s].set_ylabel(lbl); axes[2 + s].grid(True, alpha=0.3)
    axes[-1].set_xlabel('Day')
    fig.suptitle(f'Macrocycle Schedule & Latent Trajectory ({n_days} days)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'macrocycle_schedule.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


def plot_observations_with_missing(obs_data, trajectory, n_days,
                                   true_params, out_dir):
    _use_agg()
    import matplotlib.pyplot as plt
    B, F, A = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    t_all = np.arange(n_days)
    p = true_params
    true_signals = {
        'obs_RHR': p['R_base'] - p['kappa_vagal'] * B + p['kappa_chronic'] * F,
        'obs_intensity': p['I_base'] + p['c_B'] * B - p['c_F'] * F,
        'obs_duration': p['D_base'] + p['d_B'] * B - p['d_F'] * F,
        'obs_stress': p['S_base'] - p['s_A'] * A + p['s_F'] * F,
        'obs_sleep': (p['Sleep_base'] + p['sl_A'] * A
                      + p['sl_B'] * B - p['sl_F'] * F),
        'obs_timing': p['Time_base'] + p['t_A'] * A - p['t_F'] * F,
    }
    ch_info = [
        ('obs_RHR', 'RHR (bpm)', '#e74c3c'),
        ('obs_intensity', 'Intensity', '#3498db'),
        ('obs_duration', 'Duration', '#2ecc71'),
        ('obs_stress', 'Stress', '#9b59b6'),
        ('obs_sleep', 'Sleep quality', '#f39c12'),
        ('obs_timing', 'Circadian timing', '#1abc9c'),
    ]
    fig, axes = plt.subplots(6, 1, figsize=(18, 16), sharex=True)
    for i, (ch_name, ylabel, color) in enumerate(ch_info):
        ax = axes[i]
        ax.plot(t_all, true_signals[ch_name], color=color, lw=0.6,
                alpha=0.4, label='true signal')
        ch = obs_data[ch_name]
        ax.scatter(ch['t_idx'], ch['obs_value'], s=1, alpha=0.3, color='black',
                   label=f"obs ({len(ch['t_idx'])}/{n_days})", zorder=3)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel('Day')
    fig.suptitle('Observation Channels with Missing Data',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'observations_5yr.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


def plot_latent_reconstruction(results, trajectory_true, T_B_daily, Phi_daily,
                               model, n_substeps, out_dir):
    _use_agg()
    import matplotlib.pyplot as plt
    n_days = len(trajectory_true)
    t_days = np.arange(n_days)
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    state_names = ['B (fitness)', 'F (strain)', 'A (amplitude)']
    colors_true = ['steelblue', 'firebrick', 'darkgreen']
    colors_est = ['#6b7fd9', '#e74c3c', '#2ecc71']
    for s in range(3):
        axes[s].plot(t_days, trajectory_true[:, s], color=colors_true[s],
                     lw=0.8, alpha=0.6, label='true')
        axes[s].set_ylabel(state_names[s], fontsize=10)
        axes[s].grid(True, alpha=0.2)
    for r in results:
        start, end = r['start_day'], r['end_day']
        stats = r['stats']
        est_params = {name: stats[name]['mean']
                      for name in model.param_prior_config.keys()}
        est_params['mu_0'] = -est_params.pop('mu_0_abs')
        fis = r.get('fixed_init_state', np.array([0.05, 0.10, 0.01]))
        est_init = {'B_0': float(fis[0]),
                    'F_0': float(fis[1]),
                    'A_0': float(fis[2])}
        recon = simulate_sde(T_B_daily[start:end], Phi_daily[start:end],
                             est_params, est_init,
                             n_substeps=n_substeps, deterministic=True)
        window_t = np.arange(start, end)
        for s in range(3):
            axes[s].plot(window_t, recon[:, s], color=colors_est[s],
                         lw=0.5, alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color='grey', lw=1.2, alpha=0.6, label='true'),
               Line2D([0], [0], color='grey', lw=0.8, alpha=0.3,
                      label='estimated')]
    axes[0].legend(handles=handles, loc='upper right', fontsize=9)
    axes[-1].set_xlabel('Day')
    fig.suptitle('Latent State Reconstruction — Rolling Window SMC²',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'latent_reconstruction.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════
# 5. Main orchestration
# ═════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--condition', choices=('C0', 'C2', 'C3'), default='C0')
    p.add_argument('--n-smc', type=int, default=256,
                   help='SMC particles')
    p.add_argument('--n-pf', type=int, default=400,
                   help='Inner PF particles')
    p.add_argument('--windows', type=int, default=None,
                   help='Max windows (None = full; 1 = fast parity test)')
    p.add_argument('--sim-only', action='store_true',
                   help='Generate data + plots only, no SMC')
    p.add_argument('--show-checkpoint', action='store_true',
                   help='Display existing checkpoint and exit')
    return p.parse_args()


def _out_dir(condition: str, n_smc: int, seed: int) -> str:
    return os.path.join('outputs', 'fsa_real_obs_5yr_rolling',
                        f'{condition}_N{n_smc}_s{seed}')


def main():
    args = _parse_args()

    out_dir = _out_dir(args.condition, args.n_smc, args.seed)
    os.makedirs(out_dir, exist_ok=True)

    if args.show_checkpoint:
        _show_checkpoint(os.path.join(out_dir, 'rolling_checkpoint.json'))
        return 0

    # All three 'param_sets' entries point to the same DEFAULT_PARAMS in the
    # FSA reference model; the SCENARIO distinction is historical and dead.
    # See HANDOFF.md TODO #7 for the upstream cleanup.
    true_params = dict(FSA_REAL_OBS_MODEL.param_sets['recovery'])
    true_init = dict(FSA_REAL_OBS_MODEL.init_states['recovery'])
    truth = _truth_dict(true_params, true_init)

    model = FSA_REAL_OBS_ESTIMATION
    smc_cfg = SMCConfig(n_smc_particles=args.n_smc, n_pf_particles=args.n_pf)
    rolling_cfg = RollingConfig(
        window_days=120, stride_days=30, dt=DT, n_substeps=N_SUBSTEPS,
        max_windows=args.windows,
    )
    missing_cfg = MissingDataConfig(
        dropout_rate=0.15, broken_watch_days=14, rest_days_per_week=(2, 3),
        active_channels=ACTIVE_CHANNELS,
        passive_channels=PASSIVE_CHANNELS,
        all_obs_channels=OBS_CHANNEL_NAMES,
    )

    print("=" * 70)
    print(f"  ROLLING WINDOW SMC² — FSA REAL-OBS — {args.condition}  "
          f"(seed={args.seed}, N={args.n_smc}, K={args.n_pf})")
    print("=" * 70)
    print(f"  Params:      {model.n_dim} ({model.n_params} model + "
          f"{model.n_init_states} init)")
    print(f"  Simulation:  {N_DAYS_TOTAL}d @ dt={DT}, {N_SUBSTEPS} substeps")
    print(f"  Windows:     {rolling_cfg.window_days}d window, "
          f"{rolling_cfg.stride_days}d stride")
    if args.windows is not None:
        print(f"  Max windows: {args.windows} (truncated for fast test)")
    print(f"  Device:      "
          f"{'GPU' if jax.devices()[0].platform == 'gpu' else 'CPU'}")
    print()

    # Step 1: Macrocycle
    print("Step 1: Generate macrocycle schedule")
    T_B_daily, Phi_daily = _MACROCYCLE_GENERATORS[args.condition](
        N_DAYS_TOTAL, seed=args.seed)
    print(f"  T_B range: [{T_B_daily.min():.3f}, {T_B_daily.max():.3f}]")
    print(f"  Phi range: [{Phi_daily.min():.4f}, {Phi_daily.max():.4f}]")

    # Step 2: Forward simulate
    print(f"\nStep 2: Forward simulate {N_DAYS_TOTAL}-day SDE trajectory")
    t0 = time.time()
    trajectory = simulate_sde(T_B_daily, Phi_daily, true_params, true_init,
                              n_substeps=N_SUBSTEPS, seed=args.seed)
    print(f"  B: [{trajectory[:,0].min():.3f}, {trajectory[:,0].max():.3f}]")
    print(f"  F: [{trajectory[:,1].min():.3f}, {trajectory[:,1].max():.3f}]")
    print(f"  A: [{trajectory[:,2].min():.3f}, {trajectory[:,2].max():.3f}]")
    print(f"  Simulation time: {time.time()-t0:.1f}s")

    traj_csv = os.path.join(out_dir, 'trajectory_5yr.csv')
    np.savetxt(traj_csv,
               np.column_stack([np.arange(N_DAYS_TOTAL), trajectory]),
               delimiter=',', header='t_days,B,F,A', comments='')
    print(f"  -> {traj_csv}")

    # Step 3: Observations
    print("\nStep 3: Generate observations (6 channels)")
    obs_data = generate_observations(trajectory, T_B_daily, Phi_daily,
                                     true_params, seed=args.seed + 1)

    # Step 4: Missing data
    print("\nStep 4: Apply missing data masking")
    obs_data = apply_missing_data(obs_data, N_DAYS_TOTAL, missing_cfg,
                                  seed=args.seed + 2)

    # Step 5: Diagnostic plots
    print("\nStep 5: Diagnostic plots")
    plot_macrocycle_schedule(T_B_daily, Phi_daily, trajectory, out_dir)
    plot_observations_with_missing(obs_data, trajectory, N_DAYS_TOTAL,
                                   true_params, out_dir)

    if args.sim_only:
        print(f"\nDone (sim-only). Check plots in {out_dir}/")
        return 0

    # Step 6: Rolling SMC²
    print("\nStep 6: Rolling window SMC²")
    import jax.numpy as jnp
    cold_init = jnp.asarray(COLD_START_INIT)
    results, T_arr = rolling_window_smc(
        obs_data, model, N_DAYS_TOTAL, out_dir,
        smc_cfg=smc_cfg, rolling_cfg=rolling_cfg,
        cold_start_init=cold_init, truth=truth,
        obs_channel_names=OBS_CHANNEL_NAMES,
        seed=args.seed,
    )

    # Step 7: Validation plots
    print("\nStep 7: Validation plots")
    plot_parameter_tracking(results, model, truth, out_dir)
    plot_latent_reconstruction(results, trajectory, T_B_daily, Phi_daily,
                               model, N_SUBSTEPS, out_dir)
    plot_coverage_and_timing(results, out_dir)

    coverages = [r['coverage'] for r in results]
    total_smc_time = sum(r['elapsed_s'] for r in results)
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Windows:       {len(results)}")
    print(f"  Mean coverage: {np.mean(coverages)*100:.1f}%")
    print(f"  Min coverage:  {np.min(coverages)*100:.1f}%")
    print(f"  Max coverage:  {np.max(coverages)*100:.1f}%")
    n_pass = sum(1 for c in coverages if c >= 0.7)
    print(f"  PASS (>=70%):  {n_pass}/{len(results)}")
    print(f"  Total SMC:     {total_smc_time/3600:.1f}h")
    print(f"  Output:        {out_dir}/")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
