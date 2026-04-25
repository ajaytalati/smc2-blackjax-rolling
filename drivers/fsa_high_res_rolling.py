#!/usr/bin/env python3
"""
fsa_high_res_rolling.py — Scenario driver for the FSA high-res model.
=====================================================================

Thin wrapper that:
  1. Generates a 14-day C0 macrocycle daily-load schedule (reuses
     generate_macrocycle_C0 from the daily FSA driver).
  2. Expands daily Phi into 15-min bins with training bursts + sleep zeros.
  3. Forward-simulates 14 days of the FSA SDE at 15-min resolution.
  4. Samples sleep Bernoulli labels, gates HR (sleep-only) + stress
     (wake-only) + steps (wake-only), applies 5% dropout.
  5. Runs rolling-window SMC² (3-day window, 1-day stride → 12 windows).
  6. Plots diagnostics + writes JSON checkpoint.

Usage
-----
    python drivers/fsa_high_res_rolling.py --seed 42
    python drivers/fsa_high_res_rolling.py --sim-only --seed 42
    python drivers/fsa_high_res_rolling.py --seed 42 --windows 1
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
import jax.numpy as jnp

from smc2bj.estimation.config import SMCConfig, RollingConfig
from smc2bj.pipeline.rolling import rolling_window_smc
from smc2bj.io.checkpoint import show_checkpoint as _show_checkpoint
from smc2bj.plotting.rolling import (
    plot_parameter_tracking,
    plot_coverage_and_timing,
)

# Reuse the C0 macrocycle generator from the daily FSA driver
from drivers.fsa_real_obs_5yr_rolling import generate_macrocycle_C0

from models.fsa_high_res.simulation import (
    HIGH_RES_FSA_MODEL, DEFAULT_PARAMS,
    generate_phi_sub_daily, circadian,
    drift, diffusion_diagonal, noise_scale_fn,
    BINS_PER_DAY, DT_BIN_DAYS, DT_BIN_HOURS,
)
from models.fsa_high_res.estimation import (
    HIGH_RES_FSA_ESTIMATION, COLD_START_INIT, PHI_FROZEN,
)


# ═════════════════════════════════════════════════════════════════════════
# Scenario constants
# ═════════════════════════════════════════════════════════════════════════

N_DAYS_TOTAL = 14
N_SUBSTEPS = 4
DT = DT_BIN_DAYS  # 1/96 day = 15 min

OBS_CHANNEL_NAMES = ('obs_HR', 'obs_sleep', 'obs_stress', 'obs_steps')


# ═════════════════════════════════════════════════════════════════════════
# Forward simulation (numpy Euler-Maruyama on the 15-min grid)
# ═════════════════════════════════════════════════════════════════════════

def simulate_sde_high_res(T_B_arr, Phi_arr, params, init_state,
                          n_substeps=N_SUBSTEPS, seed=42):
    """Euler-Maruyama on a 15-min grid with per-bin T_B and Phi arrays."""
    rng = np.random.default_rng(seed)
    n_bins = len(T_B_arr)
    sub_dt = DT_BIN_DAYS / n_substeps
    sqrt_sub_dt = np.sqrt(sub_dt)

    trajectory = np.zeros((n_bins, 3))
    B = float(init_state['B_0'])
    F = float(init_state['F_0'])
    A = float(init_state['A_0'])
    sigma = diffusion_diagonal(params)

    aux = (T_B_arr, Phi_arr)
    for k in range(n_bins):
        t_days = k * DT_BIN_DAYS
        for _ in range(n_substeps):
            dy = drift(t_days, np.array([B, F, A]), params, aux)
            ns = noise_scale_fn(np.array([B, F, A]), params)
            z = rng.standard_normal(3)
            B = B + sub_dt * dy[0] + sqrt_sub_dt * sigma[0] * ns[0] * z[0]
            F = F + sub_dt * dy[1] + sqrt_sub_dt * sigma[1] * ns[1] * z[1]
            A = A + sub_dt * dy[2] + sqrt_sub_dt * sigma[2] * ns[2] * z[2]
            B = np.clip(B, 1e-4, 1.0 - 1e-4)
            F = max(F, 0.0)
            A = max(A, 0.0)
            t_days += sub_dt
        trajectory[k] = [B, F, A]
    return trajectory


# ═════════════════════════════════════════════════════════════════════════
# Observation generation (calls the model's gen_obs_* functions)
# ═════════════════════════════════════════════════════════════════════════

def generate_observations(trajectory, T_B_arr, Phi_arr, params, seed=42):
    """Run the 4 obs channel generators + 2 exogenous broadcasts."""
    from models.fsa_high_res.simulation import (
        gen_obs_sleep, gen_obs_hr, gen_obs_stress, gen_obs_steps,
        gen_T_B_channel, gen_Phi_channel, gen_C_channel,
    )
    n_bins = trajectory.shape[0]
    t_grid = np.arange(n_bins, dtype=np.float32) * DT_BIN_DAYS
    aux = (T_B_arr, Phi_arr)

    # Sleep first (others depend on sleep label for gating)
    sleep = gen_obs_sleep(trajectory, t_grid, params, aux, None, seed=seed)
    prior = {'obs_sleep': sleep}
    hr     = gen_obs_hr(trajectory,     t_grid, params, aux, prior, seed=seed + 1)
    stress = gen_obs_stress(trajectory, t_grid, params, aux, prior, seed=seed + 2)
    steps  = gen_obs_steps(trajectory,  t_grid, params, aux, prior, seed=seed + 3)
    T_B_ch = gen_T_B_channel(trajectory, t_grid, params, aux, None, seed=seed + 4)
    Phi_ch = gen_Phi_channel(trajectory, t_grid, params, aux, None, seed=seed + 5)
    C_ch   = gen_C_channel  (trajectory, t_grid, params, aux, None, seed=seed + 6)

    return {
        'obs_sleep':  sleep,
        'obs_HR':     hr,
        'obs_stress': stress,
        'obs_steps':  steps,
        'T_B':        T_B_ch,
        'Phi':        Phi_ch,
        'C':          C_ch,
    }


def apply_missing_data_high_res(obs_data, seed=42, dropout_rate=0.05):
    """Minimal missing-data for the proof-of-principle: 5% dropout on
    HR, stress, steps (not sleep)."""
    rng = np.random.default_rng(seed)
    for ch_name in ('obs_HR', 'obs_stress', 'obs_steps'):
        ch = obs_data[ch_name]
        idx = ch['t_idx']
        keep = rng.random(len(idx)) > dropout_rate
        ch['t_idx'] = idx[keep]
        ch['obs_value'] = ch['obs_value'][keep]
    return obs_data


# ═════════════════════════════════════════════════════════════════════════
# Truth helpers
# ═════════════════════════════════════════════════════════════════════════

def _truth_dict(true_params):
    d = dict(true_params)
    # high-res PARAM_PRIOR_CONFIG uses mu_0 directly (positive lognormal prior).
    # This is the sign-clean version of the daily FSA's mu_0_abs reparam.
    d.pop('sigma_B', None)
    d.pop('sigma_F', None)
    d.pop('sigma_A', None)
    d.pop('phi', None)
    return d


# ═════════════════════════════════════════════════════════════════════════
# Diagnostic plots (FSA-specific, scenario-specific)
# ═════════════════════════════════════════════════════════════════════════

def _use_agg():
    import matplotlib
    matplotlib.use('Agg')


def plot_high_res_channels(obs_data, trajectory, T_B_arr, Phi_arr,
                            true_params, out_dir):
    """14-day 6-panel: B/F/A, C(t), Phi, and the 4 obs channels."""
    _use_agg()
    import matplotlib.pyplot as plt
    n_bins = trajectory.shape[0]
    t_days = np.arange(n_bins, dtype=np.float32) * DT_BIN_DAYS
    t_hours = t_days * 24.0
    C = np.cos(2.0 * np.pi * t_days + PHI_FROZEN)

    fig, axes = plt.subplots(7, 1, figsize=(18, 14), sharex=True)

    axes[0].plot(t_hours, trajectory[:, 0], color='steelblue', lw=0.7, label='B')
    axes[0].plot(t_hours, trajectory[:, 1], color='firebrick', lw=0.7, label='F')
    axes[0].plot(t_hours, trajectory[:, 2], color='darkgreen', lw=0.7, label='A')
    axes[0].set_ylabel('latent')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_title(f'FSA high-res ({N_DAYS_TOTAL} days)')

    axes[1].plot(t_hours, C, color='purple', lw=0.5)
    axes[1].axhline(0, color='grey', ls=':', alpha=0.3)
    axes[1].set_ylabel('C(t)')
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(t_hours, Phi_arr, color='darkorange', lw=0.5)
    axes[2].set_ylabel(r'$\Phi(t)$')
    axes[2].grid(True, alpha=0.2)

    hr = obs_data['obs_HR']
    axes[3].scatter(hr['t_idx'] * DT_BIN_HOURS, hr['obs_value'],
                    s=3, color='#e74c3c', alpha=0.6)
    axes[3].set_ylabel('HR (bpm)')
    axes[3].grid(True, alpha=0.2)

    sl = obs_data['obs_sleep']
    axes[4].scatter(sl['t_idx'] * DT_BIN_HOURS, sl['sleep_label'],
                    s=3, color='#1abc9c', alpha=0.6)
    axes[4].set_ylabel('sleep')
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].grid(True, alpha=0.2)

    st = obs_data['obs_stress']
    axes[5].scatter(st['t_idx'] * DT_BIN_HOURS, st['obs_value'],
                    s=3, color='#9b59b6', alpha=0.6)
    axes[5].set_ylabel('stress')
    axes[5].grid(True, alpha=0.2)

    stp = obs_data['obs_steps']
    axes[6].scatter(stp['t_idx'] * DT_BIN_HOURS, stp['obs_value'],
                    s=3, color='#3498db', alpha=0.6)
    axes[6].set_ylabel('steps/bin')
    axes[6].set_xlabel('hours')
    axes[6].grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, 'high_res_channels.png')
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-smc', type=int, default=256)
    p.add_argument('--n-pf', type=int, default=400)
    p.add_argument('--windows', type=int, default=None,
                   help='Max windows (None = full 27; 1 = fast parity test)')
    p.add_argument('--bridge', choices=('gaussian', 'mog'),
                   default='gaussian',
                   help='Bridge base measure: single Gaussian (default) or '
                        'K-component mixture of Gaussians')
    p.add_argument('--bridge-K', type=int, default=2,
                   help='K for --bridge mog (default 2)')
    p.add_argument('--sim-only', action='store_true')
    p.add_argument('--show-checkpoint', action='store_true')
    p.add_argument('--scenario-artifact', default=None,
                   help='Path to a Python-Model-Scenario-Simulation artifact '
                        'directory. When set, bypasses inline data generation '
                        '(Steps 1-4) and consumes the validated artifact '
                        'instead. The downstream rolling-window SMC step is '
                        'unchanged.')
    return p.parse_args()


def _out_dir(seed: int, n_smc: int, bridge_tag: str = '') -> str:
    tag = f'_{bridge_tag}' if bridge_tag else ''
    return os.path.join('outputs', 'fsa_high_res_rolling',
                        f'C0_N{n_smc}_s{seed}{tag}')


def main():
    args = _parse_args()
    bridge_tag = '' if args.bridge == 'gaussian' else f'mog{args.bridge_K}'
    if args.scenario_artifact:
        bridge_tag = (bridge_tag + '_psim_artifact').lstrip('_')
    out_dir = _out_dir(args.seed, args.n_smc, bridge_tag)
    os.makedirs(out_dir, exist_ok=True)

    if args.show_checkpoint:
        _show_checkpoint(os.path.join(out_dir, 'rolling_checkpoint.json'))
        return 0

    true_params = dict(DEFAULT_PARAMS)
    truth = _truth_dict(true_params)
    model = HIGH_RES_FSA_ESTIMATION

    # POC sizing: 1-day windows (96 bins) with 12-hour stride (48 bins).
    # A 3-day window at 15-min resolution produces ~700 obs/window; with
    # 29 params this gives a prior→posterior contrast too large for the
    # adaptive-tempering ESS solver to bridge in feasible wall-clock.
    # Tighter priors + smaller windows is the proof-of-principle setup.
    smc_cfg = SMCConfig(
        n_smc_particles=args.n_smc,
        n_pf_particles=args.n_pf,
        target_ess_frac=0.3,              # allow larger Δλ per level
        max_lambda_inc=0.10,
        max_lambda_inc_bridge=0.15,
        bridge_type=args.bridge,
        bridge_mog_components=args.bridge_K,
    )
    WINDOW_BINS = 1 * BINS_PER_DAY        # 96 bins = 1 day
    STRIDE_BINS = BINS_PER_DAY // 2       # 48 bins = 12 hours
    rolling_cfg = RollingConfig(
        window_days=WINDOW_BINS, stride_days=STRIDE_BINS, dt=DT,
        n_substeps=N_SUBSTEPS, max_windows=args.windows,
    )

    print("=" * 70)
    print(f"  ROLLING WINDOW SMC² — FSA HIGH-RES — C0  "
          f"(seed={args.seed}, N={args.n_smc}, K={args.n_pf})")
    print("=" * 70)
    print(f"  Params:       {model.n_dim}")
    print(f"  Grid:         dt={DT:.5f} day = {DT*24*60:.0f} min")
    print(f"  Simulation:   {N_DAYS_TOTAL} days × {BINS_PER_DAY} bins/day = "
          f"{N_DAYS_TOTAL * BINS_PER_DAY} total bins")
    print(f"  Windows:      {rolling_cfg.window_days} bins "
          f"({rolling_cfg.window_days // BINS_PER_DAY}d), "
          f"stride {rolling_cfg.stride_days} bins "
          f"({rolling_cfg.stride_days // BINS_PER_DAY}d)")
    if args.windows is not None:
        print(f"  Max windows:  {args.windows}")
    print(f"  Device:       "
          f"{'GPU' if jax.devices()[0].platform == 'gpu' else 'CPU'}")
    print()

    if args.scenario_artifact:
        # Steps 1-4 (inline data generation) replaced by loading a
        # validated, packaged scenario artifact from the
        # Python-Model-Scenario-Simulation repo.
        from drivers._artifact_loader import load_scenario
        print(f"Steps 1-4 [SKIPPED]: loading scenario artifact")
        print(f"  artifact: {args.scenario_artifact}")
        bundle = load_scenario(args.scenario_artifact)
        print(f"  model:    {bundle['model_name']}/{bundle['model_version']}")
        print(f"  scenario: {bundle['scenario_name']}")
        print(f"  bins:     {bundle['n_bins_total']} "
              f"({bundle['bins_per_day']}/day)  seed={bundle['seed']}")
        T_B_arr = np.asarray(bundle['T_B_arr'], dtype=np.float32)
        Phi_arr = np.asarray(bundle['Phi_arr'], dtype=np.float32)
        trajectory = np.asarray(bundle['trajectory'], dtype=np.float64)
        obs_data = bundle['obs_data']
        # Reconcile truth with the artifact's manifest (sim/est should agree
        # on truth here; inline-path truth was DEFAULT_PARAMS).
        true_params = dict(bundle['truth_params'])
        truth = _truth_dict(true_params)
        print(f"  B range:  [{trajectory[:,0].min():.3f}, "
              f"{trajectory[:,0].max():.3f}]")
        print(f"  F range:  [{trajectory[:,1].min():.3f}, "
              f"{trajectory[:,1].max():.3f}]")
        print(f"  A range:  [{trajectory[:,2].min():.3f}, "
              f"{trajectory[:,2].max():.3f}]")
        for ch_name in ('obs_HR', 'obs_sleep', 'obs_stress', 'obs_steps'):
            n = len(obs_data[ch_name].get('t_idx', []))
            print(f"    {ch_name}: {n} obs")
        traj_csv = os.path.join(out_dir, 'trajectory.csv')
        np.savetxt(traj_csv,
                   np.column_stack([np.arange(trajectory.shape[0]) * DT_BIN_DAYS,
                                    trajectory]),
                   delimiter=',', header='t_days,B,F,A', comments='')
        print(f"  -> {traj_csv}")
    else:
        # Step 1: Macrocycle → daily Phi
        print("Step 1: Generate daily macrocycle schedule (C0) then expand to 15-min bins")
        daily_T_B, daily_Phi = generate_macrocycle_C0(N_DAYS_TOTAL, seed=args.seed)
        print(f"  daily T_B range: [{daily_T_B.min():.3f}, {daily_T_B.max():.3f}]")
        print(f"  daily Phi range: [{daily_Phi.min():.4f}, {daily_Phi.max():.4f}]")
        Phi_arr = generate_phi_sub_daily(daily_Phi, seed=args.seed)
        T_B_arr = np.repeat(daily_T_B, BINS_PER_DAY).astype(np.float32)
        print(f"  expanded Phi_arr: shape {Phi_arr.shape}, "
              f"range [{Phi_arr.min():.4f}, {Phi_arr.max():.4f}]")

        # Step 2: Forward simulate
        print(f"\nStep 2: Forward simulate ({N_SUBSTEPS} substeps/bin)")
        init = {'B_0': 0.05, 'F_0': 0.10, 'A_0': 0.55}
        t0 = time.time()
        trajectory = simulate_sde_high_res(T_B_arr, Phi_arr, true_params, init,
                                            n_substeps=N_SUBSTEPS, seed=args.seed)
        print(f"  B range: [{trajectory[:,0].min():.3f}, {trajectory[:,0].max():.3f}]")
        print(f"  F range: [{trajectory[:,1].min():.3f}, {trajectory[:,1].max():.3f}]")
        print(f"  A range: [{trajectory[:,2].min():.3f}, {trajectory[:,2].max():.3f}]")
        print(f"  sim time: {time.time()-t0:.1f}s")

        # Save trajectory
        traj_csv = os.path.join(out_dir, 'trajectory.csv')
        np.savetxt(traj_csv,
                   np.column_stack([np.arange(trajectory.shape[0]) * DT_BIN_DAYS,
                                    trajectory]),
                   delimiter=',', header='t_days,B,F,A', comments='')
        print(f"  -> {traj_csv}")

        # Step 3: Observations
        print("\nStep 3: Generate observations (4 channels)")
        obs_data = generate_observations(trajectory, T_B_arr, Phi_arr,
                                         true_params, seed=args.seed + 100)

        # Step 4: Missing data
        print("\nStep 4: Apply missing data (5% dropout on HR/stress/steps)")
        obs_data = apply_missing_data_high_res(obs_data, seed=args.seed + 200)
        for ch_name in ('obs_HR', 'obs_sleep', 'obs_stress', 'obs_steps'):
            n = (len(obs_data[ch_name]['t_idx']) if 't_idx' in obs_data[ch_name] else 0)
            print(f"    {ch_name}: {n} obs")

    # Step 5: Diagnostic plots
    print("\nStep 5: Diagnostic plots")
    plot_high_res_channels(obs_data, trajectory, T_B_arr, Phi_arr,
                            true_params, out_dir)

    if args.sim_only:
        print(f"\nDone (sim-only). Plots in {out_dir}/")
        return 0

    # Step 6: Rolling SMC²
    print("\nStep 6: Rolling window SMC²")
    cold_init = jnp.asarray(COLD_START_INIT)
    n_bins_total = trajectory.shape[0]
    results, T_arr = rolling_window_smc(
        obs_data, model, n_bins_total, out_dir,
        smc_cfg=smc_cfg, rolling_cfg=rolling_cfg,
        cold_start_init=cold_init, truth=truth,
        obs_channel_names=OBS_CHANNEL_NAMES,
        seed=args.seed,
    )

    # Step 7: Validation plots
    print("\nStep 7: Validation plots")
    plot_parameter_tracking(results, model, truth, out_dir)
    plot_coverage_and_timing(results, out_dir)

    # Summary
    coverages = [r['coverage'] for r in results]
    coverages_informed = [r['coverage_informed'] for r in results
                          if r['coverage_informed'] == r['coverage_informed']]
    total_time = sum(r['elapsed_s'] for r in results)
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Windows:              {len(results)}")
    print(f"  Mean coverage (raw):  {np.mean(coverages)*100:.1f}%")
    if coverages_informed:
        print(f"  Mean coverage (inf):  {np.mean(coverages_informed)*100:.1f}%")
    print(f"  PASS (>=70%):         {sum(1 for c in coverages if c >= 0.7)}/{len(results)}")
    print(f"  Total SMC time:       {total_time/3600:.2f}h")
    print(f"  Output:               {out_dir}/")
    print(f"{'='*70}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
