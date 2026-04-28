"""SMC² rolling-window estimation for glucose_insulin (Bergman minimal model).

Consumes a packaged scenario artifact from psim and runs SMC² rolling-
window inference. ALL glucose_insulin-specific choices (window size,
particle counts, bridge type, channel names) live in
``drivers/glucose_insulin/config.py`` as a frozen dataclass.

Default bridge: SF Path B-fixed per outputs/SF_BEST_PRACTICE_3_models.md.
glucose_insulin is the 4th independent model exercising the recommendation.

Run via:
  PYTHONPATH=. python -m drivers.glucose_insulin.rolling --seed 42 --windows 1
  PYTHONPATH=. python -m drivers.glucose_insulin.rolling --seed 42                 # full
  PYTHONPATH=. python -m drivers.glucose_insulin.rolling --seed 42 --bridge gaussian
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


# ─── public-dev glucose_insulin model lives via the namespace-package merge ──
_PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)
if not os.path.isdir(_PUBLIC_DEV_V1):
    raise SystemExit(
        f"glucose_insulin lives in the public dev repo at {_PUBLIC_DEV_V1}. "
        f"Clone https://github.com/ajaytalati/Python-Model-Development-Simulation "
        f"there or set _PUBLIC_DEV_V1."
    )
if _PUBLIC_DEV_V1 not in sys.path:
    sys.path.append(_PUBLIC_DEV_V1)


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

from drivers._artifact_loader import load_scenario
from drivers.glucose_insulin.config import (
    GI_SET_A_CONFIG, GI_SET_B_CONFIG, GI_SET_C_CONFIG, GI_SET_D_CONFIG,
    GiRollingConfig,
)
from drivers.glucose_insulin.plots import plot_glucose_insulin_channels

from models.glucose_insulin.estimation import (
    make_glucose_insulin_estimation,
    INIT_STATE_PRIOR_CONFIG,
)
from models.glucose_insulin.simulation import (
    INIT_STATE_A, INIT_STATE_B, INIT_STATE_C, INIT_STATE_D,
    _meal_schedule, _insulin_schedule,
)


_INIT_STATES_BY_SET = {
    'set_A_healthy':              INIT_STATE_A,
    'set_B_insulin_resistance':   INIT_STATE_B,
    'set_C_t1d_no_insulin':       INIT_STATE_C,
    'set_D_t1d_open_loop':        INIT_STATE_D,
}


# ═════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════

def _prior_mean(ptype: str, pargs: tuple) -> float:
    if ptype == 'lognormal':
        return math.exp(pargs[0] + pargs[1] ** 2 / 2)
    if ptype == 'normal':
        return float(pargs[0])
    if ptype == 'beta':
        a, b = pargs
        return float(a) / float(a + b)
    return 0.0


def _cold_start_init(estimation_model) -> jnp.ndarray:
    """2-D init [G_0, I_0] from prior means (X_0 derived = 0)."""
    init_cfg = estimation_model.init_state_prior_config
    return jnp.array([
        _prior_mean(*init_cfg['G_0']),
        _prior_mean(*init_cfg['I_0']),
    ], dtype=jnp.float32)


def _truth_dict(artifact_truth_params: dict, cfg: GiRollingConfig) -> dict:
    """Build the truth dict for coverage. Sources: artifact + per-set INIT_STATE."""
    d = dict(artifact_truth_params)
    init = _INIT_STATES_BY_SET.get(cfg.scenario_name, INIT_STATE_A)
    d.update(init)
    # Drop metadata + frozen / scheduled values that aren't estimated.
    for k in ('dt_hours', 't_total_hours', 'Ib', 'V_G', 'V_I', 'BW',
              'T_X', 'T_I', 'n_beta', 'h_beta', 'meal_carbs_g',
              'schedule_seed', 'insulin_schedule_active',
              'insulin_carb_ratio', 'basal_rate_U_hr', 'X_0'):
        d.pop(k, None)
    for k in cfg.frozen_param_keys:
        d.pop(k, None)
    return d


def _build_meal_schedule_from_truth(truth_params: dict) -> list:
    """Re-create the canonical meal schedule from the truth params."""
    seed = int(truth_params.get('schedule_seed', 0))
    n_days = int(round(truth_params['t_total_hours'] / 24.0))
    meal_carbs_g = float(truth_params.get('meal_carbs_g', 40.0))
    return _meal_schedule(seed, n_days, meal_carbs_g)


def _build_insulin_schedule_from_truth(truth_params: dict, meal_schedule):
    """Re-create the insulin schedule (Set D only) from truth params."""
    if not truth_params.get('insulin_schedule_active', False):
        return None
    V_I_BW = truth_params['V_I'] * truth_params['BW']
    return _insulin_schedule(
        seed=int(truth_params.get('schedule_seed', 0)),
        n_days=int(round(truth_params['t_total_hours'] / 24.0)),
        meal_schedule=meal_schedule,
        insulin_carb_ratio=truth_params.get('insulin_carb_ratio', 10.0),
        basal_rate_U_hr=truth_params.get('basal_rate_U_hr', 0.5),
        V_I_BW=V_I_BW,
    )


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

_CONFIGS = {
    'A': GI_SET_A_CONFIG,
    'B': GI_SET_B_CONFIG,
    'C': GI_SET_C_CONFIG,
    'D': GI_SET_D_CONFIG,
}


def _parse_args(cfg: GiRollingConfig):
    p = argparse.ArgumentParser(
        description="SMC² rolling-window estimation for glucose_insulin (Bergman model).")
    p.add_argument('--set', choices=tuple(_CONFIGS), default='A',
                   help='Scenario set: A (healthy paper-parity), '
                        'B (insulin resistance), C (T1D no-control), '
                        'D (T1D + open-loop insulin)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-smc', type=int, default=cfg.n_smc)
    p.add_argument('--n-pf', type=int, default=cfg.n_pf)
    p.add_argument('--windows', type=int, default=None,
                   help='Max windows (None = full)')
    p.add_argument('--scenario-artifact', default=None,
                   help='Path to a psim artifact dir (default: per-scenario)')
    p.add_argument('--bridge', choices=('gaussian', 'mog', 'schrodinger_follmer'),
                   default=cfg.bridge_type)
    p.add_argument('--bridge-K', type=int, default=cfg.bridge_mog_components)
    p.add_argument('--sf-blend', type=float, default=cfg.sf_blend)
    p.add_argument('--sf-entropy-reg', type=float, default=cfg.sf_entropy_reg)
    p.add_argument('--sf-q1-mode', choices=('is', 'annealed'),
                   default=cfg.sf_q1_mode)
    p.add_argument('--sf-annealed-n-stages', type=int,
                   default=cfg.sf_annealed_n_stages)
    p.add_argument('--sf-annealed-n-mh-steps', type=int,
                   default=cfg.sf_annealed_n_mh_steps)
    p.add_argument('--sf-annealed-proposal-scale', type=float,
                   default=cfg.sf_annealed_proposal_scale)
    p.add_argument('--sf-use-q0-cov', action='store_true',
                   default=cfg.sf_use_q0_cov)
    p.add_argument('--show-checkpoint', action='store_true')
    return p.parse_args()


def _out_dir(seed: int, n_smc: int, scenario_name: str,
             bridge_tag: str = '') -> str:
    tag = f'_{bridge_tag}' if bridge_tag else ''
    return os.path.join('outputs', 'glucose_insulin_rolling',
                        f'{scenario_name}_N{n_smc}_s{seed}{tag}')


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--set', default='A')
    pre_args, _ = pre_parser.parse_known_args()
    cfg = _CONFIGS[pre_args.set]

    args = _parse_args(cfg)
    if args.bridge == 'gaussian':
        bridge_tag = 'gauss'
    elif args.bridge == 'mog':
        bridge_tag = f'mog{args.bridge_K}'
    elif args.bridge == 'schrodinger_follmer':
        mode_tag = 'a' if args.sf_q1_mode == 'annealed' else 'i'
        cov_tag = 'q' if args.sf_use_q0_cov else ''
        bridge_tag = f'sf{mode_tag}{cov_tag}{args.sf_blend:.2f}'
    else:
        bridge_tag = args.bridge
    out_dir = _out_dir(args.seed, args.n_smc, cfg.scenario_name, bridge_tag)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'driver_config.json'), 'w') as f:
        json.dump({'config': cfg.to_dict(), 'cli': vars(args)}, f, indent=2)

    if args.show_checkpoint:
        _show_checkpoint(os.path.join(out_dir, 'rolling_checkpoint.json'))
        return 0

    # Load artifact
    artifact = os.path.expanduser(args.scenario_artifact or cfg.default_artifact_dir)
    bundle = load_scenario(artifact)

    print("=" * 70)
    print(f"  SMC² ROLLING — glucose_insulin — {cfg.scenario_name}  "
          f"(seed={args.seed}, N={args.n_smc}, K={args.n_pf})")
    print("=" * 70)
    print(f"  Artifact: {artifact}")
    print(f"  Bins:     {bundle['n_bins_total']} ({bundle['bins_per_day']}/d)")
    print(f"  Window:   {cfg.window_bins} bins "
          f"(= {cfg.window_bins / cfg.bins_per_day * 24:.1f} hours)")
    print(f"  Stride:   {cfg.stride_bins} bins")
    print(f"  Bridge:   {args.bridge}"
          + (f"  (q1={args.sf_q1_mode}, q0_cov={args.sf_use_q0_cov}, "
             f"blend={args.sf_blend}, n_mh={args.sf_annealed_n_mh_steps})"
             if args.bridge == 'schrodinger_follmer' else ""))
    print(f"  Device:   {'GPU' if jax.devices()[0].platform == 'gpu' else 'CPU'}")
    print()

    if bundle['n_bins_total'] != cfg.n_bins_total:
        raise RuntimeError(
            f"Artifact bins {bundle['n_bins_total']} != cfg "
            f"{cfg.n_bins_total}. Check artifact provenance.")

    # Re-build meal & insulin schedules from artifact truth params
    truth_params = bundle['truth_params']
    meal_schedule = _build_meal_schedule_from_truth(truth_params)
    insulin_schedule = _build_insulin_schedule_from_truth(truth_params, meal_schedule)

    # Build EstimationModel with this scenario's frozen params + schedules
    frozen_overrides = {
        'Ib':     float(truth_params.get('Ib', 7.0)),
        'n_beta': float(truth_params.get('n_beta', 8.0)),
        'h_beta': float(truth_params.get('h_beta', 90.0)),
    }
    gi_estimation = make_glucose_insulin_estimation(
        meal_schedule=meal_schedule,
        insulin_schedule=insulin_schedule,
        frozen_params=frozen_overrides,
    )

    smc_cfg = SMCConfig(
        n_smc_particles=args.n_smc, n_pf_particles=args.n_pf,
        target_ess_frac=cfg.target_ess_frac,
        max_lambda_inc=cfg.max_lambda_inc,
        max_lambda_inc_bridge=cfg.max_lambda_inc_bridge,
        bridge_type=args.bridge,
        bridge_mog_components=args.bridge_K,
        sf_blend=args.sf_blend,
        sf_entropy_reg=args.sf_entropy_reg,
        sf_q1_mode=args.sf_q1_mode,
        sf_annealed_n_stages=args.sf_annealed_n_stages,
        sf_annealed_n_mh_steps=args.sf_annealed_n_mh_steps,
        sf_annealed_proposal_scale=args.sf_annealed_proposal_scale,
        sf_use_q0_cov=args.sf_use_q0_cov,
    )
    rolling_cfg = RollingConfig(
        window_days=cfg.window_bins,
        stride_days=cfg.stride_bins,
        dt=cfg.dt_hours,
        n_substeps=cfg.n_substeps,
        max_windows=args.windows,
    )

    truth = _truth_dict(truth_params, cfg)

    # Diagnostic plot of inputs
    print("Step 1: Plot input artifact (4-panel diagnostic)")
    path = plot_glucose_insulin_channels(bundle, out_dir)
    print(f"  -> {path}")

    # Rolling SMC²
    print(f"\nStep 2: Rolling SMC² ({cfg.n_bins_total} bins, "
          f"{cfg.window_bins}-bin windows, {cfg.stride_bins}-bin stride)")
    cold_init = _cold_start_init(gi_estimation)
    t0 = time.time()
    results, _T_arr = rolling_window_smc(
        bundle['obs_data'],
        gi_estimation,
        bundle['n_bins_total'],
        out_dir,
        smc_cfg=smc_cfg,
        rolling_cfg=rolling_cfg,
        cold_start_init=cold_init,
        truth=truth,
        obs_channel_names=cfg.obs_channel_names,
        seed=args.seed,
    )

    print("\nStep 3: Validation plots")
    plot_parameter_tracking(results, gi_estimation, truth, out_dir)
    plot_coverage_and_timing(results, out_dir)

    coverages = [r['coverage'] for r in results]
    coverages_informed = [r['coverage_informed'] for r in results
                          if r['coverage_informed'] == r['coverage_informed']]
    total_time = sum(r['elapsed_s'] for r in results)
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Windows:              {len(results)}")
    print(f"  Mean coverage (raw):  {np.mean(coverages) * 100:.1f}%")
    if coverages_informed:
        print(f"  Mean coverage (inf):  {np.mean(coverages_informed) * 100:.1f}%")
    print(f"  PASS (>=70%):         "
          f"{sum(c >= 0.7 for c in coverages)}/{len(results)}")
    print(f"  Total SMC time:       {total_time / 3600:.2f}h")
    print(f"  Output:               {out_dir}/")
    print(f"{'=' * 70}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
