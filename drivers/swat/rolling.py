"""SMC² rolling-window estimation for SWAT.

Consumes a packaged scenario artifact from the
Python-Model-Scenario-Simulation repo and runs the SMC²-style rolling
inference. ALL SWAT-specific choices (window size, particle counts,
bridge type, channel names, frozen params) live in
``drivers/swat/config.py`` as a frozen dataclass — the single source
of truth for reproducibility. The dataclass is dumped to each output
dir's ``driver_config.json`` so any run can be reproduced from its
saved config.

Run via:
  PYTHONPATH=. python -m drivers.swat.rolling --seed 42 --windows 1
  PYTHONPATH=. python -m drivers.swat.rolling --seed 42                 # full
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


# ─── public-dev SWAT model lives via the namespace-package merge ─────
# Same pattern as drivers/fsa_high_res_rolling.py — SMC² root must be
# first on sys.path so models.fsa_real_obs (SMC²-edited) resolves
# locally; models.swat falls through to the public dev copy.
_PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)
if not os.path.isdir(_PUBLIC_DEV_V1):
    raise SystemExit(
        f"SWAT model lives in the public dev repo at {_PUBLIC_DEV_V1}. "
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
from drivers.swat.config import SWAT_SET_A_CONFIG, SwatRollingConfig
from drivers.swat.plots import plot_swat_channels

from models.swat.estimation import (
    SWAT_ESTIMATION,
    INIT_STATE_PRIOR_CONFIG,
)
from models.swat.simulation import INIT_STATE_A   # for Vh, Vn truth values


# ═════════════════════════════════════════════════════════════════════════
# Truth & cold-start init helpers
# ═════════════════════════════════════════════════════════════════════════

def _prior_mean(ptype: str, pargs: tuple) -> float:
    """Mean of the supported prior families (matches estimation.py)."""
    if ptype == 'lognormal':
        return math.exp(pargs[0] + pargs[1] ** 2 / 2)
    if ptype == 'normal':
        return float(pargs[0])
    if ptype == 'beta':
        a, b = pargs
        return float(a) / float(a + b)
    return 0.0


def _cold_start_init() -> jnp.ndarray:
    """4-D init [W_0, Zt_0, a_0, T_0] from prior means.

    The 7-D state is built by SWAT_ESTIMATION.shard_init_fn (which
    pulls Vh, Vn from params and computes C(0) analytically), so
    the inner PF only needs the 4 estimable init values.
    """
    return jnp.array([
        _prior_mean(*INIT_STATE_PRIOR_CONFIG['W_0']),
        _prior_mean(*INIT_STATE_PRIOR_CONFIG['Zt_0']),
        _prior_mean(*INIT_STATE_PRIOR_CONFIG['a_0']),
        _prior_mean(*INIT_STATE_PRIOR_CONFIG['T_0']),
    ], dtype=jnp.float32)


def _truth_dict(artifact_truth_params: dict, cfg: SwatRollingConfig) -> dict:
    """Build the truth dict for coverage, matching the est-side all_names.

    Sources:
      - PARAM_SET (artifact's truth_params): 31 estimable + 2 metadata
      - INIT_STATE (model package's INIT_STATE_A): 4 init + Vh, Vn
      - Filter: keep only keys in SWAT_ESTIMATION.all_names (35 entries)
    """
    d = dict(artifact_truth_params)
    d.update(INIT_STATE_A)               # adds W_0, Zt_0, a_0, T_0, Vh, Vn
    d.pop('dt_hours', None)              # metadata, not estimated
    d.pop('t_total_hours', None)
    for k in cfg.frozen_param_keys:
        d.pop(k, None)
    return d


# ═════════════════════════════════════════════════════════════════════════
# CLI + output dir
# ═════════════════════════════════════════════════════════════════════════

def _parse_args(cfg: SwatRollingConfig):
    """CLI defaults inherit from the dataclass."""
    p = argparse.ArgumentParser(
        description="SMC² rolling-window estimation for SWAT (Set A baseline).")
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-smc', type=int, default=cfg.n_smc)
    p.add_argument('--n-pf', type=int, default=cfg.n_pf)
    p.add_argument('--windows', type=int, default=None,
                   help='Max windows (None = full)')
    p.add_argument('--scenario-artifact', default=cfg.default_artifact_dir,
                   help='Path to a Python-Model-Scenario-Simulation artifact dir')
    p.add_argument('--bridge', choices=('gaussian', 'mog', 'schrodinger_follmer'),
                   default=cfg.bridge_type)
    p.add_argument('--bridge-K', type=int, default=cfg.bridge_mog_components)
    p.add_argument('--sf-blend', type=float, default=cfg.sf_blend,
                   help='Schrödinger-Föllmer t in [0, 1] along BW geodesic')
    p.add_argument('--sf-entropy-reg', type=float, default=cfg.sf_entropy_reg,
                   help='Schrödinger entropic regularisation; 0 = exact OT')
    p.add_argument('--sf-q1-mode', choices=('is', 'annealed'),
                   default=cfg.sf_q1_mode,
                   help='SF q1 estimator: is = Path A (single IS), '
                        'annealed = Path B (K-stage tempered SMC)')
    p.add_argument('--sf-annealed-n-stages', type=int,
                   default=cfg.sf_annealed_n_stages)
    p.add_argument('--sf-annealed-n-mh-steps', type=int,
                   default=cfg.sf_annealed_n_mh_steps)
    p.add_argument('--sf-annealed-proposal-scale', type=float,
                   default=cfg.sf_annealed_proposal_scale)
    p.add_argument('--sf-use-q0-cov', action='store_true',
                   default=cfg.sf_use_q0_cov,
                   help='Decoupled SF (issue #3 fix 2): bridge mean from BW interp, '
                        'cov from q0. Recommended with --sf-q1-mode annealed.')
    p.add_argument('--show-checkpoint', action='store_true')
    return p.parse_args()


def _out_dir(seed: int, n_smc: int, scenario_name: str,
             bridge_tag: str = '') -> str:
    tag = f'_{bridge_tag}' if bridge_tag else ''
    return os.path.join('outputs', 'swat_rolling',
                        f'{scenario_name}_N{n_smc}_s{seed}{tag}')


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    cfg = SWAT_SET_A_CONFIG
    args = _parse_args(cfg)
    if args.bridge == 'gaussian':
        bridge_tag = ''
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

    # 0. Save the config + CLI args for reproducibility.
    with open(os.path.join(out_dir, 'driver_config.json'), 'w') as f:
        json.dump({'config': cfg.to_dict(), 'cli': vars(args)}, f, indent=2)

    if args.show_checkpoint:
        _show_checkpoint(os.path.join(out_dir, 'rolling_checkpoint.json'))
        return 0

    # 1. Load artifact (no inline data gen — artifact is canonical)
    artifact = os.path.expanduser(args.scenario_artifact)
    bundle = load_scenario(artifact)

    print("=" * 70)
    print(f"  SMC² ROLLING — SWAT — {cfg.scenario_name}  "
          f"(seed={args.seed}, N={args.n_smc}, K={args.n_pf})")
    print("=" * 70)
    print(f"  Artifact: {artifact}")
    print(f"  Model:    {bundle['model_name']}/{bundle['model_version']}")
    print(f"  Scenario: {bundle['scenario_name']}")
    print(f"  Bins:     {bundle['n_bins_total']} ({bundle['bins_per_day']}/d)")
    print(f"  Window:   {cfg.window_bins} bins (= {cfg.window_bins // cfg.bins_per_day} day)")
    print(f"  Stride:   {cfg.stride_bins} bins")
    print(f"  Device:   {'GPU' if jax.devices()[0].platform == 'gpu' else 'CPU'}")
    print()

    if bundle['n_bins_total'] != cfg.n_bins_total:
        raise RuntimeError(
            f"Artifact bins {bundle['n_bins_total']} != cfg "
            f"{cfg.n_bins_total} (n_days={cfg.n_days} × "
            f"bins_per_day={cfg.bins_per_day}). Check artifact provenance.")

    # 2. Build SMC + rolling configs from dataclass + CLI overrides
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
        window_days=cfg.window_bins,    # (framework reads as bins; "days" is misnomer)
        stride_days=cfg.stride_bins,
        dt=cfg.dt_hours,
        n_substeps=cfg.n_substeps,
        max_windows=args.windows,
    )

    # 3. Truth dict (covers 35 estimable scalars: 31 params + 4 init + Vh/Vn merged)
    truth = _truth_dict(bundle['truth_params'], cfg)

    # 4. Diagnostic plot of inputs (SWAT-specific 4-channel)
    print("Step 1: Plot input artifact (4-channel diagnostic)")
    path = plot_swat_channels(bundle, out_dir, n_show_days=3)
    print(f"  -> {path}")

    # 5. Rolling SMC²
    print(f"\nStep 2: Rolling SMC² ({cfg.n_bins_total} bins, "
          f"{cfg.window_bins}-bin windows, {cfg.stride_bins}-bin stride)")
    cold_init = _cold_start_init()
    t0 = time.time()
    results, _T_arr = rolling_window_smc(
        bundle['obs_data'],
        SWAT_ESTIMATION,
        bundle['n_bins_total'],
        out_dir,
        smc_cfg=smc_cfg,
        rolling_cfg=rolling_cfg,
        cold_start_init=cold_init,
        truth=truth,
        obs_channel_names=cfg.obs_channel_names,
        seed=args.seed,
    )

    # 6. Result plots (generic; reused from smc2bj/plotting/rolling.py)
    print("\nStep 3: Validation plots")
    plot_parameter_tracking(results, SWAT_ESTIMATION, truth, out_dir)
    plot_coverage_and_timing(results, out_dir)

    # 7. Summary
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
