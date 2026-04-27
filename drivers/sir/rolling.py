"""SMC² rolling-window estimation for SIR.

Consumes a packaged scenario artifact from the
Python-Model-Scenario-Simulation repo and runs the SMC²-style rolling
inference. ALL SIR-specific choices (window size, particle counts,
bridge type, channel names, frozen N) live in
``drivers/sir/config.py`` as a frozen dataclass — the single source
of truth for reproducibility. The dataclass is dumped to each output
dir's ``driver_config.json`` so any run can be reproduced from its
saved config.

Default bridge: SF Path B-fixed per outputs/SF_BEST_PRACTICE_2_models.md.
Override with ``--bridge gaussian`` for the regression-baseline run.

Run via:
  PYTHONPATH=. python -m drivers.sir.rolling --seed 42 --windows 1
  PYTHONPATH=. python -m drivers.sir.rolling --seed 42                 # full
  PYTHONPATH=. python -m drivers.sir.rolling --seed 42 --bridge gaussian
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np


# ─── public-dev SIR model lives via the namespace-package merge ──────
_PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)
if not os.path.isdir(_PUBLIC_DEV_V1):
    raise SystemExit(
        f"SIR model lives in the public dev repo at {_PUBLIC_DEV_V1}. "
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
from drivers.sir.config import (
    SIR_SET_A_CONFIG, SIR_SET_B_CONFIG, SIR_SET_C_CONFIG, SIR_SET_D_CONFIG,
    SirRollingConfig,
)
from drivers.sir.plots import plot_sir_channels

from models.sir.estimation import (
    make_sir_estimation,
    INIT_STATE_PRIOR_CONFIG,
    PARAM_PRIOR_OVERRIDES_B, PARAM_PRIOR_OVERRIDES_C, PARAM_PRIOR_OVERRIDES_D,
)
from models.sir.simulation import (
    INIT_STATE_A, INIT_STATE_B, INIT_STATE_C, INIT_STATE_D,
)

_INIT_STATES_BY_SET = {
    'set_A_boarding_school': INIT_STATE_A,
    'set_B_small_outbreak':  INIT_STATE_B,
    'set_C_large_outbreak':  INIT_STATE_C,
    'set_D_vaccination':     INIT_STATE_D,
}

# Per-scenario prior overrides — Set A uses the canonical (Set-A-centered)
# defaults, B/C/D shift the rate/diffusion priors to track each scenario's
# truth. Without these, Set C inference collapses on the prior-truth gap.
_PRIOR_OVERRIDES_BY_SET = {
    'set_A_boarding_school': None,
    'set_B_small_outbreak':  PARAM_PRIOR_OVERRIDES_B,
    'set_C_large_outbreak':  PARAM_PRIOR_OVERRIDES_C,
    'set_D_vaccination':     PARAM_PRIOR_OVERRIDES_D,
}


# ═════════════════════════════════════════════════════════════════════════
# Truth & cold-start init helpers
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


def _cold_start_init() -> jnp.ndarray:
    """1-D init [I_0] from prior mean.

    The 2-D state is built by SIR_ESTIMATION.make_init_state_fn which
    derives S_0 = N - I_0 from the frozen N. Inner PF only needs I_0.
    """
    return jnp.array([
        _prior_mean(*INIT_STATE_PRIOR_CONFIG['I_0']),
    ], dtype=jnp.float32)


def _truth_dict(artifact_truth_params: dict, cfg: SirRollingConfig) -> dict:
    """Build the truth dict for coverage matching the est-side estimable names.

    Sources:
      - artifact's truth_params: 6 estimable params + frozen (N, v) + metadata
      - INIT_STATE_<set>: I_0 (not in the manifest's truth_params)
      - Filter: keep only PARAM_PRIOR_CONFIG + INIT_STATE_PRIOR_CONFIG keys
    """
    d = dict(artifact_truth_params)
    # Merge in the init_state from the model package (psim manifest doesn't
    # include I_0 in truth_params; it's stored separately).
    init = _INIT_STATES_BY_SET.get(cfg.scenario_name, INIT_STATE_A)
    d.update(init)
    # Drop metadata + frozen.
    for k in ('dt_hours', 't_total_hours', 'N', 'v', 'R_0_init'):
        d.pop(k, None)
    for k in cfg.frozen_param_keys:
        d.pop(k, None)
    return d


# ═════════════════════════════════════════════════════════════════════════
# CLI + output dir
# ═════════════════════════════════════════════════════════════════════════

_CONFIGS = {
    'A': SIR_SET_A_CONFIG,
    'B': SIR_SET_B_CONFIG,
    'C': SIR_SET_C_CONFIG,
    'D': SIR_SET_D_CONFIG,
}


def _parse_args(cfg: SirRollingConfig):
    """CLI defaults inherit from the dataclass."""
    p = argparse.ArgumentParser(
        description="SMC² rolling-window estimation for SIR (Set A default).")
    p.add_argument('--set', choices=tuple(_CONFIGS), default='A',
                   help='Scenario set: A (boarding-school, paper-parity), '
                        'B (small), C (large), D (vaccination)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-smc', type=int, default=cfg.n_smc)
    p.add_argument('--n-pf', type=int, default=cfg.n_pf)
    p.add_argument('--windows', type=int, default=None,
                   help='Max windows (None = full)')
    p.add_argument('--scenario-artifact', default=None,
                   help='Path to a Python-Model-Scenario-Simulation artifact dir '
                        '(default: per-scenario default)')
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
    return os.path.join('outputs', 'sir_rolling',
                        f'{scenario_name}_N{n_smc}_s{seed}{tag}')


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

def main():
    # Pre-parse just the --set arg so the rest of CLI defaults match its config
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

    # 0. Save the config + CLI args for reproducibility.
    with open(os.path.join(out_dir, 'driver_config.json'), 'w') as f:
        json.dump({'config': cfg.to_dict(), 'cli': vars(args)}, f, indent=2)

    if args.show_checkpoint:
        _show_checkpoint(os.path.join(out_dir, 'rolling_checkpoint.json'))
        return 0

    # 1. Load artifact
    artifact = os.path.expanduser(args.scenario_artifact or cfg.default_artifact_dir)
    bundle = load_scenario(artifact)

    print("=" * 70)
    print(f"  SMC² ROLLING — SIR — {cfg.scenario_name}  "
          f"(seed={args.seed}, N={args.n_smc}, K={args.n_pf})")
    print("=" * 70)
    print(f"  Artifact: {artifact}")
    print(f"  Model:    {bundle['model_name']}/{bundle['model_version']}")
    print(f"  Scenario: {bundle['scenario_name']}")
    print(f"  Bins:     {bundle['n_bins_total']} ({bundle['bins_per_day']}/d)")
    print(f"  Window:   {cfg.window_bins} bins "
          f"(= {cfg.window_bins // cfg.bins_per_day} day)")
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
            f"{cfg.n_bins_total} (n_days={cfg.n_days} × "
            f"bins_per_day={cfg.bins_per_day}). Check artifact provenance.")

    # 2. Scenario-specific frozen params (N varies per set) + prior overrides
    truth_params = bundle['truth_params']
    frozen_overrides = {
        'N': float(truth_params.get('N', cfg.population_N)),
        'v': float(truth_params.get('v', 0.0)),
    }
    prior_overrides = _PRIOR_OVERRIDES_BY_SET.get(cfg.scenario_name)
    sir_estimation = make_sir_estimation(
        frozen_params=frozen_overrides,
        param_prior_overrides=prior_overrides,
    )

    # 3. Build SMC + rolling configs
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

    # 4. Truth dict
    truth = _truth_dict(truth_params, cfg)

    # 5. Diagnostic plot of inputs
    print("Step 1: Plot input artifact (3-panel SIR diagnostic)")
    path = plot_sir_channels(bundle, out_dir)
    print(f"  -> {path}")

    # 6. Rolling SMC²
    print(f"\nStep 2: Rolling SMC² ({cfg.n_bins_total} bins, "
          f"{cfg.window_bins}-bin windows, {cfg.stride_bins}-bin stride)")
    cold_init = _cold_start_init()
    t0 = time.time()
    results, _T_arr = rolling_window_smc(
        bundle['obs_data'],
        sir_estimation,
        bundle['n_bins_total'],
        out_dir,
        smc_cfg=smc_cfg,
        rolling_cfg=rolling_cfg,
        cold_start_init=cold_init,
        truth=truth,
        obs_channel_names=cfg.obs_channel_names,
        seed=args.seed,
    )

    # 7. Validation plots
    print("\nStep 3: Validation plots")
    plot_parameter_tracking(results, sir_estimation, truth, out_dir)
    plot_coverage_and_timing(results, out_dir)

    # 8. Summary
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
