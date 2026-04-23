"""Rolling-window SMC² orchestration.

Runs adaptive tempered SMC² in a sliding window over an observation series,
with Gaussian-bridge warm-start between consecutive windows.

The generic flow:
  for each window w:
    1. Extract observations in [start, end)
    2. Build the per-window log-density (inner PF factory given ``model``)
    3. w=0: cold-start SMC from the prior
       w>0: bridge from the previous posterior (Gaussian fit)
    4. Convert particles to constrained space, compute coverage stats
    5. Extract smoothed state at the stride point for the next window's PF init
    6. Checkpoint and continue

No FSA-specific references remain: the five hooks needed from ``model`` are
all on ``EstimationModel`` (``align_obs_fn``, ``param_prior_config``,
``all_names``, ``n_dim``, ``init_state_prior_config``) and the log-density
factory return value (``extract_state_at_step``).
"""

from __future__ import annotations

import importlib
import math
import time
from typing import Dict, Tuple, Sequence, Optional

import jax
import jax.numpy as jnp
import numpy as np

from smc2bj.transforms.unconstrained import unconstrained_to_constrained
from smc2bj.estimation.config import SMCConfig, RollingConfig
from smc2bj.estimation.smc_window import (
    run_smc_window, run_smc_window_bridge,
)
from smc2bj.pipeline.windowing import extract_window
from smc2bj.io.checkpoint import save_checkpoint


SHRINKAGE_THRESHOLD = 0.5  # posterior_std/prior_std below this = data-informed


def _prior_stds(model) -> Dict[str, float]:
    """Precompute prior stds in the constrained space for the shrinkage metric."""
    stds: Dict[str, float] = {}
    for name, (dist, args) in model.param_prior_config.items():
        if dist == 'lognormal':
            mu_ln, sigma_ln = args
            stds[name] = math.sqrt(
                (math.exp(sigma_ln ** 2) - 1) * math.exp(2 * mu_ln + sigma_ln ** 2))
        elif dist == 'normal':
            stds[name] = args[1]
    return stds


def rolling_window_smc(
    obs_data,
    model,
    n_days_total: int,
    out_dir: str,
    *,
    smc_cfg: SMCConfig,
    rolling_cfg: RollingConfig,
    cold_start_init: jnp.ndarray,
    truth: Dict[str, float],
    obs_channel_names: Sequence[str],
    seed: int = 42,
    lik_module_path: str = 'smc2bj.log_density.gk_dpf_v3_lite',
    lik_factory_name: str = 'make_gk_dpf_v3_lite_log_density',
) -> Tuple[list, dict]:
    """Run rolling-window SMC² over all windows with Gaussian-bridge warm-start.

    Parameters
    ----------
    obs_data : dict
        ``{channel_name: {'t_idx', 'obs_value', ...}}`` over the full series.
    model : EstimationModel
        The estimation-side model object; must expose ``align_obs_fn``,
        ``param_prior_config``, ``init_state_prior_config``, ``all_names``,
        ``n_dim``.
    n_days_total : int
        Total length of the observation series (in days).
    out_dir : str
        Checkpoint directory.
    smc_cfg, rolling_cfg : config dataclasses
    cold_start_init : jnp.ndarray
        Initial state for window 1's PF (cold). For subsequent windows,
        the PF initialises from the previous window's smoothed state.
    truth : dict
        True parameter values (for coverage computation). Keys must match
        ``model.all_names`` or provide superset.
    obs_channel_names : sequence of str
        Observation channels to count during the per-window obs report.
        Typically the keys of ``obs_data`` that correspond to observed
        measurements (e.g. 'obs_RHR', 'obs_intensity', ...).
    seed : int
        Base seed; per-window seeds derived as ``seed + w * 1000``.
    lik_module_path, lik_factory_name : str
        The inner-PF factory. Defaults to GK-DPF v3-lite.

    Returns
    -------
    all_results : list of per-window result dicts
    T_arr : transform arrays (from the last window's log-density factory)
    """
    lik_module = importlib.import_module(lik_module_path)
    lik_factory = getattr(lik_module, lik_factory_name)

    window_days = rolling_cfg.window_days
    stride_days = rolling_cfg.stride_days
    dt = rolling_cfg.dt

    n_windows = (n_days_total - window_days) // stride_days + 1
    if rolling_cfg.max_windows is not None:
        n_windows = min(n_windows, rolling_cfg.max_windows)
    print(f"\n  Rolling window SMC²: {n_windows} windows")
    print(f"    window={window_days}d, stride={stride_days}d, "
          f"total={n_days_total}d")

    all_results = []
    prev_particles = None
    fixed_init_state = cold_start_init

    prior_stds = _prior_stds(model)
    total_t0 = time.time()

    for w in range(n_windows):
        start = w * stride_days
        end = start + window_days

        print(f"\n  {'='*60}")
        print(f"  Window {w+1}/{n_windows}: days {start}-{end}")
        print(f"  {'='*60}")

        # Extract window data
        window_obs = extract_window(obs_data, start, end)

        # Count obs in window
        n_obs_win = sum(len(window_obs.get(c, {}).get('t_idx', []))
                        for c in obs_channel_names)
        print(f"    Obs in window: {n_obs_win}")

        # Build the per-window log-density (inner PF)
        grid_obs = model.align_obs_fn(window_obs, window_days, dt)
        ld = lik_factory(model, grid_obs,
                         n_particles=smc_cfg.n_pf_particles,
                         bandwidth_scale=smc_cfg.bandwidth_scale,
                         ot_ess_frac=smc_cfg.ot_ess_frac,
                         ot_temperature=smc_cfg.ot_temperature,
                         ot_max_weight=smc_cfg.ot_max_weight,
                         ot_rank=smc_cfg.ot_rank,
                         ot_n_iter=smc_cfg.ot_n_iter,
                         ot_epsilon=smc_cfg.ot_epsilon,
                         dt=dt, seed=seed + w,
                         fixed_init_state=fixed_init_state)
        T_arr = ld._transforms

        if prev_particles is not None:
            print(f"    Init: bridge (Gaussian base → new posterior)")
            particles, elapsed, n_temp = run_smc_window_bridge(
                new_ld=ld, prev_particles=prev_particles,
                model=model, T_arr=T_arr, cfg=smc_cfg,
                seed=seed + w * 1000)
        else:
            print(f"    Init: cold (prior)")
            particles, elapsed, n_temp = run_smc_window(
                ld, model, T_arr, cfg=smc_cfg,
                initial_particles=None, seed=seed + w * 1000)

        print(f"    SMC done: {n_temp} levels in {elapsed:.1f}s")

        # Convert to constrained space
        n_smc, n_d = particles.shape
        samp = np.zeros((n_smc, n_d))
        for i in range(n_smc):
            samp[i] = np.array(unconstrained_to_constrained(
                jnp.array(particles[i]), T_arr))

        # Coverage + shrinkage metric
        stats, cov, cov_informed, n_informed = _compute_stats(
            samp, model.all_names, truth, prior_stds)

        print(f"    Coverage: {sum(1 for s in stats.values() if s['in_ci'])}/"
              f"{len(model.all_names)} = {cov*100:.1f}%  "
              f"(data-informed: "
              f"{sum(1 for s in stats.values() if s['data_informed'] and s['in_ci'])}/"
              f"{n_informed} = {cov_informed*100:.1f}%)")

        true_failures = [n for n, s in stats.items()
                         if s['data_informed'] and not s['in_ci']]
        uninformative_misses = [n for n, s in stats.items()
                                if not s['data_informed'] and not s['in_ci']]
        if true_failures:
            print(f"    TRUE FAILURES (shrink<0.5, miss): {true_failures}")
        if uninformative_misses:
            print(f"    Uninformative misses (shrink>=0.5): "
                  f"{uninformative_misses}")

        result = {
            'window': w,
            'start_day': start,
            'end_day': end,
            'n_temp_steps': n_temp,
            'elapsed_s': elapsed,
            'coverage': cov,
            'coverage_informed': cov_informed,
            'n_informed': n_informed,
            'stats': stats,
            'particles_constrained': samp,
            'fixed_init_state': np.array(fixed_init_state),
        }
        all_results.append(result)

        # Store for next window's bridge
        prev_particles = particles

        # Extract smoothed latent state at t=stride for next window's PF init
        n_extract = min(10, n_smc)
        target_step = int(stride_days / dt)
        extracted_states = []
        for ei in range(n_extract):
            u_draw = jnp.array(particles[ei])
            st = ld.extract_state_at_step(u_draw, target_step)
            extracted_states.append(np.array(st))
        fixed_init_state = jnp.array(np.mean(extracted_states, axis=0))
        state_str = ", ".join(f"x{i}={float(fixed_init_state[i]):.4f}"
                              for i in range(len(fixed_init_state)))
        print(f"    Extracted init state for next window: {state_str}")

        # Checkpoint after each window
        save_checkpoint(all_results, out_dir, truth, smc_cfg, rolling_cfg)

    total_elapsed = time.time() - total_t0
    print(f"\n  Total rolling SMC time: {total_elapsed/3600:.1f}h "
          f"({total_elapsed:.0f}s)")

    return all_results, T_arr


def _compute_stats(samples, param_names, truth, prior_stds):
    """Per-parameter coverage + shrinkage stats for one window's posterior."""
    stats = {}
    n_in = 0
    n_informed = 0
    n_in_informed = 0
    for j, name in enumerate(param_names):
        vals = samples[:, j]
        tv = truth.get(name, float('nan'))
        m = float(np.mean(vals))
        s = float(np.std(vals))
        q5 = float(np.percentile(vals, 5))
        q95 = float(np.percentile(vals, 95))
        ok = q5 <= tv <= q95
        ps = prior_stds.get(name, 1.0)
        shrinkage = s / ps if ps > 0 else 0.0
        informed = shrinkage < SHRINKAGE_THRESHOLD
        if ok:
            n_in += 1
        if informed:
            n_informed += 1
            if ok:
                n_in_informed += 1
        stats[name] = dict(mean=m, std=s, q05=q5, q95=q95, in_ci=ok,
                           shrinkage=shrinkage, prior_std=ps,
                           data_informed=informed)

    cov = n_in / len(param_names)
    cov_informed = (n_in_informed / n_informed
                    if n_informed > 0 else float('nan'))
    return stats, cov, cov_informed, n_informed
