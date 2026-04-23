"""GK-DPF Version 3 — ESS-scaled bandwidth + Liu-West + OT rescue (always-on).

Date:    18 April 2026
Version: 3.8 (split checkpoint: OT forward runs once, not recomputed in backward)

Extends v2 with an optimal-transport component for particle rescue when
the cloud degenerates beyond what GK-blending + Liu-West can correct.
The scientific motivation is multi-modal posteriors (e.g. sleep-wake
basin boundaries) where the kernel blend would average across modes;
OT transport preserves mass separation between modes.

Changelog
---------
v3.2 (18 Apr 2026) — OT output sanitization.
    v3.1 inherited a subtle bug: Sinkhorn with aggressive low-rank settings
    (rank=20, n_iter=5) occasionally produced NaN / ±Inf / huge-finite
    values at some theta.  Under IEEE-754, these propagate through the
    interpolation even when ot_weight = 10⁻²⁶ at healthy ESS:
        NaN × 1e-26 = NaN;   Inf × 1e-26 = Inf;   1e30 × 1e-26 = 1e4
    MCLMC's tuner saw non-finite log-density at affected proposals and
    shrank eps to 0.014 (vs v2's 3.05) — the same symptom as the original
    v1/v2 tuner regression, from a different cause.  Fix: sanitize OT
    output pointwise (replace pathological values with gk_lw), clip to
    state bounds, then stop_gradient.  Backward pass unchanged from v2.
v3.1 (18 Apr 2026) — compilation-time fix (always-on structure, no cond).
v3.0 (17 Apr 2026) — initial implementation (nested lax.cond, failed).

Algorithm (v3.8)
----------------
For each time step k = 1 … T:
    §5.1–5.4  Propagate + weight update + marginal-LL increment.
    §5.5      Always-on, no branch.
    §5.6      Compute BOTH operators in parallel:
              (a) GK+LW via smooth_resample_ess_scaled_lw (the v2 helper).
              (b) OT rescue via ot_resample_lr with stop_gradient wrap.
              Combine:
                  ot_weight = ot_max × sigmoid((ot_threshold - ess) / ot_temp)
                  resampled = (1 - ot_weight) * gk_lw + ot_weight * ot
    §5.7      Reset log_w to uniform on observed steps.

Public API
----------
    make_gk_dpf_v3_log_density(
        model, grid_obs, n_particles, bandwidth_scale,
        ot_ess_frac, ot_temperature, ot_max_weight,
        ot_rank, ot_n_iter, ot_epsilon,
        dt, seed
    ) -> Callable
"""

from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax import Array

from smc2bj.estimation_model import EstimationModel
from smc2bj.log_density._gk_kernel import smooth_resample_ess_scaled_lw, compute_ess
from smc2bj.transport.resample import ot_resample_lr
from smc2bj.transforms.unconstrained import (
    build_transform_arrays,
    unconstrained_to_constrained,
    log_prior_unconstrained,
    split_theta,
)


def make_gk_dpf_v3_log_density(
        model: EstimationModel,
        grid_obs: Dict[str, Array],
        n_particles: int = 500,
        bandwidth_scale: float = 1.0,
        ot_ess_frac: float = 0.05,
        ot_temperature: float = 5.0,
        ot_max_weight: float = 0.01,
        ot_rank: int = 5,
        ot_n_iter: int = 2,
        ot_epsilon: float = 0.5,
        dt: float = 5.0 / 60.0,
        seed: int = 42,
) -> Callable:
    """Build the v3.1 (ESS-scaled bandwidth + Liu-West + OT rescue) GK-DPF.

    Always-on structure: GK+LW and OT both run every step; smooth sigmoid
    interpolation based on ESS weights them.  No conditional branches.
    Compiles fast; matches v2 on unimodal problems.

    Args:
        model: EstimationModel with propagate_fn, obs_log_weight_fn,
               diffusion_fn, shard_init_fn, state_bounds, stochastic_indices.
        grid_obs: dict of grid-aligned JAX arrays.  Optionally contains
            'has_any_obs': shape (T,) float — 1.0 at observed steps.
        n_particles: K — particle count.
        bandwidth_scale: Silverman bandwidth multiplier (default 1.0).
        ot_ess_frac: ESS/K value at which ot_weight = 0.5.  Default 0.25
            means the OT rescue is "half engaged" when ESS = K/4.
        ot_temperature: sigmoid sharpness.  Small = sharp transition,
            large = soft transition.  Default 5.0 keeps ot_weight
            numerically zero (< 10⁻²⁰) when ESS is healthy on the OU model.
        ot_rank: Nyström anchor count for low-rank Sinkhorn (default 20).
        ot_n_iter: Sinkhorn iterations (default 5).
        ot_epsilon: Sinkhorn entropic regularisation (default 0.5).
        dt: grid step in hours.
        seed: RNG seed.

    Returns:
        JIT-compiled log_density(u) -> scalar.
        Attributes: ._transforms, ._model, ._method = 'gk_dpf_v3'.
    """
    if model.propagate_fn is None or model.obs_log_weight_fn is None:
        raise ValueError(
            f"Model '{model.name}' must provide propagate_fn and "
            f"obs_log_weight_fn for GK-DPF v3.")

    K       = int(n_particles)
    sqrt_dt = jnp.sqrt(jnp.float64(dt))
    n_s     = model.n_states

    stochastic_idx      = jnp.array(model.stochastic_indices, dtype=jnp.int32)
    stochastic_idx_list = list(model.stochastic_indices)
    bw_scale            = jnp.float64(bandwidth_scale)
    log_K               = jnp.log(jnp.float64(K))

    ot_threshold    = jnp.float64(K * ot_ess_frac)
    ot_temp         = jnp.float64(ot_temperature)
    ot_max          = jnp.float64(ot_max_weight)

    _ot_rank    = int(ot_rank)
    _ot_n_iter  = int(ot_n_iter)
    _ot_epsilon = float(ot_epsilon)

    first_key = next(k for k in grid_obs if k not in model.exogenous_keys)
    t_steps   = int(jnp.asarray(grid_obs[first_key]).shape[0])

    T_arr     = build_transform_arrays(model)
    exogenous = {k: jnp.asarray(grid_obs[k]) for k in model.exogenous_keys}

    _ot_active = ot_max_weight >= 1e-6
    print(f"    GK-DPF v3.8 (ESS-scaled + Liu-West + OT rescue, split checkpoint):")
    print(f"      {t_steps} steps, K={K} particles, bandwidth_scale={bandwidth_scale}")
    print(f"      GK+LW: smooth_resample_ess_scaled_lw (same as v2)")
    if _ot_active:
        print(f"      OT:    rank={_ot_rank}, n_iter={_ot_n_iter}, epsilon={_ot_epsilon}, "
              f"max_weight={ot_max_weight}")
        print(f"      Interpolation: sigmoid((K*{ot_ess_frac} - ESS) / {ot_temperature})")
        print(f"      Split checkpoint: core PF checkpointed, OT stored (no recompute)")
    else:
        print(f"      OT:    DISABLED (ot_max_weight={ot_max_weight} < 1e-6)")
        print(f"      Scan body compiles identically to v2 — zero OT overhead")

    @jax.jit
    def log_density(u):
        theta  = unconstrained_to_constrained(u, T_arr)
        params, init = split_theta(theta, model.n_params)
        sigma_diag   = model.diffusion_fn(params)

        base      = model.shard_init_fn(jnp.int32(0), params, exogenous, init)
        key0      = jax.random.PRNGKey(seed)
        key0, ik  = jax.random.split(key0)
        noise_init = jax.random.normal(ik, (K, n_s))
        particles  = base[None, :] + sigma_diag[None, :] * sqrt_dt * noise_init
        for i, (lo, hi) in enumerate(model.state_bounds):
            particles = particles.at[:, i].set(
                jnp.clip(particles[:, i], lo, hi))

        log_w_init = jnp.zeros(K, dtype=u.dtype)

        # ── Split checkpoint (v3.8) ──────────────────────────────────
        # Core PF (propagation, weights, GK+LW) is checkpointed —
        # same memory profile as v2.  OT computation lives OUTSIDE the
        # checkpoint, so its forward runs ONCE and intermediates are
        # stored for backward (not recomputed).  This eliminates the
        # OT recomputation that doubled per-eval cost in v3.7.
        #
        # Memory cost: ~36KB per step × 500 steps × 16 chains ≈ 288MB
        # for stored OT intermediates.  Acceptable on modern GPUs.

        @jax.checkpoint
        def _core_step(particles, log_w, key, k):
            """Core PF: propagate, weight update, GK+LW.  Checkpointed."""
            key, kp, kn = jax.random.split(key, 3)
            rk = jax.random.fold_in(key, jnp.int32(7))
            t = jnp.asarray(k * dt, dtype=u.dtype)

            noise = jax.random.normal(kn, (K, n_s))

            def _propagate_one(y, xi):
                x_new, pred_lw = model.propagate_fn(
                    y, t, dt, params, grid_obs, k, sigma_diag, xi, kp)
                obs_lw = model.obs_log_weight_fn(x_new, grid_obs, k, params)
                return x_new, pred_lw + obs_lw

            new_particles, step_lw = jax.vmap(_propagate_one)(particles, noise)
            log_w_pre = log_w + step_lw

            lik_inc = (jax.nn.logsumexp(log_w_pre)
                       - jax.nn.logsumexp(log_w))

            has_obs = grid_obs.get(
                'has_any_obs',
                jnp.ones(t_steps, dtype=u.dtype))[k]

            gk_lw = smooth_resample_ess_scaled_lw(
                new_particles, log_w_pre, stochastic_idx, K, bw_scale)

            return new_particles, log_w_pre, lik_inc, gk_lw, has_obs, key, rk

        def scan_step(carry, k):
            particles, log_w, ll_acc, key = carry

            # Core PF (checkpointed → recomputed during backward, same as v2)
            (new_particles, log_w_pre, lik_inc, gk_lw,
             has_obs, key, rk) = _core_step(particles, log_w, key, k)

            # ── OT rescue or pure GK+LW ──
            # Python-level conditional evaluated at trace time (not runtime).
            # When ot_max_weight < 1e-6, OT is effectively disabled and the
            # scan body compiles to exactly the same graph as v2.  This
            # avoids paying any OT overhead on unimodal benchmarks.
            if ot_max_weight >= 1e-6:
                # OT rescue (NOT checkpointed → forward runs once, stored)
                ot_raw = ot_resample_lr(
                    new_particles, log_w_pre, rk,
                    stochastic_indices=stochastic_idx_list,
                    epsilon=_ot_epsilon,
                    n_iter=_ot_n_iter,
                    rank=_ot_rank)
                ot_valid = jnp.isfinite(ot_raw) & (jnp.abs(ot_raw) <= 1e10)
                ot_safe  = jnp.where(ot_valid, ot_raw, gk_lw)
                for i, (lo, hi) in enumerate(model.state_bounds):
                    ot_safe = ot_safe.at[:, i].set(
                        jnp.clip(ot_safe[:, i], lo, hi))
                ot_out = jax.lax.stop_gradient(ot_safe)

                ess       = compute_ess(log_w_pre)
                ot_weight = ot_max * jax.nn.sigmoid(
                    (ot_threshold - ess) / ot_temp)

                resampled = (1.0 - ot_weight) * gk_lw + ot_weight * ot_out
            else:
                # OT disabled — identical to v2
                resampled = gk_lw

            particles_next = jnp.where(has_obs > 0.5, resampled, new_particles)
            log_w_next     = jnp.where(
                has_obs > 0.5,
                jnp.zeros(K, dtype=u.dtype),
                log_w_pre)

            for i, (lo, hi) in enumerate(model.state_bounds):
                particles_next = particles_next.at[:, i].set(
                    jnp.clip(particles_next[:, i], lo, hi))

            return (particles_next, log_w_next, ll_acc + lik_inc, key), None

        init_carry = (particles, log_w_init,
                       jnp.zeros((), dtype=u.dtype), key0)
        (_, lw_final, total_ll, _), _ = jax.lax.scan(
            scan_step, init_carry, jnp.arange(t_steps))

        total_ll = total_ll + jax.nn.logsumexp(lw_final) - log_K
        lp       = log_prior_unconstrained(u, T_arr)
        return total_ll + lp

    log_density._transforms = T_arr
    log_density._model      = model
    log_density._method     = 'gk_dpf_v3'
    return log_density