"""GK-DPF Version 2 — ESS-adaptive resampling + Liu-West correction.

Date:    17 April 2026
Version: 2.0

Levels 1 + 2 combined.  This is the production-quality single-mode filter:
ESS-triggered resampling (Level 1) plus Liu-West variance restoration
(Level 2) to counteract the progressive mode collapse introduced by the
Gaussian-kernel blend.

Four-version comparison:
    v0 — Baseline: resample at EVERY observed step, no ESS check, no Liu-West.
    v1 — Level 1:  ESS-adaptive resampling trigger (no Liu-West).
    v2 — Levels 1+2: ESS-adaptive + Liu-West shrinkage correction. ← THIS FILE
    v3 — Levels 1+2+3: ESS-adaptive + Liu-West + OT rescue.

This file is IDENTICAL to the current gk_dpf.py (v1.1) in logic; it
exists as a named version so the proof_of_principle script can dispatch
to it by name ('gk_dpf_v2') alongside v0 and v1.

Algorithm (v2)
--------------
For each time step k = 1 … T:
    §5.1–5.3  Propagate + weight update.
    §5.4      SIRS-PF marginal-LL increment.
    §5.5      ESS trigger: resample when (ESS < K/2) AND (has_any_obs > 0.5).
    §5.6      smooth_resample = GK blend + Liu-West shrinkage correction.
              Liu-West: x_corrected_i = a × x̃_i + (1−a) × μ_w
              where a = sqrt(1 − h_norm²), partially restores pre-blend spread.
    §5.7      Carry forward.

Public API
----------
    make_gk_dpf_v2_log_density(model, grid_obs, n_particles, bandwidth_scale,
                                dt, seed) -> Callable
"""

from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax import Array

from smc2bj.estimation_model import EstimationModel
from smc2bj.log_density._gk_kernel import smooth_resample, compute_ess
from smc2bj.transforms.unconstrained import (
    build_transform_arrays,
    unconstrained_to_constrained,
    log_prior_unconstrained,
    split_theta,
)


def make_gk_dpf_v2_log_density(model: EstimationModel,
                                 grid_obs: Dict[str, Array],
                                 n_particles: int = 500,
                                 bandwidth_scale: float = 1.0,
                                 dt: float = 5.0 / 60.0,
                                 seed: int = 42) -> Callable:
    """Build the v2 (ESS-adaptive + Liu-West) GK-DPF log-density evaluator.

    Resamples only when ESS < K/2 AND an observation is present.
    Applies Liu-West shrinkage correction to restore variance lost to blending.

    Args:
        model: EstimationModel with propagate_fn, obs_log_weight_fn,
               diffusion_fn, shard_init_fn, state_bounds, stochastic_indices.
        grid_obs: dict of grid-aligned JAX arrays.  Optionally contains
            'has_any_obs': shape (T,) float — 1.0 at observed steps.
        n_particles: K — particle count.
        bandwidth_scale: Silverman bandwidth multiplier (default 1.0).
        dt: grid step in hours.
        seed: RNG seed.

    Returns:
        JIT-compiled log_density(u) -> scalar.
        Attributes: ._transforms, ._model, ._method = 'gk_dpf_v2'.
    """
    if model.propagate_fn is None or model.obs_log_weight_fn is None:
        raise ValueError(
            f"Model '{model.name}' must provide propagate_fn and "
            f"obs_log_weight_fn for GK-DPF v2.")

    K       = int(n_particles)
    sqrt_dt = jnp.sqrt(jnp.float64(dt))
    n_s     = model.n_states

    stochastic_idx = jnp.array(model.stochastic_indices, dtype=jnp.int32)
    bw_scale       = jnp.float64(bandwidth_scale)
    log_K          = jnp.log(jnp.float64(K))
    ess_threshold  = jnp.float64(K * 0.5)

    first_key = next(k for k in grid_obs if k not in model.exogenous_keys)
    t_steps   = int(jnp.asarray(grid_obs[first_key]).shape[0])

    T_arr     = build_transform_arrays(model)
    exogenous = {k: jnp.asarray(grid_obs[k]) for k in model.exogenous_keys}

    print(f"    GK-DPF v2 (ESS-adaptive + Liu-West): {t_steps} steps, "
          f"K={K} particles, bandwidth_scale={bandwidth_scale}")
    print(f"    Resampling: ESS-triggered (ESS < K/2={K//2}), "
          f"Liu-West shrinkage correction ENABLED")

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

        @jax.checkpoint
        def scan_step(carry, k):
            particles, log_w, ll_acc, key = carry
            key, kp, kn = jax.random.split(key, 3)
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

            ess     = compute_ess(log_w_pre)
            has_obs = grid_obs.get(
                'has_any_obs',
                jnp.ones(t_steps, dtype=u.dtype))[k]
            do_resample = (ess < ess_threshold) & (has_obs > 0.5)

            # smooth_resample = GK blend + Liu-West correction.
            resampled = smooth_resample(
                new_particles, log_w_pre, stochastic_idx, K, bw_scale)

            particles_next = jnp.where(do_resample, resampled, new_particles)
            log_w_next     = jnp.where(
                do_resample,
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
    log_density._method     = 'gk_dpf_v2'
    return log_density
