"""GK-DPF Version 2-lite — Systematic resampling (no K×K kernel).

Date:    19 April 2026
Version: 1.0

Identical to gk_dpf_v2 except the O(K²) Gaussian-kernel blend + Liu-West
correction is replaced with O(K) systematic resampling.  This removes the
dominant per-step cost in the inner particle filter.

Motivation
----------
The smooth Gaussian kernel in v2 was designed for MCLMC tuner compatibility
(continuous gradients through the resampling step).  For SMC² usage, HMC
mutations operate on the OUTER parameter space; the inner PF only needs to
provide a reasonable likelihood estimate.  Systematic resampling is the
standard choice in most SMC² implementations (Chopin et al. 2013,
Andrieu et al. 2010).

Gradient flow
-------------
The log-likelihood increment  lik_inc = logsumexp(log_w_pre) - logsumexp(log_w)
is computed BEFORE resampling, so ∂LL/∂θ flows cleanly through the weights.
After resampling, new_particles[indices] is a gather — JAX propagates
gradients through particle values, only the index selection is non-differentiable.

Cost comparison (per scan step, K particles, n_st stochastic dims):
    v2:      O(K² × n_st)  — pairwise kernel matrix + K×K matmul
    v2-lite: O(K)          — cumsum + searchsorted

Public API
----------
    make_gk_dpf_v2_lite_log_density(model, grid_obs, n_particles,
                                     bandwidth_scale, dt, seed) -> Callable
"""

from __future__ import annotations

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax import Array

from smc2bj.estimation_model import EstimationModel
from smc2bj.transforms.unconstrained import (
    build_transform_arrays,
    unconstrained_to_constrained,
    log_prior_unconstrained,
    split_theta,
)


def make_gk_dpf_v2_lite_log_density(model: EstimationModel,
                                     grid_obs: Dict[str, Array],
                                     n_particles: int = 500,
                                     bandwidth_scale: float = 1.0,
                                     dt: float = 5.0 / 60.0,
                                     seed: int = 42) -> Callable:
    """Build the v2-lite (systematic resampling) GK-DPF log-density.

    Same propagation, observation model, and likelihood accumulation as v2,
    but replaces the O(K²) Gaussian-kernel blend with O(K) systematic
    resampling.

    Args:
        model: EstimationModel with propagate_fn, obs_log_weight_fn,
               diffusion_fn, shard_init_fn, state_bounds, stochastic_indices.
        grid_obs: dict of grid-aligned JAX arrays.
        n_particles: K — particle count.
        bandwidth_scale: unused (kept for API compatibility with v2).
        dt: grid step in hours.
        seed: RNG seed.

    Returns:
        JIT-compiled log_density(u) -> scalar.
        Attributes: ._transforms, ._model, ._method = 'gk_dpf_v2_lite'.
    """
    if model.propagate_fn is None or model.obs_log_weight_fn is None:
        raise ValueError(
            f"Model '{model.name}' must provide propagate_fn and "
            f"obs_log_weight_fn for GK-DPF v2-lite.")

    K       = int(n_particles)
    sqrt_dt = jnp.sqrt(jnp.float64(dt))
    n_s     = model.n_states

    log_K   = jnp.log(jnp.float64(K))
    K_float = jnp.float64(K)

    first_key = next(k for k in grid_obs if k not in model.exogenous_keys)
    t_steps   = int(jnp.asarray(grid_obs[first_key]).shape[0])

    T_arr     = build_transform_arrays(model)
    exogenous = {k: jnp.asarray(grid_obs[k]) for k in model.exogenous_keys}

    # Pre-compute the uniform offsets for systematic resampling (fixed)
    sys_offsets = jnp.arange(K, dtype=jnp.float64) / K_float

    print(f"    GK-DPF v2-lite (systematic resampling): "
          f"{t_steps} steps, K={K} particles")
    print(f"    Resampling: systematic at observed steps (O(K) per step)")

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
            key, kp, kn, kr = jax.random.split(key, 4)
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

            # ── Systematic resampling — O(K) ──────────────────────────
            has_obs = grid_obs.get(
                'has_any_obs',
                jnp.ones(t_steps, dtype=u.dtype))[k]

            log_w_norm = log_w_pre - jax.nn.logsumexp(log_w_pre)
            weights = jnp.exp(log_w_norm)
            cumsum = jnp.cumsum(weights)
            # Single uniform offset for systematic resampling
            u_shift = jax.random.uniform(kr, (), dtype=u.dtype) / K_float
            u_points = sys_offsets + u_shift
            indices = jnp.searchsorted(cumsum, u_points)
            indices = jnp.clip(indices, 0, K - 1)
            resampled = new_particles[indices]

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
    log_density._method     = 'gk_dpf_v2_lite'
    return log_density
