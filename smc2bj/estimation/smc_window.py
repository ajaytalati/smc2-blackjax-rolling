"""Per-window adaptive tempered SMC² with two entry points:

  - ``run_smc_window``        — cold start (prior → posterior).
  - ``run_smc_window_bridge`` — bridge (Gaussian fit of previous posterior
                                → new posterior). Used for warm-start
                                between rolling windows.

Both use BlackJAX's tempered SMC kernel with a clamped adaptive schedule:
the ESS solver finds the optimal lambda increment, then we clamp it to
``max_lambda_inc`` (or ``max_lambda_inc_bridge``) to guarantee a minimum
number of tempering levels.

The per-level MCMC kernel is HMC with a diagonal mass matrix re-estimated
from the current particle cloud after each step.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

import blackjax
import blackjax.smc.tempered as tempered
import blackjax.smc.ess as smc_ess
import blackjax.smc.solver as solver

from smc2bj.transforms.unconstrained import log_prior_unconstrained
from smc2bj.estimation.config import SMCConfig
from smc2bj.estimation.mass_matrix import estimate_mass_matrix
from smc2bj.estimation.sampling import sample_from_prior


# ─────────────────────────────────────────────────────────────────────────────
# Cold-start
# ─────────────────────────────────────────────────────────────────────────────

def run_smc_window(full_log_density, model, T_arr, cfg: SMCConfig,
                   initial_particles=None, seed: int = 42):
    """Adaptive tempered SMC from the prior to the posterior.

    If ``initial_particles`` is None, draws from the prior (cold start).
    Otherwise warm-starts from the provided particles (rare; use
    ``run_smc_window_bridge`` for the standard warm-start).

    Returns ``(particles, elapsed_s, n_tempering_steps)``.
    """
    n_dim = model.n_dim
    n_smc = cfg.n_smc_particles

    @jax.jit
    def logprior_fn(u):
        return log_prior_unconstrained(u, T_arr)

    @jax.jit
    def loglikelihood_fn(u):
        return full_log_density(u) - log_prior_unconstrained(u, T_arr)

    if initial_particles is None:
        init_key = jax.random.PRNGKey(seed)
        initial_particles = sample_from_prior(n_smc, T_arr, n_dim, init_key)

    hmc_kernel = blackjax.mcmc.hmc.build_kernel()
    smc_kernel = tempered.build_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=blackjax.mcmc.hmc.init,
        resampling_fn=blackjax.smc.resampling.systematic,
    )
    smc_kernel_jit = jax.jit(smc_kernel, static_argnums=(2,))

    state = tempered.init(initial_particles)
    inv_mass = estimate_mass_matrix(initial_particles)

    rng_key = jax.random.PRNGKey(seed + 123)
    step_idx = 0
    t0 = time.time()
    prev_lam = 0.0

    while float(state.tempering_param) < 1.0:
        rng_key, step_key = jax.random.split(rng_key)

        current_lam = float(state.tempering_param)
        max_delta = 1.0 - current_lam
        delta = smc_ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            cfg.target_ess_frac,
            max_delta,
            solver.dichotomy,
        )
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, cfg.max_lambda_inc)
        next_lam = current_lam + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0

        mcmc_parameters = {
            'step_size': jnp.array([cfg.hmc_step_size]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array([cfg.hmc_num_leapfrog],
                                                dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(
            step_key, state, cfg.num_mcmc_steps,
            jnp.float64(next_lam), mcmc_parameters)
        inv_mass = estimate_mass_matrix(state.particles)

        lam = float(state.tempering_param)
        step_idx += 1
        actual_delta = lam - prev_lam
        prev_lam = lam

        try:
            acc = float(jnp.mean(info.update_info.acceptance_rate))
        except Exception:
            acc = float('nan')

        elapsed = time.time() - t0
        compile_note = " (JIT)" if step_idx == 1 else ""
        print(f"      step {step_idx:3d}  lam={lam:.6f}  "
              f"d={actual_delta:.4f}  acc={acc:.3f}  "
              f"[{elapsed:.0f}s{compile_note}]",
              flush=True)

    elapsed = time.time() - t0
    particles = np.array(jax.device_get(state.particles))
    return particles, elapsed, step_idx


# ─────────────────────────────────────────────────────────────────────────────
# Bridge (warm-start between rolling windows)
# ─────────────────────────────────────────────────────────────────────────────

def run_smc_window_bridge(new_ld, prev_particles, model, T_arr,
                          cfg: SMCConfig, seed: int = 42):
    """Bridge tempered SMC: Gaussian base measure → new posterior.

    Data-annealing bridge (tempered SMC²):
      logprior_fn(u)      = log N(u; mu_hat, Sigma_hat_LW)   [Gaussian fit]
      loglikelihood_fn(u) = new_ld(u) - logprior_fn(u)       [1 PF eval]

    At lambda=0 the target is N(mu_hat, Sigma_hat) ~ old posterior
    (where particles start). At lambda=1 the target is new_ld(u) =
    correct new-window posterior. Uses Ledoit-Wolf shrinkage for
    stable covariance estimation.
    """
    prev = jnp.array(prev_particles, dtype=jnp.float64)
    N, d = prev.shape

    mu = jnp.mean(prev, axis=0)
    S = jnp.cov(prev.T)

    # Ledoit-Wolf optimal shrinkage (Ledoit & Wolf 2004, Eq. 2)
    X_c = prev - mu[None, :]
    mu_target = jnp.trace(S) / d
    delta_mat = S - mu_target * jnp.eye(d)
    delta_sq_sum = jnp.sum(delta_mat ** 2)
    X2 = (X_c[:, :, None] * X_c[:, None, :])
    b_bar = jnp.sum((X2 - S[None, :, :]) ** 2) / (N * N)
    alpha = min(float(b_bar / jnp.maximum(delta_sq_sum, 1e-10)), 1.0)

    cov_lw = (1.0 - alpha) * S + alpha * mu_target * jnp.eye(d)
    cov_reg = cov_lw + 1e-4 * jnp.eye(d)
    L_chol = jnp.linalg.cholesky(cov_reg)
    L_inv = jax.scipy.linalg.solve_triangular(
        L_chol, jnp.eye(d), lower=True)
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L_chol)))
    const = -0.5 * (d * jnp.log(2.0 * jnp.pi) + log_det)

    print(f"      Gaussian base: LW shrinkage={alpha:.3f}, "
          f"log_det={float(log_det):.1f}", flush=True)

    @jax.jit
    def logprior_fn(u):
        diff = u - mu
        maha = jnp.sum((L_inv @ diff) ** 2)
        return const - 0.5 * maha

    @jax.jit
    def loglikelihood_fn(u):
        return new_ld(u) - logprior_fn(u)

    init_key = jax.random.PRNGKey(seed)
    z = jax.random.normal(init_key, (cfg.n_smc_particles, d),
                          dtype=jnp.float64)
    initial_particles = mu[None, :] + z @ L_chol.T

    hmc_kernel = blackjax.mcmc.hmc.build_kernel()
    smc_kernel = tempered.build_kernel(
        logprior_fn=logprior_fn,
        loglikelihood_fn=loglikelihood_fn,
        mcmc_step_fn=hmc_kernel,
        mcmc_init_fn=blackjax.mcmc.hmc.init,
        resampling_fn=blackjax.smc.resampling.systematic,
    )
    smc_kernel_jit = jax.jit(smc_kernel, static_argnums=(2,))

    state = tempered.init(initial_particles)
    inv_mass = estimate_mass_matrix(initial_particles)

    rng_key = jax.random.PRNGKey(seed + 123)
    step_idx = 0
    t0 = time.time()
    prev_lam = 0.0

    incr_ll_init = jax.vmap(loglikelihood_fn)(initial_particles)
    incr_var_init = float(jnp.var(incr_ll_init))
    incr_range_init = float(jnp.max(incr_ll_init) - jnp.min(incr_ll_init))
    print(f"      Bridge init: incr_ll var={incr_var_init:.1f} "
          f"range={incr_range_init:.1f}", flush=True)

    while float(state.tempering_param) < 1.0:
        rng_key, step_key = jax.random.split(rng_key)

        current_lam = float(state.tempering_param)
        max_delta = 1.0 - current_lam
        delta = smc_ess.ess_solver(
            jax.vmap(loglikelihood_fn),
            state.particles,
            cfg.target_ess_frac,
            max_delta,
            solver.dichotomy,
        )
        delta = float(jnp.clip(delta, 0.0, max_delta))
        delta = min(delta, cfg.max_lambda_inc_bridge)
        next_lam = current_lam + delta
        if 1.0 - next_lam < 1e-6:
            next_lam = 1.0

        mcmc_parameters = {
            'step_size': jnp.array([cfg.hmc_step_size]),
            'inverse_mass_matrix': inv_mass,
            'num_integration_steps': jnp.array([cfg.hmc_num_leapfrog],
                                                dtype=jnp.int32),
        }
        state, info = smc_kernel_jit(
            step_key, state, cfg.num_mcmc_steps_bridge,
            jnp.float64(next_lam), mcmc_parameters)
        inv_mass = estimate_mass_matrix(state.particles)

        lam = float(state.tempering_param)
        step_idx += 1
        actual_delta = lam - prev_lam
        prev_lam = lam

        try:
            acc = float(jnp.mean(info.update_info.acceptance_rate))
        except Exception:
            acc = float('nan')

        incr_ll = jax.vmap(loglikelihood_fn)(state.particles)
        incr_var = float(jnp.var(incr_ll))

        elapsed = time.time() - t0
        compile_note = " (JIT)" if step_idx == 1 else ""
        print(f"      step {step_idx:3d}  lam={lam:.6f}  "
              f"d={actual_delta:.4f}  acc={acc:.3f}  "
              f"incr_var={incr_var:.1f}  "
              f"[{elapsed:.0f}s{compile_note}]",
              flush=True)

    elapsed = time.time() - t0
    particles = np.array(jax.device_get(state.particles))
    return particles, elapsed, step_idx
