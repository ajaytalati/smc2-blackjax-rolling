"""
Schrödinger-Föllmer initialisation via BlackJAX.
==================================================
Date:    15 April 2026
Version: 6.0 (BlackJAX-centric)

Uses blackjax.vi.schrodinger_follmer to generate K diverse starting
positions near posterior modes.  SFS transports samples from a simple
prior toward the target using an OT-based generative SDE with Stein's
lemma for drift estimation (gradient-free).

This handles multimodal posteriors far better than prior-mean init.
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, List


def run_sfs_init(log_density: Callable, n_chains: int,
                  n_dim: int, u0: Array, *,
                  n_steps: int = 50,
                  n_inner: int = 100,
                  step_size: float = 0.01,
                  seed: int = 42) -> Array:
    """Generate K diverse starting positions via SFS.

    Args:
        log_density: JIT-compiled fn(u) -> scalar.
        n_chains: Number of starting positions to produce.
        n_dim: Dimensionality.
        u0: Base position (prior mean in unconstrained space).
        n_steps: SFS integration steps.
        n_inner: Inner samples for Stein's lemma drift.
        step_size: SFS step size.
        seed: RNG seed.

    Returns:
        Array of shape (n_chains, n_dim) — diverse starting positions.
    """
    import blackjax

    rng_key = jax.random.PRNGKey(seed)

    # Generate perturbed starting points for each chain
    rng_key, init_key = jax.random.split(rng_key)
    perturbations = jax.random.normal(init_key, (n_chains, n_dim)) * 0.1
    initial_positions = u0[None, :] + perturbations

    print(f"    SFS init: {n_chains} chains, {n_steps} steps, "
          f"{n_inner} inner samples")

    sf = blackjax.vi.schrodinger_follmer

    # Vectorised SFS: run all chains in parallel
    sf_states = jax.vmap(sf.init)(initial_positions)

    def sf_step_all(states, key):
        keys = jax.random.split(key, n_chains)

        def one_step(state, key):
            return sf.step(key, state, log_density,
                           step_size=step_size,
                           n_samples=n_inner)

        new_states, infos = jax.vmap(one_step)(states, keys)
        return new_states, None

    # Run SFS steps
    rng_key, run_key = jax.random.split(rng_key)
    step_keys = jax.random.split(run_key, n_steps)
    final_states, _ = jax.lax.scan(sf_step_all, sf_states, step_keys)

    positions = final_states.position
    print(f"    SFS done: {n_chains} positions generated")

    return positions
