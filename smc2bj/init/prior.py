"""
Prior mean initialisation — fast, no BlackJAX VI needed.
==========================================================
Date:    15 April 2026
Version: 6.0 (BlackJAX-centric)

Uses model.get_init_theta_fn or prior means for all chains.
All chains start from the same point (differentiated by MCLMC tuning).
"""

import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable

from smc2bj.estimation_model import EstimationModel
from smc2bj.transforms.unconstrained import (
    build_transform_arrays,
    constrained_to_unconstrained,
)
from smc2bj.transforms.priors import get_prior_means_dict


def run_prior_init(model: EstimationModel, n_chains: int,
                    seed: int = 42) -> Array:
    """Generate K starting positions from prior means.

    All chains start from the same unconstrained position (with small
    perturbations to break symmetry).

    Args:
        model: EstimationModel.
        n_chains: Number of starting positions.
        seed: RNG seed.

    Returns:
        Array of shape (n_chains, n_dim).
    """
    T_arr = build_transform_arrays(model)

    if model.get_init_theta_fn is not None:
        theta0 = jnp.array(model.get_init_theta_fn())
    else:
        pm = get_prior_means_dict(model)
        theta0 = jnp.array([pm[n] for n in model.all_names],
                            dtype=jnp.float32)

    u0 = constrained_to_unconstrained(theta0, T_arr)

    # Small perturbations to break symmetry between chains
    rng_key = jax.random.PRNGKey(seed)
    perturbations = jax.random.normal(rng_key, (n_chains, model.n_dim)) * 0.01
    positions = u0[None, :] + perturbations

    print(f"    Prior mean init: {n_chains} chains (perturbed)")

    return positions
