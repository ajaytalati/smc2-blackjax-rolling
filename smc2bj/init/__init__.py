"""init — Three interchangeable initialisation strategies.

All return Array of shape (n_chains, n_dim).

    'sfs'         — Schrödinger-Föllmer (DEFAULT, handles multimodality)
    'pathfinder'  — Pathfinder (fast, assumes unimodal)
    'prior_mean'  — Prior means + perturbation (fastest, no VI)
"""

from smc2bj.init.sfs import run_sfs_init
from smc2bj.init.prior import run_prior_init


def build_initial_positions(model, log_density, config):
    """Factory: find starting positions from config.init_method.

    Args:
        model: EstimationModel.
        log_density: JIT-compiled fn(u) -> scalar.
        config: InferenceConfig.

    Returns:
        Array of shape (n_chains, n_dim).
    """
    from transforms.unconstrained import (
        build_transform_arrays, constrained_to_unconstrained)
    import jax.numpy as jnp

    method = config.init_method

    if method == 'prior_mean':
        return run_prior_init(model, config.n_chains, seed=config.seed)

    # Get u0 for VI methods
    T_arr = build_transform_arrays(model)
    if model.get_init_theta_fn is not None:
        theta0 = jnp.array(model.get_init_theta_fn())
    else:
        from transforms.priors import get_prior_means_dict
        pm = get_prior_means_dict(model)
        theta0 = jnp.array([pm[n] for n in model.all_names],
                            dtype=jnp.float32)
    u0 = constrained_to_unconstrained(theta0, T_arr)

    if method == 'sfs':
        return run_sfs_init(
            log_density, config.n_chains, model.n_dim, u0,
            n_steps=config.n_sfs_steps, n_inner=config.n_sfs_inner,
            step_size=config.sfs_step_size, seed=config.seed)

    elif method == 'pathfinder':
        return run_pathfinder_init(
            log_density, config.n_chains, model.n_dim, u0,
            seed=config.seed)

    else:
        raise ValueError(f"Unknown init_method: '{method}'. "
                         f"Choose from: sfs, pathfinder, prior_mean")
