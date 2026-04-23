"""
sde_solver_scipy.py — Generic SDE Solvers (numpy + scipy)
==========================================================
Date:    19 April 2026
Version: 1.1

Model-agnostic solvers. Receive an SDEModel as first argument.
Never reference any specific state variable or parameter name.

Changelog from 1.0:
  - solve_sde now supports DIFFUSION_DIAGONAL_STATE in addition to
    DIFFUSION_DIAGONAL_CONSTANT.  State-dependent noise is evaluated
    at each Euler substep as sigma_i(params) * g_i(y, params) where
    g_i comes from model.noise_scale_fn.
"""

import math
import numpy as np
from scipy.integrate import solve_ivp

from sde_model import (SDEModel, DIFFUSION_DIAGONAL_CONSTANT,
                        DIFFUSION_DIAGONAL_STATE)


# =========================================================================
# CONSTRAINT ENFORCEMENT
# =========================================================================

def apply_constraints(trajectory, bounds):
    """Clip each state to its declared [lower, upper] bound."""
    traj = trajectory.copy()
    for i, (lo, hi) in enumerate(bounds):
        traj[:, i] = np.clip(traj[:, i], lo, hi)
    return traj


def clip_state(y, bounds):
    """Clip a single state vector to bounds."""
    y_out = y.copy()
    for i, (lo, hi) in enumerate(bounds):
        y_out[i] = np.clip(y_out[i], lo, hi)
    return y_out


# =========================================================================
# DETERMINISTIC STATE OVERWRITE
# =========================================================================

def overwrite_deterministic(trajectory, t_grid, model, params):
    """Replace deterministic states with their exact analytical values."""
    traj = trajectory.copy()
    for idx in model.deterministic_indices:
        fn = model.states[idx].analytical_fn
        if fn is not None:
            traj[:, idx] = np.array([fn(t, params) for t in t_grid])
    return traj


# =========================================================================
# DETERMINISTIC ODE SOLVER (scipy BDF)
# =========================================================================

def solve_deterministic(model, params, init_state, t_grid, exogenous=None):
    """Solve the ODE deterministically with scipy BDF (3rd-order implicit).

    Parameters
    ----------
    model : SDEModel
        The model specification (drift, bounds, etc.).
    params : dict
        Parameter values.
    init_state : dict
        Initial state values (model-specific keys).
    t_grid : ndarray
        Times at which to save the solution.
    exogenous : dict, optional
        External inputs (model-specific, opaque to this function).

    Returns
    -------
    trajectory : ndarray (len(t_grid), n_states)
    """
    exogenous = exogenous or {}
    y0 = model.make_y0_fn(init_state, params)
    aux = model.make_aux_fn(params, init_state, t_grid, exogenous) if model.make_aux_fn else None

    def ode_rhs(t, y):
        return model.drift_fn(t, y, params, aux)

    sol = solve_ivp(
        ode_rhs,
        t_span=(float(t_grid[0]), float(t_grid[-1])),
        y0=y0,
        method='BDF',
        t_eval=t_grid,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.1,
    )

    if sol.status != 0:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    trajectory = sol.y.T
    trajectory = overwrite_deterministic(trajectory, t_grid, model, params)
    trajectory = apply_constraints(trajectory, model.bounds)
    return trajectory


# =========================================================================
# STOCHASTIC SDE SOLVER (fine Euler-Maruyama)
# =========================================================================

def solve_sde(model, params, init_state, t_grid, exogenous=None,
              seed=42, n_substeps=10):
    """Solve the SDE with fine-grained Euler-Maruyama.

    Uses n_substeps per grid interval for numerical stability.
    At n_substeps=10 with dt_grid=5min: dt_sub=0.5min.

    Parameters
    ----------
    model : SDEModel
    params : dict
    init_state : dict
    t_grid : ndarray
    exogenous : dict, optional
    seed : int
    n_substeps : int
        Euler-Maruyama substeps per grid interval.

    Returns
    -------
    trajectory : ndarray (len(t_grid), n_states)
    """
    exogenous = exogenous or {}
    rng = np.random.default_rng(seed)
    y0 = model.make_y0_fn(init_state, params)
    aux = model.make_aux_fn(params, init_state, t_grid, exogenous) if model.make_aux_fn else None
    n_states = model.n_states
    bounds = model.bounds

    # Build the per-step diffusion vector function.
    # Both supported types use diffusion_fn(params) as the per-state
    # sigma magnitudes; DIFFUSION_DIAGONAL_STATE multiplies by the
    # state-dependent noise_scale_fn(y, params).
    if model.diffusion_fn is None:
        raise ValueError(
            f"Model '{model.name}' does not provide diffusion_fn."
        )
    sigma_const = np.asarray(model.diffusion_fn(params), dtype=np.float64)

    if model.diffusion_type == DIFFUSION_DIAGONAL_CONSTANT:
        def _sigma_of(y):
            return sigma_const
    elif model.diffusion_type == DIFFUSION_DIAGONAL_STATE:
        if model.noise_scale_fn is None:
            raise ValueError(
                f"Model '{model.name}' declares DIFFUSION_DIAGONAL_STATE "
                f"but does not provide noise_scale_fn.  Pass "
                f"noise_scale_fn=<fn> to SDEModel(...)."
            )
        _scale_fn = model.noise_scale_fn
        def _sigma_of(y):
            return sigma_const * np.asarray(_scale_fn(y, params),
                                             dtype=np.float64)
    else:
        raise NotImplementedError(
            f"Diffusion type '{model.diffusion_type}' not yet supported "
            f"in scipy solver.  Supported: DIFFUSION_DIAGONAL_CONSTANT, "
            f"DIFFUSION_DIAGONAL_STATE."
        )

    dt_grid = float(t_grid[1] - t_grid[0])
    dt_sub = dt_grid / n_substeps
    sqrt_dt_sub = math.sqrt(dt_sub)

    n_grid = len(t_grid)
    trajectory = np.zeros((n_grid, n_states))
    y = clip_state(y0, bounds)
    trajectory[0] = y.copy()

    for k in range(n_grid - 1):
        t_start = t_grid[k]
        for s in range(n_substeps):
            t_now = t_start + s * dt_sub
            dy = model.drift_fn(t_now, y, params, aux)
            sigma_now = _sigma_of(y)
            noise = rng.standard_normal(n_states) * sigma_now * sqrt_dt_sub
            y = y + dt_sub * dy + noise
            # Overwrite deterministic states
            t_next = t_now + dt_sub
            for idx in model.deterministic_indices:
                fn = model.states[idx].analytical_fn
                if fn is not None:
                    y[idx] = fn(t_next, params)
            y = clip_state(y, bounds)
        trajectory[k + 1] = y.copy()

    return trajectory