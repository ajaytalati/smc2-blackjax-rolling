"""models/fsa_real_obs/simulation.py — 3-State FSA SDE with Real Obs.

Date:    19 April 2026
Version: 1.0

Same 3-state latent SDE as fitness_strain_amplitude (B, F, A).
Observation model replaced: 6 physiological channels instead of
direct state observations.

SDE (identical to base FSA)
---
    dB = [(1 + alpha_A * A)/tau_B * (T_B(t) - B)] dt
         + sigma_B * sqrt(B(1-B)) dW_B                      (Jacobi)

    dF = [Phi(t) - (1 + lambda_B B + lambda_A A)/tau_F * F] dt
         + sigma_F * sqrt(F) dW_F                            (CIR)

    dA = [mu(B,F) * A - eta * A^3] dt
         + sigma_A * sqrt(A + eps_A) dW_A         (regularised Landau)

    mu(B, F) = mu_0 + mu_B * B - mu_F * F - mu_FF * F^2

Observation channels (NEW)
--------------------------
    Ch1 RHR:       R_base - kappa_vagal*B + kappa_chronic*F + N(0, sigma_obs_R^2)
    Ch2 Intensity: I_base + c_B*B - c_F*F + N(0, sigma_obs_I^2)
    Ch3 Duration:  D_base + d_B*B - d_F*F + N(0, sigma_obs_D^2)
    Ch4 Stress:    S_base - s_A*A + s_F*F + N(0, sigma_obs_S^2)
    Ch5 Sleep:     Sleep_base + sl_A*A + sl_B*B - sl_F*F + N(0, sigma_obs_Sleep^2)
    Ch6 Timing:    Time_base + t_A*A - t_F*F + N(0, sigma_obs_Time^2)
"""

import math
import numpy as np

from smc2bj.simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec,
    DIFFUSION_DIAGONAL_STATE)
from models.fsa_real_obs.sim_plots import plot_fsa_real_obs


# =========================================================================
# FROZEN CONSTANTS (same as base FSA)
# =========================================================================

EPS_A_FROZEN = 1.0e-4
EPS_B_FROZEN = 1.0e-4


# =========================================================================
# INPUT SCHEDULES (same as base FSA)
# =========================================================================

def _T_B(t, T_B_const):
    return T_B_const


def _Phi(t, Phi_1, Phi_2, T_jump):
    return Phi_1 if t < T_jump else Phi_2


# =========================================================================
# DRIFT (same as base FSA)
# =========================================================================

def drift(t, y, params, aux):
    T_B_const, Phi_1, Phi_2, T_jump = aux
    p = params
    B = y[0]; F = y[1]; A = y[2]

    T_B_t = _T_B(t, T_B_const)
    Phi_t = _Phi(t, Phi_1, Phi_2, T_jump)

    mu = (p['mu_0'] + p['mu_B'] * B
          - p['mu_F'] * F - p['mu_FF'] * F * F)

    dB = (1.0 + p['alpha_A'] * A) / p['tau_B'] * (T_B_t - B)
    dF = Phi_t - (1.0 + p['lambda_B'] * B
                  + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return np.array([dB, dF, dA])


def drift_jax(t, y, args):
    import jax.numpy as jnp
    p, T_B_const, Phi_1, Phi_2, T_jump = args
    B = y[0]; F = y[1]; A = y[2]

    T_B_t = T_B_const
    Phi_t = jnp.where(t < T_jump, Phi_1, Phi_2)

    mu = (p['mu_0'] + p['mu_B'] * B
          - p['mu_F'] * F - p['mu_FF'] * F * F)

    dB = (1.0 + p['alpha_A'] * A) / p['tau_B'] * (T_B_t - B)
    dF = Phi_t - (1.0 + p['lambda_B'] * B
                  + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return jnp.array([dB, dF, dA])


# =========================================================================
# DIFFUSION (same as base FSA — state-dependent)
# =========================================================================

def diffusion_diagonal(params):
    return np.array([params['sigma_B'],
                     params['sigma_F'],
                     params['sigma_A']])


def noise_scale_fn(y, params):
    del params
    B = np.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F = max(y[1], 0.0)
    A = max(y[2], 0.0)
    return np.array([math.sqrt(B * (1.0 - B)),
                     math.sqrt(F),
                     math.sqrt(A + EPS_A_FROZEN)])


def noise_scale_fn_jax(y, params):
    import jax.numpy as jnp
    del params
    B = jnp.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F = jnp.maximum(y[1], 0.0)
    A = jnp.maximum(y[2], 0.0)
    return jnp.array([jnp.sqrt(B * (1.0 - B)),
                      jnp.sqrt(F),
                      jnp.sqrt(A + EPS_A_FROZEN)])


# =========================================================================
# AUXILIARY / INITIAL STATE (same as base FSA)
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    del params, init_state, t_grid
    return (exogenous['T_B_const'],
            exogenous['Phi_1'],
            exogenous['Phi_2'],
            exogenous['T_jump'])


def make_aux_jax(params, init_state, t_grid, exogenous):
    import jax.numpy as jnp
    del init_state, t_grid
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax,
            jnp.float64(exogenous['T_B_const']),
            jnp.float64(exogenous['Phi_1']),
            jnp.float64(exogenous['Phi_2']),
            jnp.float64(exogenous['T_jump']))


def make_y0(init_dict, params):
    del params
    return np.array([init_dict['B_0'], init_dict['F_0'], init_dict['A_0']])


# =========================================================================
# OBSERVATION CHANNELS — 6 physiological channels
# =========================================================================

def gen_obs_RHR(trajectory, t_grid, params, aux, prior_channels, seed):
    """Ch1: RHR = R_base - kappa_vagal*B + kappa_chronic*F + noise."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    F = trajectory[:, 1]
    RHR_true = (params['R_base']
                - params['kappa_vagal'] * B
                + params['kappa_chronic'] * F)
    noise = rng.normal(0.0, params['sigma_obs_R'], size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'obs_value': (RHR_true + noise).astype(np.float32),
    }


def gen_obs_intensity(trajectory, t_grid, params, aux, prior_channels, seed):
    """Ch2: I_norm = I_base + c_B*B - c_F*F + noise."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    F = trajectory[:, 1]
    I_true = params['I_base'] + params['c_B'] * B - params['c_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_I'], size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'obs_value': (I_true + noise).astype(np.float32),
    }


def gen_obs_duration(trajectory, t_grid, params, aux, prior_channels, seed):
    """Ch3: D_norm = D_base + d_B*B - d_F*F + noise."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    F = trajectory[:, 1]
    D_true = params['D_base'] + params['d_B'] * B - params['d_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_D'], size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'obs_value': (D_true + noise).astype(np.float32),
    }


def gen_obs_stress(trajectory, t_grid, params, aux, prior_channels, seed):
    """Ch4: S_obs = S_base - s_A*A + s_F*F + noise."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    F = trajectory[:, 1]
    A = trajectory[:, 2]
    S_true = params['S_base'] - params['s_A'] * A + params['s_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_S'], size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'obs_value': (S_true + noise).astype(np.float32),
    }


def gen_obs_sleep(trajectory, t_grid, params, aux, prior_channels, seed):
    """Ch5: Sleep = Sleep_base + sl_A*A + sl_B*B - sl_F*F + noise."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    F = trajectory[:, 1]
    A = trajectory[:, 2]
    Sleep_true = (params['Sleep_base']
                  + params['sl_A'] * A
                  + params['sl_B'] * B
                  - params['sl_F'] * F)
    noise = rng.normal(0.0, params['sigma_obs_Sleep'], size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'obs_value': (Sleep_true + noise).astype(np.float32),
    }


def gen_obs_timing(trajectory, t_grid, params, aux, prior_channels, seed):
    """Ch6: Time_logit = Time_base + t_A*A - t_F*F + noise."""
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    F = trajectory[:, 1]
    A = trajectory[:, 2]
    Time_true = params['Time_base'] + params['t_A'] * A - params['t_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_Time'], size=len(t_grid))
    return {
        't_idx':     np.arange(len(t_grid), dtype=np.int32),
        'obs_value': (Time_true + noise).astype(np.float32),
    }


def gen_T_B_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Emit T_B(t) schedule as exogenous input record."""
    del trajectory, params, prior_channels, seed
    T_B_const, _Phi_1, _Phi_2, _T_jump = aux
    val = np.full(len(t_grid), T_B_const, dtype=np.float32)
    return {'t_idx':    np.arange(len(t_grid), dtype=np.int32),
            'T_B_value': val}


def gen_Phi_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Emit Phi(t) schedule as exogenous input record."""
    del trajectory, params, prior_channels, seed
    _T_B_const, Phi_1, Phi_2, T_jump = aux
    t = t_grid
    val = np.where(t < T_jump, Phi_1, Phi_2).astype(np.float32)
    return {'t_idx':     np.arange(len(t), dtype=np.int32),
            'Phi_value': val}


# =========================================================================
# PHYSICS VERIFICATION (same as base FSA)
# =========================================================================

def _mu_of(B, F, p):
    return p['mu_0'] + p['mu_B']*B - p['mu_F']*F - p['mu_FF']*F*F


def verify_physics(trajectory, t_grid, params):
    B = trajectory[:, 0]; F = trajectory[:, 1]; A = trajectory[:, 2]
    mu_traj = _mu_of(B, F, params)
    B_crit_0 = abs(params['mu_0']) / params['mu_B']
    disc = params['mu_F']**2 + 4.0 * params['mu_FF'] * (params['mu_B'] * 1.0 + params['mu_0'])
    F_max_1 = ((-params['mu_F'] + math.sqrt(disc)) / (2.0 * params['mu_FF'])
               if disc >= 0 else float('nan'))
    return {
        'B_min': float(B.min()), 'B_max': float(B.max()),
        'F_min': float(F.min()), 'F_max': float(F.max()),
        'A_min': float(A.min()), 'A_max': float(A.max()),
        'B_final': float(B[-1]), 'F_final': float(F[-1]),
        'A_final': float(A[-1]),
        'mu_min': float(mu_traj.min()), 'mu_max': float(mu_traj.max()),
        'mu_crosses_zero': bool(mu_traj.min() < 0 < mu_traj.max()),
        'landmark_B_crit_0': float(B_crit_0),
        'landmark_F_max_1':  float(F_max_1),
        'A_activated':       bool(A.max() > 0.3),
        'all_finite':        bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETERS — same SDE params as base FSA + observation model params
# =========================================================================

DEFAULT_PARAMS = {
    # --- SDE dynamics (same as base FSA) ---
    'tau_B':    14.0,
    'alpha_A':   1.0,
    'tau_F':     7.0,
    'lambda_B':  3.0,
    'lambda_A':  1.5,
    'mu_0':     -0.10,
    'mu_B':      0.30,
    'mu_F':      0.10,
    'mu_FF':     0.40,
    'eta':       0.20,
    'sigma_B':   0.01,
    'sigma_F':   0.005,
    'sigma_A':   0.02,

    # --- Ch1: RHR ---
    'R_base':        62.0,
    'kappa_vagal':   12.0,
    'kappa_chronic': 10.0,
    'sigma_obs_R':    1.5,

    # --- Ch2: Intensity ---
    'I_base':        0.5,
    'c_B':           0.2,
    'c_F':           0.1,
    'sigma_obs_I':   0.05,

    # --- Ch3: Duration ---
    'D_base':        0.5,
    'd_B':           0.3,
    'd_F':           0.2,
    'sigma_obs_D':   0.08,

    # --- Ch4: Stress ---
    'S_base':       30.0,
    's_A':          15.0,
    's_F':          20.0,
    'sigma_obs_S':   5.0,

    # --- Ch5: Sleep ---
    'Sleep_base':    0.5,
    'sl_A':          0.2,
    'sl_B':          0.1,
    'sl_F':          0.2,
    'sigma_obs_Sleep': 0.1,

    # --- Ch6: Timing ---
    'Time_base':     0.0,
    't_A':           1.0,
    't_F':           0.5,
    'sigma_obs_Time': 0.5,
}

DEFAULT_INIT = {'B_0': 0.05, 'F_0': 0.10, 'A_0': 0.01}


# =========================================================================
# THREE EXOGENOUS INPUT SCHEDULES (same as base FSA)
# =========================================================================

EXO_SEDENTARY = {
    'T_B_const': 0.0,
    'Phi_1':     0.0,
    'Phi_2':     0.0,
    'T_jump':  1000.0,
    'T_end':    120.0,
}

EXO_RECOVERY = {
    'T_B_const': 0.6,
    'Phi_1':     0.03,
    'Phi_2':     0.03,
    'T_jump':   1000.0,
    'T_end':    200.0,
}

EXO_OVERTRAINING = {
    'T_B_const': 0.6,
    'Phi_1':     0.03,
    'Phi_2':     0.20,
    'T_jump':    150.0,
    'T_end':     240.0,
}


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

FSA_REAL_OBS_MODEL = SDEModel(
    name="fsa_real_obs",
    version="1.0",

    states=(
        StateSpec("B", 0.0, 1.0),
        StateSpec("F", 0.0, 10.0),
        StateSpec("A", 0.0, 5.0),
    ),

    drift_fn=drift,
    drift_fn_jax=drift_jax,
    diffusion_type=DIFFUSION_DIAGONAL_STATE,
    diffusion_fn=diffusion_diagonal,
    noise_scale_fn=noise_scale_fn,
    noise_scale_fn_jax=noise_scale_fn_jax,
    make_aux_fn=make_aux,
    make_aux_fn_jax=make_aux_jax,
    make_y0_fn=make_y0,

    channels=(
        ChannelSpec("obs_RHR",       depends_on=(), generate_fn=gen_obs_RHR),
        ChannelSpec("obs_intensity", depends_on=(), generate_fn=gen_obs_intensity),
        ChannelSpec("obs_duration",  depends_on=(), generate_fn=gen_obs_duration),
        ChannelSpec("obs_stress",    depends_on=(), generate_fn=gen_obs_stress),
        ChannelSpec("obs_sleep",     depends_on=(), generate_fn=gen_obs_sleep),
        ChannelSpec("obs_timing",    depends_on=(), generate_fn=gen_obs_timing),
        ChannelSpec("T_B",           depends_on=(), generate_fn=gen_T_B_channel),
        ChannelSpec("Phi",           depends_on=(), generate_fn=gen_Phi_channel),
    ),

    plot_fn=plot_fsa_real_obs,
    verify_physics_fn=verify_physics,

    param_sets={
        'sedentary':    DEFAULT_PARAMS,
        'recovery':     DEFAULT_PARAMS,
        'overtraining': DEFAULT_PARAMS,
    },
    init_states={
        'sedentary':    DEFAULT_INIT,
        'recovery':     DEFAULT_INIT,
        'overtraining': DEFAULT_INIT,
    },
    exogenous_inputs={
        'sedentary':    EXO_SEDENTARY,
        'recovery':     EXO_RECOVERY,
        'overtraining': EXO_OVERTRAINING,
    },
)
