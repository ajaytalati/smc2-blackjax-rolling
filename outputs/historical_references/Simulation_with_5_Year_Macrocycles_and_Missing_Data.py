"""models/fsa_real_obs/simulation.py — 3-State FSA SDE with Real Obs.

Date:    19 April 2026
Version: 1.1

Updates:
- Replaced static jump schedules with 5-year rolling macrocycles (load/deload).
- Added realistic missing data masks (background dropout, rest days, 14-day gaps)
  to stress-test the SMC^2 Particle Filter.
- Refactored `drift` and `make_aux` to accept full `T_B_arr` and `Phi_arr` 
  time-series arrays instead of static constants.
"""

import math
import numpy as np

from smc2bj.simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec,
    DIFFUSION_DIAGONAL_STATE)
from models.fsa_real_obs.sim_plots import plot_fsa_real_obs

EPS_A_FROZEN = 1.0e-4
EPS_B_FROZEN = 1.0e-4

# =========================================================================
# 5-YEAR EXOGENOUS SCHEDULE GENERATOR (from Protocol)
# =========================================================================

def generate_5_year_schedule(days=1825, seed=42):
    """Generates 5 years of procedural load/deload macrocycles."""
    rng = np.random.default_rng(seed)
    T_B = np.zeros(days, dtype=np.float32)
    Phi = np.zeros(days, dtype=np.float32)
    
    for day in range(days):
        week_of_cycle = (day % 28) // 7
        if week_of_cycle < 3:
            # Weeks 1-3: Loading Phase
            T_B[day] = rng.uniform(0.6, 0.85)
            Phi[day] = rng.uniform(0.08, 0.15)
        else:
            # Week 4: Deload / Taper Phase
            T_B[day] = rng.uniform(0.3, 0.5)
            Phi[day] = rng.uniform(0.01, 0.04)
            
        # Add ~10% daily organic noise
        T_B[day] *= rng.uniform(0.9, 1.1)
        Phi[day] *= rng.uniform(0.9, 1.1)
        
    return {'T_B_arr': T_B, 'Phi_arr': Phi}

# Generate the global 5-year dataset config
EXO_5_YEAR_MACRO = generate_5_year_schedule(days=1825, seed=42)

# =========================================================================
# MISSING DATA GENERATOR
# =========================================================================

def get_missing_data_mask(t_grid, channel_type, seed):
    """Generates boolean masks to punch realistic holes in the synthetic data."""
    rng = np.random.default_rng(seed)
    days = len(t_grid)
    present = np.ones(days, dtype=bool)

    if channel_type == 'continuous':
        # 1. Background Dropout (15%) for RHR, Stress, Sleep
        dropouts = rng.random(days) < 0.15
        present[dropouts] = False
        
        # 2. "Broken Watch" Gap (14 continuous days missing)
        if days > 30:
            start_gap = rng.integers(10, days - 15)
            present[start_gap:start_gap+14] = False

    elif channel_type == 'exercise':
        # 3. Rest Days (2-3 days a week missing for Intensity, Duration, Timing)
        for week in range(days // 7 + 1):
            num_rest = rng.integers(2, 4)
            rest_days = rng.choice(7, size=num_rest, replace=False)
            for d in rest_days:
                idx = week * 7 + d
                if idx < days:
                    present[idx] = False

    return present

# =========================================================================
# DRIFT (Updated to read from full arrays)
# =========================================================================

def drift(t, y, params, aux):
    T_B_arr, Phi_arr = aux
    idx = min(int(t), len(T_B_arr) - 1)
    
    T_B_t = T_B_arr[idx]
    Phi_t = Phi_arr[idx]

    p = params
    B, F, A = y[0], y[1], y[2]

    mu = (p['mu_0'] + p['mu_B'] * B - p['mu_F'] * F - p['mu_FF'] * F * F)
    dB = (1.0 + p['alpha_A'] * A) / p['tau_B'] * (T_B_t - B)
    dF = Phi_t - (1.0 + p['lambda_B'] * B + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return np.array([dB, dF, dA])

def drift_jax(t, y, args):
    import jax.numpy as jnp
    p, T_B_arr, Phi_arr = args
    
    # Fast array lookup by day index in JAX
    idx = jnp.minimum(t.astype(jnp.int32), len(T_B_arr) - 1)
    T_B_t = T_B_arr[idx]
    Phi_t = Phi_arr[idx]

    B, F, A = y[0], y[1], y[2]
    mu = (p['mu_0'] + p['mu_B'] * B - p['mu_F'] * F - p['mu_FF'] * F * F)
    dB = (1.0 + p['alpha_A'] * A) / p['tau_B'] * (T_B_t - B)
    dF = Phi_t - (1.0 + p['lambda_B'] * B + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return jnp.array([dB, dF, dA])

def make_aux(params, init_state, t_grid, exogenous):
    del params, init_state, t_grid
    return (np.asarray(exogenous['T_B_arr']), np.asarray(exogenous['Phi_arr']))

def make_aux_jax(params, init_state, t_grid, exogenous):
    import jax.numpy as jnp
    del init_state, t_grid
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax, jnp.asarray(exogenous['T_B_arr']), jnp.asarray(exogenous['Phi_arr']))

# =========================================================================
# DIFFUSION & HELPERS
# =========================================================================
def diffusion_diagonal(params):
    return np.array([params['sigma_B'], params['sigma_F'], params['sigma_A']])

def noise_scale_fn(y, params):
    del params
    B = np.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F = max(y[1], 0.0)
    A = max(y[2], 0.0)
    return np.array([math.sqrt(B * (1.0 - B)), math.sqrt(F), math.sqrt(A + EPS_A_FROZEN)])

def noise_scale_fn_jax(y, params):
    import jax.numpy as jnp
    del params
    B = jnp.clip(y[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F = jnp.maximum(y[1], 0.0)
    A = jnp.maximum(y[2], 0.0)
    return jnp.array([jnp.sqrt(B * (1.0 - B)), jnp.sqrt(F), jnp.sqrt(A + EPS_A_FROZEN)])

def make_y0(init_dict, params):
    del params
    return np.array([init_dict['B_0'], init_dict['F_0'], init_dict['A_0']])

# =========================================================================
# OBSERVATION CHANNELS (Now with missing data masks applied)
# =========================================================================

def gen_obs_RHR(trajectory, t_grid, params, aux, prior_channels, seed):
    rng = np.random.default_rng(seed)
    B, F = trajectory[:, 0], trajectory[:, 1]
    RHR_true = params['R_base'] - params['kappa_vagal'] * B + params['kappa_chronic'] * F
    noise = rng.normal(0.0, params['sigma_obs_R'], size=len(t_grid))
    
    mask = get_missing_data_mask(t_grid, 'continuous', seed + 1)
    return {
        't_idx': np.arange(len(t_grid), dtype=np.int32)[mask],
        'obs_value': (RHR_true + noise).astype(np.float32)[mask]
    }

def gen_obs_intensity(trajectory, t_grid, params, aux, prior_channels, seed):
    rng = np.random.default_rng(seed)
    B, F = trajectory[:, 0], trajectory[:, 1]
    I_true = params['I_base'] + params['c_B'] * B - params['c_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_I'], size=len(t_grid))
    
    mask = get_missing_data_mask(t_grid, 'exercise', seed + 2)
    return {
        't_idx': np.arange(len(t_grid), dtype=np.int32)[mask],
        'obs_value': (I_true + noise).astype(np.float32)[mask]
    }

def gen_obs_duration(trajectory, t_grid, params, aux, prior_channels, seed):
    rng = np.random.default_rng(seed)
    B, F = trajectory[:, 0], trajectory[:, 1]
    D_true = params['D_base'] + params['d_B'] * B - params['d_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_D'], size=len(t_grid))
    
    mask = get_missing_data_mask(t_grid, 'exercise', seed + 3) # Same shape as intensity ideally
    return {
        't_idx': np.arange(len(t_grid), dtype=np.int32)[mask],
        'obs_value': (D_true + noise).astype(np.float32)[mask]
    }

def gen_obs_stress(trajectory, t_grid, params, aux, prior_channels, seed):
    rng = np.random.default_rng(seed)
    F, A = trajectory[:, 1], trajectory[:, 2]
    S_true = params['S_base'] - params['s_A'] * A + params['s_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_S'], size=len(t_grid))
    
    mask = get_missing_data_mask(t_grid, 'continuous', seed + 4)
    return {
        't_idx': np.arange(len(t_grid), dtype=np.int32)[mask],
        'obs_value': (S_true + noise).astype(np.float32)[mask]
    }

def gen_obs_sleep(trajectory, t_grid, params, aux, prior_channels, seed):
    rng = np.random.default_rng(seed)
    B, F, A = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    Sleep_true = params['Sleep_base'] + params['sl_A'] * A + params['sl_B'] * B - params['sl_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_Sleep'], size=len(t_grid))
    
    mask = get_missing_data_mask(t_grid, 'continuous', seed + 5)
    return {
        't_idx': np.arange(len(t_grid), dtype=np.int32)[mask],
        'obs_value': (Sleep_true + noise).astype(np.float32)[mask]
    }

def gen_obs_timing(trajectory, t_grid, params, aux, prior_channels, seed):
    rng = np.random.default_rng(seed)
    F, A = trajectory[:, 1], trajectory[:, 2]
    Time_true = params['Time_base'] + params['t_A'] * A - params['t_F'] * F
    noise = rng.normal(0.0, params['sigma_obs_Time'], size=len(t_grid))
    
    mask = get_missing_data_mask(t_grid, 'exercise', seed + 6)
    return {
        't_idx': np.arange(len(t_grid), dtype=np.int32)[mask],
        'obs_value': (Time_true + noise).astype(np.float32)[mask]
    }

def gen_T_B_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    T_B_arr, _ = aux
    return {'t_idx': np.arange(len(t_grid), dtype=np.int32), 'T_B_value': T_B_arr.astype(np.float32)}

def gen_Phi_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    _, Phi_arr = aux
    return {'t_idx': np.arange(len(t_grid), dtype=np.int32), 'Phi_value': Phi_arr.astype(np.float32)}

# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================
def verify_physics(trajectory, t_grid, params):
    B, F, A = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]
    mu_traj = params['mu_0'] + params['mu_B']*B - params['mu_F']*F - params['mu_FF']*F*F
    return {
        'mu_crosses_zero': bool(mu_traj.min() < 0 < mu_traj.max()),
        'all_finite': bool(np.all(np.isfinite(trajectory))),
    }

# =========================================================================
# ASSEMBLE MODEL
# =========================================================================

DEFAULT_PARAMS = {
    'tau_B': 14.0, 'alpha_A': 1.0, 'tau_F': 7.0, 'lambda_B': 3.0, 'lambda_A': 1.5,
    'mu_0': -0.10, 'mu_B': 0.30, 'mu_F': 0.10, 'mu_FF': 0.40, 'eta': 0.20,
    'sigma_B': 0.01, 'sigma_F': 0.005, 'sigma_A': 0.02,
    'R_base': 62.0, 'kappa_vagal': 12.0, 'kappa_chronic': 10.0, 'sigma_obs_R': 1.5,
    'I_base': 0.5, 'c_B': 0.2, 'c_F': 0.1, 'sigma_obs_I': 0.05,
    'D_base': 0.5, 'd_B': 0.3, 'd_F': 0.2, 'sigma_obs_D': 0.08,
    'S_base': 30.0, 's_A': 15.0, 's_F': 20.0, 'sigma_obs_S': 5.0,
    'Sleep_base': 0.5, 'sl_A': 0.2, 'sl_B': 0.1, 'sl_F': 0.2, 'sigma_obs_Sleep': 0.1,
    'Time_base': 0.0, 't_A': 1.0, 't_F': 0.5, 'sigma_obs_Time': 0.5,
}

DEFAULT_INIT = {'B_0': 0.05, 'F_0': 0.10, 'A_0': 0.01}

FSA_REAL_OBS_MODEL = SDEModel(
    name="fsa_real_obs",
    version="1.1",
    states=(StateSpec("B", 0.0, 1.0), StateSpec("F", 0.0, 10.0), StateSpec("A", 0.0, 5.0)),
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
    param_sets={'five_year_macro': DEFAULT_PARAMS},
    init_states={'five_year_macro': DEFAULT_INIT},
    exogenous_inputs={'five_year_macro': EXO_5_YEAR_MACRO},
)