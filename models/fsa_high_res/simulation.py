"""
models/fsa_high_res/simulation.py — High-resolution (15-min) FSA variant.
==========================================================================

Same latent dynamics as ``models/fsa_real_obs/`` (3-state B/F/A SDE) but:

  - Observation grid is **15-min bins** (dt = 1/96 day) rather than daily.
  - Exogenous Phi(t) and T_B(t) are per-bin arrays (not piecewise-constant
    with T_jump); driven by a training-burst pattern that's zero overnight.
  - Deterministic circadian forcing C(t) = sin(2 pi t / 1 day + phi) enters
    the observation links to give the sub-daily data diurnal structure.
    It is NOT a latent state — just a time function.

  - Observations (4 channels, SWAT-style):
      HR     ~ N(HR_base - kappa_B * B + alpha_A_HR * A + beta_C_HR * C(t), sigma_HR^2)
                                           [Gaussian, sleep-gated]
      sleep  ~ Bernoulli(sigmoid(k_C * C(t) + k_A * A - c_tilde))
                                           [Bernoulli, always observed]
      stress ~ N(S_base + k_F * F - k_A_S * A - beta_C_S * C(t), sigma_S^2)
                                           [Gaussian, wake-gated]
      steps  ~ log-Normal: log(steps+1) ~ N(mu_step0 + beta_B_st * B - beta_F_st * F
                                            + beta_A_st * A + beta_C_st * C(t), sigma_st^2)
                                           [log-Gaussian, wake-gated]

Circadian phase `phi` is frozen at 0 for the proof-of-principle; the rest
of the obs coefficients are estimated.
"""

import math
import numpy as np

from smc2bj.simulator.sde_model import (
    SDEModel, StateSpec, ChannelSpec,
    DIFFUSION_DIAGONAL_STATE)
from models.fsa_high_res.sim_plots import plot_fsa_high_res


# =========================================================================
# FROZEN CONSTANTS
# =========================================================================

EPS_A_FROZEN = 1.0e-4
EPS_B_FROZEN = 1.0e-4

# Time-grid constants — one "day" is 1.0 in t-units.
DT_BIN_DAYS = 1.0 / 96.0           # 15 minutes
DT_BIN_HOURS = 24.0 / 96.0         # = 0.25
BINS_PER_DAY = 96


# =========================================================================
# Circadian forcing (deterministic, not a state)
# =========================================================================

def circadian(t_days, phi=0.0):
    """C(t) = cos(2 pi t + phi), t in days. Period = 1 day.

    With phi=0: peak at midnight (t_days integer), trough at noon.
    Physiological convention: a "healthy morning type" has phi ≈ 0 and
    sleep aligns with C > 0, waking peak activity with C < 0.
    """
    return np.cos(2.0 * np.pi * t_days + phi)


def circadian_jax(t_days, phi=0.0):
    import jax.numpy as jnp
    return jnp.cos(2.0 * jnp.pi * t_days + phi)


# =========================================================================
# Exogenous Phi(t) — sub-daily with training bursts
# =========================================================================

def generate_phi_sub_daily(daily_phi, seed=42,
                            wake_hour=7.0, sleep_hour=23.0,
                            peak_hours_post_wake=2.0, tau_hours=3.0,
                            noise_frac=0.15):
    """Expand a per-day Phi schedule to a per-15-min-bin array with a
    **morning-loaded** activity profile (hunter-gatherer pattern).

    Profile for each wake bin:
        t = hour_of_day - wake_hour      (zero at wake)
        shape(t) = t * exp(-t / tau)      (Gamma(k=2) shape)
                                          peaks at t = tau (~3h post-wake)
    Normalised so the daily-integrated Phi equals ~24 * daily_phi
    (i.e. the slow FSA dynamics see the same daily load as the base model).

    Concretely: activity ramps from 0 at wake, peaks 2-3h later, and
    tapers exponentially through the afternoon. By ~8h post-wake (3pm if
    waking at 7am) activity is ~20% of peak; by sleep (16h post-wake)
    ~0.3% of peak. Matches the "~75% of activity in the morning" pattern.

    Sleep hours [sleep_hour, wake_hour+24] (wrapping midnight): Phi = 0.

    Small multiplicative Gaussian noise (noise_frac std) is added per bin
    to avoid an unrealistically smooth signal.
    """
    rng = np.random.default_rng(seed)
    n_days = len(daily_phi)
    phi = np.zeros(n_days * BINS_PER_DAY, dtype=np.float32)

    wake_duration = sleep_hour - wake_hour  # 16.0 h

    # Normalisation: ∫_0^T t * exp(-t/tau) dt = tau^2 * (1 - e^{-T/tau} * (1 + T/tau))
    T = wake_duration
    gamma_integral = tau_hours ** 2 * (
        1.0 - np.exp(-T / tau_hours) * (1.0 + T / tau_hours)
    )
    # Want daily_phi_integral(per-day) = daily_phi * 24 (so slow-scale FSA sees the same load).
    # sum of Phi(t)*dt over wake hours = amplitude * gamma_integral.
    # amplitude = daily_phi * 24 / gamma_integral.

    for d in range(n_days):
        phi_d = float(daily_phi[d])
        amplitude = phi_d * 24.0 / max(gamma_integral, 1e-12)
        for k in range(BINS_PER_DAY):
            h = k * DT_BIN_HOURS          # hour-of-day in [0, 24)
            if h < wake_hour or h >= sleep_hour:
                phi[d * BINS_PER_DAY + k] = 0.0
                continue
            t = h - wake_hour
            shape = t * np.exp(-t / tau_hours)
            base = amplitude * shape
            noise = rng.normal(0.0, noise_frac) if noise_frac > 0 else 0.0
            phi[d * BINS_PER_DAY + k] = max(base * (1.0 + noise), 0.0)

    return phi


def sleep_mask_from_hours(n_days, sleep_hour_lo=23.0, sleep_hour_hi=7.0):
    """Deterministic a-priori sleep mask (1 if 'nominally asleep' at bin).

    Used to build T_B if we wanted T_B-during-sleep=0; otherwise just a
    helper for callers. The actual sleep OBSERVATION is Bernoulli — this
    is the training-burst zero-out mask.
    """
    mask = np.zeros(n_days * BINS_PER_DAY, dtype=np.float32)
    for d in range(n_days):
        for k in range(BINS_PER_DAY):
            h = k * DT_BIN_HOURS
            in_sleep = (h >= sleep_hour_lo) or (h < sleep_hour_hi)
            mask[d * BINS_PER_DAY + k] = 1.0 if in_sleep else 0.0
    return mask


# =========================================================================
# DRIFT (same dynamics as FSA; aux now holds per-bin T_B and Phi arrays)
# =========================================================================

def _bin_lookup(t_days, array, dt_bin_days=DT_BIN_DAYS):
    k = int(t_days / dt_bin_days)
    k = max(0, min(k, len(array) - 1))
    return float(array[k])


def drift(t, y, params, aux):
    """Classical numpy drift for scipy / Euler-Maruyama. t in days."""
    T_B_arr, Phi_arr = aux
    p = params
    B = y[0]; F = y[1]; A = y[2]

    T_B_t = _bin_lookup(t, T_B_arr)
    Phi_t = _bin_lookup(t, Phi_arr)

    mu = (p['mu_0'] + p['mu_B'] * B
          - p['mu_F'] * F - p['mu_FF'] * F * F)

    dB = (1.0 + p['alpha_A'] * A) / p['tau_B'] * (T_B_t - B)
    dF = Phi_t - (1.0 + p['lambda_B'] * B
                  + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return np.array([dB, dF, dA])


def drift_jax(t, y, args):
    """JAX drift. t in days, args = (params_dict, T_B_arr, Phi_arr)."""
    import jax.numpy as jnp
    p, T_B_arr, Phi_arr = args
    B = y[0]; F = y[1]; A = y[2]

    # Bin lookup via integer index
    k = jnp.clip((t / DT_BIN_DAYS).astype(jnp.int32), 0, T_B_arr.shape[0] - 1)
    T_B_t = T_B_arr[k]
    Phi_t = Phi_arr[k]

    mu = (p['mu_0'] + p['mu_B'] * B
          - p['mu_F'] * F - p['mu_FF'] * F * F)

    dB = (1.0 + p['alpha_A'] * A) / p['tau_B'] * (T_B_t - B)
    dF = Phi_t - (1.0 + p['lambda_B'] * B
                  + p['lambda_A'] * A) / p['tau_F'] * F
    dA = mu * A - p['eta'] * A * A * A

    return jnp.array([dB, dF, dA])


# =========================================================================
# DIFFUSION (state-dependent, same as FSA)
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
# AUX / INITIAL STATE
# =========================================================================

def make_aux(params, init_state, t_grid, exogenous):
    """aux = (T_B_arr, Phi_arr) — per-bin arrays for drift lookup."""
    del params, init_state, t_grid
    return (np.asarray(exogenous['T_B_arr'], dtype=np.float32),
            np.asarray(exogenous['Phi_arr'], dtype=np.float32))


def make_aux_jax(params, init_state, t_grid, exogenous):
    import jax.numpy as jnp
    del init_state, t_grid
    p_jax = {k: jnp.float64(v) for k, v in params.items()}
    return (p_jax,
            jnp.asarray(exogenous['T_B_arr'], dtype=jnp.float32),
            jnp.asarray(exogenous['Phi_arr'], dtype=jnp.float32))


def make_y0(init_dict, params):
    del params
    return np.array([init_dict['B_0'], init_dict['F_0'], init_dict['A_0']])


# =========================================================================
# OBSERVATION CHANNELS (4 physiological + 2 exogenous broadcasts)
# =========================================================================

def _sleep_prob(A, C, k_C, k_A, c_tilde):
    z = k_C * C + k_A * A - c_tilde
    return 1.0 / (1.0 + np.exp(-z))


def gen_obs_sleep(trajectory, t_grid, params, aux, prior_channels, seed):
    """Bernoulli sleep label at every 15-min bin.

    Always observed. Sleep probability is sigmoid(k_C * C(t) + k_A * A - c_tilde).
    Also stores the binary label as 'sleep_label' (not 'obs_value') so the
    estimator can distinguish.
    """
    del aux, prior_channels
    rng = np.random.default_rng(seed)
    A = trajectory[:, 2]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
    labels = (rng.random(len(t_grid)) < p).astype(np.int32)
    return {
        't_idx':       np.arange(len(t_grid), dtype=np.int32),
        'sleep_label': labels,
    }


def gen_obs_hr(trajectory, t_grid, params, aux, prior_channels, seed):
    """HR, Gaussian, measured only during sleep.

    hr = HR_base - kappa_B * B + alpha_A_HR * A + beta_C_HR * C(t) + noise.

    Gated by the sleep_label from prior_channels['obs_sleep'] if present;
    if not present, uses the underlying sleep probability threshold at 0.5.
    """
    del aux
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    A = trajectory[:, 2]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    hr_mean = (params['HR_base']
               - params['kappa_B'] * B
               + params['alpha_A_HR'] * A
               + params['beta_C_HR'] * C)
    hr_obs = hr_mean + rng.normal(0.0, params['sigma_HR'], size=len(t_grid))

    # Sleep gating
    if prior_channels is not None and 'obs_sleep' in prior_channels:
        sleep_label = prior_channels['obs_sleep']['sleep_label']
        present = sleep_label.astype(np.int32)
    else:
        # Fallback: threshold on sleep prob
        p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
        present = (p > 0.5).astype(np.int32)

    idx_present = np.where(present == 1)[0]
    return {
        't_idx':     idx_present.astype(np.int32),
        'obs_value': hr_obs[idx_present].astype(np.float32),
    }


def gen_obs_stress(trajectory, t_grid, params, aux, prior_channels, seed):
    """Stress, Gaussian, measured only during waking.

    stress = S_base + k_F * F - k_A_S * A - beta_C_S * C(t) + noise
    """
    del aux
    rng = np.random.default_rng(seed)
    F = trajectory[:, 1]
    A = trajectory[:, 2]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    s_mean = (params['S_base']
              + params['k_F'] * F
              - params['k_A_S'] * A
              + params['beta_C_S'] * C)
    s_obs = s_mean + rng.normal(0.0, params['sigma_S'], size=len(t_grid))

    # Wake gating
    if prior_channels is not None and 'obs_sleep' in prior_channels:
        sleep_label = prior_channels['obs_sleep']['sleep_label']
        present = (1 - sleep_label).astype(np.int32)
    else:
        p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
        present = (p <= 0.5).astype(np.int32)

    idx_present = np.where(present == 1)[0]
    return {
        't_idx':     idx_present.astype(np.int32),
        'obs_value': s_obs[idx_present].astype(np.float32),
    }


def gen_obs_steps(trajectory, t_grid, params, aux, prior_channels, seed):
    """Step count, log-Gaussian, measured only during waking.

    log(steps + 1) = mu_step0 + beta_B_st * B - beta_F_st * F
                     + beta_A_st * A + beta_C_st * C(t) + noise.
    """
    del aux
    rng = np.random.default_rng(seed)
    B = trajectory[:, 0]
    F = trajectory[:, 1]
    A = trajectory[:, 2]
    C = circadian(t_grid, phi=params.get('phi', 0.0))
    log_mean = (params['mu_step0']
                + params['beta_B_st'] * B
                - params['beta_F_st'] * F
                + params['beta_A_st'] * A
                + params['beta_C_st'] * C)
    log_obs = log_mean + rng.normal(0.0, params['sigma_st'], size=len(t_grid))
    # Count is exp(log_obs) - 1, clipped at 0.
    step_count = np.maximum(np.exp(log_obs) - 1.0, 0.0)

    # Wake gating
    if prior_channels is not None and 'obs_sleep' in prior_channels:
        sleep_label = prior_channels['obs_sleep']['sleep_label']
        present = (1 - sleep_label).astype(np.int32)
    else:
        p = _sleep_prob(A, C, params['k_C'], params['k_A'], params['c_tilde'])
        present = (p <= 0.5).astype(np.int32)

    idx_present = np.where(present == 1)[0]
    return {
        't_idx':     idx_present.astype(np.int32),
        'obs_value': step_count[idx_present].astype(np.float32),
    }


def gen_T_B_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Broadcast per-bin T_B from aux."""
    del trajectory, params, prior_channels, seed
    T_B_arr, _Phi_arr = aux
    val = np.asarray(T_B_arr, dtype=np.float32)[: len(t_grid)]
    return {'t_idx':    np.arange(len(t_grid), dtype=np.int32),
            'T_B_value': val}


def gen_Phi_channel(trajectory, t_grid, params, aux, prior_channels, seed):
    """Broadcast per-bin Phi from aux."""
    del trajectory, params, prior_channels, seed
    _T_B_arr, Phi_arr = aux
    val = np.asarray(Phi_arr, dtype=np.float32)[: len(t_grid)]
    return {'t_idx':     np.arange(len(t_grid), dtype=np.int32),
            'Phi_value': val}


# =========================================================================
# PHYSICS VERIFICATION
# =========================================================================

def _mu_of(B, F, p):
    return p['mu_0'] + p['mu_B'] * B - p['mu_F'] * F - p['mu_FF'] * F * F


def verify_physics(trajectory, t_grid, params):
    B = trajectory[:, 0]; F = trajectory[:, 1]; A = trajectory[:, 2]
    mu_traj = _mu_of(B, F, params)
    return {
        'B_min': float(B.min()), 'B_max': float(B.max()),
        'F_min': float(F.min()), 'F_max': float(F.max()),
        'A_min': float(A.min()), 'A_max': float(A.max()),
        'B_final': float(B[-1]), 'F_final': float(F[-1]), 'A_final': float(A[-1]),
        'mu_min': float(mu_traj.min()), 'mu_max': float(mu_traj.max()),
        'mu_crosses_zero': bool(mu_traj.min() < 0 < mu_traj.max()),
        'all_finite': bool(np.all(np.isfinite(trajectory))),
    }


# =========================================================================
# PARAMETERS (truth values for the recovery scenario)
# =========================================================================

DEFAULT_PARAMS = {
    # --- SDE dynamics ---
    # Tuned from FSA v4.1 defaults to activate the Landau bifurcation
    # within a 14-day POC horizon: mu_0 less negative (-0.05 vs -0.10)
    # so mu(B,F) crosses zero earlier as B rises.
    'tau_B':    14.0,
    'alpha_A':   1.0,
    'tau_F':     7.0,
    'lambda_B':  3.0,
    'lambda_A':  1.5,
    'mu_0':      0.02,
    'mu_B':      0.30,
    'mu_F':      0.10,
    'mu_FF':     0.40,
    'eta':       0.20,
    'sigma_B':   0.01,
    'sigma_F':   0.005,
    'sigma_A':   0.02,

    # --- Circadian (frozen) ---
    'phi':       0.0,

    # --- Ch1: HR (sleep-gated) ---
    'HR_base':      62.0,
    'kappa_B':      12.0,     # fitness lowers HR during sleep (vagal)
    'alpha_A_HR':    3.0,     # A raises HR mildly
    'beta_C_HR':    -2.5,     # circadian dip at t*24 = 3-6am (late-night nadir)
    'sigma_HR':      2.0,

    # --- Ch2: Sleep (Bernoulli) ---
    'k_C':           3.0,     # circadian dominates (sin(2pi t) is negative overnight → need sign flip; see plan)
    'k_A':           2.0,     # A modulates sleep quality
    'c_tilde':       0.5,

    # --- Ch3: Stress (wake-gated). All coeffs as additive; cos-circadian
    # peaks at midnight, so negative beta_C_S → stress up at noon ---
    'S_base':       30.0,
    'k_F':          20.0,     # strain raises stress
    'k_A_S':         8.0,     # A lowers stress
    'beta_C_S':     -4.0,     # stress peaks when C negative (noon)
    'sigma_S':       4.0,

    # --- Ch4: Steps (log-Gaussian, wake-gated) ---
    'mu_step0':      5.5,     # ~e^5.5 ≈ 245 steps/bin baseline = ~980 steps/hour
    'beta_B_st':     0.8,     # fitness enables more steps
    'beta_F_st':     0.5,     # fatigue suppresses
    'beta_A_st':     0.3,     # A boosts activity
    'beta_C_st':    -0.8,     # steps peak when C negative (waking / afternoon)
    'sigma_st':      0.5,
}

DEFAULT_INIT = {'B_0': 0.05, 'F_0': 0.10, 'A_0': 0.55}
# A_0 raised from 0.01 → 0.55 to start A near the Stuart-Landau
# fixed point A* = sqrt(mu/eta) for typical (B, F). With mu_0=+0.02,
# mu_B=0.30, mu_F=0.10, eta=0.20, and mid-run (B,F)=(0.3, 0.15),
# mu ≈ 0.095 → A* ≈ 0.69. We start below A* so there's some
# convergence to watch, but above the quasi-absorbing boundary.


# =========================================================================
# SCENARIO PRESETS (scenario distinction collapsed; single entry)
# =========================================================================

EXO_RECOVERY = {
    # placeholder — driver generates T_B_arr, Phi_arr at runtime
    'T_B_arr': np.array([0.6], dtype=np.float32),
    'Phi_arr': np.array([0.03], dtype=np.float32),
}


# =========================================================================
# THE MODEL OBJECT
# =========================================================================

HIGH_RES_FSA_MODEL = SDEModel(
    name="fsa_high_res",
    version="0.1",

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
        # Sleep is generated FIRST so HR/stress/steps can use it for gating.
        ChannelSpec("obs_sleep",  depends_on=(),             generate_fn=gen_obs_sleep),
        ChannelSpec("obs_HR",     depends_on=("obs_sleep",), generate_fn=gen_obs_hr),
        ChannelSpec("obs_stress", depends_on=("obs_sleep",), generate_fn=gen_obs_stress),
        ChannelSpec("obs_steps",  depends_on=("obs_sleep",), generate_fn=gen_obs_steps),
        ChannelSpec("T_B",        depends_on=(),             generate_fn=gen_T_B_channel),
        ChannelSpec("Phi",        depends_on=(),             generate_fn=gen_Phi_channel),
    ),

    plot_fn=plot_fsa_high_res,
    verify_physics_fn=verify_physics,

    param_sets={'recovery': DEFAULT_PARAMS},
    init_states={'recovery': DEFAULT_INIT},
    exogenous_inputs={'recovery': EXO_RECOVERY},
)
