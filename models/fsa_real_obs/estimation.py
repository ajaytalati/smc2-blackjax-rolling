"""models/fsa_real_obs/estimation.py — FSA with Real Observation Channels.

Date:    19 April 2026
Version: 1.0

Estimation model for the 3-state FSA SDE with 6 physiological
observation channels replacing the direct-state observations.

LATENT SDE (same as base FSA)
-----------------------------
    dB = (1 + alpha_A A)/tau_B * (T_B(t) - B) dt
         + sigma_B * sqrt(B(1-B)) dW_B

    dF = [Phi(t) - (1 + lambda_B B + lambda_A A)/tau_F * F] dt
         + sigma_F * sqrt(F) dW_F

    dA = [mu(B,F) A - eta A^3] dt
         + sigma_A * sqrt(A + eps_A) dW_A

    mu(B, F) = mu_0 + mu_B B - mu_F F - mu_FF F^2

OBSERVATION MODEL (6 channels)
------------------------------
    Ch1 RHR:       -kappa_vagal*B + kappa_chronic*F + N(0, sigma_obs_R^2)  [mean-centered]
    Ch2 Intensity: I_base + c_B*B - c_F*F + N(0, sigma_obs_I^2)
    Ch3 Duration:  D_base + d_B*B - d_F*F + N(0, sigma_obs_D^2)
    Ch4 Stress:    S_base - s_A*A + s_F*F + N(0, sigma_obs_S^2)
    Ch5 Sleep:     Sleep_base + sl_A*A + sl_B*B - sl_F*F + N(0, sigma_obs_Sleep^2)
    Ch6 Timing:    Time_base + t_A*A - t_F*F + N(0, sigma_obs_Time^2)

Estimated parameters (34 total):
    10 dynamical: tau_B, alpha_A, tau_F, lambda_B, lambda_A,
                  mu_0_abs, mu_B, mu_F, mu_FF, eta
    24 obs model: kappa_vagal, kappa_chronic, sigma_obs_R,
                  I_base, c_B, c_F, sigma_obs_I,
                  D_base, d_B, d_F, sigma_obs_D,
                  S_base, s_A, s_F, sigma_obs_S,
                  Sleep_base, sl_A, sl_B, sl_F, sigma_obs_Sleep,
                  Time_base, t_A, t_F, sigma_obs_Time

Initial states (3): B_0, F_0, A_0

Total: 38 estimated dimensions.

Proposal: JOINT GAUSSIAN GUIDED via sequential scalar Kalman fusion.
All 6 channels are LINEAR in (B, F, A), so Gaussian fusion is exact:
    obs_i = H_i @ [B, F, A] + bias_i + N(0, sigma_i^2)
The propagate_fn builds H (6x3), performs 6 sequential Kalman updates
on the SDE-predicted prior N(x_pred, P_process), samples from the
fused N(mu_fused, P_fused), and returns a weight correction so that
the total weight equals the sample-independent predictive likelihood.

Frozen constants (not estimated):
    EPS_A_FROZEN, EPS_B_FROZEN, SIGMA_B_FROZEN, SIGMA_F_FROZEN,
    SIGMA_A_FROZEN
"""

import math
import numpy as np
from collections import OrderedDict

import jax
import jax.numpy as jnp

from smc2bj.estimation_model import EstimationModel
from smc2bj._likelihood_constants import HALF_LOG_2PI


# =========================================================================
# FROZEN CONSTANTS (same as base FSA)
# =========================================================================

EPS_A_FROZEN    = 1.0e-4
EPS_B_FROZEN    = 1.0e-4
SIGMA_B_FROZEN  = 0.01
SIGMA_F_FROZEN  = 0.005
SIGMA_A_FROZEN  = 0.02


# =========================================================================
# PRIORS
# =========================================================================

PARAM_PRIOR_CONFIG = OrderedDict([
    # --- SDE dynamics (same priors as base FSA) ---
    ('tau_B',     ('lognormal', (math.log(14.0), 0.08))),  # tightened: 95% CI ≈ [11.9, 16.4]
    ('alpha_A',   ('lognormal', (math.log( 1.0), 0.4))),
    ('tau_F',     ('lognormal', (math.log( 7.0), 0.3))),
    ('lambda_B',  ('lognormal', (math.log( 3.0), 0.3))),
    ('lambda_A',  ('lognormal', (math.log( 1.5), 0.3))),
    ('mu_0_abs',  ('lognormal', (math.log(0.10), 0.4))),
    ('mu_B',      ('lognormal', (math.log(0.30), 0.4))),
    ('mu_F',      ('lognormal', (math.log(0.10), 0.4))),
    ('mu_FF',     ('lognormal', (math.log(0.40), 0.4))),
    ('eta',       ('lognormal', (math.log(0.20), 0.3))),

    # --- Ch1: RHR (R_base & kappa_chronic frozen for identifiability) ---
    ('kappa_vagal',   ('lognormal', (math.log(12.0), 0.3))),   # truth=12.0
    ('sigma_obs_R',   ('lognormal', (math.log(1.5), 0.4))),

    # --- Ch2: Intensity ---
    ('I_base',       ('normal',    (0.5, 0.1))),
    ('c_B',          ('lognormal', (math.log(0.2), 0.5))),
    ('c_F',          ('lognormal', (math.log(0.1), 0.5))),
    ('sigma_obs_I',  ('lognormal', (math.log(0.05), 0.4))),

    # --- Ch3: Duration ---
    ('D_base',       ('normal',    (0.5, 0.1))),
    ('d_B',          ('lognormal', (math.log(0.3), 0.5))),
    ('d_F',          ('lognormal', (math.log(0.2), 0.5))),
    ('sigma_obs_D',  ('lognormal', (math.log(0.08), 0.4))),

    # --- Ch4: Stress ---
    ('S_base',       ('normal',    (30.0, 10.0))),
    ('s_A',          ('lognormal', (math.log(15.0), 0.5))),
    ('s_F',          ('lognormal', (math.log(20.0), 0.5))),
    ('sigma_obs_S',  ('lognormal', (math.log(5.0), 0.4))),

    # --- Ch5: Sleep ---
    ('Sleep_base',      ('normal',    (0.5, 0.1))),
    ('sl_A',            ('lognormal', (math.log(0.2), 0.5))),
    ('sl_B',            ('lognormal', (math.log(0.1), 0.5))),
    ('sl_F',            ('lognormal', (math.log(0.2), 0.5))),
    ('sigma_obs_Sleep', ('lognormal', (math.log(0.1), 0.4))),

    # --- Ch6: Timing ---
    ('Time_base',      ('normal',    (0.0, 1.0))),
    ('t_A',            ('lognormal', (math.log(1.0), 0.5))),
    ('t_F',            ('lognormal', (math.log(0.5), 0.5))),
    ('sigma_obs_Time', ('lognormal', (math.log(0.5), 0.4))),
])

# Init states are NOT estimated — fixed externally (see COLD_START_INIT)
INIT_STATE_PRIOR_CONFIG = OrderedDict()

# Cold-start initial conditions (prior means from v3)
COLD_START_INIT = jnp.array([0.05, 0.10, 0.01])  # [B_0, F_0, A_0]

# kappa_chronic frozen — structurally non-identifiable jointly with kappa_vagal
# from a single mean-centered RHR channel when B and F are correlated
KAPPA_CHRONIC_FROZEN = 10.0
R_BASE_FROZEN        = 62.0

_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# =========================================================================
# SDE DYNAMICS — JOINT GAUSSIAN GUIDED PROPOSAL (Kalman fusion)
# =========================================================================

def propagate_fn(y, t, dt, params, grid_obs, k,
                 sigma_diag, noise, rng_key):
    """Joint Gaussian guided proposal via sequential scalar Kalman fusion.

    All 6 observation channels are linear in (B, F, A):
        obs_i = H_i @ [B, F, A] + bias_i + N(0, r_i)

    Algorithm:
      1. Prior from SDE Euler step: N(x_pred, diag(var_B, var_F, var_A))
      2. Sequential scalar Kalman updates for each present channel
      3. Sample from fused N(mu_fused, P_fused) via Cholesky
      4. Weight correction: pred_lw = log_pred_total - obs_ll(x_new)
         so that total_weight = log_pred_total (sample-independent)

    Degrades gracefully to bootstrap when no observations are present
    (all Kalman updates are masked, pred_lw = 0).
    """
    del t, rng_key, sigma_diag

    # --- Unpack dynamics params ---
    tau_B     = params[_PI['tau_B']]
    alpha_A   = params[_PI['alpha_A']]
    tau_F     = params[_PI['tau_F']]
    lambda_B  = params[_PI['lambda_B']]
    lambda_A  = params[_PI['lambda_A']]
    mu_0      = -params[_PI['mu_0_abs']]
    mu_B      = params[_PI['mu_B']]
    mu_F      = params[_PI['mu_F']]
    mu_FF     = params[_PI['mu_FF']]
    eta       = params[_PI['eta']]

    B = y[0]; F = y[1]; A = y[2]
    T_B_k = grid_obs['T_B'][k]
    Phi_k = grid_obs['Phi'][k]

    # --- Deterministic Euler predictions ---
    mu_bif  = mu_0 + mu_B * B - mu_F * F - mu_FF * F * F
    drift_B = (1.0 + alpha_A * A) / tau_B * (T_B_k - B)
    drift_F = Phi_k - (1.0 + lambda_B * B + lambda_A * A) / tau_F * F
    drift_A = mu_bif * A - eta * A * A * A

    B_pred = B + dt * drift_B
    F_pred = F + dt * drift_F
    A_pred = A + dt * drift_A

    # --- Prior covariance from state-dependent process noise ---
    B_cl = jnp.clip(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F_cl = jnp.maximum(F, 0.0)
    A_cl = jnp.maximum(A, 0.0)

    var_B = jnp.maximum(SIGMA_B_FROZEN**2 * B_cl * (1.0 - B_cl) * dt, 1e-12)
    var_F = jnp.maximum(SIGMA_F_FROZEN**2 * F_cl * dt, 1e-12)
    var_A = jnp.maximum(SIGMA_A_FROZEN**2 * (A_cl + EPS_A_FROZEN) * dt, 1e-12)

    mu_prior = jnp.array([B_pred, F_pred, A_pred])
    P_prior  = jnp.diag(jnp.array([var_B, var_F, var_A]))

    # --- Unpack observation params ---
    kappa_vagal   = params[_PI['kappa_vagal']]
    kappa_chronic = KAPPA_CHRONIC_FROZEN
    sigma_R       = params[_PI['sigma_obs_R']]

    I_base   = params[_PI['I_base']]
    c_B      = params[_PI['c_B']]
    c_F      = params[_PI['c_F']]
    sigma_I  = params[_PI['sigma_obs_I']]

    D_base   = params[_PI['D_base']]
    d_B      = params[_PI['d_B']]
    d_F      = params[_PI['d_F']]
    sigma_D  = params[_PI['sigma_obs_D']]

    S_base   = params[_PI['S_base']]
    s_A      = params[_PI['s_A']]
    s_F      = params[_PI['s_F']]
    sigma_S  = params[_PI['sigma_obs_S']]

    Sleep_base    = params[_PI['Sleep_base']]
    sl_A          = params[_PI['sl_A']]
    sl_B          = params[_PI['sl_B']]
    sl_F          = params[_PI['sl_F']]
    sigma_Sleep   = params[_PI['sigma_obs_Sleep']]

    Time_base    = params[_PI['Time_base']]
    t_A          = params[_PI['t_A']]
    t_F          = params[_PI['t_F']]
    sigma_Time   = params[_PI['sigma_obs_Time']]

    # --- Build H matrix (6x3): obs_i = H_i @ [B,F,A] + bias_i ---
    H = jnp.array([
        [-kappa_vagal,  kappa_chronic, 0.0],       # Ch1: RHR
        [c_B,          -c_F,           0.0],       # Ch2: Intensity
        [d_B,          -d_F,           0.0],       # Ch3: Duration
        [0.0,           s_F,          -s_A],       # Ch4: Stress
        [sl_B,         -sl_F,          sl_A],      # Ch5: Sleep
        [0.0,          -t_F,           t_A],       # Ch6: Timing
    ])
    bias = jnp.array([R_BASE_FROZEN, I_base, D_base, S_base, Sleep_base, Time_base])
    R_diag = jnp.array([sigma_R**2, sigma_I**2, sigma_D**2,
                         sigma_S**2, sigma_Sleep**2, sigma_Time**2])

    # --- Observations and presence flags at step k ---
    obs_vals = jnp.array([
        grid_obs['obs_RHR_value'][k],
        grid_obs['obs_intensity_value'][k],
        grid_obs['obs_duration_value'][k],
        grid_obs['obs_stress_value'][k],
        grid_obs['obs_sleep_value'][k],
        grid_obs['obs_timing_value'][k],
    ])
    obs_pres = jnp.array([
        grid_obs['obs_RHR_present'][k],
        grid_obs['obs_intensity_present'][k],
        grid_obs['obs_duration_present'][k],
        grid_obs['obs_stress_present'][k],
        grid_obs['obs_sleep_present'][k],
        grid_obs['obs_timing_present'][k],
    ])

    # --- Sequential scalar Kalman updates (6 channels) ---
    def _kalman_step(carry, ch):
        mu, P, lp = carry
        h_i, b_i, r_i, y_i, pres_i = ch
        innov = y_i - (h_i @ mu + b_i)
        Ph    = P @ h_i                             # (3,)
        S_i   = h_i @ Ph + r_i                      # scalar
        K_i   = Ph / S_i                             # (3,)
        ll_i  = -0.5 * jnp.log(2.0 * jnp.pi * S_i) - 0.5 * innov**2 / S_i
        mu = mu + pres_i * K_i * innov
        P  = P  - pres_i * jnp.outer(K_i, Ph)
        lp = lp + pres_i * ll_i
        return (mu, P, lp), None

    (mu_fused, P_fused, log_pred_total), _ = jax.lax.scan(
        _kalman_step,
        (mu_prior, P_prior, 0.0),
        (H, bias, R_diag, obs_vals, obs_pres),
    )

    # --- Sample from fused Gaussian ---
    P_safe = P_fused + 1e-10 * jnp.eye(3)
    L = jnp.linalg.cholesky(P_safe)
    x_new = mu_fused + L @ noise

    # --- Enforce physical bounds ---
    B_new = jnp.clip(x_new[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F_new = jnp.maximum(x_new[1], 0.0)
    A_new = jnp.maximum(x_new[2], 0.0)
    y_new = jnp.array([B_new, F_new, A_new])

    # --- Weight correction: pred_lw + obs_log_weight(y_new) = log_pred ---
    preds_new  = H @ y_new + bias
    resids_new = obs_vals - preds_new
    obs_ll_new = jnp.sum(obs_pres * (-0.5 * resids_new**2 / R_diag
                                      - 0.5 * jnp.log(R_diag) - HALF_LOG_2PI))
    pred_lw = log_pred_total - obs_ll_new

    return y_new, pred_lw


def diffusion_fn(params):
    del params
    return jnp.array([SIGMA_B_FROZEN, SIGMA_F_FROZEN, SIGMA_A_FROZEN])


def imex_step_fn(y, t, dt, params, grid_obs):
    del t
    tau_B     = params[_PI['tau_B']]
    alpha_A   = params[_PI['alpha_A']]
    tau_F     = params[_PI['tau_F']]
    lambda_B  = params[_PI['lambda_B']]
    lambda_A  = params[_PI['lambda_A']]
    mu_0      = -params[_PI['mu_0_abs']]
    mu_B      = params[_PI['mu_B']]
    mu_F      = params[_PI['mu_F']]
    mu_FF     = params[_PI['mu_FF']]
    eta       = params[_PI['eta']]
    B = y[0]; F = y[1]; A = y[2]
    T_B_k = grid_obs.get('T_B_k', 0.0)
    Phi_k = grid_obs.get('Phi_k', 0.0)
    mu = mu_0 + mu_B * B - mu_F * F - mu_FF * F * F
    drift_B = (1.0 + alpha_A * A) / tau_B * (T_B_k - B)
    drift_F = Phi_k - (1.0 + lambda_B * B + lambda_A * A) / tau_F * F
    drift_A = mu * A - eta * A * A * A
    return jnp.array([B + dt*drift_B, F + dt*drift_F, A + dt*drift_A])


# =========================================================================
# OBSERVATION MODEL — 6 independent Gaussian channels
# =========================================================================

def _obs_predictions(y, params):
    """Compute predicted observation for all 6 channels given state y.

    Returns dict of (predicted_mean, sigma) per channel.
    """
    B, F, A = y[0], y[1], y[2]

    kappa_vagal   = params[_PI['kappa_vagal']]
    kappa_chronic = KAPPA_CHRONIC_FROZEN

    I_base = params[_PI['I_base']]
    c_B    = params[_PI['c_B']]
    c_F    = params[_PI['c_F']]

    D_base = params[_PI['D_base']]
    d_B    = params[_PI['d_B']]
    d_F    = params[_PI['d_F']]

    S_base = params[_PI['S_base']]
    s_A    = params[_PI['s_A']]
    s_F    = params[_PI['s_F']]

    Sleep_base = params[_PI['Sleep_base']]
    sl_A       = params[_PI['sl_A']]
    sl_B       = params[_PI['sl_B']]
    sl_F       = params[_PI['sl_F']]

    Time_base = params[_PI['Time_base']]
    t_A       = params[_PI['t_A']]
    t_F       = params[_PI['t_F']]

    pred_RHR   = R_BASE_FROZEN - kappa_vagal * B + kappa_chronic * F
    pred_I     = I_base + c_B * B - c_F * F
    pred_D     = D_base + d_B * B - d_F * F
    pred_S     = S_base - s_A * A + s_F * F
    pred_Sleep = Sleep_base + sl_A * A + sl_B * B - sl_F * F
    pred_Time  = Time_base + t_A * A - t_F * F

    return (pred_RHR, pred_I, pred_D, pred_S, pred_Sleep, pred_Time)


def obs_log_prob_fn(y, grid_obs, k, params):
    """Summed Gaussian log-probability over all 6 obs channels."""
    preds = _obs_predictions(y, params)
    pred_RHR, pred_I, pred_D, pred_S, pred_Sleep, pred_Time = preds

    sigma_R     = params[_PI['sigma_obs_R']]
    sigma_I     = params[_PI['sigma_obs_I']]
    sigma_D     = params[_PI['sigma_obs_D']]
    sigma_S     = params[_PI['sigma_obs_S']]
    sigma_Sleep = params[_PI['sigma_obs_Sleep']]
    sigma_Time  = params[_PI['sigma_obs_Time']]

    def _ll(pred, obs_val, obs_pres, sigma):
        resid = obs_val - pred
        return obs_pres * (-0.5 * (resid / sigma) ** 2
                           - jnp.log(sigma) - HALF_LOG_2PI)

    lp  = _ll(pred_RHR,   grid_obs['obs_RHR_value'][k],
              grid_obs['obs_RHR_present'][k],       sigma_R)
    lp += _ll(pred_I,     grid_obs['obs_intensity_value'][k],
              grid_obs['obs_intensity_present'][k],  sigma_I)
    lp += _ll(pred_D,     grid_obs['obs_duration_value'][k],
              grid_obs['obs_duration_present'][k],   sigma_D)
    lp += _ll(pred_S,     grid_obs['obs_stress_value'][k],
              grid_obs['obs_stress_present'][k],     sigma_S)
    lp += _ll(pred_Sleep, grid_obs['obs_sleep_value'][k],
              grid_obs['obs_sleep_present'][k],      sigma_Sleep)
    lp += _ll(pred_Time,  grid_obs['obs_timing_value'][k],
              grid_obs['obs_timing_present'][k],     sigma_Time)
    return lp


def obs_log_weight_fn(x_new, grid_obs, k, params):
    return obs_log_prob_fn(x_new, grid_obs, k, params)


def gaussian_obs_fn(y, grid_obs, k, params):
    """Per-step Gaussian info for EKF (stack all 6 channels)."""
    preds = _obs_predictions(y, params)
    pred_RHR, pred_I, pred_D, pred_S, pred_Sleep, pred_Time = preds

    sigma_R     = params[_PI['sigma_obs_R']]
    sigma_I     = params[_PI['sigma_obs_I']]
    sigma_D     = params[_PI['sigma_obs_D']]
    sigma_S     = params[_PI['sigma_obs_S']]
    sigma_Sleep = params[_PI['sigma_obs_Sleep']]
    sigma_Time  = params[_PI['sigma_obs_Time']]

    return {
        'mean':     jnp.array([pred_RHR, pred_I, pred_D,
                               pred_S, pred_Sleep, pred_Time]),
        'value':    jnp.array([grid_obs['obs_RHR_value'][k],
                               grid_obs['obs_intensity_value'][k],
                               grid_obs['obs_duration_value'][k],
                               grid_obs['obs_stress_value'][k],
                               grid_obs['obs_sleep_value'][k],
                               grid_obs['obs_timing_value'][k]]),
        'cov_diag': jnp.array([sigma_R**2, sigma_I**2, sigma_D**2,
                               sigma_S**2, sigma_Sleep**2, sigma_Time**2]),
        'present':  jnp.array([grid_obs['obs_RHR_present'][k],
                               grid_obs['obs_intensity_present'][k],
                               grid_obs['obs_duration_present'][k],
                               grid_obs['obs_stress_present'][k],
                               grid_obs['obs_sleep_present'][k],
                               grid_obs['obs_timing_present'][k]]),
    }


def obs_sample_fn(y, exog, k, params, rng_key):
    """Sample one observation from each channel given state y."""
    del exog, k
    preds = _obs_predictions(y, params)
    pred_RHR, pred_I, pred_D, pred_S, pred_Sleep, pred_Time = preds

    sigma_R     = params[_PI['sigma_obs_R']]
    sigma_I     = params[_PI['sigma_obs_I']]
    sigma_D     = params[_PI['sigma_obs_D']]
    sigma_S     = params[_PI['sigma_obs_S']]
    sigma_Sleep = params[_PI['sigma_obs_Sleep']]
    sigma_Time  = params[_PI['sigma_obs_Time']]

    keys = jax.random.split(rng_key, 6)
    return {
        'obs_RHR_value':       pred_RHR   + sigma_R     * jax.random.normal(keys[0], dtype=y.dtype),
        'obs_intensity_value': pred_I     + sigma_I     * jax.random.normal(keys[1], dtype=y.dtype),
        'obs_duration_value':  pred_D     + sigma_D     * jax.random.normal(keys[2], dtype=y.dtype),
        'obs_stress_value':    pred_S     + sigma_S     * jax.random.normal(keys[3], dtype=y.dtype),
        'obs_sleep_value':     pred_Sleep + sigma_Sleep * jax.random.normal(keys[4], dtype=y.dtype),
        'obs_timing_value':    pred_Time  + sigma_Time  * jax.random.normal(keys[5], dtype=y.dtype),
    }


# =========================================================================
# GRID ALIGNMENT — 6 obs channels + 2 exogenous (T_B, Phi)
# =========================================================================

def _align_channel(obs_data, t_steps, value_key):
    val  = np.zeros(t_steps, dtype=np.float32)
    pres = np.zeros(t_steps, dtype=np.float32)
    if obs_data is not None and 't_idx' in obs_data and value_key in obs_data:
        idx = np.asarray(obs_data['t_idx']).astype(int)
        val[idx]  = np.asarray(obs_data[value_key]).astype(np.float32)
        pres[idx] = 1.0
    return val, pres


def align_obs_fn(obs_data, t_steps, dt_hours):
    """Align all 6 obs channels + T_B and Phi exogenous channels."""
    del dt_hours

    def _get(name):
        return obs_data.get(name) if isinstance(obs_data, dict) else None

    RHR_ch   = _get('obs_RHR')
    I_ch     = _get('obs_intensity')
    D_ch     = _get('obs_duration')
    S_ch     = _get('obs_stress')
    Sleep_ch = _get('obs_sleep')
    Time_ch  = _get('obs_timing')
    T_ch     = _get('T_B')
    P_ch     = _get('Phi')

    RHR_val,   RHR_pres   = _align_channel(RHR_ch,   t_steps, 'obs_value')
    I_val,     I_pres     = _align_channel(I_ch,     t_steps, 'obs_value')
    D_val,     D_pres     = _align_channel(D_ch,     t_steps, 'obs_value')
    S_val,     S_pres     = _align_channel(S_ch,     t_steps, 'obs_value')
    Sleep_val, Sleep_pres = _align_channel(Sleep_ch, t_steps, 'obs_value')
    Time_val,  Time_pres  = _align_channel(Time_ch,  t_steps, 'obs_value')
    T_B_val, _            = _align_channel(T_ch,     t_steps, 'T_B_value')
    Phi_val, _            = _align_channel(P_ch,     t_steps, 'Phi_value')

    has_any = np.maximum.reduce([RHR_pres, I_pres, D_pres,
                                  S_pres, Sleep_pres, Time_pres])

    return {
        'obs_RHR_value':          jnp.array(RHR_val),
        'obs_RHR_present':        jnp.array(RHR_pres),
        'obs_intensity_value':    jnp.array(I_val),
        'obs_intensity_present':  jnp.array(I_pres),
        'obs_duration_value':     jnp.array(D_val),
        'obs_duration_present':   jnp.array(D_pres),
        'obs_stress_value':       jnp.array(S_val),
        'obs_stress_present':     jnp.array(S_pres),
        'obs_sleep_value':        jnp.array(Sleep_val),
        'obs_sleep_present':      jnp.array(Sleep_pres),
        'obs_timing_value':       jnp.array(Time_val),
        'obs_timing_present':     jnp.array(Time_pres),
        'has_any_obs':            jnp.array(has_any),
        'T_B':                    jnp.array(T_B_val),
        'Phi':                    jnp.array(Phi_val),
    }


# =========================================================================
# FORWARD INTEGRATION (latent SDE only — same as base FSA)
# =========================================================================

def forward_sde_stochastic(init_state, params, exogenous, dt, n_steps,
                            rng_key=None):
    tau_B     = params[_PI['tau_B']]
    alpha_A   = params[_PI['alpha_A']]
    tau_F     = params[_PI['tau_F']]
    lambda_B  = params[_PI['lambda_B']]
    lambda_A  = params[_PI['lambda_A']]
    mu_0      = -params[_PI['mu_0_abs']]
    mu_B      = params[_PI['mu_B']]
    mu_F      = params[_PI['mu_F']]
    mu_FF     = params[_PI['mu_FF']]
    eta       = params[_PI['eta']]
    sqrt_dt   = jnp.sqrt(dt)
    T_B_arr = jnp.asarray(exogenous['T_B'])
    Phi_arr = jnp.asarray(exogenous['Phi'])
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    def step(carry, i):
        y, key = carry
        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, (3,))
        B, F, A = y[0], y[1], y[2]
        mu = mu_0 + mu_B*B - mu_F*F - mu_FF*F*F
        dB = (1.0 + alpha_A*A)/tau_B * (T_B_arr[i] - B)
        dF = Phi_arr[i] - (1.0 + lambda_B*B + lambda_A*A)/tau_F * F
        dA = mu*A - eta*A*A*A
        B_cl = jnp.clip(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
        F_cl = jnp.maximum(F, 0.0); A_cl = jnp.maximum(A, 0.0)
        B_new = B + dt*dB + SIGMA_B_FROZEN*jnp.sqrt(B_cl*(1-B_cl))*sqrt_dt*noise[0]
        F_new = F + dt*dF + SIGMA_F_FROZEN*jnp.sqrt(F_cl)*sqrt_dt*noise[1]
        A_new = A + dt*dA + SIGMA_A_FROZEN*jnp.sqrt(A_cl + EPS_A_FROZEN)*sqrt_dt*noise[2]
        B_new = jnp.clip(B_new, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
        F_new = jnp.maximum(F_new, 0.0); A_new = jnp.maximum(A_new, 0.0)
        y_new = jnp.array([B_new, F_new, A_new])
        return (y_new, key), y_new

    (_, _), traj = jax.lax.scan(step, (init_state, rng_key),
                                 jnp.arange(n_steps))
    return traj


# =========================================================================
# MISC HELPERS
# =========================================================================

def shard_init_fn(time_offset, params, exogenous, global_init):
    del time_offset, params, exogenous
    return global_init


def make_init_state_fn(init_estimates, params):
    del params
    return init_estimates


def _prior_mean(ptype, pargs):
    if ptype == 'lognormal':
        return math.exp(pargs[0] + pargs[1]**2 / 2)
    elif ptype == 'normal':
        return pargs[0]
    return 0.0


def get_init_theta():
    all_config = OrderedDict()
    all_config.update(PARAM_PRIOR_CONFIG)
    all_config.update(INIT_STATE_PRIOR_CONFIG)
    return np.array([_prior_mean(pt, pa) for _, (pt, pa) in all_config.items()],
                    dtype=np.float32)


# =========================================================================
# ASSEMBLE
# =========================================================================

FSA_REAL_OBS_ESTIMATION = EstimationModel(
    name="fsa_real_obs",
    version="1.0",
    n_states=3,
    n_stochastic=3,
    stochastic_indices=(0, 1, 2),
    state_bounds=((0.0, 1.0), (0.0, 10.0), (0.0, 5.0)),
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params={
        'eps_A':   EPS_A_FROZEN,
        'eps_B':   EPS_B_FROZEN,
        'sigma_B': SIGMA_B_FROZEN,
        'sigma_F': SIGMA_F_FROZEN,
        'sigma_A': SIGMA_A_FROZEN,
    },
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    forward_sde_fn=forward_sde_stochastic,
    get_init_theta_fn=get_init_theta,
    imex_step_fn=imex_step_fn,
    obs_log_prob_fn=obs_log_prob_fn,
    make_init_state_fn=make_init_state_fn,
    obs_sample_fn=obs_sample_fn,
    gaussian_obs_fn=gaussian_obs_fn,
    exogenous_keys=('T_B', 'Phi'),
)
