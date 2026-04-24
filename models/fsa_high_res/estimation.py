"""models/fsa_high_res/estimation.py — 15-min FSA with 4-channel mixed obs.

Latent dynamics: identical to FSA v4.1 (3-state B/F/A SDE).

Observation model (4 channels):
  - HR  (Gaussian, sleep-gated)      : HR_base - kappa_B B + alpha_A_HR A + beta_C_HR C(t)
  - sleep (Bernoulli)                : sigmoid(k_C C(t) + k_A A - c_tilde)
  - stress (Gaussian, wake-gated)    : S_base + k_F F - k_A_S A + beta_C_S C(t)
  - log(steps+1) (Gaussian, wake)    : mu_step0 + beta_B_st B - beta_F_st F + beta_A_st A + beta_C_st C(t)

HR / stress / steps go through the sequential-scalar-Kalman fusion in
propagate_fn (same pattern as FSA v4.1's 6-channel fusion, but with gating
masks from the sleep/wake label). The Bernoulli sleep is NOT part of the
Kalman fusion — it's added as an extra term in obs_log_weight_fn.

Time grid: dt = 1/96 day = 15 min. Circadian C(t) = cos(2 pi t_days + phi)
is precomputed by align_obs_fn and stored in grid_obs['C'].
"""

import math
import numpy as np
from collections import OrderedDict

import jax
import jax.numpy as jnp

from smc2bj.estimation_model import EstimationModel
from smc2bj._likelihood_constants import HALF_LOG_2PI


# =========================================================================
# FROZEN CONSTANTS
# =========================================================================

EPS_A_FROZEN   = 1.0e-4
EPS_B_FROZEN   = 1.0e-4
SIGMA_B_FROZEN = 0.01
SIGMA_F_FROZEN = 0.005
SIGMA_A_FROZEN = 0.02
PHI_FROZEN     = 0.0  # circadian phase, 0 = healthy morning chronotype


# =========================================================================
# PRIORS (29 params)
# =========================================================================

PARAM_PRIOR_CONFIG = OrderedDict([
    # Priors TIGHTENED vs. what the daily FSA uses. Rationale: at 15-min
    # resolution a 1-day window already carries ~60-90 informative obs,
    # and the adaptive-tempering ESS solver picks tiny Δλ if the prior is
    # too wide. These widths give the POC a tractable number of tempering
    # levels while still leaving 3-5× room around the truth for the
    # posterior to move.
    # --- SDE dynamics ---
    ('tau_B',     ('lognormal', (math.log(14.0), 0.05))),
    ('alpha_A',   ('lognormal', (math.log( 1.0), 0.20))),
    ('tau_F',     ('lognormal', (math.log( 7.0), 0.15))),
    ('lambda_B',  ('lognormal', (math.log( 3.0), 0.15))),
    ('lambda_A',  ('lognormal', (math.log( 1.5), 0.15))),
    # High-res uses mu_0 > 0 directly (A sits at a stable Stuart-Landau
    # fixed point rather than being activated by a bifurcation crossing).
    # This is the sign-clean version of the daily FSA's 'mu_0_abs' reparam.
    ('mu_0',      ('lognormal', (math.log(0.02), 0.20))),
    ('mu_B',      ('lognormal', (math.log(0.30), 0.20))),
    ('mu_F',      ('lognormal', (math.log(0.10), 0.20))),
    ('mu_FF',     ('lognormal', (math.log(0.40), 0.20))),
    ('eta',       ('lognormal', (math.log(0.20), 0.15))),

    # --- Ch1: HR (sleep-gated, Gaussian) ---
    ('HR_base',     ('normal',    (62.0, 2.0))),
    ('kappa_B',     ('lognormal', (math.log(12.0), 0.15))),
    ('alpha_A_HR',  ('lognormal', (math.log(3.0),  0.20))),
    ('beta_C_HR',   ('normal',    (-2.5, 0.5))),
    ('sigma_HR',    ('lognormal', (math.log(2.0), 0.20))),

    # --- Ch2: Sleep (Bernoulli) ---
    ('k_C',         ('lognormal', (math.log(3.0), 0.15))),
    ('k_A',         ('lognormal', (math.log(2.0), 0.25))),
    ('c_tilde',     ('normal',    (0.5, 0.25))),

    # --- Ch3: Stress (wake-gated, Gaussian) ---
    ('S_base',      ('normal',    (30.0, 3.0))),
    ('k_F',         ('lognormal', (math.log(20.0), 0.20))),
    ('k_A_S',       ('lognormal', (math.log(8.0),  0.25))),
    ('beta_C_S',    ('normal',    (-4.0, 0.8))),
    ('sigma_S',     ('lognormal', (math.log(4.0), 0.20))),

    # --- Ch4: Steps (log-Gaussian, wake-gated) ---
    ('mu_step0',    ('normal',    (5.5, 0.3))),
    ('beta_B_st',   ('lognormal', (math.log(0.8), 0.20))),
    ('beta_F_st',   ('lognormal', (math.log(0.5), 0.25))),
    ('beta_A_st',   ('lognormal', (math.log(0.3), 0.25))),
    ('beta_C_st',   ('normal',    (-0.8, 0.2))),
    ('sigma_st',    ('lognormal', (math.log(0.5), 0.15))),
])

INIT_STATE_PRIOR_CONFIG = OrderedDict()
# A_0 raised to 0.55 — starts A near the Stuart-Landau fixed point
# A* = sqrt(mu/eta) for typical (B,F) where mu > 0. Matches
# DEFAULT_INIT in simulation.py.
COLD_START_INIT = jnp.array([0.05, 0.10, 0.55])

_PK = list(PARAM_PRIOR_CONFIG.keys())
_PI = {k: i for i, k in enumerate(_PK)}


# =========================================================================
# PROPAGATE_FN — Joint Gaussian guided proposal (3 Gaussian channels fused)
# =========================================================================

def propagate_fn(y, t, dt, params, grid_obs, k,
                 sigma_diag, noise, rng_key):
    """Sequential-scalar Kalman fusion over 3 Gaussian obs channels.

    Channels (linear in [B, F, A]):
      HR     = HR_base - kappa_B*B + alpha_A_HR*A + beta_C_HR*C
               → H_HR    = [-kappa_B, 0, alpha_A_HR]
               → bias_HR = HR_base + beta_C_HR * C(t_k)
      stress = S_base + k_F*F - k_A_S*A + beta_C_S*C
               → H_S     = [0, k_F, -k_A_S]
               → bias_S  = S_base + beta_C_S * C(t_k)
      log_steps = mu_step0 + beta_B_st*B - beta_F_st*F + beta_A_st*A + beta_C_st*C
               → H_ST    = [beta_B_st, -beta_F_st, beta_A_st]
               → bias_ST = mu_step0 + beta_C_st * C(t_k)

    Gating: HR_present, stress_present, steps_present read from grid_obs.
    Bernoulli sleep is handled in obs_log_weight_fn, NOT here.

    Returns y_new, pred_lw where pred_lw is the sample-independent Gaussian
    predictive log-marginal minus obs_ll_Gaussian(y_new) — i.e. the Gaussian
    part of the particle weight is absorbed into pred_lw, and
    obs_log_weight_fn only needs to add the Bernoulli sleep term.
    """
    del t, rng_key, sigma_diag

    # --- Dynamics params ---
    tau_B    = params[_PI['tau_B']]
    alpha_A  = params[_PI['alpha_A']]
    tau_F    = params[_PI['tau_F']]
    lambda_B = params[_PI['lambda_B']]
    lambda_A = params[_PI['lambda_A']]
    mu_0     = params[_PI['mu_0']]
    mu_B     = params[_PI['mu_B']]
    mu_F     = params[_PI['mu_F']]
    mu_FF    = params[_PI['mu_FF']]
    eta      = params[_PI['eta']]

    B, F, A = y[0], y[1], y[2]
    T_B_k = grid_obs['T_B'][k]
    Phi_k = grid_obs['Phi'][k]
    C_k   = grid_obs['C'][k]

    # --- Euler drift predictions ---
    mu_bif  = mu_0 + mu_B * B - mu_F * F - mu_FF * F * F
    drift_B = (1.0 + alpha_A * A) / tau_B * (T_B_k - B)
    drift_F = Phi_k - (1.0 + lambda_B * B + lambda_A * A) / tau_F * F
    drift_A = mu_bif * A - eta * A * A * A
    B_pred = B + dt * drift_B
    F_pred = F + dt * drift_F
    A_pred = A + dt * drift_A

    # --- Prior predictive covariance from state-dependent process noise ---
    B_cl = jnp.clip(B, EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F_cl = jnp.maximum(F, 0.0)
    A_cl = jnp.maximum(A, 0.0)
    var_B = jnp.maximum(SIGMA_B_FROZEN ** 2 * B_cl * (1.0 - B_cl) * dt, 1e-12)
    var_F = jnp.maximum(SIGMA_F_FROZEN ** 2 * F_cl * dt, 1e-12)
    var_A = jnp.maximum(SIGMA_A_FROZEN ** 2 * (A_cl + EPS_A_FROZEN) * dt, 1e-12)

    mu_prior = jnp.array([B_pred, F_pred, A_pred])
    P_prior = jnp.diag(jnp.array([var_B, var_F, var_A]))

    # --- Observation params ---
    HR_base    = params[_PI['HR_base']]
    kappa_B    = params[_PI['kappa_B']]
    alpha_A_HR = params[_PI['alpha_A_HR']]
    beta_C_HR  = params[_PI['beta_C_HR']]
    sigma_HR   = params[_PI['sigma_HR']]

    S_base     = params[_PI['S_base']]
    k_F        = params[_PI['k_F']]
    k_A_S      = params[_PI['k_A_S']]
    beta_C_S   = params[_PI['beta_C_S']]
    sigma_S    = params[_PI['sigma_S']]

    mu_step0   = params[_PI['mu_step0']]
    beta_B_st  = params[_PI['beta_B_st']]
    beta_F_st  = params[_PI['beta_F_st']]
    beta_A_st  = params[_PI['beta_A_st']]
    beta_C_st  = params[_PI['beta_C_st']]
    sigma_st   = params[_PI['sigma_st']]

    # --- Linear obs model: y_c = H_c @ [B,F,A] + bias_c + N(0, R_c) ---
    H = jnp.array([
        [-kappa_B,    0.0,         alpha_A_HR],  # HR
        [0.0,         k_F,        -k_A_S],       # stress
        [beta_B_st,  -beta_F_st,   beta_A_st],   # log(steps+1)
    ])
    bias = jnp.array([
        HR_base + beta_C_HR * C_k,
        S_base  + beta_C_S  * C_k,
        mu_step0 + beta_C_st * C_k,
    ])
    R_diag = jnp.array([sigma_HR ** 2, sigma_S ** 2, sigma_st ** 2])

    obs_vals = jnp.array([
        grid_obs['hr_value'][k],
        grid_obs['stress_value'][k],
        grid_obs['log_steps_value'][k],
    ])
    obs_pres = jnp.array([
        grid_obs['hr_present'][k],
        grid_obs['stress_present'][k],
        grid_obs['steps_present'][k],
    ])

    # --- Sequential scalar Kalman fusion (3 channels) ---
    def _kalman_step(carry, ch):
        mu, P, lp = carry
        h_i, b_i, r_i, y_i, pres_i = ch
        innov = y_i - (h_i @ mu + b_i)
        Ph    = P @ h_i
        S_i   = h_i @ Ph + r_i
        K_i   = Ph / S_i
        ll_i  = -0.5 * jnp.log(2.0 * jnp.pi * S_i) - 0.5 * innov ** 2 / S_i
        mu = mu + pres_i * K_i * innov
        P  = P  - pres_i * jnp.outer(K_i, Ph)
        lp = lp + pres_i * ll_i
        return (mu, P, lp), None

    (mu_fused, P_fused, log_pred_total), _ = jax.lax.scan(
        _kalman_step,
        (mu_prior, P_prior, 0.0),
        (H, bias, R_diag, obs_vals, obs_pres),
    )

    # --- Sample from fused Gaussian posterior ---
    P_safe = P_fused + 1e-10 * jnp.eye(3)
    L = jnp.linalg.cholesky(P_safe)
    x_new = mu_fused + L @ noise

    # --- Physical bounds ---
    B_new = jnp.clip(x_new[0], EPS_B_FROZEN, 1.0 - EPS_B_FROZEN)
    F_new = jnp.maximum(x_new[1], 0.0)
    A_new = jnp.maximum(x_new[2], 0.0)
    y_new = jnp.array([B_new, F_new, A_new])

    # --- Weight correction: pred_lw = log_pred_total - obs_ll_Gaussian(y_new) ---
    preds_new = H @ y_new + bias
    resids_new = obs_vals - preds_new
    obs_ll_new = jnp.sum(obs_pres * (-0.5 * resids_new ** 2 / R_diag
                                      - 0.5 * jnp.log(R_diag) - HALF_LOG_2PI))
    pred_lw = log_pred_total - obs_ll_new

    return y_new, pred_lw


def diffusion_fn(params):
    del params
    return jnp.array([SIGMA_B_FROZEN, SIGMA_F_FROZEN, SIGMA_A_FROZEN])


# =========================================================================
# OBS LOG WEIGHT FN — Gaussian channels (for weight correction) + Bernoulli
# =========================================================================

def _sleep_log_prob(A, C_k, sleep_label, sleep_present, params):
    k_C     = params[_PI['k_C']]
    k_A     = params[_PI['k_A']]
    c_tilde = params[_PI['c_tilde']]
    z = k_C * C_k + k_A * A - c_tilde
    p = jax.nn.sigmoid(z)
    p_safe = jnp.clip(p, 1e-8, 1.0 - 1e-8)
    s = sleep_label.astype(p_safe.dtype)
    return sleep_present * (s * jnp.log(p_safe)
                             + (1.0 - s) * jnp.log(1.0 - p_safe))


def _gaussian_obs_ll(y, grid_obs, k, params):
    """Sum of log-Gaussian terms for HR, stress, log(steps+1) on y_new.

    Matches what propagate_fn's pred_lw absorbs: the final particle weight
    is pred_lw + _gaussian_obs_ll(y_new) + bernoulli_sleep_ll(y_new) =
    log_pred_total_Gaussian + bernoulli_sleep_ll.
    """
    B, F, A = y[0], y[1], y[2]
    C_k = grid_obs['C'][k]

    HR_base    = params[_PI['HR_base']]
    kappa_B    = params[_PI['kappa_B']]
    alpha_A_HR = params[_PI['alpha_A_HR']]
    beta_C_HR  = params[_PI['beta_C_HR']]
    sigma_HR   = params[_PI['sigma_HR']]

    S_base     = params[_PI['S_base']]
    k_F        = params[_PI['k_F']]
    k_A_S      = params[_PI['k_A_S']]
    beta_C_S   = params[_PI['beta_C_S']]
    sigma_S    = params[_PI['sigma_S']]

    mu_step0   = params[_PI['mu_step0']]
    beta_B_st  = params[_PI['beta_B_st']]
    beta_F_st  = params[_PI['beta_F_st']]
    beta_A_st  = params[_PI['beta_A_st']]
    beta_C_st  = params[_PI['beta_C_st']]
    sigma_st   = params[_PI['sigma_st']]

    pred_HR  = HR_base - kappa_B * B + alpha_A_HR * A + beta_C_HR * C_k
    pred_S   = S_base + k_F * F - k_A_S * A + beta_C_S * C_k
    pred_ST  = mu_step0 + beta_B_st * B - beta_F_st * F + beta_A_st * A + beta_C_st * C_k

    def _ll(pred, obs_val, obs_pres, sigma):
        resid = obs_val - pred
        return obs_pres * (-0.5 * (resid / sigma) ** 2
                           - jnp.log(sigma) - HALF_LOG_2PI)

    lp  = _ll(pred_HR,  grid_obs['hr_value'][k],
              grid_obs['hr_present'][k], sigma_HR)
    lp += _ll(pred_S,   grid_obs['stress_value'][k],
              grid_obs['stress_present'][k], sigma_S)
    lp += _ll(pred_ST,  grid_obs['log_steps_value'][k],
              grid_obs['steps_present'][k], sigma_st)
    return lp


def obs_log_weight_fn(x_new, grid_obs, k, params):
    """Total observation log-weight for the particle at x_new.

    Adds the Gaussian obs log-likelihood (matches what propagate_fn's
    pred_lw absorbed with opposite sign — so total weight = predictive
    marginal) AND the Bernoulli sleep log-prob (standalone).
    """
    gauss_ll = _gaussian_obs_ll(x_new, grid_obs, k, params)
    C_k = grid_obs['C'][k]
    bern_ll = _sleep_log_prob(x_new[2], C_k,
                               grid_obs['sleep_label'][k],
                               grid_obs['sleep_present'][k],
                               params)
    return gauss_ll + bern_ll


def obs_log_prob_fn(y, grid_obs, k, params):
    """Full observation log-probability (all 4 channels). Same as
    obs_log_weight_fn — kept for symmetry with FSA's interface."""
    return obs_log_weight_fn(y, grid_obs, k, params)


# =========================================================================
# ALIGN_OBS_FN — grid-align per-channel obs into JAX arrays
# =========================================================================

def align_obs_fn(obs_data, t_steps, dt):
    """Align the 4 obs channels + T_B/Phi exogenous + precomputed C(t).

    t_steps : number of time-steps (bins) in the window.
    dt      : bin width in **days** (consumed from rolling_cfg.dt). For the
              15-min variant, dt = 1/96 day.

    Output keys:
      hr_value, hr_present
      stress_value, stress_present
      log_steps_value, steps_present  (log-transformed at align time)
      sleep_label (int), sleep_present
      T_B (n,), Phi (n,)   (per-bin, filled from exogenous)
      C (n,)               (cos(2 pi t_days + phi_frozen) per bin)
      has_any_obs (n,)
    """
    T = t_steps

    def _get(name):
        return obs_data.get(name) if isinstance(obs_data, dict) else None

    # -- HR (Gaussian, sleep-gated) --
    hr_val = np.zeros(T, dtype=np.float32)
    hr_pres = np.zeros(T, dtype=np.float32)
    hr_ch = _get('obs_HR')
    if hr_ch and 't_idx' in hr_ch:
        idx = np.asarray(hr_ch['t_idx']).astype(int)
        mask = (idx >= 0) & (idx < T)
        hr_val[idx[mask]] = np.asarray(hr_ch['obs_value'])[mask]
        hr_pres[idx[mask]] = 1.0

    # -- stress (Gaussian, wake-gated) --
    s_val = np.zeros(T, dtype=np.float32)
    s_pres = np.zeros(T, dtype=np.float32)
    s_ch = _get('obs_stress')
    if s_ch and 't_idx' in s_ch:
        idx = np.asarray(s_ch['t_idx']).astype(int)
        mask = (idx >= 0) & (idx < T)
        s_val[idx[mask]] = np.asarray(s_ch['obs_value'])[mask]
        s_pres[idx[mask]] = 1.0

    # -- steps (log-transformed, wake-gated) --
    log_st_val = np.zeros(T, dtype=np.float32)
    st_pres = np.zeros(T, dtype=np.float32)
    st_ch = _get('obs_steps')
    if st_ch and 't_idx' in st_ch:
        idx = np.asarray(st_ch['t_idx']).astype(int)
        mask = (idx >= 0) & (idx < T)
        raw = np.asarray(st_ch['obs_value'])[mask]
        log_st_val[idx[mask]] = np.log(raw + 1.0).astype(np.float32)
        st_pres[idx[mask]] = 1.0

    # -- sleep (Bernoulli, always observed) --
    sl_label = np.zeros(T, dtype=np.int32)
    sl_pres = np.zeros(T, dtype=np.float32)
    sl_ch = _get('obs_sleep')
    if sl_ch and 't_idx' in sl_ch:
        idx = np.asarray(sl_ch['t_idx']).astype(int)
        mask = (idx >= 0) & (idx < T)
        sl_label[idx[mask]] = np.asarray(sl_ch['sleep_label'])[mask].astype(np.int32)
        sl_pres[idx[mask]] = 1.0

    # -- T_B / Phi exogenous (per-bin arrays) --
    t_ch = _get('T_B')
    p_ch = _get('Phi')
    T_B_val = np.zeros(T, dtype=np.float32)
    Phi_val = np.zeros(T, dtype=np.float32)
    if t_ch and 'T_B_value' in t_ch:
        raw = np.asarray(t_ch['T_B_value']).astype(np.float32)
        n = min(len(raw), T)
        T_B_val[:n] = raw[:n]
    if p_ch and 'Phi_value' in p_ch:
        raw = np.asarray(p_ch['Phi_value']).astype(np.float32)
        n = min(len(raw), T)
        Phi_val[:n] = raw[:n]

    # -- Circadian C(t) per bin (day units, phi frozen) --
    # dt arg is in days (rolling_cfg.dt convention).
    t_days = np.arange(T, dtype=np.float32) * float(dt)
    C_val = np.cos(2.0 * np.pi * t_days + PHI_FROZEN).astype(np.float32)

    has_any = np.maximum.reduce([hr_pres, s_pres, st_pres, sl_pres])

    return {
        'hr_value':         jnp.array(hr_val),
        'hr_present':       jnp.array(hr_pres),
        'stress_value':     jnp.array(s_val),
        'stress_present':   jnp.array(s_pres),
        'log_steps_value':  jnp.array(log_st_val),
        'steps_present':    jnp.array(st_pres),
        'sleep_label':      jnp.array(sl_label),
        'sleep_present':    jnp.array(sl_pres),
        'T_B':              jnp.array(T_B_val),
        'Phi':              jnp.array(Phi_val),
        'C':                jnp.array(C_val),
        'has_any_obs':      jnp.array(has_any),
    }


# =========================================================================
# MISC HELPERS (parallel FSA interface)
# =========================================================================

def shard_init_fn(time_offset, params, exogenous, global_init):
    del time_offset, params, exogenous
    return global_init


def imex_step_fn(y, t, dt, params, grid_obs):
    del t
    tau_B    = params[_PI['tau_B']]
    alpha_A  = params[_PI['alpha_A']]
    tau_F    = params[_PI['tau_F']]
    lambda_B = params[_PI['lambda_B']]
    lambda_A = params[_PI['lambda_A']]
    mu_0     = params[_PI['mu_0']]
    mu_B     = params[_PI['mu_B']]
    mu_F     = params[_PI['mu_F']]
    mu_FF    = params[_PI['mu_FF']]
    eta      = params[_PI['eta']]
    B = y[0]; F = y[1]; A = y[2]
    T_B_k = grid_obs.get('T_B_k', 0.0)
    Phi_k = grid_obs.get('Phi_k', 0.0)
    mu = mu_0 + mu_B * B - mu_F * F - mu_FF * F * F
    drift_B = (1.0 + alpha_A * A) / tau_B * (T_B_k - B)
    drift_F = Phi_k - (1.0 + lambda_B * B + lambda_A * A) / tau_F * F
    drift_A = mu * A - eta * A * A * A
    return jnp.array([B + dt * drift_B, F + dt * drift_F, A + dt * drift_A])


def forward_sde_stochastic(init_state, params, exogenous, dt, n_steps,
                            rng_key=None):
    tau_B    = params[_PI['tau_B']]
    alpha_A  = params[_PI['alpha_A']]
    tau_F    = params[_PI['tau_F']]
    lambda_B = params[_PI['lambda_B']]
    lambda_A = params[_PI['lambda_A']]
    mu_0     = params[_PI['mu_0']]
    mu_B     = params[_PI['mu_B']]
    mu_F     = params[_PI['mu_F']]
    mu_FF    = params[_PI['mu_FF']]
    eta      = params[_PI['eta']]
    sqrt_dt = jnp.sqrt(dt)
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


def make_init_state_fn(init_estimates, params):
    del params
    return init_estimates


def _prior_mean(ptype, pargs):
    if ptype == 'lognormal':
        return math.exp(pargs[0] + pargs[1] ** 2 / 2)
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

HIGH_RES_FSA_ESTIMATION = EstimationModel(
    name="fsa_high_res",
    version="0.1",
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
        'phi':     PHI_FROZEN,
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
)
