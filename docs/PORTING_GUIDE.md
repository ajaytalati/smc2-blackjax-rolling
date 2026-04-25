# Porting Guide — bring your own model

> For what the framework does algorithmically see [SMC2_ALGORITHM_SPECIFICATION.md](SMC2_ALGORITHM_SPECIFICATION.md).
> For a worked concrete example see [models/fsa_real_obs/](../models/fsa_real_obs/) and
> [drivers/fsa_real_obs_5yr_rolling.py](../drivers/fsa_real_obs_5yr_rolling.py).

This guide explains how another agent or user can plug a **different** latent /
observation model into the framework while leaving the SMC² + rolling-window
machinery untouched.

---

## 1. The 3-file model convention

Every model is a Python package under `models/<your_name>/` with three files:

```
models/<your_name>/
├── __init__.py          # re-exports the two module-level instances below
├── simulation.py        # generative side — drift, diffusion, obs generators
├── estimation.py        # inference side — priors + 5 PF callbacks + EstimationModel
└── sim_plots.py         # per-model diagnostic plots
```

The responsibilities of each file are sharp and worth memorising.

### 1.1 `simulation.py` — the generative model

Defines the continuous-time SDE and the per-channel observation generators
for **synthetic-data generation**. This is the side used by the data-flow
pipeline (forward-integrate to get a trajectory, then sample obs from it),
not by the particle filter directly.

Required exports (match the function signatures used by
`smc2bj.simulator`):

```python
from smc2bj.simulator.sde_model import SDEModel, StateSpec, ChannelSpec

def drift(t, y, params, aux): ...              # deterministic drift of the SDE
def diffusion_diagonal(params): ...             # per-state noise scales
def make_aux(params, init_state, t_grid, exog): ...  # per-integration-call context
def make_y0(init_dict, params): ...             # pack initial state

# Per-channel generators (one per observation channel)
def gen_obs_<channel>(trajectory, t_grid, params, aux, prior_channels, seed): ...

YOUR_MODEL: SDEModel = SDEModel(
    name='your_model',
    states=[StateSpec(...), ...],
    channels=[ChannelSpec(name='obs_X', generate_fn=gen_obs_X, dependencies=()),
              ...],
    param_sets={'default': {...}, 'recovery': {...}, ...},
    init_states={'default': {...}, ...},
)
```

The `param_sets` dict holds named "scenarios" (true parameter values used
for synthetic-data experiments).
`init_states` holds the matching true initial states.

Reference: [models/fsa_real_obs/simulation.py](../models/fsa_real_obs/simulation.py).

### 1.2 `estimation.py` — the inference-side model

**This file is where you specialise the particle filter and SMC² for your
model.** The framework's generic code calls into five model-specific
callbacks you write here. Nothing FSA-specific in the framework — only
these callbacks bind the framework to a specific dynamical system.

Required exports:

```python
from collections import OrderedDict
from smc2bj.estimation_model import EstimationModel

# Priors: one entry per estimated parameter, in canonical order
PARAM_PRIOR_CONFIG = OrderedDict([
    ('param_1', ('lognormal', (mu_ln, sigma_ln))),
    ('param_2', ('normal',    (mu, sigma))),
    # ...
])
INIT_STATE_PRIOR_CONFIG = OrderedDict()          # usually empty if init states are frozen

# Fixed scalars the framework needs
COLD_START_INIT = jnp.array([...])               # Window-1 latent init
FROZEN_PARAMS = {'name': value, ...}             # params not estimated (optional)

# --- The five PF callbacks -------------------------------------------------

def propagate_fn(y, t, dt, params, grid_obs, k,
                 sigma_diag, xi, rng_key) -> (y_new, pred_lw):
    """One PF substep. Return new state and the log-weight from dynamics."""

def diffusion_fn(params) -> sigma_diag:          # per-state noise scales
    ...

def obs_log_weight_fn(x_new, grid_obs, k, params) -> float:
    """Log observation weight at step k given state x_new."""

def align_obs_fn(obs_data, t_steps, dt_hours) -> grid_obs:
    """Per-window preprocessing: grid observations onto the PF t-axis.
       Do per-window mean-centering / logit transforms / etc. here."""

def shard_init_fn(time_offset, params, exogenous, global_init) -> y_0:
    """Produce the initial state for each PF particle."""

# --- The instance ---------------------------------------------------------

YOUR_MODEL_ESTIMATION = EstimationModel(
    name='your_model',
    version='0.1',
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params=FROZEN_PARAMS,
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    # ... plus introspection fields: n_states, state_bounds, exogenous_keys
)
```

The generic particle filter in
[smc2bj/log_density/gk_dpf_v3_lite.py](../smc2bj/log_density/gk_dpf_v3_lite.py)
reads the five callbacks from `YOUR_MODEL_ESTIMATION` and never inspects any
other attribute. The outer SMC in
[smc2bj/estimation/smc_window.py](../smc2bj/estimation/smc_window.py) reads
`model.n_dim` (derived from `PARAM_PRIOR_CONFIG` + `INIT_STATE_PRIOR_CONFIG`)
and the transform arrays built from the priors.

Reference: [models/fsa_real_obs/estimation.py](../models/fsa_real_obs/estimation.py).

### 1.3 `sim_plots.py` — diagnostics

Per-model plots: trajectory overlay, per-channel residuals, landscape slice,
etc. Used for smoke-testing the generative model and sanity-checking
estimation runs. Entirely optional (the generic rolling-SMC diagnostics in
the driver don't depend on this).

Reference: [models/fsa_real_obs/sim_plots.py](../models/fsa_real_obs/sim_plots.py).

---

## 1.4 ⚠ Sim-Est Consistency Validation — DO THIS BEFORE ANY SMC RUN

**Mandatory step for every new model.** Skip at your own peril.

The simulator/scenario generator IS the ground truth your estimator
will be compared against. If the simulator and the estimator interpret
the data even slightly differently — wrong sign on a covariate, wrong
phase on a periodic forcing, wrong bin index in an out-of-bounds array
read — the SMC² will produce confidently-wrong posteriors. The
estimator doesn't know it's wrong; it just fits the data the
**estimator** thinks it's seeing, which differs from what the
**simulator** generated.

Real cost data from the high-res FSA development (see
[outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md](../outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md)):
**three sim/est consistency bugs cost ~15 hours of wasted GPU and
analyst time** before each was found, and one of them prompted two
separate "principled fix" plans for a phantom problem (the apparent
bridge cascade was actually a phase-misaligned covariate).

### 1.4.1 The three checks every new model must pass

Run all three before launching even a 1-window SMC. Total cost: ~30 min
of test-writing for a 3-state, 4-channel model. Catches >90% of the
bug class above.

#### Check A — Drift parity at truth

For the truth parameter set, the **simulator's drift** at any state
must equal the **estimator's drift** at the same state and parameters.
If they don't match, you have a sign/sign-convention bug somewhere.

```python
def test_drift_parity():
    from models.<your_model>.simulation import drift, DEFAULT_PARAMS
    from models.<your_model>.estimation import (
        propagate_fn, get_init_theta_fn, _PI,
    )
    p = DEFAULT_PARAMS                          # truth
    y = jnp.array([0.05, 0.10, 0.55])           # any plausible state
    aux = (build_T_B_arr(...), build_Phi_arr(...))

    # Simulator drift (numpy)
    sim_dy = drift(0.0, np.asarray(y), p, aux)

    # Estimator drift via propagate_fn at dt → 0 with zero noise:
    # x_pred ≈ y + dt * drift, so (x_pred - y) / dt ≈ drift.
    params_vec = get_init_theta_fn()
    grid_obs = build_minimal_grid_obs(...)      # T_B, Phi at k=0
    dt = 1e-4
    sigma_diag = jnp.zeros(3)
    noise = jnp.zeros(3)
    y_next, _ = propagate_fn(y, 0.0, dt, params_vec, grid_obs, 0,
                              sigma_diag, noise, None)
    est_dy = (np.asarray(y_next) - np.asarray(y)) / dt

    np.testing.assert_allclose(sim_dy, est_dy, atol=1e-3)
```

If this fails, you have a drift-formula mismatch (sign, missing term,
wrong parameter index).

#### Check B — Forward-prediction parity for every observation channel

For every Gaussian-linear observation channel, the simulator's
noiseless prediction must equal the estimator's `_obs_predictions`
(or whatever the equivalent is for your model) at the same state and
truth parameters.

```python
def test_obs_prediction_parity():
    p = DEFAULT_PARAMS
    y = jnp.array([0.30, 0.15, 0.60])
    C_t = float(circadian(t=0.0))   # midnight

    # Simulator: hr_mean (no noise)
    sim_hr = (p['HR_base'] - p['kappa_B']*y[0]
              + p['alpha_A_HR']*y[2] + p['beta_C_HR']*C_t)

    # Estimator's _gaussian_obs_ll computes pred internally; expose
    # _obs_predictions or replicate the formula from align_obs_fn output:
    grid = align_obs_fn(fake_obs_data_with_known_C, ...)
    est_hr = compute_pred_hr_from_grid(y, grid, k=0, params)

    np.testing.assert_allclose(sim_hr, est_hr, atol=1e-5)
```

Repeat for every Gaussian channel. For Bernoulli channels, compare
the success probability `p = sigmoid(...)` rather than the integer
sample. **This single test would have caught the C-phase bug
immediately:** at noon (C=−1) the sim's predicted HR for the same
state would have differed from the estimator's by `2 · |β_C_HR|` ≈ 5
bpm, which `assert_allclose(atol=1e-5)` flags instantly.

#### Check C — Cold-start coverage on a single window with truth-centred prior

Run **one** window of the SMC with priors centered tightly around the
truth (e.g. lognormal sigma 0.05). With correct sim/est alignment,
coverage should be ~100% on every parameter. If it's not, there is a
sim/est discrepancy — fix it before adding bridges, more particles,
or any other complexity.

```bash
python drivers/<your_scenario>.py --seed 42 --windows 1 --prior tight
```

Cold-start at a tight prior is the cheapest possible diagnostic of
sim/est consistency. If coverage isn't ~100%, **the rest of the
pipeline can't help you** — work backward from the failing parameters
to find which obs channel or covariate is mis-aligned.

### 1.4.2 Side-by-side data-flow plots (recommended)

For each exogenous covariate (T_B, Phi, periodic forcings like the
FSA's C(t), etc.), plot **side-by-side**:
- The array the simulator used to generate the data for one window.
- The array the estimator's `align_obs_fn` produces for the same window.

These should be **pointwise identical** (within float32 precision).
This is the test that would have caught the C-phase bug at the cost
of one matplotlib figure per covariate × the first non-W1 window.

A reference helper for this lives in
`/tmp/diff_old_vs_new_sim.py` (used during the high-res development to
diagnose the C-phase bug — adapt for your model).

### 1.4.3 Why module-level code review isn't enough

Six independent code-audit passes ran during the high-res FSA
debugging. The first three found two real bugs (mu_0 sign,
extract_state_at_step k/K). The next three were clean. **None caught
the C-phase bug.** Why: the C-phase bug isn't in any single function
— `align_obs_fn` is correct in isolation, the simulator is correct
in isolation, the estimator's `propagate_fn` is correct in isolation.
The bug is in the **interaction across the window boundary**, which
no single-file audit can catch.

End-to-end data-flow tests of the kind described in 1.4.1 and 1.4.2
DO catch this class of bug. Always run them before SMC.

---

## 2. The per-window data pipeline

Orchestrated by `drivers/<scenario>.py` + `smc2bj.pipeline.rolling`. A run
goes through seven stages; for each you need to know what's generic vs.
what you provide.

| Stage | Provided by you (model-specific) | Provided by framework (generic) |
|-------|----------------------------------|---------------------------------|
| 1. Exogenous-input schedule | Driver's macrocycle / input function | — |
| 2. Latent-state simulation | `simulation.py` drift + diffusion | `smc2bj.simulator.sde_solver_scipy` |
| 3. Observation generation | `simulation.py` per-channel generators | `smc2bj.simulator.generate_all_channels` (respects channel DAG) |
| 4. Missing-data corruption | `MissingDataConfig` channel names | `smc2bj.pipeline.missing_data.apply_missing_data` |
| 5. Window extraction | — | `smc2bj.pipeline.windowing.extract_window` |
| 6. Per-window preprocessing | `estimation.py::align_obs_fn` | Called from `rolling_window_smc` |
| 7. SMC² estimation | `estimation.py` (5 callbacks + priors) | `smc2bj.pipeline.rolling.rolling_window_smc` |

---

## 3. What's replaceable, what isn't

### Replaceable without algorithm work
- **Exogenous inputs.** Swap the scenario driver's input generator. E.g. replace FSA macrocycles with a sinusoid, square wave, real-world schedule.
- **Missing-data patterns.** Configure `MissingDataConfig` for your channel grouping (active vs. passive vs. all); or replace the function entirely if your gap structure is different (e.g. irregular sampling, clock-drift).
- **Diagnostic plots.** Move `sim_plots.py` functions around freely.
- **Inner-PF hyperparameters.** `SMCConfig` controls everything tempering-schedule-related, OT-rescue-related, and HMC-kernel-related. All tunable at driver-invocation time.

### Replaceable with algorithm work
- **Inner PF when observations aren't Gaussian.** The guided-Kalman proposal in GK-DPF v3-lite assumes $\log p(y_t \mid x_t, \theta)$ is quadratic in $x_t$. For Poisson counts, Bernoulli, or any non-Gaussian observation, the Kalman fusion step breaks down. You'd need to swap the inner PF to a bootstrap PF or auxiliary PF. The `lik_module_path` argument of `rolling_window_smc` makes this swap explicit — but you'd write the new PF.
- **Prior support.** The HMC-unconstrained transform in [smc2bj/transforms/unconstrained.py](../smc2bj/transforms/unconstrained.py) supports log-normal and normal priors. For hard support constraints (simplex, compact, bounded), you'd need a new transform (softmax reparameterisation for simplex, beta for bounded). HMC remains; only the `is_ln` / `is_norm` flags get a third option.
- **Strongly non-Gaussian posterior.** The Gaussian bridge assumes the previous posterior is approximately Gaussian; it's a first-order approximation. For multimodal posteriors, consider replacing the bridge with a mixture fit or annealed-importance-sampling restart.

### Not replaceable (structural)
- **Rolling-window structure.** `smc2bj.pipeline.rolling` assumes $\theta$ is constant within a window. Time-varying $\theta$ requires a different orchestration (state-space model over $\theta$ too).
- **Adaptive-tempered-SMC outer loop.** We use BlackJAX's tempered SMC; if your prior has zero density at sensible init, you'd need particle initialisation changes.

---

## 4. Worked sketch: 2-state OU model

To make the contract concrete, here's the file-level sketch for a simple
2-state OU model $dx = \theta(x - \mu) dt + \sigma dW$ with a single
observed channel $y_t = x_t + \eta_t$, $\eta \sim \mathcal{N}(0, \tau^2)$.
**This is a sketch — not a working implementation — showing what moves
where.**

### `models/ou_2state/simulation.py`

```python
import jax.numpy as jnp
from smc2bj.simulator.sde_model import SDEModel, StateSpec, ChannelSpec

# Drift + diffusion
def drift(t, y, params, aux):
    x1, x2 = y[0], y[1]
    dx1 = params['theta'] * (params['mu'] - x1)
    dx2 = params['theta2'] * (x1 - x2)  # x2 tracks filtered x1
    return jnp.stack([dx1, dx2])

def diffusion_diagonal(params):
    return jnp.array([params['sigma'], params['sigma2']])

# One observation channel
def gen_obs_x1(trajectory, t_grid, params, aux, prior_channels, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    x1 = trajectory[:, 0]
    obs = x1 + rng.normal(0, params['tau'], len(t_grid))
    return {'t_idx': np.arange(len(t_grid)), 'obs_value': obs}

def make_aux(params, init_state, t_grid, exog):  return {}
def make_y0(init_dict, params):  return jnp.array([init_dict['x1_0'], init_dict['x2_0']])

OU_2STATE_MODEL = SDEModel(
    name='ou_2state',
    states=[StateSpec('x1'), StateSpec('x2')],
    channels=[ChannelSpec(name='obs_x1', generate_fn=gen_obs_x1, dependencies=())],
    param_sets={'default': {'theta': 0.5, 'mu': 1.0, 'theta2': 0.3,
                            'sigma': 0.1, 'sigma2': 0.05, 'tau': 0.1}},
    init_states={'default': {'x1_0': 0.0, 'x2_0': 0.0}},
)
```

### `models/ou_2state/estimation.py`

```python
from collections import OrderedDict
import jax.numpy as jnp
from smc2bj.estimation_model import EstimationModel

PARAM_PRIOR_CONFIG = OrderedDict([
    ('theta',  ('lognormal', (jnp.log(0.5), 0.4))),
    ('mu',     ('normal',    (1.0, 0.3))),
    ('theta2', ('lognormal', (jnp.log(0.3), 0.4))),
    ('sigma',  ('lognormal', (jnp.log(0.1), 0.3))),
    ('sigma2', ('lognormal', (jnp.log(0.05), 0.3))),
    ('tau',    ('lognormal', (jnp.log(0.1), 0.3))),
])
INIT_STATE_PRIOR_CONFIG = OrderedDict()   # init states frozen
FROZEN_PARAMS = {}
COLD_START_INIT = jnp.array([0.0, 0.0])

def propagate_fn(y, t, dt, params, grid_obs, k, sigma_diag, xi, key):
    # Linearised one-step update (2D). See the FSA reference for the full
    # guided-Kalman form; for OU this is exact since dynamics are linear.
    ...

def diffusion_fn(params):
    return jnp.array([params['sigma'], params['sigma2']])

def obs_log_weight_fn(x_new, grid_obs, k, params):
    # Gaussian log-likelihood on channel x1
    y = grid_obs['obs_x1_value'][k]
    m = grid_obs['obs_x1_mask'][k]
    pred = x_new[0]
    resid = y - pred
    return m * (-0.5 * (resid / params['tau'])**2 - jnp.log(params['tau']))

def align_obs_fn(obs_data, t_steps, dt_hours):
    # Grid obs_x1 onto t_steps, produce value + mask arrays
    ...

def shard_init_fn(time_offset, params, exogenous, global_init):
    return global_init

OU_2STATE_ESTIMATION = EstimationModel(
    name='ou_2state',
    version='0.1',
    param_prior_config=PARAM_PRIOR_CONFIG,
    init_state_prior_config=INIT_STATE_PRIOR_CONFIG,
    frozen_params=FROZEN_PARAMS,
    propagate_fn=propagate_fn,
    diffusion_fn=diffusion_fn,
    obs_log_weight_fn=obs_log_weight_fn,
    align_obs_fn=align_obs_fn,
    shard_init_fn=shard_init_fn,
    # ... n_states=2, exogenous_keys=(), etc.
)
```

### `drivers/ou_2state_rolling.py`

```python
from smc2bj.estimation.config import SMCConfig, RollingConfig, MissingDataConfig
from smc2bj.pipeline.rolling import rolling_window_smc
from smc2bj.pipeline.missing_data import apply_missing_data
from models.ou_2state.estimation import OU_2STATE_ESTIMATION, COLD_START_INIT

# No macrocycle for OU — just a constant forcing, or none at all.
# Generate synthetic data, apply missing-data corruption,
# then pass obs_data to rolling_window_smc with OU_2STATE_ESTIMATION.
```

### What didn't change
- [smc2bj/](../smc2bj/) — everything: SMC² loop, bridge, tempering schedule, OT rescue, HMC, transforms, checkpoints, plots (generic ones).

### What changed
- One new `models/ou_2state/` package (3 files).
- One new `drivers/ou_2state_rolling.py` that wires it up.

---

## 5. FAQ / common pitfalls

**"My cold-start fails with NaN log-likelihood in the first tempering
step."** Check that `PARAM_PRIOR_CONFIG` covers sensible initial values
(the prior mean should give finite PF likelihood). Shrink priors if they
put mass on pathological parameter values (e.g. $\tau \to 0$ with
$\text{LogN}(\ln 0.01, 1.0)$ will often generate pathological draws).

**"The bridge collapses — ESS drops to 1/N on the first bridge step."**
Previous posterior is concentrated far from the prior; the Gaussian fit
is accurate but the new window's likelihood is very sharp around a
nearby mode. Raise `n_smc_particles`, or lengthen the overlap by
raising `window_days` relative to `stride_days` (smaller stride =
smaller posterior shift).

**"Seed changes change conclusions."** Single-seed results on this
class of model have ~5-10 pp coverage variance. Always replicate across
seeds before making claims about identifiability. See
[outputs/robustness_check_report.md](../outputs/robustness_check_report.md)
for a concrete cross-seed study.

**"My observation is a count / Bernoulli / bounded-support."** Guided
Kalman won't work: it assumes locally-Gaussian observations. Swap the
inner PF module by writing a new `smc2bj/log_density/<your_pf>.py` (use
bootstrap PF or auxiliary PF) and passing its factory via
`lik_module_path` / `lik_factory_name` to `rolling_window_smc`.

**"My prior has hard support constraints (simplex / compact)."** The
HMC unconstrained transform needs a third bijection option. Add it to
[smc2bj/transforms/unconstrained.py](../smc2bj/transforms/unconstrained.py)
and the corresponding `is_<dist>` flag to the T_arr producer.

**"Parameters are non-identifiable."** Check the shrinkage metric
printed after each window: $s_j = \sigma_{\text{post},j} / \sigma_{\text{prior},j}$.
If $s_j \geq 0.5$, the data did not move the posterior — the parameter
is prior-dominated. Either reparameterise (as FSA did for $\kappa$ via
the ratio), freeze it, or find a new observation channel that
constrains it.

**"Window 1 works but window 2 fails."** The bridge carries forward
degeneracy. Check ESS at the end of window 1 — if it's already low,
the Gaussian fit is over a degenerate cloud and the bridge becomes
ill-posed. Increase `n_smc_particles` or shrink `max_lambda_inc`.
