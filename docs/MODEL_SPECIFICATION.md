# Model Specification — FSA Real-Obs v4.1

> Reference model. For algorithm spec see [SMC2_ALGORITHM_SPECIFICATION.md](SMC2_ALGORITHM_SPECIFICATION.md);
> for porting your own model see [PORTING_GUIDE.md](PORTING_GUIDE.md).

**Model version:** 4.1 (real-obs, rolling SMC²)
**Latent state dimension:** 3 ($B, F, A$)
**Observation dimension:** 6 physiological channels
**Parameter dimension:** **33** (10 dynamical + 23 observational)
**Implementation:** [models/fsa_real_obs/](../models/fsa_real_obs/) (3-file package: `simulation.py`, `estimation.py`, `sim_plots.py`)

This document defines the complete generative model: latent SDE + observation
equations + missing-data corruption + priors + the HMC-unconstrained
reparameterisation. It is sufficient to simulate synthetic data and evaluate
the log-likelihood without reading the Python code.

---

## 1. Notation and conventions

| Symbol | Meaning |
|--------|---------|
| $t$ | continuous time in **days** |
| $\Delta t$ | observation-grid step size; default $\Delta t = 1$ day |
| $n_{\text{sub}}$ | SDE sub-steps per $\Delta t$; default 10 |
| $\theta \in \mathbb{R}^{33}$ | parameter vector (constrained space) |
| $u \in \mathbb{R}^{33}$ | parameter vector (unconstrained space, used by HMC) |
| $x_t = (B_t, F_t, A_t)$ | latent state at time $t$ |
| $y_{t,c}$ | observation in channel $c$ at time $t$ |
| $m_{t,c} \in \{0,1\}$ | observation mask (1 = observed) |
| PRNG | JAX PRNGKey; seed discipline: one key per independent random draw, split via `jax.random.split` |

---

## 2. Exogenous inputs

Two scalar input channels feed into the SDE as daily piecewise-constant signals:

- $T_B(t) \in [0.05, 0.95]$: **adaptation target** (normalised training load)
- $\Phi(t) \in [0.005, 0.28]$: **strain production** (normalised volume × intensity)

For the FSA scenario, these are generated via a **macrocycle schedule**. Three
variants are implemented in [drivers/fsa_real_obs_5yr_rolling.py](../drivers/fsa_real_obs_5yr_rolling.py):

**C0 (baseline).** 28-day mesocycles overlaid with:
- Off-season tapers every 180d, 21d duration: $T_B \sim \mathcal{U}(0.15, 0.30)$, $\Phi \sim \mathcal{U}(0.01, 0.02)$.
- Overreach spikes every 90d, 14d duration (skipped when overlapping a taper): $T_B \sim \mathcal{U}(0.80, 0.95)$, $\Phi \sim \mathcal{U}(0.20, 0.25)$.

**C2 (strong excitation).** Same mesocycle base + 35-day deep tapers every 90d
followed immediately by 21-day overreach. Designed to drive large $B$-$F$
decoupling and activate the Landau bifurcation.

**C3 (maximal excitation).** No base layer; a repeating 75-day cycle of 30d
moderate + 30d deep taper + 15d intense overreach.

The n=3 cross-seed robustness experiments in
[outputs/robustness_check_report.md](../outputs/robustness_check_report.md)
show C3 is reliably worse than baseline for identifiability; C0 and C2 are
statistically indistinguishable on average.

---

## 3. Latent dynamical system

Itô SDE system with three states, two dyadic feedbacks, and one bifurcation.

### 3.1 Fitness $B$ — Jacobi diffusion on $[0, 1]$

$$
dB_t = \frac{1 + \alpha_A A_t}{\tau_B}\bigl(T_B(t) - B_t\bigr)\,dt
     + \sigma_B \sqrt{B_t(1 - B_t)}\,dW_{B,t}
$$

Mean-reverting to $T_B(t)$ with timescale $\tau_B$, amplified by endocrine
state $A_t$. The Jacobi diffusion ($\sqrt{B(1-B)}$) makes $[0, 1]$ invariant.

### 3.2 Strain $F$ — CIR diffusion on $[0, \infty)$

$$
dF_t = \Bigl[\,\Phi(t) - \frac{1 + \lambda_B B_t + \lambda_A A_t}{\tau_F}\,F_t\,\Bigr]\,dt
     + \sigma_F \sqrt{F_t}\,dW_{F,t}
$$

Input-driven (by $\Phi$), with recovery rate enhanced by $B$ and $A$.

### 3.3 Endocrine amplitude $A$ — regularised Landau bifurcation

$$
dA_t = \bigl(\mu(B_t, F_t) A_t - \eta A_t^3\bigr)\,dt
     + \sigma_A \sqrt{A_t + \epsilon_A}\,dW_{A,t}
$$

where the bifurcation parameter is

$$
\mu(B_t, F_t) = \mu_0 + \mu_B B_t - \mu_F F_t - \mu_{FF} F_t^2.
$$

Subcritical pitchfork: $\mu > 0$ makes $A = 0$ unstable (amplitude grows),
$\mu < 0$ makes it stable. The $+\epsilon_A$ in the diffusion term is a
non-absorbing-boundary regulariser with $\epsilon_A = 10^{-4}$.

**Known limitation:** $\epsilon_A = 10^{-4}$ is small enough that in practice
$A$ is weakly absorbing — if $A$ decays close to 0, it struggles to escape.
This is flagged in the [excitation experiment report](../outputs/excitation_experiment_report.md)
and is a model-architecture issue, not an estimator issue.

### 3.4 Fixed constants (not estimated)

| Symbol | Value | Role |
|--------|-------|------|
| $\sigma_B$ | $0.01$ | Jacobi noise scale |
| $\sigma_F$ | $0.005$ | CIR noise scale |
| $\sigma_A$ | $0.02$ | Landau noise scale |
| $\epsilon_A$ | $10^{-4}$ | $A$-boundary regulariser |
| $\epsilon_B$ | $10^{-4}$ | $B$-clipping regulariser |

These are hardcoded in [models/fsa_real_obs/simulation.py](../models/fsa_real_obs/simulation.py)
(`EPS_A_FROZEN`, `EPS_B_FROZEN`) and the driver's SDE integrator. They were
frozen (removed from the estimation block) to guarantee structural
identifiability of the remaining 33 parameters — noise scales and obs
intercepts are strongly coupled.

### 3.5 Initial states

| Regime | $B_0$ | $F_0$ | $A_0$ |
|--------|-------|-------|-------|
| Cold start (Window 1) | $0.05$ | $0.10$ | $0.01$ |
| Warm bridge (Windows 2+) | extracted from previous window's posterior at $t = \text{stride\_days}$ |

The Window-1 cold-start vector lives in [models/fsa_real_obs/estimation.py](../models/fsa_real_obs/estimation.py#L137)
as `COLD_START_INIT`. Initial states are **frozen, not estimated** — they
contribute zero parameters to $\theta$. Extraction policy for warm bridge:
average the smoothed state at $t = \text{stride\_days}$ across the top 10
posterior draws; see [smc2bj/pipeline/rolling.py](../smc2bj/pipeline/rolling.py).

---

## 4. Observation model — 6 independent Gaussian channels

All six channels share the form $y_c \sim \mathcal{N}(\text{link}_c(x, \theta),
\sigma_{\text{obs},c}^2)$. Channels are conditionally independent given $x_t$
and $\theta$.

### 4.1 Channel 1 — Resting heart rate (mean-centered)

The most identifiability-sensitive channel. To break the $(B, F)$
collinearity that kills $\kappa_{\text{vagal}}$ / $\kappa_{\text{chronic}}$
joint identifiability, v4.1 applies **two identifiability fixes**:

1. **Mean-center RHR within each rolling window.** Removes $R_{\text{base}}$
   from the estimation block (it's a nuisance intercept whose value drifts
   over years of real data anyway). `align_obs_fn` subtracts the rolling
   mean before the PF sees the data.
2. **Freeze $\kappa_{\text{chronic}} = 10.0$.** After freezing $R_{\text{base}}$,
   $\kappa_{\text{vagal}}$ and $\kappa_{\text{chronic}}$ are still structurally
   non-identifiable jointly (they appear as a linear combination in a single
   channel's mean); fixing $\kappa_{\text{chronic}}$ makes $\kappa_{\text{vagal}}$
   uniquely identifiable.

The observation equation, after mean-centering per window:

$$
\mathrm{RHR}_{\text{centered}}(t) \sim
\mathcal{N}\bigl(-\kappa_{\text{vagal}} B(t) + \kappa_{\text{chronic}} F(t),\;
\sigma_{\text{obs},R}^2\bigr)
$$

Mean-centering is a **per-window operation**, applied inside
`align_obs_fn` (see [models/fsa_real_obs/estimation.py:490](../models/fsa_real_obs/estimation.py#L490)).
It subtracts the rolling mean before the PF sees it. $R_{\text{base}}$
and $\kappa_{\text{chronic}}$ are `frozen_params` on the `EstimationModel`
instance and do not appear in `PARAM_PRIOR_CONFIG`.

### 4.2 Channels 2-6

| # | Channel | Link $\text{link}_c(x, \theta)$ |
|---|---------|---------------------------------|
| 2 | Intensity | $I_{\text{base}} + c_B B - c_F F$ |
| 3 | Duration | $D_{\text{base}} + d_B B - d_F F$ |
| 4 | Stress | $S_{\text{base}} - s_A A + s_F F$ |
| 5 | Sleep | $\text{Sleep}_{\text{base}} + sl_B B - sl_F F + sl_A A$ |
| 6 | Circadian timing | $\text{Time}_{\text{base}} + t_A A - t_F F$ (on logit scale) |

Channel 6 is bounded in $(0, 1)$; pre-transform via logit to map to $\mathbb{R}$
so the Gaussian observation is well-defined.

---

## 5. Missing-data corruption model

Three overlapping gap patterns. Implementation:
[smc2bj/pipeline/missing_data.py](../smc2bj/pipeline/missing_data.py)
(generic) + scenario params in the driver.

1. **Rest days.** Each week, sample $n_{\text{rest}} \sim \mathcal{U}\{2, 3\}$
   days and mask the **active** channels (intensity, duration, timing).
2. **Random dropout.** Independently drop 15% of observations on the
   **passive** channels (RHR, stress, sleep).
3. **Broken-watch gap.** A single contiguous 14-day gap placed uniformly in
   the interior of the series (90 days from either end); all channels masked.

All three are applied with a fixed (seed-dependent) schedule: the mask is not
random at evaluation time — it's a deterministic function of `seed`.

---

## 6. Parameter vector $\theta \in \mathbb{R}^{33}$ and priors

Source of truth: [models/fsa_real_obs/estimation.py](../models/fsa_real_obs/estimation.py#L84)
(`PARAM_PRIOR_CONFIG`). The machine-generated table below is the
authoritative numerical spec; the hand-written §6.1 / §6.2 tables that
follow annotate the same information with semantic descriptions.

<!-- AUTO-GENERATED-PRIORS-START -->
<!-- regenerate via: python tools/dump_model_spec.py models.fsa_real_obs.estimation --update-docs -->

| # | Parameter | Distribution | Location (constrained mean) | Scale |
|---|-----------|--------------|------------------------------|-------|
| 1 | `tau_B` | LogNormal | 14 | 0.08 (log-space) |
| 2 | `alpha_A` | LogNormal | 1 | 0.4 (log-space) |
| 3 | `tau_F` | LogNormal | 7 | 0.3 (log-space) |
| 4 | `lambda_B` | LogNormal | 3 | 0.3 (log-space) |
| 5 | `lambda_A` | LogNormal | 1.5 | 0.3 (log-space) |
| 6 | `mu_0_abs` | LogNormal | 0.1 | 0.4 (log-space) |
| 7 | `mu_B` | LogNormal | 0.3 | 0.4 (log-space) |
| 8 | `mu_F` | LogNormal | 0.1 | 0.4 (log-space) |
| 9 | `mu_FF` | LogNormal | 0.4 | 0.4 (log-space) |
| 10 | `eta` | LogNormal | 0.2 | 0.3 (log-space) |
| 11 | `kappa_vagal` | LogNormal | 12 | 0.3 (log-space) |
| 12 | `sigma_obs_R` | LogNormal | 1.5 | 0.4 (log-space) |
| 13 | `I_base` | Normal | 0.5 | 0.1 |
| 14 | `c_B` | LogNormal | 0.2 | 0.5 (log-space) |
| 15 | `c_F` | LogNormal | 0.1 | 0.5 (log-space) |
| 16 | `sigma_obs_I` | LogNormal | 0.05 | 0.4 (log-space) |
| 17 | `D_base` | Normal | 0.5 | 0.1 |
| 18 | `d_B` | LogNormal | 0.3 | 0.5 (log-space) |
| 19 | `d_F` | LogNormal | 0.2 | 0.5 (log-space) |
| 20 | `sigma_obs_D` | LogNormal | 0.08 | 0.4 (log-space) |
| 21 | `S_base` | Normal | 30 | 10 |
| 22 | `s_A` | LogNormal | 15 | 0.5 (log-space) |
| 23 | `s_F` | LogNormal | 20 | 0.5 (log-space) |
| 24 | `sigma_obs_S` | LogNormal | 5 | 0.4 (log-space) |
| 25 | `Sleep_base` | Normal | 0.5 | 0.1 |
| 26 | `sl_A` | LogNormal | 0.2 | 0.5 (log-space) |
| 27 | `sl_B` | LogNormal | 0.1 | 0.5 (log-space) |
| 28 | `sl_F` | LogNormal | 0.2 | 0.5 (log-space) |
| 29 | `sigma_obs_Sleep` | LogNormal | 0.1 | 0.4 (log-space) |
| 30 | `Time_base` | Normal | 0 | 1 |
| 31 | `t_A` | LogNormal | 1 | 0.5 (log-space) |
| 32 | `t_F` | LogNormal | 0.5 | 0.5 (log-space) |
| 33 | `sigma_obs_Time` | LogNormal | 0.5 | 0.4 (log-space) |

<!-- AUTO-GENERATED-PRIORS-END -->

### 6.1 Dynamical parameters (10)

| Parameter | Role | Prior |
|-----------|------|-------|
| $\tau_B$ | fitness timescale | $\text{LogNormal}(\ln 14, 0.08)$ |
| $\alpha_A$ | $A$-enhancement of fitness | $\text{LogNormal}(\ln 1.0, 0.4)$ |
| $\tau_F$ | strain timescale | $\text{LogNormal}(\ln 7.0, 0.3)$ |
| $\lambda_B$ | $B$-enhancement of recovery | $\text{LogNormal}(\ln 3.0, 0.3)$ |
| $\lambda_A$ | $A$-enhancement of recovery | $\text{LogNormal}(\ln 1.5, 0.3)$ |
| $\mu_{0,\text{abs}}$ | $\|\mu_0\|$ bifurcation baseline | $\text{LogNormal}(\ln 0.10, 0.4)$ |
| $\mu_B$ | fitness → bifurcation | $\text{LogNormal}(\ln 0.30, 0.4)$ |
| $\mu_F$ | strain → bifurcation (linear) | $\text{LogNormal}(\ln 0.10, 0.4)$ |
| $\mu_{FF}$ | strain → bifurcation (quadratic) | $\text{LogNormal}(\ln 0.40, 0.4)$ |
| $\eta$ | Landau restoring force | $\text{LogNormal}(\ln 0.20, 0.3)$ |

Note $\mu_0$ is estimated via its absolute value $\mu_{0,\text{abs}}$ and
reconstructed via $\mu_0 = -\mu_{0,\text{abs}}$; see the driver's
`plot_latent_reconstruction` for the inverse map.

### 6.2 Observational parameters (23)

**RHR (2):** $\kappa_{\text{vagal}} \sim \text{LogN}(\ln 12, 0.3)$;
$\sigma_{\text{obs},R} \sim \text{LogN}(\ln 1.5, 0.4)$.
(*$\kappa_{\text{chronic}}$ frozen at 10.0, $R_{\text{base}}$ frozen at 62.0;
both appear in the forward model but are removed from the estimation block.*)

**Intensity (4):** $I_{\text{base}} \sim \mathcal{N}(0.5, 0.1)$;
$c_B \sim \text{LogN}(\ln 0.2, 0.5)$;
$c_F \sim \text{LogN}(\ln 0.1, 0.5)$;
$\sigma_{\text{obs},I} \sim \text{LogN}(\ln 0.05, 0.4)$.

**Duration (4):** $D_{\text{base}} \sim \mathcal{N}(0.5, 0.1)$;
$d_B \sim \text{LogN}(\ln 0.3, 0.5)$; $d_F \sim \text{LogN}(\ln 0.2, 0.5)$;
$\sigma_{\text{obs},D} \sim \text{LogN}(\ln 0.08, 0.4)$.

**Stress (4):** $S_{\text{base}} \sim \mathcal{N}(30, 10)$;
$s_A \sim \text{LogN}(\ln 15, 0.5)$; $s_F \sim \text{LogN}(\ln 20, 0.5)$;
$\sigma_{\text{obs},S} \sim \text{LogN}(\ln 5, 0.4)$.

**Sleep (5):** $\text{Sleep}_{\text{base}} \sim \mathcal{N}(0.5, 0.1)$;
$sl_A \sim \text{LogN}(\ln 0.2, 0.5)$; $sl_B \sim \text{LogN}(\ln 0.1, 0.5)$;
$sl_F \sim \text{LogN}(\ln 0.2, 0.5)$;
$\sigma_{\text{obs, Sleep}} \sim \text{LogN}(\ln 0.1, 0.4)$.

**Timing (4):** $\text{Time}_{\text{base}} \sim \mathcal{N}(0, 1)$;
$t_A \sim \text{LogN}(\ln 1.0, 0.5)$; $t_F \sim \text{LogN}(\ln 0.5, 0.5)$;
$\sigma_{\text{obs, Time}} \sim \text{LogN}(\ln 0.5, 0.4)$.

Total: **10 dynamical + 23 observational = 33 parameters.** *(The prior
prose in the archived pre-refactor doc said "34"; this is a typo — there
are 33, as enforced by `n_dim` on the `EstimationModel` instance.)*

---

## 7. Unconstrained reparameterisation for HMC

HMC operates in $\mathbb{R}^d$ without boundaries. Each constrained parameter
is mapped through a bijection:

- **LogNormal** $\theta \sim \text{LogN}(\mu, \sigma)$ $\Rightarrow$ $u = \log\theta$ so $u \sim \mathcal{N}(\mu, \sigma^2)$.
- **Normal** $\theta \sim \mathcal{N}(\mu, \sigma)$ $\Rightarrow$ $u = \theta$ (identity).

The per-dimension transform flags `is_ln[i]`, `is_norm[i]` and the moments
`ln_mu[i]`, `ln_sigma[i]`, `n_mu[i]`, `n_sigma[i]` are stored in `T_arr`,
which the prior-evaluation and sampling code consume. Both maps have closed
Jacobians, absorbed into the prior log-density:

$$
\log \pi(u) =
\sum_{i \in \mathcal{L}} \log \mathcal{N}(u_i; \mu_{\ln,i}, \sigma_{\ln,i})
+ \sum_{i \in \mathcal{N}} \log \mathcal{N}(u_i; \mu_{n,i}, \sigma_{n,i})
$$

— no Jacobian correction needed because $u$ is defined directly in the
prior's natural parameterisation.

Implementation:
[smc2bj/transforms/unconstrained.py](../smc2bj/transforms/unconstrained.py).

---

## 8. End-to-end likelihood

For a rolling window with observations $\{y_{t,c}, m_{t,c}\}_{t=1,\ldots,T;\, c=1,\ldots,6}$,
the marginal likelihood is

$$
p(y_{1:T} \mid \theta) = \int p(x_{0:T} \mid \theta) \prod_{t,c : m_{t,c}=1} p(y_{t,c} \mid x_t, \theta)\,dx_{0:T}
$$

with the SDE transition $p(x_{t+1} \mid x_t, \theta)$ factorised via the
Euler-Maruyama scheme in $n_{\text{sub}}$ sub-steps per $\Delta t$. The
inner particle filter (GK-DPF v3-lite) estimates $\log p(y_{1:T} \mid \theta)$
via importance sampling with a guided-Kalman proposal; see
[SMC2_ALGORITHM_SPECIFICATION.md](SMC2_ALGORITHM_SPECIFICATION.md) §3 for
the exact proposal, resampling, and OT-rescue formulas.

---

## 9. Source-code map

| Section here | Source file | Lines |
|--------------|-------------|-------|
| §3 latent SDE (drift, diffusion) | [models/fsa_real_obs/simulation.py](../models/fsa_real_obs/simulation.py) | `drift` 66-84, `diffusion_diagonal` 108-112 |
| §3 per-substep stochastic step | [models/fsa_real_obs/estimation.py](../models/fsa_real_obs/estimation.py) | `propagate_fn` 152-310, `imex_step_fn` 317-341 |
| §4 obs generators (synthetic) | [models/fsa_real_obs/simulation.py](../models/fsa_real_obs/simulation.py) | `gen_obs_*` 167-277 |
| §4 obs log-likelihoods (inference) | [models/fsa_real_obs/estimation.py](../models/fsa_real_obs/estimation.py) | `_obs_predictions` 343-382, `obs_log_prob_fn` 384-414 |
| §5 missing-data corruption | [smc2bj/pipeline/missing_data.py](../smc2bj/pipeline/missing_data.py) | all |
| §6 priors (`PARAM_PRIOR_CONFIG`) | [models/fsa_real_obs/estimation.py](../models/fsa_real_obs/estimation.py#L84) | 84-133 |
| §7 unconstrained transforms | [smc2bj/transforms/unconstrained.py](../smc2bj/transforms/unconstrained.py) | all |
| Exogenous schedules (macrocycles) | [drivers/fsa_real_obs_5yr_rolling.py](../drivers/fsa_real_obs_5yr_rolling.py) | `generate_macrocycle_C{0,2,3}` |
