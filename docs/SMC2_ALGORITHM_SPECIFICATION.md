# SMC² Algorithm Specification

> Implementation-independent spec. For model-side definitions see
> [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md); for the locked numerical
> fingerprint see [NUMERICAL_FINGERPRINT.md](NUMERICAL_FINGERPRINT.md).

This document gives the exact inference algorithm, stated at a level of
precision sufficient to re-implement it in another language and benchmark
against the reference implementation in [smc2bj/](../smc2bj/).

---

## 1. Problem statement

**Given**
- Observations $\{y_{t,c}, m_{t,c}\}_{t=1,\ldots,T,\; c=1,\ldots,C}$ with mask
  $m_{t,c} \in \{0, 1\}$.
- Prior $\pi(\theta)$ on $\theta \in \mathbb{R}^{n_d}$ (unconstrained).
- SDE transition density $p(x_{t+1} \mid x_t, \theta)$ via Euler-Maruyama
  with $n_{\text{sub}}$ sub-steps per observation step.
- Observation density $p(y_{t,c} \mid x_t, \theta)$ (per-channel Gaussian in
  the reference model).

**Target** — for each rolling window $w$, the posterior
$p(\theta \mid y_{1:T_w})$, approximated by $N_{\text{smc}}$ weighted particles.

**Structure.** Two nested SMC loops:
- **Outer loop**: tempered SMC over $\theta$ with adaptive $\lambda$ schedule
  and an HMC kernel per tempering level.
- **Inner loop**: a particle filter (GK-DPF v3-lite) evaluates
  $\widehat{\log p}(y_{1:T_w} \mid \theta)$ by importance sampling over
  latent paths $x_{0:T_w}$.

---

## 2. Outer loop: adaptive-tempered SMC over $\theta$

**Tempered target** at temperature $\lambda \in [0, 1]$:

$$
\pi_\lambda(\theta \mid y) \;\propto\; \pi(\theta) \cdot \hat L(\theta; y)^\lambda,
\qquad \hat L(\theta; y) = \exp\!\bigl(\widehat{\log p}(y_{1:T_w} \mid \theta)\bigr).
$$

At $\lambda = 0$ the target is the prior; at $\lambda = 1$ it is the
posterior. The schedule $\lambda_0 = 0, \lambda_1, \ldots, \lambda_K = 1$
is chosen adaptively.

### 2.1 Adaptive schedule via ESS bisection

At each step $k$, compute the next increment $\Delta\lambda$ as the largest
value such that the effective sample size (ESS) under reweighting remains
at least $\alpha_{\text{target}} \cdot N_{\text{smc}}$. Concretely, find
$\Delta$ such that

$$
\mathrm{ESS}\bigl(\{w_i \exp(\Delta \cdot \ell_i)\}_{i=1}^{N_{\text{smc}}}\bigr) = \alpha_{\text{target}} \cdot N_{\text{smc}}
$$

where $\ell_i = \log \hat L(\theta_i; y) - \log \pi(\theta_i) = \widehat{\log p}(y \mid \theta_i)$
is the incremental log-likelihood for particle $i$. The bisection is
BlackJAX's `smc.ess.ess_solver` with `solver.dichotomy`.

**Clamping.** The solver's $\Delta$ is then clamped:

$$
\Delta\lambda \leftarrow \min(\Delta, \Delta_{\max})
$$

with $\Delta_{\max} = 0.05$ for cold starts and $0.10$ for bridges. This
guarantees a minimum number of tempering levels even when the adaptive
solver would otherwise take a large jump.

**Termination.** If $1 - (\lambda_k + \Delta\lambda) < 10^{-6}$, snap to
$\lambda_{k+1} = 1$. Otherwise continue.

### 2.2 Within-temperature HMC kernel

For each tempering level $\lambda_{k+1}$, apply an HMC update:

| Parameter | Value (cold) | Value (bridge) |
|-----------|--------------|----------------|
| Leapfrog step size | $\epsilon = 0.025$ | $\epsilon = 0.025$ |
| Num leapfrog steps | $L = 8$ | $L = 8$ |
| Inverse mass matrix | diagonal, $\widehat{\mathrm{diag}}(\mathrm{Var}(\theta))$ | same |
| Num MCMC steps | $M_{\text{cold}} = 5$ | $M_{\text{bridge}} = 3$ |

The mass matrix is re-estimated from the current particle cloud after every
tempering level:

$$
M^{-1}_{\text{diag},i} = \max\bigl(\mathrm{Var}(\theta_{\cdot,i}), \;10^{-4}\bigr).
$$

Full (non-diagonal) mass matrices were tried and failed: the particle-filter
likelihood landscape punishes correlated HMC proposals and acceptance
collapses to zero by $\lambda \approx 0.3$. Diagonal is stable.

Implementation: [smc2bj/estimation/smc_window.py](../smc2bj/estimation/smc_window.py);
mass matrix in [smc2bj/estimation/mass_matrix.py](../smc2bj/estimation/mass_matrix.py).

### 2.3 Resampling

Systematic resampling (BlackJAX's `smc.resampling.systematic`), applied
every tempering level as part of BlackJAX's tempered SMC kernel. The
adaptive schedule implicitly keeps ESS at $\alpha_{\text{target}} \cdot N$
so resampling typically does not trigger particle degeneracy.

### 2.4 Cold-start initialisation

Particles are drawn independently from the prior in unconstrained space:

$$
u_i \sim \pi(\cdot), \qquad u_i \in \mathbb{R}^{n_d}, \quad i = 1, \ldots, N_{\text{smc}}.
$$

For each dimension $j$, if `is_ln[j]=1` the sample is
$u_{i,j} \sim \mathcal{N}(\mu_{\ln,j}, \sigma_{\ln,j}^2)$;
if `is_norm[j]=1` then $u_{i,j} \sim \mathcal{N}(\mu_{n,j}, \sigma_{n,j}^2)$.

Implementation: `sample_from_prior` in [smc2bj/estimation/sampling.py](../smc2bj/estimation/sampling.py).

### 2.5 Warm-start bridge (Windows 2+)

Between rolling windows, a **Gaussian-base bridge** replaces the prior with
a Gaussian fit to the previous posterior, so $\lambda = 0$ starts close to
the new posterior rather than at the broad prior.

**Step 1. Fit a Gaussian via Ledoit-Wolf shrinkage.** From the previous
window's particles $\{\theta^{\text{old}}_i\}$:

- $\hat\mu = \bar\theta^{\text{old}}$, $\hat S = \mathrm{Cov}(\theta^{\text{old}})$.
- LW shrinkage target: $\hat m = \mathrm{tr}(\hat S) / d$ (scaled identity).
- Shrinkage coefficient:

  $$
  \alpha = \min\!\left(\frac{\hat b^2}{\hat \delta^2}, \; 1\right)
  $$

  with $\hat \delta^2 = \|\hat S - \hat m I\|_F^2$ and
  $\hat b^2 = \frac{1}{N^2}\sum_i \|(\theta^{\text{old}}_i - \hat\mu)(\theta^{\text{old}}_i - \hat\mu)^\top - \hat S\|_F^2$.

- Shrunk covariance: $\hat\Sigma = (1 - \alpha)\hat S + \alpha \hat m I$,
  plus a tiny eigenvalue floor $10^{-4} I$ for numerical safety.

**Step 2. Bridge target.** Define

$$
\tilde\pi_0(u) = \mathcal{N}(u; \hat\mu, \hat\Sigma), \qquad
\tilde\ell(u) = \widehat{\log p}(y \mid u) - \log\tilde\pi_0(u),
$$

so the tempered target $\tilde\pi_\lambda(u) \propto \tilde\pi_0(u) \cdot
\exp(\lambda\tilde\ell(u))$ starts at the Gaussian fit ($\lambda=0$) and
ends at the new posterior ($\lambda=1$).

**Step 3. Initial particles.** Draw fresh samples from $\tilde\pi_0$:
$u_i = \hat\mu + L z_i$ with $z_i \sim \mathcal{N}(0, I)$, $L = \mathrm{chol}(\hat\Sigma)$.

The rest is as in §2.2-2.3 with the bridge clamp and $M_{\text{bridge}}$.

Implementation: `run_smc_window_bridge` in
[smc2bj/estimation/smc_window.py](../smc2bj/estimation/smc_window.py).

---

## 3. Inner loop: GK-DPF v3-lite particle filter

For each parameter $\theta$ (one per outer-loop particle), the inner loop
estimates $\widehat{\log p}(y_{1:T_w} \mid \theta)$. The variant used is
**GK-DPF v3-lite**: guided-Kalman proposal, systematic resampling,
Liu-West shrinkage correction, and optimal-transport (OT) rescue on low
ESS.

Implementation: [smc2bj/log_density/gk_dpf_v3_lite.py](../smc2bj/log_density/gk_dpf_v3_lite.py).

### 3.1 State proposal: guided Kalman

At each observation step $t$, given ancestor states $\{x^j_{t-1}\}$ and the
current observation $y_t$, draw new states via a **locally Gaussian proposal**
centered on the one-step linearisation of the drift at the current state.

Specifically, linearise the drift $f(x, t; \theta) = \partial_t \mathbb{E}[x]$
at $x^j_{t-1}$ to obtain a predictive Gaussian:

$$
\tilde x^j_t \sim \mathcal{N}\bigl(\mu^j_t(\theta), \Sigma^j_t(\theta)\bigr),
$$

and Kalman-fuse with the linearised observation $y_t = H x + \eta$,
$\eta \sim \mathcal{N}(0, R)$, to form the optimal Gaussian proposal:

$$
q(x^j_t \mid x^j_{t-1}, y_t, \theta)
= \mathcal{N}(x^j_t; \bar x^j_t, \bar \Sigma^j_t).
$$

The weight increment per particle is the standard guided-importance-sampling
log-weight. When the true dynamics are strongly nonlinear (as in C3's
75-day overreach-taper cycles), this proposal degrades.

### 3.2 Liu-West shrinkage correction

Weights are corrected via an ESS-scaled Liu-West shrinkage term — the standard
Liu-West rejuvenation factor, applied adaptively as ESS drops. See
[smc2bj/log_density/_gk_kernel.py](../smc2bj/log_density/_gk_kernel.py)
(`smooth_resample_ess_scaled_lw`).

### 3.3 Systematic resampling

Resample at every observed step with O(K) systematic resampling (see
BlackJAX's implementation). Every-step resampling removes ancestor bookkeeping
and makes the PF fully differentiable.

### 3.4 Optimal-transport rescue (on low ESS)

When ESS drops below a threshold, trigger an **OT rescue** that replaces
the degenerate particle cloud with an OT-coupled version of a proposal
measure. Parameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `ot_ess_frac` | $0.05$ | Activate when $\text{ESS} < 0.05 \cdot K$ |
| `ot_temperature` | $5.0$ | Sinkhorn regularisation strength |
| `ot_max_weight` | $0.01$ | Cap on any single transported weight |
| `ot_rank` | $5$ | Low-rank factorisation rank |
| `ot_n_iter` | $2$ | Sinkhorn iterations |
| `ot_epsilon` | $0.5$ | Sinkhorn kernel bandwidth |

The rescue output is **interpolated** with the vanilla PF output via a
sigmoid on ESS:

$$
\text{final} = \sigma\!\bigl((K \cdot 0.05 - \text{ESS}) / 5\bigr) \cdot \text{rescue}
             + \bigl(1 - \sigma(\cdot)\bigr) \cdot \text{vanilla}.
$$

This ensures the rescue only kicks in meaningfully when ESS is very low;
far from collapse, it has no effect.

Implementation: [smc2bj/transport/](../smc2bj/transport/) (the `kernel`,
`project`, `resample`, `sinkhorn` modules).

### 3.5 Marginal log-likelihood estimator

$$
\widehat{\log p}(y_{1:T_w} \mid \theta) = \sum_{t=1}^{T_w} \log \Bigl(\frac{1}{K} \sum_{j=1}^K w^j_t\Bigr)
$$

with the per-step unnormalised incremental weights $w^j_t$ summing to the
per-step log-marginal. This is then used in the outer SMC (§2) to form
$\log \hat L(\theta) = \widehat{\log p}(y \mid \theta)$.

---

## 4. Rolling-window orchestration

Given observations over $[0, T_{\text{total}})$, split into overlapping
windows of length $W$ days with stride $S$:

$$
w = 0, 1, \ldots, \quad [w \cdot S, \; w \cdot S + W).
$$

**Per-window state**
- $w = 0$ (cold): draw initial SMC particles from the prior (§2.4), run
  adaptive tempered SMC (§2.1-2.2).
- $w \geq 1$ (warm): fit Gaussian to previous posterior (§2.5), run bridged
  tempered SMC.

After each window:
1. Convert particles to constrained space via the inverse transform.
2. Compute per-parameter posterior mean / std / 5% / 95% quantiles and
   coverage (is the truth in the 90% CI?).
3. Compute **shrinkage** $s_j = \sigma_{\text{post},j} / \sigma_{\text{prior},j}$
   and flag parameters with $s_j < 0.5$ as "data-informed". Coverage restricted
   to data-informed parameters is the primary identifiability metric.
4. Extract smoothed latent state at $t = \text{stride}$, averaged over the
   top 10 posterior draws — used as the PF initial state for the next window.
5. Checkpoint to JSON.

Implementation: `rolling_window_smc` in
[smc2bj/pipeline/rolling.py](../smc2bj/pipeline/rolling.py).

---

## 5. Algorithm pseudocode

```
Input: obs y, mask m, model M, priors, config (smc_cfg, rolling_cfg),
       cold_start_init x0, truth dict, total days T_total
Output: sequence of per-window posterior particle clouds

# Precompute
compute prior_stds[name] for every param
n_windows = min(max_windows, (T_total - window_days) // stride + 1)
prev_particles = None
init_state = x0

for w in 0 .. n_windows - 1:
    start = w * stride; end = start + window_days
    window_obs = extract_window(y, start, end)
    grid_obs = M.align_obs_fn(window_obs, window_days, dt)          # per-window preprocessing: RHR mean-centering, etc.
    ld = make_gk_dpf_v3_lite_log_density(M, grid_obs,
                                         n_pf_particles, bandwidth,
                                         ot_cfg, dt, seed+w,
                                         fixed_init_state=init_state)

    if prev_particles is None:
        # Cold-start adaptive tempered SMC
        particles = run_smc_window(ld, M, T_arr, smc_cfg,
                                   initial_particles=None, seed=seed + w*1000)
    else:
        # Bridge from previous posterior (Gaussian base measure)
        particles = run_smc_window_bridge(ld, prev_particles, M, T_arr,
                                          smc_cfg, seed=seed + w*1000)

    # Posterior summaries in constrained space
    samp = [unconstrained_to_constrained(u, T_arr) for u in particles]
    stats, coverage, coverage_informed, n_informed =
        compute_coverage_and_shrinkage(samp, M.all_names, truth, prior_stds)

    # Warm-start state extraction
    target_step = stride / dt
    extracted = mean over top-10 particles of ld.extract_state_at_step(u, target_step)
    init_state = extracted
    prev_particles = particles

    save_checkpoint(results, truth, smc_cfg, rolling_cfg)
```

### Adaptive tempered SMC (inside run_smc_window)

```
state = SMC_init(initial_particles, lam=0)
inv_mass = diag(var(initial_particles))
while state.lambda < 1:
    delta = ess_bisection(ll_fn, state.particles, target_ess_frac)
    delta = clip(delta, 0, 1 - state.lambda)
    delta = min(delta, max_lambda_inc)
    next_lam = state.lambda + delta
    if 1 - next_lam < 1e-6: next_lam = 1.0
    # HMC kernel at this tempering level
    state = blackjax_tempered_smc_kernel(state, M_mcmc_steps, next_lam,
                                         hmc_params(inv_mass, eps, L))
    inv_mass = diag(var(state.particles))
return state.particles
```

---

## 6. Known approximations and failure regimes

| Approximation | Regime of validity | Symptom when violated |
|---------------|-------------------|-----------------------|
| Guided-Kalman proposal assumes locally linear drift | State trajectories vary slowly relative to $\Delta t$ | Low ESS, acceptance collapse during rapid regime changes (e.g. FSA condition C3's 75-day taper cycles — see [outputs/robustness_check_report.md](../outputs/robustness_check_report.md)) |
| Diagonal mass matrix | Posterior has weakly correlated dimensions | Acceptance drops when dimensions are highly correlated (observed with full mass matrices on this problem) |
| Adaptive-$\lambda$ clamp ($\Delta_{\max}$) | Small enough to land many levels | If too small, wall-clock blows up; if too large, tempering steps miss the posterior |
| Gaussian bridge | Previous posterior is approximately Gaussian | Bridge imprecision on multimodal posteriors; the Ledoit-Wolf shrinkage mitigates but does not eliminate |

---

## 7. Reference numerical output

See [NUMERICAL_FINGERPRINT.md](NUMERICAL_FINGERPRINT.md) for the expected
log-marginal-likelihood, per-parameter posterior mean, and per-parameter
posterior stdev at a locked seed (`--seed 42 --condition C0 --windows 1`).
`tests/test_smc2_fingerprint.py` asserts these to `rtol=1e-3`.

If a re-implementation matches the fingerprint, it is numerically equivalent;
if it differs beyond rtol, compare intermediate quantities against
`rolling_checkpoint.json`'s per-tempering-level acceptance and lambda
schedule.
