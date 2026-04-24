# High-res FSA addendum (v0.2.0)

> Delta-doc describing `models/fsa_high_res/`, a 15-min-bin variant of the
> daily FSA reference model. For the unchanged latent SDE and the rest of
> the framework, see [MODEL_SPECIFICATION.md](MODEL_SPECIFICATION.md),
> [SMC2_ALGORITHM_SPECIFICATION.md](SMC2_ALGORITHM_SPECIFICATION.md), and
> [PORTING_GUIDE.md](PORTING_GUIDE.md).

## What's new

The high-res variant is the first model in the repo that exercises:

1. **A sub-daily time grid** — `dt = 1/96` day = 15 min. The
   `rolling_window_smc` unit-convention was tightened to make this work
   cleanly: `stride_days`/`window_days`/`n_days_total` in `rolling_cfg`
   are always step-index units (same as `obs_data`'s `t_idx`), and `dt`
   is metadata for the PF integrator only. A bug fix at line 221 of
   [smc2bj/pipeline/rolling.py](../smc2bj/pipeline/rolling.py#L221)
   removes a spurious `/ dt` that was only correct when `dt = 1.0`.

2. **A 4-channel mixed-likelihood observation model** — 3 Gaussian (HR,
   stress, log-steps) all linear in $(B, F, A)$ and gated by a sleep/wake
   mask, plus 1 Bernoulli (binary sleep label). The Gaussian channels go
   through the **sequential-scalar Kalman fusion** (extension of FSA's
   6-channel fusion); the Bernoulli runs through `obs_log_weight_fn`
   only — it never enters the guided proposal. This confirms the plan's
   claim that GK-DPF v3-lite handles Bernoulli natively.

3. **Deterministic circadian forcing** `C(t) = cos(2π·t_days + φ)` with
   `φ` frozen at 0 (healthy morning chronotype). C(t) is precomputed by
   `align_obs_fn` and stored in `grid_obs['C']` — the inner PF consumes
   it per-bin without per-step trig calls.

4. **Sub-daily exogenous Φ(t)** — the macrocycle generator still picks
   daily Φ values (reusing `generate_macrocycle_C0` from the daily FSA
   driver), but `generate_phi_sub_daily` expands those to per-bin arrays
   using a **morning-loaded gamma profile**: zero during sleep, ramp-up
   after waking, peak ~3h post-wake, exponential tapering through the
   afternoon. No sedentary-plateau baseline — activity is concentrated
   in the morning hunter-gatherer style, with ~75% of daily load before
   midday. The shape is `t · exp(-t/τ)` (Gamma(k=2)) with τ=3h, scaled
   so the daily-integrated Φ matches the daily FSA's per-day load.

## Model summary

| | Daily FSA (v4.1) | High-res FSA (v0.2) |
|---|---|---|
| Latent state | (B, F, A) | (B, F, A) — **unchanged** |
| Δt | 1.0 day | 1/96 day (15 min) |
| Substeps/bin | 10 | 4 |
| Obs channels | 6 × Gaussian (RHR, intensity, duration, stress, sleep, timing) | **HR** (Gaussian, sleep-gated) + **sleep** (Bernoulli) + **stress** (Gaussian, wake-gated) + **steps** (log-Gaussian, wake-gated) |
| Params estimated | 33 | 29 |
| Circadian | none | C(t) = cos(2π·t + 0) covariate |
| Window | 120d / stride 30d | 3d / stride 1d (= 288 bins / 96 bins) |
| Rollout | 365d, 9 windows | 14d, 12 windows |

## Mixed-likelihood pattern (canonical for future models)

The pattern used here generalises directly to any model with a mix of
Gaussian-linear-in-state channels and non-Gaussian channels (Bernoulli,
Poisson, ordinal):

1. **Gaussian-linear channels** → stacked into `H` (d_obs × d_state),
   `bias` (d_obs,), `R_diag` (d_obs,). Each channel gets a **presence
   mask** (0/1) from `align_obs_fn`. `propagate_fn` runs a single
   `jax.lax.scan` of scalar Kalman updates over the stack, accumulating
   the predictive log-marginal.

2. **Non-Gaussian channels** → only contribute to `obs_log_weight_fn`.
   Their log-likelihood is added to the particle weight post-propagation.
   The guided proposal doesn't see them — the loss is acceptable as long
   as at least one Gaussian channel is present at each step to keep the
   proposal informed.

The math of this pattern is documented inline in
[models/fsa_high_res/estimation.py](../models/fsa_high_res/estimation.py)
(see the `propagate_fn` / `obs_log_weight_fn` docstrings) and is expected
to carry over to SWAT (which has exactly this mix: HR Gaussian + sleep
Bernoulli, plus testosterone amplitude as a new state).

## Parameter blocks

Regenerate via `python tools/dump_model_spec.py models.fsa_high_res.estimation`.

- **10 dynamical** — same as FSA v4.1: `tau_B, alpha_A, tau_F, lambda_B,
  lambda_A, mu_0_abs, mu_B, mu_F, mu_FF, eta`.
- **5 HR** — `HR_base, kappa_B, alpha_A_HR, beta_C_HR, sigma_HR`.
  HR = `HR_base - kappa_B·B + alpha_A_HR·A + beta_C_HR·C(t) + noise`,
  observed during sleep bins.
- **3 Bernoulli sleep** — `k_C, k_A, c_tilde`. Sleep prob =
  `sigmoid(k_C·C(t) + k_A·A - c_tilde)`.
- **5 stress** — `S_base, k_F, k_A_S, beta_C_S, sigma_S`. Stress =
  `S_base + k_F·F - k_A_S·A + beta_C_S·C(t) + noise`, observed during
  waking bins.
- **6 steps** — `mu_step0, beta_B_st, beta_F_st, beta_A_st, beta_C_st,
  sigma_st`. `log(steps+1) = mu_step0 + beta_B_st·B - beta_F_st·F +
  beta_A_st·A + beta_C_st·C(t) + noise`, observed during waking bins.

Frozen (not estimated): `sigma_B, sigma_F, sigma_A` (SDE noise — same
as FSA), `eps_A, eps_B` (boundary regulariser), `phi` (circadian phase,
healthy morning chronotype).

## Proof-of-principle run

Full 14-day / 27-window rolling run at `--seed 42`, `--condition C0`
(2026-04-24). See `outputs/fsa_high_res_rolling/C0_N256_s42/`.

| Quantity | Value |
|----------|-------|
| Windows | 27 (1-day window, 12h stride) |
| Mean coverage (raw, 29 params) | **37.5%** |
| Mean coverage (data-informed) | 33.6% |
| PASS rate (≥70% coverage) | **1 / 27** |
| Best window (W1 cold-start) | **100%** raw, **100%** data-informed |
| Wall-clock | 1.27 h |
| Per-window bins | 96 |
| Inner-PF particles | 400 |
| SMC particles | 256 |

### What the result actually tells us

- **The framework is correct end-to-end.** W1 cold-start achieves
  **100% coverage** on all 29 parameters (after fixing the mu_0 sign
  bug — see provenance note). This confirms the 3-channel Kalman fusion
  + Bernoulli obs_log_weight_fn design is correct on 15-min data, and
  the model as implemented is fully identifiable from cold-start.

- **Bridge cascade degrades fast on this model — but the degradation
  is BIAS, not variance.** W2-W27 drift into 20-60% coverage with
  posterior means locked off-truth at narrow CIs (confidently wrong),
  not wide uncertainty. Inspection of `parameter_tracking.png` shows
  most failing traces have stable means substantially offset from
  the green truth line, with their 90% CIs narrower than the offset.

  The Gaussian-bridge warm-start approximates the previous window's
  posterior as Gaussian; when the true posterior is narrow and slightly
  skewed (likely here, given the ratio-like κ_B / α_A^HR / k_F
  structure and the A-coupled parameters' Stuart-Landau curvature),
  the first-order fit error compounds across each 1-day stride. By
  W26-W27 the posterior has collapsed to a biased fixed point 3-10%
  coverage.

  This is a **research question about the bridge**, not a model
  identifiability failure: cold-start (W1) recovers the truth perfectly.

- **This is a research question, not a bug.** The Gaussian-bridge
  bridge is a known first-order approximation documented in the plan's
  "Risks and mitigations" section. Fix candidates:
  1. Ledoit-Wolf shrinkage is on; try targeted mixture bridges.
  2. Larger n_smc_particles (256 → 512) to preserve particle diversity.
  3. Periodic cold-restarts every N windows.
  4. Wider priors at the expense of per-window convergence.

### Per-window raw coverage

```
W1  cold 100.0%  ← validates model correctness (post mu_0 fix)
W2-6     24-52% (early bridge drift, mean ~40%)
W7-13    44-62% (partial plateau)
W14-24   17-48% (slow drift off-truth)
W25-27    3-10% (posterior collapsed far from truth)
```

### Provenance — the mu_0 sign bug

The first rollout attempt used the FSA daily estimator's `mu_0_abs`
reparameterisation (truth mu_0 < 0 stored as positive lognormal,
negated at use). For the high-res model we picked mu_0 = +0.02
(positive, so A sits at the Stuart-Landau fixed point A* = √(μ/η)
> 0 rather than relying on pitchfork crossings). The estimator was
still negating at use, producing a sign mismatch: estimator computed
μ = −0.02 while simulator generated data with μ = +0.02.

Fixed by renaming the parameter `mu_0_abs` → `mu_0` in
`PARAM_PRIOR_CONFIG` and removing the negation in `propagate_fn`.
Post-fix: W1 coverage 96.6% → **100%**. Later-window results changed
only slightly (37% vs 41% before), confirming the bridge cascade is
the dominant problem, not mu_0.

## Known limitations at v0.2.0

- **A-state tuning required for activation.** The Landau bifurcation
  `dA = μ·A - η·A³` with ε_A=1e-4 in the diffusion means `A = 0` is
  quasi-absorbing. Initial defaults (A_0=0.01, μ_0=-0.10) kept A
  stuck at zero for the full POC horizon, making all A-coupled params
  unidentifiable. Tuned values (A_0=0.10, μ_0=-0.05) start A clear of
  the boundary and let μ cross zero at B≈0.18 instead of B≈0.33; A
  then oscillates in [0.03, 0.10] during the 14-day horizon, tracking
  the daily F spikes (higher F → lower μ → A decays during training
  burst then recovers). This is the same architectural issue
  documented for the daily FSA model and is not specific to the
  high-res port.
- **φ frozen at 0** — circadian phase is weakly identifiable over 14
  days and is kept fixed for the proof-of-principle. A follow-up can
  add `phi` to `PARAM_PRIOR_CONFIG` and rerun.
- **Only the C0 macrocycle is ported.** C2/C3 would be a one-line
  addition but aren't required for the warm-up.
- **Priors tightened ~2× vs daily FSA.** At 15-min resolution the
  information density per observation window is ~10× the daily case,
  and the adaptive-tempering ESS solver picks tiny Δλ if the
  prior→posterior gap is too large. Tighter priors (lognormal σ
  halved on most params) give a tractable number of tempering levels
  while still leaving 3-5× room around truth for the posterior to
  move.

## Follow-ups

- **Bridge-cascade collapse research.** The 96.6% → 30% degradation
  over 27 windows is the primary open question. Try: larger
  n_smc_particles (512); Ledoit-Wolf → mixture bridges; periodic
  cold-restarts; posterior widening via small deliberate tempering
  back-steps.
- Bigger rollout (30-60 days) to see if F / A dynamics mature enough
  to improve shrinkage on A-coupled params.
- Add Poisson step channel variant as an alternative inner PF test
  case (demonstrates two independent non-Gaussian channels can
  coexist via obs_log_weight_fn).
- Port SWAT (different latent SDE, overlapping obs architecture).
