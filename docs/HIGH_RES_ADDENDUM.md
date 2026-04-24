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
   with training bursts (5× daily Φ during a randomly-placed 45-90 min
   window) and zero during sleep hours.

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

*(populated by Phase D after the 14-day / 12-window SMC² run completes —
see `outputs/fsa_high_res_rolling/C0_N256_s42/`).*

| Quantity | Value |
|----------|-------|
| Mean coverage (raw, 29 params) | **TBD** |
| Mean coverage (data-informed) | **TBD** |
| PASS rate (≥70% coverage) | **TBD** / 12 windows |
| Wall-clock | **TBD** |
| Per-window bins | 288 |
| Inner-PF particles | 400 |
| SMC particles | 256 |

## Known limitations at v0.2.0

- **A-state often remains near zero.** The Landau bifurcation
  `dA = μ·A - η·A³` with `ε_A = 1e-4` in the diffusion means `A = 0` is
  quasi-absorbing. Under the default `recovery` scenario's moderate
  load, A barely escapes the boundary in a 14-day window. A-coupled
  parameters (`alpha_A`, `α_A^HR`, `k_A`, `k_A_S`, `beta_A_st`) may
  therefore be prior-dominated. This is the same architectural issue
  documented for the daily FSA model — not new in the high-res port.
- **φ frozen at 0** — circadian phase is weakly identifiable over 14
  days and is kept fixed for the proof-of-principle. A follow-up can
  add `phi` to `PARAM_PRIOR_CONFIG` and rerun.
- **Only the C0 macrocycle is ported.** C2/C3 would be a one-line
  addition but aren't required for the warm-up.

## Follow-ups

- Bigger rollout (30-60 days) to see if F / A dynamics mature enough
  to improve shrinkage on A-coupled params.
- Add Poisson Bernoulli-style step channel as an alternative inner PF
  test case.
- Port SWAT (different latent SDE, overlapping obs architecture).
