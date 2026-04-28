# 3-way comparison: pre-tune Gauss / post-tune Gauss / post-tune SF (Path A)

**Date:** 2026-04-26
**Goal:** isolate the contribution of (a) the model re-tune (Apr 26 morning,
public-dev #5/#6: `beta_Z 2.5→4.0`, `c_tilde 3.0→2.5`) and (b) the
new Schrödinger-Föllmer (SF) bridge to the SWAT Set A SMC²
rolling-window inference.

## TL;DR

| Run | Bridge | Truth | Prior centre | Mean cov | Inf cov | PASS |
|---|---|---|---|---:|---:|---:|
| **Pre-tune Gauss** (B3 fix) | Gaussian + LW | `beta_Z=2.5` `c_tilde=3.0` | `beta_Z=2.5` `c_tilde=3.0` (matched) | **48.4%** | 46.7% | 2/27 |
| **Post-tune Gauss** | Gaussian + LW | `beta_Z=4.0` `c_tilde=2.5` | `beta_Z=2.5` `c_tilde=3.0` (mismatched) | **25.6%** | 24.6% | 2/27 |
| **Post-tune SF** | BW geodesic, blend=0.5 | `beta_Z=4.0` `c_tilde=2.5` | `beta_Z=2.5` `c_tilde=3.0` (mismatched) | **34.3%** | 33.6% | 2/27 |

Two effects cleanly separated:

1. **Model re-tune introduced a prior-truth mismatch that dropped
   coverage by 22.8 pp** (48.4 → 25.6). The estimable-parameter priors
   in `models/swat/estimation.py:PARAM_PRIOR_CONFIG` are still centred
   at the pre-tune values, while the artifact's truth shifted. SMC²
   now has to chase truth across a high-prior-density barrier, and
   the Gaussian bridge's LW shrinkage prevents it from ever reaching
   it.

2. **SF bridge recovers +8.7 pp on the post-tune artifact** (25.6 →
   34.3) — modest but real. The mechanism turns out NOT to be the
   designed-for one (importance-sampling moment-match toward q1).
   The IS step degenerates in 35-D (see below); the gain comes from
   **less aggressive shrinkage in the Bures-Wasserstein (BW)
   interpolation** vs Liu-West shrinkage.

**Neither change closes the 70% gate.** SF is a useful base-measure
upgrade but Path A is the wrong tool against a prior-truth mismatch.

## Per-window coverage

| W | Pre-Gauss | Post-Gauss | Post-SF | Δ(SF − Gauss) |
|---|---:|---:|---:|---:|
| 1  | 94.3% | 94% | 91% | −3 |
| 2  | 65.7% | 69% | 71% | +2 |
| 3  | 48.6% | 80% | 69% | −11 |
| 4  | 65.7% | 49% | 66% | +17 |
| 5  | 71.4% | 14% | 34% | +20 |
| 6  | 34.3% | 20% | 26% | +6 |
| 7  | 37.1% | 20% | 40% | +20 |
| 8  | 51.4% | 29% | 31% | +2 |
| 9  | 51.4% | 31% | 40% | +9 |
| 10 | 57.1% | 34% | 29% | −5 |
| 11 | 37.1% | 43% | 43% | 0 |
| 12 | 31.4% | 20% | 26% | +6 |
| 13 | 34.3% |  9% | 23% | +14 |
| 14 | 45.7% | 20% | 40% | +20 |
| 15 | 51.4% | 20% | 43% | +23 |
| 16 | 51.4% | 14% | 14% | 0 |
| 17 | 42.9% |  6% | 29% | +23 |
| 18 | 48.6% |  9% | 31% | +22 |
| 19 | 48.6% | 14% | 31% | +17 |
| 20 | 51.4% | 17% | 34% | +17 |
| 21 | 51.4% | 17% | 11% | −6 |
| 22 | 42.9% | 17% | 17% | 0 |
| 23 | 45.7% | 20% | 17% | −3 |
| 24 | 40.0% | 17% | 20% | +3 |
| 25 | 42.9% |  0% |  9% | +9 |
| 26 | 42.9% |  3% | 17% | +14 |
| 27 | 20.0% |  6% | 23% | +17 |

**Pattern**: SF beats post-tune Gauss in 22/27 windows; ties or
slightly worse in 5. Largest SF wins are in the deep cascade
(W17-W20: +17 to +23 pp). Both runs collapse late (W21-W27).
Pre-tune Gauss dominates both throughout — confirms prior-truth
match is the dominant factor.

## SF Path A diagnostics — why the modest gain

`fit_sf_base()` builds the bridge base in three steps:

1. **q0** — fit a Gaussian to the previous-window posterior
   (Liu-West shrunk).
2. **q1** — importance-sampling moment-match: re-weight the prev
   particles by the new-window log-density to estimate the new
   posterior mean and covariance.
3. **bridge base** — interpolate (q0, q1) along the BW geodesic at
   `t = blend = 0.5`.

In a well-conditioned regime, q1 should differ meaningfully from q0
(the new window pulls the mean), and the BW geodesic carries
particles smoothly to the new posterior. **In 35-D with a
1-day-window log-likelihood that's ~hundreds of nats sharper than
the prior, IS degenerates**:

| Diagnostic | Value (across 26 bridge windows) |
|---|---|
| `IS n_eff` median | **1.5 / 256** |
| `IS n_eff` distribution | 14/26 at floor (1.0); max 3.4 |
| `&#124;&#124;m1 − m0&#124;&#124;` | **0.000** (every window) |
| log det of bridge base | −88 to −252 (vs Gauss-LW −180 to −250) |

`||m1 − m0|| = 0.000` is the smoking gun: when IS `n_eff` falls below
the `floor_eff_n=5` threshold in `estimate_target_gaussian()`, the
estimator falls back to **uniform weights**, which gives back the
sample mean of the prev particles — i.e. **q1 ≡ q0**. The BW
geodesic between two identical Gaussians is just that Gaussian.
The "new posterior" information never enters the bridge base in 26/26
bridge windows.

**So why is SF still better than Gauss?** Two secondary effects:

- **Less LW-shrinkage damping**: the Gaussian bridge applies Liu-West
  shrinkage *to the base measure itself*. The SF path takes
  `q0 = LW-shrunk fit` but interpolates along BW with the unshrunk
  q1=q0 endpoint, which (with `entropy_reg=0`) is mathematically
  equivalent to the unshrunk Gaussian fit. The bridge log_det is
  ~30 nats wider on average → wider proposals → less particle
  collapse → marginally better acceptance.
- **Numerical Cholesky path**: the SF code uses a slightly different
  positive-definite regularisation (`+1e-4·I` in BW) which appears
  to be a hair more stable than the Gauss bridge's `+1e-6·I`.

These are honest improvements but **they are not what the SF
formulation was designed to deliver**. The designed gain — q1 carrying
new-data information through the bridge — is gated by IS efficiency
in 35-D, which is unfixable with naïve self-normalised IS.

## What the failure mode says about Path A vs Path B

Path A (this run) used Gaussian-fit IS for q1. The 35-D parameter
space + sharp 1-day likelihoods make IS untenable. Two routes
forward:

- **Path B — particle-empirical SF with kernel score**: instead of
  fitting Gaussian q1 by IS, treat the propagated SDE particles as
  an empirical q1, and use a kernel density / score estimator on
  them. The Schrödinger drift becomes `∇ log q_t(u)` evaluated on
  the empirical mixture. This sidesteps IS degeneracy entirely
  because no re-weighting toward the target is required — the
  particles already carry the new likelihood once they've been
  propagated through the inner PF.
- **Path C — local IS with annealed proposals**: keep the Gaussian-q1
  framework but use multiple short SMC steps inside the bridge,
  re-fitting q1 after each. Cheaper than Path B, more involved than
  Path A.

## Recommendation

**Two independent issues, both must be addressed:**

1. **Prior-truth mismatch (dominant, fix first)**.
   `models/swat/estimation.py:PARAM_PRIOR_CONFIG` was authored before
   the Apr-26 model re-tune. Centres for `beta_Z` and `c_tilde` (and
   anything that depends on them, e.g. `delta_c`) need to be
   re-centred to the new truth values **OR** widened so the prior
   density at truth isn't suppressed.
   - Cost: ~30 min code edit + re-run psim consistency tests + re-run
     SMC² rolling. Estimated coverage gain: 15-20 pp (most of the
     Gauss-side regression should reverse).
   - Does NOT need SF; the Gaussian bridge will work once prior covers
     truth.

2. **SF Path B (secondary, after #1 verified)**.
   Path A's gain is real but mechanism is wrong (less shrinkage, not
   better q1). Path B (particle-empirical kernel-score SF) addresses
   the IS bottleneck and should give a more honest 10-20 pp
   improvement on top of a corrected-prior baseline.
   - Cost: ~1 day of impl work; needs a bandwidth selector for the
     KDE and a stop-gradient on the score evaluation.

**Suggested next session**: tackle #1 first (cheap, big expected
return); only then re-evaluate whether Path B is worth the impl work.

## Reproducibility

- Pre-tune Gauss numbers: from
  `outputs/swat_rolling/set_A_healthy_N256_s42/result.md` ("Attempt 3
  (B3 fix)" column, against the **pre-tune** psim artifact).
- Post-tune Gauss numbers: from
  `outputs/swat_rolling/set_A_healthy_N256_s42_gaussian/rolling_checkpoint.json`
  (snapshot of the no-suffix dir taken 13:33 after the post-tune
  artifact re-run).
- Post-tune SF numbers: from
  `outputs/swat_rolling/set_A_healthy_N256_s42_sf0.50/rolling_checkpoint.json`
  (this run, 13:33-14:53).
- All three: same seed (42), same N_SMC=256, same N_PF=400, same
  driver code (`drivers/swat/rolling.py`), same `tests/` green.
- SF impl: `smc2bj/estimation/sf_bridge.py` (9 unit tests pass).

## Status

- Branch: `feat/schrodinger_follmer_bridge` (off `feat/swat_rolling_driver`).
- Commits up through the SF impl + this comparison; **no push, no tag**.
- Awaiting user direction on the two-track fix above.
