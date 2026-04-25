# Postmortem: three sim/est consistency bugs in the high-res FSA model

**Date:** 2026-04-23 → 2026-04-25
**Scope:** v0.2.0 → v0.2.1-fsa-high-res
**Outcome:** mean coverage 37.5% → **96.8%**, PASS 1/27 → **27/27**, after fixing three bugs.

This is a record of what went wrong, how it was found, what the cost was,
and what process would have caught each one *before* burning GPU time.

---

## Bug 1: `mu_0` sign mismatch

**Location.** `models/fsa_high_res/estimation.py` (was at three sites:
`propagate_fn`, `imex_step_fn`, `forward_sde_stochastic`).

**Mistake.** Copied the daily-FSA `mu_0_abs` reparameterisation into
the high-res estimator without re-reading what it did. In daily FSA
the truth `mu_0 = −0.10` is negative; we store its absolute value as
`mu_0_abs` (always positive, lognormal-prior-friendly) and negate it
back in the drift:

```python
mu_0 = -params[_PI['mu_0_abs']]   # ← daily FSA pattern
```

For high-res we deliberately set `mu_0 = +0.02` so that A sits at the
Stuart-Landau fixed point A* = √(μ/η) > 0 instead of needing a
pitchfork crossing. The estimator kept the negation, so it was running
with `mu_0 = −0.02` against simulator data generated with `mu_0 = +0.02`.

**Detection.** Round 3 audit (algorithmic-correctness Explore agent
read both files and traced the parameter through the drift function).

**Fix.** [Commit `41997bf`.] Renamed `mu_0_abs → mu_0` in
`PARAM_PRIOR_CONFIG`, removed the negation at all three call sites.
Result: W1 cold-start coverage 96.6% → 100%, but bridge cascade still
present at this point (other bugs still active).

**What would have caught it cheaply.** A 5-line test that asserts the
estimator's drift evaluation at the truth parameters matches the
simulator's drift at the same state and parameters:

```python
def test_drift_parity():
    p_truth = DEFAULT_PARAMS
    y = jnp.array([0.05, 0.10, 0.55])
    aux = (T_B_arr, Phi_arr)
    sim_dy = simulator_drift(0.0, y, p_truth, aux)
    est_dy = estimator_drift_at_truth(0.0, y, p_truth, aux)
    assert jnp.allclose(sim_dy, est_dy, atol=1e-6)
```

This is **15 minutes of test-writing** vs the **3+ hours** spent debugging
posterior plots and chasing nonexistent "bridge cascade" theories.

---

## Bug 2: `extract_state_at_step` passes `K` (particle count) where `k` (step index) is expected

**Location.** `smc2bj/log_density/gk_dpf_v3_lite.py:323`.

**Mistake.** The signature of `model.propagate_fn` is
`(y, t, dt, params, grid_obs, k, sigma_diag, noise, rng_key)` —
position 6 is the **step index**. Inside the `extract_state_at_step`
helper a stale variable `K` (the **particle count**, a constant ≈ 400)
was being passed at position 6 by mistake:

```python
x_new, pred_lw = model.propagate_fn(
    y, k, dt, params, grid_obs, K,         # ← K here, should be k
    sigma_diag, xi, None)
```

JAX silently returns the **last in-bounds value** (or NaN-equivalent)
for `grid_obs['T_B'][K]` when the window has only `t_steps = 96` bins.
So during the bridge state-extraction PF rerun, the model integrated
its B/F dynamics with a constant garbage `T_B` for every step.

The main PF code path (line 191-192) was correct; only the
`extract_state_at_step` helper was wrong, and it's only called by
`rolling_window_smc` to compute the next window's initial latent
state.

**Detection.** Round 3 audit (same Explore agent that found bug 1).

**Fix.** [Commit `8814d5d`.] Changed `K → k` at line 323. This left
the next-window's `fixed_init_state` computation correct.

**What would have caught it cheaply.** A unit test that sends a known
state through `extract_state_at_step` at `target_step` and compares to
the simulator's trajectory at the same global step. With one window
that's a 50-line test. Cost: ~30 minutes.

The bug had been there for the full development of the high-res
estimator; the daily FSA tests didn't catch it because they don't
exercise `extract_state_at_step` (or they do, but with `t_steps = 120`
which happens to overlap with `K`-ish indices in a way that masked the
issue).

---

## Bug 3: C(t) phase mis-alignment (THE big one)

**Location.** `models/fsa_high_res/estimation.py:align_obs_fn` (the
old C-array computation block).

**Mistake.** `align_obs_fn` was generating the circadian forcing C(t)
from **window-local time** that always started at 0:

```python
t_days = np.arange(T, dtype=np.float32) * float(dt)   # window-local!
C_val = np.cos(2.0 * np.pi * t_days + PHI_FROZEN)
```

The simulator generates HR / stress / steps using **global** time:

```python
# in models/fsa_high_res/simulation.py
C = circadian(t_grid, phi=...)        # t_grid is global
hr_mean = HR_base - kappa_B*B + alpha_A_HR*A + beta_C_HR * C
```

With `stride = 48 bins (12h)`, every other rolling window starts at a
noon offset within the day. The estimator's window-local C in those
windows is `cos(2π · t_local)` ranging over [+1, -1, +1] across the
day — but the **simulator's C in those same global bins** is
`cos(2π · (t_local + 0.5))` ranging over [-1, +1, -1]. Exact sign
inversion, on alternate windows.

When SMC averaged the per-window posteriors, all three β_C_*
coefficients were pulled toward 0 — bias = `mean(truth, -truth) = 0`
in the limit of many alternating windows. The visual signature was
**narrow CIs locked at ~50% of truth value, all three β_C_*
coefficients biased the same way.**

**Effect at scale.**

| Run | Mean raw cov | PASS |
|-----|--------------|------|
| Pre-fix (with bugs 1, 2 already fixed) | 37.5% | 1/27 |
| Post-fix | **96.8%** | **27/27** |

This single bug accounted for the entire apparent "bridge cascade".

**Detection.** **The user**, by visual inspection of the
`parameter_tracking.png` plot. The bias signature (all β_C_* shifted
toward zero by similar fractions, narrow CIs) was diagnostic in a way
that no algorithm-level audit caught. Six independent Explore-agent
audits had passed before the user spotted it.

**Why the audits missed it.** The agents looked at code-paths
individually:
- `align_obs_fn` "looks right" by itself (it correctly computes a cosine over the window).
- The simulator "looks right" by itself.
- The estimator's `obs_log_weight_fn` matches its own `propagate_fn`'s sign convention.

The bug is **not in any single function** — it's in the *interaction
of two functions across a window boundary*. End-to-end data-flow
audits weren't done; module-level audits were.

**Fix.** [Commit `7621d1e`.] Treat C(t) as an exogenous channel like
T_B and Phi. The simulator's `gen_C_channel` emits per-bin C values
for the full trajectory in global frame. `extract_window` slices it by
global bin index. `align_obs_fn` consumes the sliced C directly
instead of recomputing it locally.

**What would have caught it cheaply.** A side-by-side plot of:
- The C(t) array the simulator used to generate one window's HR data.
- The C(t) array the estimator's `align_obs_fn` produced for the same window.

For W2 (the first window where the bug manifests), these two arrays
would have been pointwise opposite. **Cost: 5 minutes of plotting.**

A simpler test would have been: pick a window that doesn't start at
midnight, run sim → align → compare:

```python
def test_C_phase_alignment():
    # generate 3 days of obs at sim-time
    obs = generate_obs(...)
    # extract a window starting at noon (bin 48)
    w = extract_window(obs, 48, 144)
    grid = align_obs_fn(w, t_steps=96, dt=1/96)
    # the C value at window-bin 0 should be cos(2π·0.5) = -1
    assert jnp.isclose(grid['C'][0], -1.0, atol=1e-3)
```

**5 lines of test code.** Would have flipped the entire investigation
from "the bridge is broken" to "the data alignment is broken" inside
the first hour of debugging.

---

## What this cost

| Activity | Time |
|---------|------|
| Initial dev (W1=100% achieved) | ~6h |
| First "bridge cascade" debugging round (mu_0 fix) | ~3h |
| Second round (audits + extract_state_at_step) | ~3h |
| Bridge experiments (N=512, MoG K=2) — built on top of the bugs | ~5h GPU + 2h coding |
| Three plans (principled fixes, OT flows) — both retrospectively unnecessary | ~3h writing |
| User diagnosis of C-phase bug + fix + verification rerun | ~2h |
| **Total burnt** | **~20h, of which ~15h would have been saved** |

The N=512 (+9pp) and MoG K=2 (+3pp) experiments were trying to make
the bridge cope with **mis-aligned data**. With aligned data, the
single-Gaussian bridge at N=256 just works. Two whole research tracks
(longer stride, OT-regularised normalising flows) were planned to fix
a phantom problem.

---

## The single takeaway

> **The simulator/scenario generator IS the ground truth. Three of the
> three real bugs we found were sim/est consistency bugs. Catch these
> before any SMC run.**

A new section "Sim-Est Consistency Validation" in the porting guide
makes this discipline mandatory for new models. See
[`docs/PORTING_GUIDE.md`](../../docs/PORTING_GUIDE.md).

---

## Other audit findings (not the source of the cascade, but worth noting)

These were turned up by the same audits but had no measurable effect on
the final coverage:

- Bernoulli sigmoid clipped at `[1e-8, 1-1e-8]`. Could in principle
  cause gradient issues for HMC near saturation; in practice the data
  doesn't push p that close to the boundary.
- Stress and steps independent dropout (5% each, separate masks). Mild
  heteroscedasticity but no observable bias.
- `extract_state_at_step` uses one RNG seed across all 10 draws — they
  aren't independent. Doesn't matter for a 10-particle average.

These can be tightened in a future cleanup pass without affecting
correctness.
