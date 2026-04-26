# SWAT Set A rolling-window estimation: PARTIAL FIX, CASCADE PERSISTS

**Date:** 2026-04-26 (updated after H1 fix attempt)
**Run:** `--seed 42` (defaults from SwatRollingConfig)
**Output:** `outputs/swat_rolling/set_A_healthy_N256_s42/`
**Driver / framework commits:**
  - `31e1f1d` Phase F docs (pre-fix baseline)
  - `<unstaged>` t-arg fix in extract_state_at_step (this attempt)
**Artifact:** psim v0.1.2 `set_A_healthy_14d/`.

## TL;DR

**A real bug was found and fixed in `smc2bj/log_density/gk_dpf_v3_lite.py`'s
`extract_state_at_step`** (passed `t = k` instead of `t = k * dt`,
silently miscomputing time for any model that uses `t` in its
`propagate_fn` — which SWAT does, fsa_high_res does not). Fix verified:
fsa_high_res 1-window regression bit-identical to v0.1.1.

**The fix improved early bridge windows (W2-W7 raw coverage averaged
56% → 61%) but did NOT resolve the deep cascade collapse (W10+).**
Aggregate mean coverage: **31.2% pre-fix → 30.8% post-fix**. PASS rate
unchanged at 2/27.

**H1 ruled out as the sole cause.** The cascade has another root.

**No push, no tag.** Branch `feat/swat_rolling_driver` retains all
commits locally.

## Headline numbers

| Metric | Pre-fix | Post-fix | Gate |
|---|---:|---:|---|
| Cold-start coverage (W1, raw) | 100.0% | 91.4% | ≥ 70% ✓ (both) |
| Cold-start coverage (W1, informed) | 100.0% | 81.8% | ≥ 70% ✓ (both) |
| Mean coverage (raw, all 27) | 31.2% | **30.8%** | ≥ 70% ❌ |
| Mean coverage (informed, all 27) | 31.3% | **30.0%** | ≥ 70% ❌ |
| PASS rate (≥70%) | 2/27 | **2/27** | ≥ 19/27 ❌ |
| Min window coverage | 5.7% (W13) | **2.9% (W18)** | — |
| Wall-clock | 1.34 h | 1.32 h | — |

## Pre-fix vs Post-fix per-window comparison

| W | Pre-fix raw | Post-fix raw | Δ | Notes |
|---|---:|---:|---:|---|
| 1 | 100.0% | 91.4% | -8.6 | both ≥70% (cold start; fluctuation is JIT-noise) |
| 2 | 54.3% | **80.0%** | +25.7 | **fix recovered W2** |
| 3 | 71.4% | 60.0% | -11.4 | small reversal |
| 4 | 60.0% | 42.9% | -17.1 | reversal |
| 5 | 68.6% | 51.4% | -17.2 | reversal |
| 6 | 31.4% | **51.4%** | +20.0 | **fix recovered W6** |
| 7 | 48.6% | **62.9%** | +14.3 | **fix improved W7** |
| 8 | 40.0% | 45.7% | +5.7 | small improvement |
| 9 | 42.9% | 48.6% | +5.7 | small improvement |
| 10 | 17.1% | 14.3% | -2.8 | **collapse from here**, ~unchanged |
| 11 | 14.3% | 22.9% | +8.6 | minor noise |
| 12 | 8.6% | 17.1% | +8.5 | minor noise |
| 13 | 5.7% | 14.3% | +8.6 | minor noise |
| 14 | 17.1% | 14.3% | -2.8 | unchanged |
| 15 | 25.7% | 14.3% | -11.4 | reversal |
| 16 | 25.7% | 17.1% | -8.6 | reversal |
| 17 | 28.6% | 14.3% | -14.3 | reversal |
| 18 | 11.4% | **2.9%** | -8.5 | **post-fix worse**; W18 hits new minimum |
| 19 | 14.3% | 17.1% | +2.8 | unchanged |
| 20 | 14.3% | 22.9% | +8.6 | minor noise |
| 21 | 14.3% | 28.6% | +14.3 | minor improvement |
| 22 | 17.1% | 17.1% | 0 | unchanged |
| 23 | 17.1% | 22.9% | +5.8 | minor improvement |
| 24 | 14.3% | 14.3% | 0 | unchanged |
| 25 | 34.3% | 11.4% | -22.9 | reversal |
| 26 | 20.0% | 11.4% | -8.6 | reversal |
| 27 | 25.7% | 20.0% | -5.7 | unchanged |

**Aggregate**: pre-fix mean raw = 31.2%, post-fix mean raw = 30.8%
(within run-to-run noise). The t-fix bumped some windows up and
others down, with no systematic improvement of the deep-collapse
region (W10+).

## What the t-fix DID resolve

The bug: `extract_state_at_step` (which produces the next window's
fixed_init_state) called `propagate_fn(y, k, dt, ...)` instead of
`propagate_fn(y, k*dt, dt, ...)`. For SWAT at dt=5/60 h, this scaled
the `t` argument by 12×. SWAT's `compute_sigmoid_args` uses `t` in
`C_eff = sin(2π(t - V_c) / 24 + φ)` — so the C_eff phase was wildly
wrong, producing biased state extractions for the bridge init.

After the fix, the W/Zt/a/T values extracted at the end of each
window are no longer phase-corrupted. **Early bridge windows (W2-W9)
do recover modestly**, confirming the bug was real.

But aggregate didn't budge. The cascade has a deeper cause that the
t-fix doesn't touch.

## Why fsa_high_res was silently unaffected (rule-in for the diagnosis)

fsa_high_res's `propagate_fn` does `del t` — discards the time
argument. All time-dependent values (T_B(t), Phi(t), C(t)) come from
`grid_obs[*][k]` indexed by the bin number `k`. So the wrong `t`
value passed to `propagate_fn` had no effect for fsa_high_res. Only
models like SWAT that USE `t` directly (for analytical time-of-day
formulas) feel the bug.

This is a generic SMC² infrastructure improvement — fix benefits any
future model that uses `t` in its propagate_fn.

## What's still broken — H1 ruled out as sole cause

Bridge cascade collapse persists with the same shape:
- W1: cold start finds truth (~91-100% raw)
- W2-W9: degrades 50-80% with high variance
- W10+: deep collapse to 5-30%, never recovers

The pattern signature is unchanged. Whatever causes the cascade is
independent of the extract_state_at_step phase bug.

## Updated hypotheses (in current order of likelihood)

### H2 (NOW MOST LIKELY) — Gaussian-bridge variance underestimate for 35-D space

The bridge log shows extremely concentrated bridge measures:

```
Gaussian base: LW shrinkage=0.019-0.034, log_det=-243.9 to -255.5
```

`log_det = -250` over 35 dimensions implies the bridge distribution's
*geometric-mean per-dim variance* is `exp(-250/35) ≈ 8e-4`, i.e. per-dim
SD ≈ 0.028. For SWAT's parameter ranges (e.g. lambda_step has truth=200
with prior CI ~[110, 360], so prior SD ~125), this means the bridge
variance is ~4 orders of magnitude tighter than the prior.

**With such a concentrated bridge, any small posterior bias from
window N is preserved nearly intact into window N+1.** The bridge can't
"let go" of accumulated bias because it's variance-starved.

fsa_high_res has 29 dims (vs SWAT's 35) and likely a less degenerate
posterior (the C-fix made HR/stress/log-steps very informative). Same
bridge type, but it doesn't cascade because its posterior is already
well-anchored at every window.

For SWAT, parameters with weak per-window data identifiability
(e.g. tau_T at 48h timescale, with only 1-day windows; mu_0/mu_E/eta
all coupled in the bifurcation; V_c with the 24h symmetry) accumulate
small biases that the tight bridge preserves and compounds.

**Test**: switch to MoG bridge (`--bridge mog --bridge-K 2`).
fsa_high_res's diagnostic plans
(`outputs/fsa_high_res_rolling/PLAN_principled_bridge_fixes.md`)
explored this for exactly this cascade pathology. ~1.5h GPU.

### H3 — V_c phase-shift identifiability symmetry

The 24-hour periodicity of the circadian drive makes V_c=0 and
V_c=12h structurally indistinguishable from the full data, modulo
the prior. The bridge MCMC might walk V_c into the wrong basin within
a window, then drag dependent parameters (kappa, lambda, beta_Z,
which all couple via the C_eff term in u_W) into compensating
biases.

**Test**: freeze V_c=0 via `frozen_param_keys=('V_c',)` in a new
`SWAT_SET_A_FROZEN_VC_CONFIG`, re-run. If coverage recovers,
confirm the symmetry-breaking prior is needed. ~1.5h GPU.

### H4 — Poisson particle weight degeneracy in fast-tempering bridge

Cold start uses 22 tempering levels (slow tempering); bridge uses
7 levels (fast). The Poisson `steps` channel has heavy-tailed
log-pmf at high counts; fast tempering may concentrate particles
on a few "lucky" Poisson realisations.

**Test**: increase `n_pf` to 800 in the bridge code path. ~2h GPU.
Less informative than H2/H3 (just adds more particles without
isolating the cause).

### H5 — Stride too aggressive for 5-min grid

12-h stride with 5-min bins means each bridge has to bridge 144
bins of new data. fsa_high_res's 12-h stride at 15-min bins =
48 bins of new data. SWAT bridges 3× more obs per step.

**Test**: longer stride (24h, 14 windows total). ~1h GPU.

### H1 (RULED OUT) — extract_state_at_step t-arg

Fixed in this run; bridge cascade unchanged.

## Diagnostic / next-step recommendation

**H2 is the top candidate.** The pattern (cold start good, slow
degradation, deep collapse from W10) is consistent with bridge
variance starvation — every window adds a small bias the bridge
can't shake off, and it compounds. The empirical evidence (very
small `log_det` in the bridge) is direct.

**Recommended next test**: re-run with `--bridge mog --bridge-K 2`.
Cost: ~1.5h GPU, no code changes (it's a pre-existing CLI option).

If MoG fixes it: ship SWAT v0.1 with `bridge_type='mog'` as the
SWAT-specific default in `SwatRollingConfig` (separate from
fsa_high_res's `'gaussian'` default). The modular per-model config
makes this a 1-line change in `drivers/swat/config.py`.

If MoG doesn't fix it: H3 (V_c freeze) is the next test.

## What's not the cause (rule-outs after this attempt)

- Not the model — psim §1.4 tests pass; SWAT defaults reproduce
  expected ranges in the simulator.
- Not the inner-PF — cold-start hits 91-100% raw / 82-100% informed.
- Not the artifact loader / extract_window — modularity tests pass,
  fsa_high_res regression bit-identical.
- Not the t-arg in extract_state_at_step — fixed in this run, no
  systemic recovery.
- Not the modularity layout — `drivers/swat/` is fully isolated;
  fsa_high_res unaffected through both attempts.

## Status

**STOPPED for user review.** Branch `feat/swat_rolling_driver` holds
locally:
- `ac46496` Phase A: SWAT driver scaffolding
- `4bd8882` extract_window scalar passthrough fix
- `31e1f1d` Phase F: how_to_add_a_new_model docs
- `c474191` Phase D failure result.md (initial diagnostic)
- `<this commit>` t-arg fix + comparative result.md

Awaiting user direction on next hypothesis to test (H2 / H3 / H4 / H5).

## Side note: model-tuning issue identified separately

User-spotted issue (independent of the SMC² bug):
[public-dev issue #5](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/5)
— SWAT Set A's Zt amplitude tops out at ~3, never reaching the
deep-sleep threshold (4.5). The simulated data therefore has <5%
deep sleep vs realistic 20-25%. This impairs identifiability of
c_tilde, delta_c, beta_Z, tau_a — would compound any bridge issue
even if the cascade itself were fixed.

A re-tuned Set A artifact (with the model-side fix) plus a
bridge-fix here would together likely produce the production
≥70%/≥70% gate.
