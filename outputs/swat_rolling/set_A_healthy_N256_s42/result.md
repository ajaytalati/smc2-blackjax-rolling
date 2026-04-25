# SWAT Set A rolling-window estimation: BRIDGE CASCADE FAILURE

**Date:** 2026-04-26
**Run:** `--seed 42` (defaults from SwatRollingConfig)
**Output:** `outputs/swat_rolling/set_A_healthy_N256_s42/`
**Driver commit:** `31e1f1d` (Phase F, post-extract_window fix)
**Artifact:** `~/Repos/Python-Model-Scenario-Simulation/outputs/swat/set_A_healthy_14d/`
  produced by psim v0.1.2 (commit `1327885`).

## TL;DR

**The run completed but failed the production gate.** The driver,
inner-PF, mixed-likelihood obs handling, modularity invariants, and
artifact loading all worked. The failure is in the SMC² **bridge
cascade**: the cold-start window finds the truth perfectly, but the
warm-start bridge to subsequent windows loses it and the loss
compounds. By W6 coverage drops to 31%; by W13 to 5.7%.

Pattern is **strongly reminiscent of fsa_high_res's pre-fix C-phase
bug** documented in `outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md`
— same shape (cold-start clean, bridge degrades monotonically with
narrow CIs locked off-truth).

**No push, no tag.** The result is recorded for diagnostic
investigation. See "Hypotheses" and "Recommended next steps" below.

## Summary

| Metric | Value | Gate |
|--------|------:|------|
| Windows | 27 | — |
| Mean coverage (raw, 35 params) | **31.2%** | ≥ 70% ❌ |
| Mean coverage (data-informed) | **31.3%** | ≥ 70% ❌ |
| PASS rate (≥70%) | **2 / 27** | ≥ 70% (≈19/27) ❌ |
| Min window coverage | 5.7% (W13) | — |
| Max window coverage | 100% (W1 only) | — |
| Wall-clock | 1.34 h | — |

## What worked (pre-bridge stages)

The entire upstream pipeline behaved correctly:

- **Phase B (fsa_high_res regression): bit-identical** to v0.1.1's
  reference (27/29 = 93.1% raw / 100% informed / 1/1 PASS in 419s).
  The SWAT addition didn't perturb fsa_high_res.
- **Phase C (SWAT 1-window cold-start): 33/35 = 94.3% raw / 84.6%
  informed / 1/1 PASS in 619s.** Mixed-likelihood SMC² works on the
  Poisson `steps` and 3-level ordinal `sleep` channels — the inner
  GK-DPF didn't degenerate on non-Gaussian likelihoods.
- **Phase A modularity tests**: 4/4 pass, fsa_high_res 9/9 pass,
  generic suite 22/22 pass. Architecture is clean.
- **Phase D Window 1 (cold-start within full run):** 35/35 = 100% raw
  / 12/12 = 100% informed. The truth is unambiguously found at the
  cold start.

## What broke (bridge windows)

Bridge cascade — coverage degrades window-by-window as the warm-start
mechanism amplifies any mismatch:

```
W 1: 100.0% / 100.0%   (cold start — perfect)
W 2:  54.3% /  55.6%   ← bridge initialises, immediate ~46pp loss
W 3:  71.4% /  80.0%   (transient recovery)
W 4:  60.0% /  57.7%
W 5:  68.6% /  70.0%
W 6:  31.4% /  31.4%   ← phase change; informed coverage now equals raw
W 7:  48.6% /  47.1%       (every miss is data-informed)
W 8:  40.0% /  41.2%
W 9:  42.9% /  39.4%
W10:  17.1% /  17.1%   ← deep collapse begins
W11:  14.3% /  14.3%
W12:   8.6% /   8.6%
W13:   5.7% /   5.7%   ← min: 2/35 params cover truth
W14:  17.1% /  17.1%
W15:  25.7% /  23.5%
W16:  25.7% /  25.7%
W17:  28.6% /  25.0%
W18:  11.4% /  11.4%
W19:  14.3% /  14.3%
W20:  14.3% /  14.7%
W21:  14.3% /  14.7%
W22:  17.1% /  17.6%
W23:  17.1% /  17.6%
W24:  14.3% /  14.7%
W25:  34.3% /  35.3%
W26:  20.0% /  20.6%
W27:  25.7% /  25.7%
```

The "informed coverage = raw coverage" pattern from W6 onward is the
diagnostic signature: every parameter that the bridge has narrowed
(reducing the CI) is missing the truth. The bridge is concentrating
probability mass in the wrong region.

## TRUE FAILURES (representative)

The full TRUE FAILURES list at W22 (28 of 35 params miss):

```
['kappa', 'lmbda', 'V_c', 'HR_base', 'alpha_HR', 'sigma_HR',
 'c_tilde', 'tau_a', 'beta_Z', 'Vh', 'Vn', 'T_W', 'T_Z', 'T_a',
 'mu_0', 'mu_E', 'eta', 'tau_T', 'alpha_T', 'delta_c', 'lambda_base',
 's_base', 'alpha_s', 'sigma_s', 'W_0', 'Zt_0', 'a_0', 'T_0']
```

Almost everything except `gamma_3`, `tau_W`, `tau_Z`, `lambda_step`,
`W_thresh`, `mu_E`, `T_T`, `beta_s` is missing. The miss is broad-based,
not concentrated on one parameter family — consistent with a **systemic
posterior mis-anchoring** rather than a per-parameter identifiability
issue.

## Comparison to fsa_high_res's C-phase bug pattern

This is structurally identical to fsa_high_res's pre-fix behaviour
([POSTMORTEM_three_bugs.md](../fsa_high_res_rolling/POSTMORTEM_three_bugs.md)):

| | fsa_high_res pre-C-fix | SWAT v0.1 (this run) |
|---|---|---|
| Cold-start coverage | high (>90%) | 100% |
| W2 onwards | rapid collapse | rapid collapse |
| Mean coverage | 37.5% | 31.2% |
| PASS rate | 1/27 | 2/27 |
| Symptom | narrow CIs locked off-truth on β_C_* coefficients | narrow CIs locked off-truth on majority of params |
| Fix that worked | gen_C_channel emits global C(t); align_obs_fn reads it (no per-window recomputation) | TBD — diagnostic investigation needed |

The fsa_high_res fix took mean coverage from 37.5% → 96.8%. The cause
was a single sim/est inconsistency in C(t) phase across windows that
the §1.4 obs-prediction parity test would have caught — but only if
extended to also exercise per-window slicing rather than just
single-state predictions.

## Hypotheses (in rough order of likelihood)

### H1 — Per-window obs alignment for SWAT's bare-key channels (highest priority)

SWAT's channels are named `'hr'`, `'sleep'`, `'steps'`, `'stress'`
(no `'obs_'` prefix). The new `obs_channel_names` parameter to
`synthesise_scenario` correctly classifies them as obs vs exogenous,
but the SMC²-side `extract_window` walks **all** of `obs_data` —
which includes the channels merged in via `_artifact_loader.load_scenario`.
Worth a careful look at:

- Does `extract_window` handle the SWAT 4-channel structure
  identically to fsa_high_res's `obs_HR/obs_sleep/...` structure?
- The Poisson `steps` channel is sparse on 15-min bins; does
  `extract_window` correctly re-index its `t_idx` after the windowing
  shift?
- Does the new `bin_hours` scalar pass-through (the
  windowing.py defensive fix in commit `4bd8882`) interact correctly
  with `align_obs_fn`'s consumption of `bin_hours`?

The §1.4 round-trip test in psim passes a 1-day scenario through
zero-noise propagation and confirms recovery within tolerance — but
that test does NOT exercise the SMC² rolling-window's per-window
extract+align pipeline. It just checks `align_obs_fn(obs_data, T, dt)`
on the full-trial obs_data, not on per-window slices.

### H2 — Bridge variance underestimate for SWAT's 35-D parameter space

The Gaussian bridge with Liu-West shrinkage may be too tight for
SWAT's 35-dim parameter space (vs fsa_high_res's 29-dim). The
bridge's `log_det` values (visible in the log) are `-243.9` to
`-250.6` — extremely small bridge variance. This concentrates
particles too tightly around the cold-start posterior mean, and any
small mis-anchoring at W2 cascades.

Test: re-run with `--bridge mog --bridge-K 2` (mixture-of-Gaussians
bridge) which fsa_high_res's diagnostic suite shipped for exactly
this pathology.

### H3 — V_c identifiability + ambiguity from sleep + steps gating

V_c (phase shift) enters drift only via `C_eff = sin(2π(t-V_c)/24 + φ)`.
At V_c=0 (Set A truth), the partial derivative of the likelihood w.r.t.
V_c at small V_c shifts is small. The cold start finds V_c=0 because
the prior is centred there, but the bridge's first MCMC moves can
walk V_c into the symmetric V_c=12h region (24-h period symmetry not
broken by the prior). Once V_c shifts, every other phase-coupled
parameter (kappa, lambda, beta_Z) re-fits incorrectly to compensate.

Test: re-run with `frozen_param_keys=('V_c',)` in the dataclass to
remove V_c from estimation. If coverage recovers, V_c is the cascade
seed.

### H4 — Poisson particle weight degeneracy in bridge windows only

Cold start uses gradual tempering (23 levels at λ=1.0) which gives
the Poisson `steps` channel time to inform W via `obs_log_weight_fn`
without crashing the ESS. The bridge uses fast tempering (7 levels
at λ=1.0). The Poisson likelihood may be heavy-tailed enough that
the bridge's larger λ-jumps cause local particle collapse, even
though the cold start was clean.

Test: increase `n_pf` from 400 → 800 in the bridge code path only
(or globally), and/or increase `target_ess_frac` from 0.3 → 0.5 in
the bridge code path.

### H5 — Stride too aggressive for SWAT's 5-min grid

Stride is 144 bins = 12 hours. With 288 bins/window and 12-hour
stride, consecutive windows share 50% of their data. fsa_high_res's
1-day window with 12-hour stride at 15-min bins shares the same 50%.
But the per-bin obs density is 3× higher for SWAT (5-min vs 15-min);
the bridge has to span more obs from window-to-window, potentially
overwhelming its variance budget.

Test: longer stride (e.g. `stride_bins=288` = 24h, 14 windows total
with no overlap).

## Recommended next steps (diagnostic investigation plan)

In order, smallest-effort first:

### Step 1 — Per-parameter posterior diagnostic (~30 min, no GPU)

Look at `parameter_tracking.png` (already generated by this run) for
the 35 parameters individually. Identify:
- Which parameters are missing the truth from W2 onwards (broad?
  concentrated in one block?)
- Whether the posterior mean DRIFTS systematically with windows
  (cascade) or jumps once and stays (single-step error)
- Whether posterior CI WIDTH collapses faster than the mean DRIFTS
  (narrow CIs locked off-truth = the C-phase signature)

The plot is at:
`outputs/swat_rolling/set_A_healthy_N256_s42/parameter_tracking.png`

### Step 2 — V_c freeze test (~1.5h GPU)

Add `frozen_param_keys=('V_c',)` to a new `SWAT_SET_A_FROZEN_VC_CONFIG`
instance, re-run. If mean coverage > 70%, V_c is the cascade seed
and we have an identifiability finding (and a path forward: add a
prior that breaks the V_c=12h symmetry).

### Step 3 — MoG bridge test (~1.5h GPU)

Re-run with `--bridge mog --bridge-K 2`. fsa_high_res's diagnostic
plans (`outputs/fsa_high_res_rolling/PLAN_principled_bridge_fixes.md`,
`PLAN_ot_regularized_flow_bridge.md`) explored this for exactly the
cascade pathology. If MoG recovers coverage, the issue is
Gaussian-bridge variance underestimate and we have a tested fix.

### Step 4 — psim §1.4 extension (no GPU)

Extend the psim round-trip test to exercise per-window
extract+align — currently only the full-trial alignment is tested.
This would catch H1-class bugs at the workflow gate, before the
SMC² port. (Also a candidate fix to add to psim's
ADDING_A_MODEL.md.)

### Step 5 — Larger N_PF (~2h GPU, last resort)

Re-run with `--n-pf 800`. Cheap to do but the answer it produces
("more particles helped") doesn't isolate the cause; only useful
to triangulate.

## What's not the cause (rule-outs)

- **Not `extract_window`** — the defensive scalar passthrough fix in
  commit `4bd8882` is correct (verified by SWAT 1-window cold start
  reaching 94.3% on the same artifact).
- **Not `_artifact_loader`** — same artifact, same loader produced
  the cold-start posterior that hit 100% in the full run's W1.
- **Not the modularity** — fsa_high_res 1-window regression is
  bit-identical, modularity tests pass, no `smc2bj/` change touched
  the inner-PF.
- **Not the cold start init** — W1 reached 100% raw / 100% informed.
- **Not the obs-prediction parity per channel** — psim's §1.4 tests
  pass for HR (Gauss), sleep (3-level ordinal CDFs), steps (Poisson
  rate), and stress (Gauss).

The cause is **specifically in the bridge mechanism** (warm-start
from W_n's posterior to W_{n+1}'s prior), interacting with one of
the four hypotheses above.

## Files inspected during diagnosis

- `/tmp/swat_full.log` — full run log (raw)
- `outputs/swat_rolling/set_A_healthy_N256_s42/parameter_tracking.png` — generated, not yet inspected
- `outputs/swat_rolling/set_A_healthy_N256_s42/coverage_and_timing.png` — generated, not yet inspected
- `outputs/swat_rolling/set_A_healthy_N256_s42/rolling_checkpoint.json` — full per-window posterior summaries
- `outputs/swat_rolling/set_A_healthy_N256_s42/swat_channels.png` — input artifact diagnostic (unchanged from psim)

## Status

**STOPPED for user review.** No git push, no tag. The
`feat/swat_rolling_driver` branch holds the local commits:
- `ac46496` Phase A: SWAT driver scaffolding
- `4bd8882` extract_window scalar passthrough fix
- `31e1f1d` Phase F: how_to_add_a_new_model docs

This `result.md` is to be added in a 4th commit on the same branch
documenting the failed first attempt.
