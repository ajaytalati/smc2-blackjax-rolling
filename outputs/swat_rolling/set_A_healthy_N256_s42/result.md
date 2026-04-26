# SWAT Set A rolling-window estimation: 3 fixes, dramatic recovery, gate not yet met

**Date:** 2026-04-26 (3rd attempt; updated after C-phase bug fix)
**Run:** `--seed 42` (defaults from SwatRollingConfig)
**Output:** `outputs/swat_rolling/set_A_healthy_N256_s42/`
**Driver / framework commits:**
  - `ac46496` Phase A driver scaffolding
  - `4bd8882` B1: extract_window scalar passthrough
  - `1ce6cee` B2: t-arg in extract_state_at_step
  - `fa6e978` **B3: window_start_bin threading (C-phase analog) ← THIS RUN**
  - `331b330` Postmortem doc
**Artifact:** psim v0.1.2 `set_A_healthy_14d/`.

## TL;DR

Three real SMC²-side bugs found and fixed (B1, B2, B3 — see
[POSTMORTEM_swat_port_bugs.md](../POSTMORTEM_swat_port_bugs.md)).
**The C-phase analog (B3) was the dominant cause of the cascade** —
fixing it raised mean coverage by **+17.6 pp (31% → 49%)** and
**eliminated the deep cascade collapse** (W10-W22 went from 14% → 47%
average).

But the gate isn't yet met (49% mean / 2-of-27 PASS, vs gate
≥70% mean / ≥19-of-27 PASS). **Most parameters are still
narrow-CI biased** — same conspiracy structure as before but
attenuated. **Hypothesised remaining causes:**

1. **Model-tuning issue M1** ([public-dev #5](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/5))
   — Set A's Zt amplitude doesn't reach c2=4.5; deep sleep happens
   only via the stochastic noise floor. This impairs identifiability
   of (c_tilde, delta_c, beta_Z, tau_a) from the sleep channel alone.
2. **Genuine identifiability limit on slow T dynamics** — tau_T=48h
   means each 1-day window sees ≤0.5 of one T cycle; the bridge can't
   reliably constrain (mu_0, mu_E, eta, tau_T, T_T, alpha_T).
3. **(V_h, V_n, s_base, beta_s) joint identifiability** — stress
   channel has the unidentified manifold s_base + beta_s·V_n = const;
   only (V_h - V_n) sum is identified by u_W.

NO push, no tag. Stopped for user review.

## Comparative summary (3 attempts)

| Metric | Attempt 1 (initial) | Attempt 2 (B2 fix) | Attempt 3 (B3 fix) | Gate |
|---|---:|---:|---:|---|
| Cold-start coverage W1 (raw) | 100.0% | 91.4% | 94.3% | ≥ 70% ✓ |
| Mean coverage (raw) | 31.2% | 30.8% | **48.4%** | ≥ 70% ❌ |
| Mean coverage (informed) | 31.3% | 30.0% | **46.7%** | ≥ 70% ❌ |
| PASS rate | 2/27 | 2/27 | **2/27** | ≥ 19/27 ❌ |
| Min window coverage | 5.7% (W13) | 2.9% (W18) | **20.0% (W27)** | — |
| Wall-clock | 1.34 h | 1.32 h | 1.31 h | — |

**Headline progression**:
- Attempt 1 → 2: B2 fix (extract_state_at_step t=k → t=k*dt) bumped
  early-bridge windows but left late cascade unchanged.
- Attempt 2 → 3: B3 fix (window_start_bin threaded through inner-PF
  for SWAT's analytical C_eff) **broke the cascade compounding** and
  bumped the deep-cascade region from ~15% to ~45%.
- Combined effect: gate not yet met, but **the failure mode has
  changed** — from monotone catastrophic cascade collapse to a
  stable-but-low equilibrium around 45-50%.

## Per-window comparison

| W | Initial | After B2 | After B3 (this run) | Δ from B2 |
|---|---:|---:|---:|---:|
| 1 | 100.0% | 91.4% | 94.3% | +2.9 |
| 2 | 54.3% | 80.0% | 65.7% | -14.3 |
| 3 | 71.4% | 60.0% | 48.6% | -11.4 |
| 4 | 60.0% | 42.9% | 65.7% | +22.8 |
| 5 | 68.6% | 51.4% | 71.4% | +20.0 |
| 6 | 31.4% | 51.4% | 34.3% | -17.1 |
| 7 | 48.6% | 62.9% | 37.1% | -25.8 |
| 8 | 40.0% | 45.7% | 51.4% | +5.7 |
| 9 | 42.9% | 48.6% | 51.4% | +2.8 |
| **10** | **17.1%** | **14.3%** | **57.1%** | **+42.8** |
| **11** | **14.3%** | **22.9%** | **37.1%** | **+14.2** |
| **12** | **8.6%** | **17.1%** | **31.4%** | **+14.3** |
| **13** | **5.7%** | **14.3%** | **34.3%** | **+20.0** |
| **14** | **17.1%** | **14.3%** | **45.7%** | **+31.4** |
| **15** | **25.7%** | **14.3%** | **51.4%** | **+37.1** |
| **16** | **25.7%** | **17.1%** | **51.4%** | **+34.3** |
| **17** | **28.6%** | **14.3%** | **42.9%** | **+28.6** |
| **18** | **11.4%** | **2.9%** | **48.6%** | **+45.7** |
| **19** | **14.3%** | **17.1%** | **48.6%** | **+31.5** |
| **20** | **14.3%** | **22.9%** | **51.4%** | **+28.5** |
| **21** | **14.3%** | **28.6%** | **51.4%** | **+22.8** |
| **22** | 17.1% | 17.1% | 42.9% | +25.8 |
| **23** | 17.1% | 22.9% | 45.7% | +22.8 |
| **24** | 14.3% | 14.3% | 40.0% | +25.7 |
| **25** | 34.3% | 11.4% | 42.9% | +31.5 |
| **26** | 20.0% | 11.4% | 42.9% | +31.5 |
| **27** | 25.7% | 20.0% | 20.0% | 0 |

The **deep-cascade region (W10+) went from 5-30% to 30-60%** — clear
evidence the C-phase fix addressed the dominant cause. But early
bridge windows (W2-W7) jitter slightly (some up, some down) suggesting
multiple weak attractors rather than a single strong one.

## TRUE FAILURES at W27 (post-Cfix)

```
['kappa', 'lmbda', 'tau_Z', 'V_c', 'HR_base', 'alpha_HR',
 'c_tilde', 'beta_Z', 'Vh', 'Vn', 'T_W', 'T_Z', 'T_a',
 'mu_0', 'eta', 'tau_T', 'T_T', 'delta_c', 'lambda_base',
 'lambda_step', 'W_thresh', 's_base', 'alpha_s', 'beta_s',
 'sigma_s', 'Zt_0', 'a_0', 'T_0']
```

**28 of 35 still missing.** The same "broad-based bias" pattern
persists, just at attenuated amplitude. The remaining bias is split
across:

- **u_W conspiracy**: kappa, lmbda, V_c, Vh, Vn (5 params)
- **Sleep dynamics**: c_tilde, delta_c, tau_Z, beta_Z (4 params; M1
  identifiability)
- **T (Stuart-Landau) bifurcation**: mu_0, eta, tau_T, T_T (4 params;
  slow timescale vs 14-day window)
- **Stress channel manifold**: s_base, alpha_s, beta_s, sigma_s (4 params;
  joint with Vn)
- **Steps Poisson**: lambda_base, lambda_step, W_thresh (3 params;
  noisier than expected post-fix?)
- **Init states**: Zt_0, a_0, T_0 (3 params)
- **Diffusion temperatures**: T_W, T_Z, T_a (3 params; only weakly
  data-informed)

That `lambda_step` and `HR_base` are now in TRUE FAILURES (they were
on-truth pre-fix) is curious — possibly the fix's W trajectory
re-anchoring shifts the W-likelihood-implied lambda_step/HR_base.

## Hypotheses for the remaining gap

### H6 (NEW, top candidate) — Model-tuning M1 limits identifiability

Public-dev [issue #5](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/5)
documents that SWAT Set A's Zt only reaches ~4.3 (never crossing
c2=4.5), so deep-sleep events are stochastic-noise-floor only.
Without clean Zt-driven deep-sleep dynamics, the joint
(c_tilde, delta_c, beta_Z, tau_a) is partially unidentified from
sleep alone.

**Testable**: re-tune Set A's truth params per [#5](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/5)
suggestion (`beta_Z: 2.5 → 4.0`), regenerate the psim artifact,
re-run SMC². If coverage hits ≥70%, M1 is the remaining cause.
This is a model-side change in public dev, not SMC².

### H7 — Slow T dynamics on 14-day window

`tau_T = 48h` means T evolves with a 2-day timescale. With 1-day
windows, each window sees ≤0.5 of one T cycle. The Stuart-Landau
bifurcation parameters (mu_0, mu_E, eta) are jointly determined by
T(t) trajectory; identifying them needs multiple T cycles.

**Testable**: longer windows (3 days) — would see ~1.5 T cycles per
window. Cost: ~2.5h GPU. Trade-off: more bridges.

### H8 — Vn / s_base manifold genuinely unidentified

Stress channel: `mean = s_base + alpha_s·W + beta_s·Vn`. The
direction (Δs_base, Δbeta_s·Vn) = (-Δ, +Δ/Vn) leaves stress
unchanged. Only u_W's `+ Vn` term breaks this manifold (Vn enters W
dynamics independently of s_base, beta_s). With B3 fix, u_W is now
correctly computed, so this should be identifiable in principle —
but the trace shows Vn locked at ~0.6-1.5 (truth=0.3) which means
the u_W constraint isn't strong enough to break the stress manifold
on the 1-day window.

**Mitigations**: tighter prior on Vn (currently centered at 1.0,
truth at 0.3 = ~1 SD off); or longer windows.

### H9 (NOW LESS LIKELY) — Bridge variance underestimate

The original H2 (Gaussian bridge variance underestimate). With B3
fix the bridge log_det values dropped from -250 to ~-180 (still
small but less concentrated). The improvement from B3 alone was
+17pp; if H9 were the dominant remaining cause, MoG bridge would
add another large jump.

**Testable**: re-run with `--bridge mog --bridge-K 2`. ~1.5h GPU.
Cheap to do but no longer the lead candidate after B3.

## Recommended next steps (in priority order)

1. **Re-run with M1 fix** (model-side, public dev): bump
   `beta_Z: 2.5 → 4.0` in SWAT Set A. Regenerate psim artifact.
   Re-run SMC². If coverage hits ≥70%, M1 was the remaining cause
   and the SWAT port is shippable.

2. **Read parameter_tracking.png for the post-Cfix run carefully** and
   confirm:
   - Vh trace: should be at ~1.0 (truth) with narrower CIs than
     pre-fix (was at ~1.5)
   - lmbda trace: should be near 32 (truth) instead of locked at 13
   - Vn trace: should be closer to 0.3 (truth) instead of locked at
     1.5
   - If these show clear improvement: B3 was the right fix; remaining
     bias is identifiability-limited

3. **MoG bridge test**: re-run with `--bridge mog --bridge-K 2`. Cheap
   to do; if it adds another +10 pp, bridge variance is a secondary
   contributor.

4. **Frozen-V_c test**: V_c=0 (truth) is in PARAM_PRIOR_CONFIG with
   N(0, 3) prior; freezing it removes one degree of conspiracy. Test
   would isolate whether V_c contributes.

5. **3-day windows**: address H7. Most expensive test (~3h GPU per
   run), least confident hypothesis.

## What's not the cause (rule-outs after this run)

- **Not the C-phase analog (B3)** — fixed; recovered +17 pp.
  Cascade no longer compounds.
- **Not extract_state_at_step (B2)** — fixed; partial improvement.
- **Not extract_window scalar handling (B1)** — fixed; required for
  SWAT to run at all.
- **Not the modular SMC²-side architecture** — fsa_high_res
  bit-identical at every step; modularity tests pass.
- **Not the §1.4 sim/est consistency** — psim's tests pass on the
  static parameter set.

## Status

**STOPPED for user review.** Branch `feat/swat_rolling_driver` holds
locally:
- `ac46496` Phase A: SWAT driver scaffolding
- `4bd8882` B1 fix: extract_window scalar passthrough
- `31e1f1d` Phase F: how_to_add_a_new_model docs
- `c474191` Phase D failure: initial diagnostic
- `1ce6cee` B2 fix: t-arg in extract_state_at_step
- `fa6e978` B3 fix: window_start_bin threading (C-phase analog)
- `331b330` Postmortem: SWAT port bugs
- `<this commit>` 3-way comparative result.md

Awaiting user direction on next investigation: Recommended first is
to retune Set A per public-dev #5 and re-run; that's a model-side
change and tests the most likely remaining cause.
