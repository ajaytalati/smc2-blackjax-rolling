# SWAT SMC² port — postmortem of bugs found during the first end-to-end run

**Date**: 2026-04-26
**Context**: First port of SWAT (7-state Sleep-Wake-Adenosine-Testosterone)
through the three-repo workflow (public dev → psim → SMC²) ending in
SMC² rolling-window estimation against the psim-validated Set A
healthy-baseline artifact.
**Outcome**: 3 real bugs found in 3 independent passes, plus 1
model-tuning concern raised with the upstream public dev repo. All
fixes contained to the SMC² repo; no model-side changes needed for
inference (model-tuning is a separate quality-of-result concern).

This postmortem mirrors the structural format of
[`outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md`](../fsa_high_res_rolling/POSTMORTEM_three_bugs.md)
which documented the analogous fsa_high_res bugs.

---

## TL;DR

| # | Bug | Impact | Fix commit | Caught by |
|---|---|---|---|---|
| **B1** | `extract_window` blindly applied boolean mask to every channel field, crashing on SWAT's scalar `bin_hours` metadata | Phase C (SWAT 1-window cold-start) crashed at first window-extract | `4bd8882` | First SWAT-via-SMC² launch |
| **B2** | `extract_state_at_step` passed `t = k` (step index) instead of `t = k * dt` (time) to `propagate_fn` | Bridge state extracted with wildly-wrong C_eff phase (12× scale at SWAT's dt=5/60h); fsa_high_res unaffected because its propagate_fn does `del t` | `1ce6cee` | Phase D (full 27-window) cascade collapse → 3-pass code audit pass 2 |
| **B3** | `make_gk_dpf_v3_lite_log_density` hard-coded `time_offset=0` to `shard_init_fn` and computed `t = k * dt` (within-window time, not global) — so for window N>0 with stride * N > 0 bins, the analytical C_eff(t) phase resets every window | **C-phase bug analog**: SWAT estimator's lmbda biased to ~13 (truth=32) because alternating windows had wrong-sign C_eff in u_W. fsa_high_res unaffected (same reason as B2). | `fa6e978` | Code audit pass 3 (rolling pipeline) after Phase D2 (post-B2) full run still cascaded |
| **M1** | Set A's truth params produce Zt amplitude max ≈ 4.3, never crossing c2 = 4.5 deep-sleep threshold; deep-sleep labels generated only via stochastic sigmoid noise floor | Model-design fragility; impairs identifiability of (c_tilde, delta_c, beta_Z, tau_a) | Tracked at [public-dev #5](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/5); not in this repo | User visual inspection of the artifact's swat_channels.png |

---

## Why the user reminded us about this story

After the first failed full-27-window run produced 31% mean coverage,
the immediate impulse was to attribute the cascade to "Gaussian bridge
variance underestimate for 35-D space" and try a Mixture-of-Gaussians
bridge. The user pushed back:

> Remember previously you claimed that the bridge was the cause of all
> the problem — when actually it was a simple bug in the code!!!!
> Attached are the latest parameters traces — again there is clearly
> bias NOT variance. Check the code again 3 times and learn from the
> parameter trace !!! Then you may try mixture of Gaussians.

That redirection — read the parameter trace bias signature, look for
code first — is exactly what enabled finding B3. The trace showed:
- `lmbda` locked at 13 (truth=32, bias DOWN)
- `kappa` locked at 9 (truth=6.67, bias UP)
- `Vn` locked at 1.5 (truth=0.3, bias UP)
- `(s_base, beta_s)` compensating: `s_base + beta_s · Vn` ≈ truth's
  s_base + beta_s · 0.3
- Narrow CIs, locked across windows — classic bias-not-variance
  signature

The lmbda-down + kappa-up + Vn-up pattern in u_W = -kappa·Zt + lmbda·C_eff + Vh + Vn − a + alpha_T·T
is exactly what one would predict if half the windows saw flipped-sign
C_eff: lmbda gets pulled toward zero (compromise across opposite signs),
the other u_W terms compensate.

This is the **second time the C-phase bug pattern has appeared in this
project**. fsa_high_res had it in its `gen_C_channel` (model-side); now
SWAT had it in `gk_dpf_v3_lite.py` (framework-side). Same root cause:
recomputing time-of-day quantities window-locally instead of using
global time.

---

## B1 — `extract_window` crashes on scalar metadata field

### Symptom

Phase C (the first attempt to run SWAT through SMC²) crashed
immediately:

```
File "smc2bj/pipeline/windowing.py", line 23, in extract_window
    new_ch[key] = ch_data[key][mask]
                  ~~~~~~~~~~~~^^^^^^
IndexError: too many indices for array: array is 0-dimensional
```

### Cause

`extract_window` walked every non-`t_idx` field of each channel and
applied the boolean mask:

```python
for key in ch_data:
    if key != 't_idx':
        new_ch[key] = ch_data[key][mask]
```

SWAT's `gen_steps` channel returns:

```python
{
    't_idx':      bin_t_idx,
    'steps':      k,                       # per-bin Poisson counts
    'bin_hours':  np.float32(bin_hours),   # SCALAR metadata
}
```

`bin_hours = 0.25` (the 15-min Poisson-bin width) is per-channel
metadata, not a per-step value. fsa_high_res had no scalar metadata
fields, so the bug had never surfaced.

### Fix

`smc2bj/pipeline/windowing.py:extract_window` now passes through any
field whose shape doesn't match `len(t_idx)`:

```python
arr = np.asarray(val)
if arr.ndim >= 1 and len(arr) == n_t:
    new_ch[key] = arr[mask]
else:
    # Scalar or non-per-step metadata — pass through.
    new_ch[key] = val
```

Defensive change benefiting any future model with scalar channel
metadata.

### Lessons

- The §1.4 sim-est consistency tests in psim DON'T exercise the SMC²
  rolling-window's `extract_window` — that's strictly framework-level.
  Worth considering whether psim should have a "round-trip via SMC²
  windowing" test as part of the workflow gate.

---

## B2 — `extract_state_at_step` passed step-index as time

### Symptom

Phase D (full 27-window) ran to completion but produced 31% mean
coverage with all 27 windows showing severe bridge cascade. Cold start
W1 was 100%; bridge windows degraded monotonically.

### Cause

`smc2bj/log_density/gk_dpf_v3_lite.py:329` (in the
`extract_state_at_step` JIT'd inner function):

```python
x_new, pred_lw = model.propagate_fn(
    y, k, dt, params, grid_obs, k, sigma_diag, xi, None)
   #  ^                        ^
   # position 2: should be t = k * dt, was k (step index)
   # position 6: k (correct, after the 2026-04-25 K→k fix)
```

Position 2 of `propagate_fn(y, t, dt, params, grid_obs, k, sigma, noise, rng)`
is the **time** argument. Passing `k` (step index) instead of `k * dt`
(time in the model's natural units) scales time by `1/dt`. For SWAT
at dt=5/60h, that's a 12× scaling of t. In SWAT's
`compute_sigmoid_args`:

```python
C_eff = jnp.sin(2.0 * jnp.pi * (t - V_c) / 24.0 + PHI_MORNING_TYPE)
```

with t scaled 12×, C_eff was at a wildly different phase (and sign)
than it should be. The state values extracted at end-of-window for
the next window's bridge init were therefore biased.

The MAIN log_density's `_core_step` was correct
(`t = jnp.asarray(k * dt, dtype=u.dtype)`); only the
`extract_state_at_step` helper had the bug. A previous fix on
2026-04-25 had corrected position 6 (was `K=400`, fixed to `k`); the
position-2 issue was overlooked at the time.

### Fix (`commit 1ce6cee`)

Compute time correctly:

```python
t_step = jnp.asarray(k * dt, dtype=u.dtype)
x_new, pred_lw = model.propagate_fn(
    y, t_step, dt, params, grid_obs, k, sigma_diag, xi, None)
```

### Effect

W2-W7 bridge windows recovered (W2: 54% → 80%, W6: 31% → 51%, W7:
49% → 63%). But aggregate mean coverage barely moved (31.2% → 30.8%);
deep cascade (W10+) unchanged. **B2 was real but not the root cause** —
ruled out as the sole cause of the cascade.

### Why fsa_high_res was silently unaffected

fsa_high_res's `propagate_fn` does `del t` (its
`models/fsa_high_res/estimation.py` line 134). It reads all
time-dependent values from `grid_obs[*][k]` (T_B, Phi, C precomputed
on the global grid by psim and sliced by extract_window). Passing
`t=k` vs `t=k*dt` had zero effect.

---

## B3 — Window-local time vs global time (C-phase bug analog)

### Symptom

After B2 fix, full 27-window run STILL produced 30.8% mean coverage
with the same bridge cascade pattern. Cold start at 91% (W1), bridge
windows still collapsed by W10.

User observed in the parameter traces: narrow CIs **locked off-truth**,
direction-consistent across all windows after W10. The bias signature:
- lmbda ↓ (truth=32, est=13)
- kappa ↑ (truth=6.67, est=9)
- Vn ↑ (truth=0.3, est=1.5)
- (s_base ↓, beta_s ↓) compensating

This is the bias-not-variance signature, exactly the
fsa_high_res C-phase bug pattern.

### Cause

`smc2bj/log_density/gk_dpf_v3_lite.py` (the production filter):

1. **`shard_init_fn` was called with `time_offset = jnp.int32(0)` for
   every window** (line 165, pre-fix). For window N with stride·N > 0
   bins of global offset, this told the model "you're starting at
   time 0" when actually the window starts at a non-zero global time.
   For SWAT, this meant the analytical C(0) initialization was always
   computed as if the window started at midnight.

2. **Within-window time `t = k * dt` was used in `_core_step`'s
   propagate_fn call** (line 186, pre-fix). For window N, k=0 at the
   FIRST bin of window N, so t=0. SWAT's `compute_sigmoid_args` then
   computed `C_eff = sin(2π·0/24 - V_c·... + φ) = sin(φ) = -0.866` at
   the start of EVERY window. But the global time at window N's first
   bin is `N · stride · dt`, not 0.

For SWAT with stride=144 bins (12h):

| Window N | Global start time | True C_eff at start | Est C_eff at start | OK? |
|---|---|---|---|---|
| 0 | t=0 | sin(φ) = -0.866 | sin(φ) = -0.866 | ✓ |
| 1 | t=12h | sin(π+φ) = +0.866 | sin(φ) = -0.866 | ✗ wrong sign |
| 2 | t=24h ≡ 0h | -0.866 | -0.866 | ✓ |
| 3 | t=36h ≡ 12h | +0.866 | -0.866 | ✗ wrong sign |
| ... alternating ... |

Half the windows had **wrong-sign C_eff** in `u_W = -kappa·Zt + lmbda·C_eff + Vh + Vn − a + alpha_T·T`.

When fitting `lmbda` against alternating-sign data, the MLE collapses
toward zero (compromise between +λ in good windows and -λ-equivalent
in bad windows). Empirically: lmbda ≈ 13, biased way below truth=32 —
exactly the predicted direction.

### Fix (`commit fa6e978`)

`make_gk_dpf_v3_lite_log_density` gained a `window_start_bin: int = 0`
kwarg. Used in three places:

1. `shard_init_fn` called with `time_offset=jnp.int32(window_start_bin)`
   (was 0). For SWAT this fixes C(0) initialization.

2. `_core_step` (main log_density): `t = jnp.asarray((_w_start + k) * dt, dtype)`
   (was `k * dt`). Time-of-day-dependent dynamics see GLOBAL time.

3. `extract_state_at_step._prop_one`: same fix as (2). Bridge state
   extracted with correct phase.

`smc2bj/pipeline/rolling.py` updated to pass `window_start_bin=int(start)`
(where `start = w * stride_days`) to the lik_factory call.

### Why fsa_high_res STILL unaffected

Same as B2: `del t` in propagate_fn, `del time_offset` in
shard_init_fn. fsa_high_res's regression test is bit-identical to
the v0.1.1 baseline (27/29 = 93.1% raw / 4/4 = 100% informed / 1/1
PASS in 419s) before AND after this fix.

### Verification (in flight at write-time)

Full SWAT 27-window run with both B2 and B3 fixes is launched. If
mean coverage recovers to ≥70% with PASS rate ≥70%, B3 was the root
cause and the SWAT port is shippable.

---

## M1 — SWAT Set A Zt amplitude doesn't reach deep-sleep threshold

### Symptom

User-spotted, looking at the SWAT input artifact's `swat_channels.png`:
- Top panel: latent Zt/6 trace peaks at ~0.5 (so Zt peaks at ~3),
  never reaching c2 = 4.5 deep-sleep threshold
- Sleep level panel: deep sleep (level 2) appears RARE relative to
  what a healthy adult should show

### Quantitative analysis

```
Total bins (4032 over 14 days at 5-min):
  level=0 (wake):      78.8%
  level=1 (light+REM): 14.0%
  level=2 (deep):       7.1%
  Total sleep:         21.2% (low — humans average 30-35%)

Of total sleep: deep = 33.7% (slightly above healthy 20-25%)

Zt range: [0.00, 4.30], mean=1.34
Zt > c2=4.5 (deep threshold):    0.0% of bins
Zt > c1=3.0 (sleep threshold):   5.7% of bins
```

The deep-sleep labels ARE generated (7% of bins / 33.7% of sleep),
but **purely via the stochastic sigmoid noise floor** — not via the
deterministic dynamics crossing c2. With Zt peaks at 4.3 and
c2=4.5, sigmoid(4.3-4.5) = sigmoid(-0.2) ≈ 0.45, so deep-sleep fires
~45% of the time at peak Zt purely from stochastic sampling.

### Why this matters for inference

The model parameters c_tilde, delta_c, beta_Z, tau_a are partially
unidentifiable from this data:
- The fraction of (level-0, level-1, level-2) is informative
- But the Zt-driven structure (clear above-threshold deep-sleep
  windows) doesn't manifest cleanly
- Multiple (delta_c, beta_Z, tau_a) triples produce equivalent
  observed sleep statistics

This contributes to (but is not the sole cause of) the parameter
identifiability issues observed in SWAT inference.

### Tracked at [public-dev #5](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/5)

Recommended re-tuning (model-side change):
- **Boost beta_Z from 2.5 → 4.0** (primary): puts u_Z at ~3.7
  overnight, sigmoid(3.7)·6 = 5.8 — clean deep-sleep dynamics
- (Alternatives: shorten tau_a, lower delta_c)

This is independent of the SMC² C-phase fixes and complementary —
both improvements together would tighten identifiability further.

---

## Lessons learned

### 1. Trust the parameter trace's bias signature

Narrow CIs locked off-truth across windows = code bug, not particle
degeneracy. The user's redirection was decisive: I had jumped to
"bridge variance" without first checking whether the trace pattern
was consistent with a code-bug source. The trace pattern (lmbda
collapsed toward zero, others compensating) was a CHARACTERISTIC
fingerprint of the C-phase bug class.

### 2. Within-window vs global time is a recurring class of bug

This is the SECOND time the same conceptual bug has bitten this
codebase:

- **fsa_high_res**: `gen_C_channel` originally computed C(t)
  window-locally; fixed by emitting C as a global-grid exogenous
  channel that align_obs_fn slices via extract_window
  ([POSTMORTEM_three_bugs.md](../fsa_high_res_rolling/POSTMORTEM_three_bugs.md))
- **SWAT (this port)**: SMC² inner-PF computed `t = k * dt` and
  passed `time_offset = 0` to shard_init_fn — both window-local

The systematic fix going forward: any analytical time-of-day
calculation in `propagate_fn` or `shard_init_fn` MUST receive the
window's GLOBAL offset.

### 3. The §1.4 consistency discipline doesn't catch framework bugs

psim's §1.4 tests verify sim/est consistency at the LIKELIHOOD level
(drift parity, obs-prediction parity, round-trip on a 1-day
scenario). They do NOT exercise the SMC² rolling-window's
extract_window or the per-window time-offset machinery. A SWAT-style
analytical-C-eff bug in the inner PF passes psim's tests cleanly.

**Suggested workflow gate addition**: psim's round-trip test should
exercise per-window slicing (currently only full-trial alignment is
tested). This would catch the SMC² C-phase analog at the workflow
gate, before any GPU goes into a full SMC² run.

### 4. fsa_high_res's `del t` + `del time_offset` was lucky

The fsa_high_res estimation.py discards both `t` and `time_offset`
arguments — all time-dependent values come from precomputed grid_obs
arrays. This made it COMPLETELY IMMUNE to both B2 and B3. If the
fsa_high_res estimation had used `t` (or even just used the value),
the C-phase bug would have surfaced there too — but might have been
attributed to model code rather than framework code, costing more
debugging cycles.

The right fix is at the FRAMEWORK level (this commit), benefiting
any future model that uses `t` analytically.

### 5. Per-model modularity worked

`drivers/swat/` is fully isolated. fsa_high_res driver untouched
through the entire port. The `smc2bj/` framework changes (windowing
+ inner-PF time fix) are GENERIC defensive improvements, not
SWAT-specific hacks. The architecture invariants from the plan held.

---

## Status

- **Branch**: `feat/swat_rolling_driver` (local only, not pushed)
- **Commits**:
  - `ac46496` Phase A: SWAT driver scaffolding
  - `4bd8882` B1 fix: extract_window scalar passthrough
  - `31e1f1d` Phase F: how_to_add_a_new_model docs
  - `c474191` Phase D failure: initial diagnostic
  - `1ce6cee` B2 fix: t-arg in extract_state_at_step
  - `fa6e978` B3 fix: window_start_bin threading (this postmortem will be added in next commit)
- **Pending**: SWAT full 27-window run with B3 fix (PID 1245820, ETA ~80 min)
- **Public-dev**: issues [#2](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/2), [#3](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/3) (closed by [PR #4](https://github.com/ajaytalati/Python-Model-Development-Simulation/pull/4)), [#5](https://github.com/ajaytalati/Python-Model-Development-Simulation/issues/5) (M1)

If the post-B3 run hits ≥70% mean coverage AND ≥70% PASS rate:
ship the SWAT port (push branch, merge, tag swat-v0.1).
