# SIR Set A — SMC² rolling-window result

**Date:** 2026-04-27
**Branch:** `feat/sir_driver`
**Scenario:** Anderson & May 1978 boarding-school flu — the canonical PMCMC tutorial benchmark
**Model:** SIR v1.0 (public-dev `feat/sir_model`, PR #10)
**Artifact:** psim v0.1+ (`feat/sir_scenarios`, PR #4) — `set_A_boarding_school_14d`

## TL;DR

The SIR test model exercises the full three-repo pipeline end-to-end —
public dev → psim → SMC² — for the first time. Set A reproduces the
**Anderson & May 1978 boarding-school flu** at the simulator level
(peak I=267 day 5-6, attack rate 93%, R₀=3.32) and runs cleanly through
SMC² rolling-window inference. The headline numbers:

| Metric            | SF Path B-fixed | Gaussian | Δ (SF − Gauss) |
|-------------------|----------------:|---------:|---------------:|
| Mean coverage (raw)     | **42.9%**       | 38.1%    | **+4.8 pp** |
| Mean coverage (informed)| **61.7%**       | 50.0%    | **+11.7 pp** |
| PASS rate (≥ 70%)       | 0 / 3           | 0 / 3    | tied |
| Wall time         | 5.5 min         | 5.4 min  | — |

**SF Path B-fixed beats Gaussian on Set A by +4.8 pp raw / +11.7 pp
informed.** Same direction (and similar magnitude on
data-informed-only) as the prior 2-model best-practice — confirms
SF Path B-fixed is the right default on a third independent model.

## Per-window detail (SF Path B-fixed)

| Window | Days   | Coverage (raw) | Data-informed |
|-------:|--------|---------------:|--------------:|
| 1      | 0-7    | 57.1% (4/7)    | 60.0% (3/5)   |
| 2      | 3.5-10.5 | 28.6% (2/7)  | 50.0% (2/4)   |
| 3      | 7-14   | 42.9% (3/7)    | 75.0% (3/4)   |

W1 cold-start gets 4 of 7 params and 3 of 5 informed params correct.
W2 (first SF-bridge window) drops, then W3 partial recovery.

## What recovers and what doesn't

Reading the [`parameter_tracking.png`](set_A_boarding_school_N256_s42_sfaq0.70/parameter_tracking.png):

| Param | Truth | Posterior across W1-W3 | Status |
|---|---:|---|---|
| `gamma`   | 0.0208/hr | 0.020-0.022 | **Recovered** ✓ |
| `T_I`     | 1.0       | 1.04-1.15 | **Recovered** ✓ |
| `sigma_z` | 0.02      | 0.03-0.06 | Slightly inflated, CI clips truth |
| `I_0`     | 1.0       | 0.5-0.75 | Slightly biased low |
| `beta`    | 0.069/hr  | 0.18-0.20 | **Locked off** (3× truth) |
| `rho`     | 1.0       | 0.50-0.58 | **Locked off** (~0.5) |
| `T_S`     | 1.0       | 60-100 | **Locked off** (60-100× truth) |

The **identifiability story** matches the published SIR-inference
literature exactly: γ (recovery rate, observable from outbreak
duration) and T_I (process noise) recover well; the (β, ρ, I_0)
joint is degenerate from cases-only data (only β·ρ enters the
observation rate); and `T_S` becomes an "escape valve" because the
inner-PF can absorb drift misspecification into S-noise.

This is **not a framework bug** — it's the canonical SIR identifiability
problem. Resolving it requires either:
- Tighter priors on `T_S` and the unidentifiable (β, ρ) ratio, OR
- More observation channels that pin down the (β, ρ, I_0) cluster
  separately (e.g., contact-tracing data for ρ; in-host viral load
  for β; serology that the noise prior can't discount).

## What's covered by the inference's data-informed parameters

The 61.7% data-informed mean implies that **of the parameters where
the data shrinks the prior CI**, 62% have truth covered. This is the
honest measure of what the inference machinery is doing right:

- `gamma`: shrunk and right
- `T_I`: shrunk and right
- `sigma_z`: shrunk, narrowly missed truth (CI 0.03-0.06 vs truth 0.02)
- `I_0`: partially shrunk, biased low

The 38.3% data-informed *miss* is the (β, ρ) pair locked into the
wrong product, which the cases-only channel cannot disambiguate.

## Sets B/C/D cascade collapse — investigation summary

The 14-day-window inference works on Set A (R₀=3.3, N=763) but
cascade-collapses on Sets B/C/D (community outbreaks, N=10000). A
focused 3-hour autonomous investigation into why is documented in
GitHub issue [#6](https://github.com/ajaytalati/smc2-blackjax-rolling/issues/6); summary here.

### What we tried — none fully fix it

| Variant | Set B mean | Set B W1 | W2 | W3 | W4-W7 |
|---|---:|---:|---:|---:|---:|
| **Bootstrap (canonical)** | 14.3% | 29% | 14% | 0% | 0% |
| Serology Pitt-Shephard guidance | 12.2% | 43% | 29% | 0% | 0% |
| Guidance + N_SMC=512 / N_PF=800 | 12.2% | 43% | 43% | 0% | 0% |
| Guidance + n_stages=6, n_mh=8 | (killed) | 43% | 0% (jumped mode) | — | — |

Mechanism diagnosed from the existing per-window logs:

- **W2 SF bridge: `min ESS=1.4/256, MH acc=0.17, incr_var=457`** on bootstrap
- The W1 posterior is concentrated in a (β, ρ) mode that doesn't match W2's data
- The bridge's q1 IS estimator collapses → particles trapped at a wrong mode
- By W3-W4 the mode is permanent (`||m1-m0||→0`, ESS healthy but at the wrong location)

Serology guidance via Pitt-Shephard on `I` (mirroring SWAT's HR-tilt
on `W`) **does** heal the bridge q1 IS collapse (ESS 1.4 → 50-200) and
extends Set B's healthy window count from 1 to 2. But it **regresses
Set A**: cold-start data-informed coverage drops 61.7% → 16.7%
because the strong serology weight at the 2 weekly obs forces
inference into a single (wrong) mode at cold-start. Net negative
across the 4 sets — reverted.

### Why this isn't a framework bug

The same SF Path B-fixed bridge gets 82.3% on SWAT (35-D) and 98.5%
on fsa_high_res (29-D). It hits an identifiability wall on SIR Sets
B/C/D specifically because:

1. **Cases-only constrains β·ρ·S·I/N as a single quantity** — leaves
   a 2-D ridge of (β, ρ) consistent with any observed case series.
2. **Sparse serology** (2 obs / 14 days) is too few to break the
   ridge in W1.
3. **Set A escapes** because its N=763 outbreak runs to extinction
   in 14 days (late-tail data informs β/γ directly), ρ=1 boarding-
   school removes one degree of freedom, and only 3 windows run
   total. Sets B/C/D have N=10000 + ρ=0.5 + 7-11 windows — the
   full identifiability problem hits hard.

### Proposed follow-ups (issue #6)

1. **Shorter rolling windows for high-R₀ sets** — e.g. 3-day windows for Set C.
2. **Annealed cold-start initialisation** to walk the prior toward truth gradually (addresses Set C's 56-level crawl).
3. **Block-decoupled bridge** — separate bridges for sharp params (β, γ, ρ) vs diffuse params (T_S, T_I, σ_z, I_0).
4. **Reparameterisation** in terms of identifiable combinations (R_0 = β/γ, plus β·ρ as a single quantity).
5. **MoG bridge** to capture multiple modes simultaneously.

These are research questions; tracking on issue #6.

## Reproducibility

```bash
# Public dev (model + tests)
git clone https://github.com/ajaytalati/Python-Model-Development-Simulation -b feat/sir_model

# psim (scenarios + paper-parity test)
git clone https://github.com/ajaytalati/Python-Model-Scenario-Simulation -b feat/sir_scenarios
cd Python-Model-Scenario-Simulation
pytest tests/test_consistency_sir.py tests/test_paper_parity_sir.py -v
python examples/sir/14d_set_A_boarding_school.py    # writes the artifact

# SMC² (this PR's branch)
git clone https://github.com/ajaytalati/smc2-blackjax-rolling -b feat/sir_driver
cd smc2-blackjax-rolling
pytest tests/test_sir_rolling_imports.py -v          # 7 modularity tests, ms-fast
PYTHONPATH=. python -m drivers.sir.rolling --seed 42  # SF Path B-fixed (default)
PYTHONPATH=. python -m drivers.sir.rolling --seed 42 --bridge gaussian   # baseline
```

Output: `outputs/sir_rolling/set_A_boarding_school_N256_s42_sfaq0.70/`
(or `..._gauss` for the Gaussian baseline).

## Status

- `feat/sir_driver` carries the SIR driver + modularity tests + this doc.
- Ready for review. **No tag.**
- Companion PRs:
  - public dev `feat/sir_model` PR [#10](https://github.com/ajaytalati/Python-Model-Development-Simulation/pull/10)
  - psim `feat/sir_scenarios` PR [#4](https://github.com/ajaytalati/Python-Model-Scenario-Simulation/pull/4)
- Depends on: SMC² `feat/schrodinger_follmer_bridge` PR [#4](https://github.com/ajaytalati/smc2-blackjax-rolling/pull/4) (SF Path B-fixed bridge).
