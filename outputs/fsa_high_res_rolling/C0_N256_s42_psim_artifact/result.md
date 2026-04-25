# psim-artifact reproduction: 96.7% / 27 of 27 PASS

**Date:** 2026-04-25
**Run:** `--scenario-artifact <psim-artifact> --seed 42`
**Output:** `outputs/fsa_high_res_rolling/C0_N256_s42_psim_artifact/`
**Driver commit:** `b798a1c` (bridge to Python-Model-Scenario-Simulation)
**Artifact:** `~/Repos/Python-Model-Scenario-Simulation/outputs/fsa_high_res/C0_recovery_14d/`
  produced by `examples/fsa_high_res/14d_C0_recovery.py` at psim commit `73b6d17`.

## Summary

| Metric | This run (psim artifact) | C-fix reference (inline) |
|--------|------:|------:|
| Windows | 27 | 27 |
| Mean coverage (raw, 29 params) | **96.7%** | 96.8% |
| Mean coverage (data-informed) | **92.5%** | 92.2% |
| PASS rate (≥70%) | **27 / 27** | 27 / 27 |
| Wall-clock | 1.21 h | 1.24 h |

**Reproduction succeeded.** Aggregate metrics match the C-fix reference
within stochastic noise (mean raw within 0.1 pp; PASS rate identical).
Per-window coverage differs slightly because identical-seed runs branch
differently after small floating-point divergences in the SDE substeps
(npy + jnp accumulation orders), but the aggregate behaviour is
statistically indistinguishable.

## What was reproduced

The acceptance test for the
[Python-Model-Scenario-Simulation](https://github.com/ajaytalati/Python-Model-Scenario-Simulation)
v0.1.0 bridge: the existing rolling-window SMC² driver, fed only a
packaged scenario artifact via the new `--scenario-artifact` flag,
reproduces the published `C_phase_fix_result.md` headline result.

This proves:

1. **Bundle parity.** The artifact's trajectory, observation channels
   (706 HR / 1344 sleep / 567 stress / 560 steps after dropout), and
   exogenous arrays (T_B, Phi, C on the global time grid) are
   sufficient input to drive SMC² to the published outcome.
2. **Loose coupling works.** No invasive change in `smc2bj/`. The bridge
   is the ~50-line `drivers/_artifact_loader.py` and a single `if`
   branch in `drivers/fsa_high_res_rolling.py`. The inline data-generation
   path remains unchanged for backward compatibility.
3. **The middle-repo workflow is sound.** Future model ports
   (SWAT, ...) follow the same pattern — develop & validate the
   scenario in `psim`, then port the SMC² driver as a thin adapter.

## Per-window detail

```
W 1:  93.1% / 100.0%   W10:  93.1% /  80.0%   W19:  96.6% /  85.7%
W 2:  96.6% / 100.0%   W11:  93.1% /  87.5%   W20: 100.0% / 100.0%
W 3:  96.6% /  80.0%   W12:  93.1% /  88.9%   W21: 100.0% / 100.0%
W 4: 100.0% / 100.0%   W13: 100.0% / 100.0%   W22: 100.0% / 100.0%
W 5:  93.1% /  66.7%   W14:  93.1% /  92.9%   W23: 100.0% / 100.0%
W 6: 100.0% / 100.0%   W15:  96.6% /  94.4%   W24:  96.6% /  87.5%
W 7: 100.0% / 100.0%   W16:  93.1% /  93.8%   W25:  96.6% /  87.5%
W 8: 100.0% / 100.0%   W17:  96.6% / 100.0%   W26:  96.6% /  90.0%
W 9:  93.1% /  88.9%   W18:  96.6% /  85.7%   W27:  96.6% /  88.9%
```

Min window: 93.1% (still 23 pp above the 70% PASS threshold).

## Conclusion

The bridge's acceptance test passes. **psim v0.1.0** is ready to tag.

---

## 2026-04-25 (later): namespace-package move regression

After moving the canonical `fsa_high_res` model out of this repo and
into the public dev repo
([Python-Model-Development-Simulation#1](https://github.com/ajaytalati/Python-Model-Development-Simulation/pull/1)),
the same artifact + driver was re-run via the new namespace-package
import path:

| Metric | Pre-move | Post-move (namespace package) |
|--------|------:|------:|
| Mean coverage (raw) | 96.7% | **96.7%** |
| Mean coverage (data-informed) | 92.5% | **92.5%** |
| PASS rate | 27 / 27 | **27 / 27** |
| Wall-clock | 1.21 h | 1.02 h |

**Per-window numbers are bit-identical** — confirming the model code
is now in only one place (the public dev repo) and the SMC² driver
imports it transparently. The 0.19h wall-clock difference is JAX
cache warming.

This finalises Phase E of the cosmic-giggling-wadler plan and
authorises the public-dev PR merge + psim v0.1.1 tag.
