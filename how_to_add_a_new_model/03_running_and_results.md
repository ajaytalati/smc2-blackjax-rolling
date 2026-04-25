# 03 — Running, regression gates, and results

The protocol for running your driver and writing up the result. Three
gates: pre-run sanity, 1-window dry run, and the full rolling run.

## Pre-run gates

Before you launch any GPU job:

### Gate 1 — `pytest tests/`

```bash
PYTHONPATH=. python -m pytest tests/ -m "not slow" -v
```

All fast tests must pass. The `-m "not slow"` flag skips the
fingerprint regression test which spawns its own SMC² run; that gets
exercised by Gate 2.

### Gate 2 — Existing-model regression (NON-NEGOTIABLE)

Your additions must NOT break any production-currently-running model.
Run the existing model's 1-window cold-start against its existing
artifact and confirm bit-identical output.

For example, with fsa_high_res as the production reference:

```bash
PYTHONPATH=. python drivers/fsa_high_res_rolling.py \
    --scenario-artifact ~/Repos/Python-Model-Scenario-Simulation/outputs/fsa_high_res/C0_recovery_14d \
    --windows 1 --seed 42
```

Compare the summary lines (raw + informed coverage, PASS rate, final
state values, uninformative misses) to the previous run's recorded
values. They must be **identical**, not "close" — the inner-PF is
deterministic at fixed seed.

If the values differ, **STOP and investigate**. Common causes:
- A `smc2bj/` change you made unintentionally affected the inner-PF
  numerics. Revert and find a less-invasive fix.
- A namespace-package import order change is now resolving
  `models.fsa_high_res` differently. Check `sys.path` order.

The existing-model regression is your guarantee that production runs
keep reproducing. Never skip it.

## 1-window cold-start dry run

```bash
PYTHONPATH=. python -m drivers.<model>.rolling --windows 1 --seed 42
```

What to look for:

- **No NaN.** Particle weights and trajectories must stay finite.
- **Tempering converges.** The cold-start loop should reach `λ=1.0`
  in fewer than ~30 levels. fsa_high_res uses 15; SWAT uses 23.
  Many more than that suggests the prior is too wide for the data
  or the inner-PF is too noisy.
- **Inner-PF doesn't collapse.** The acceptance rates should stay
  > 0.5 by the time `λ=1.0` is reached. If acceptance drops below
  0.3 with many levels, particle degeneracy is biting.
- **W1 coverage > 50% raw** as a sanity floor. The actual production
  target is ≥ 70%, but cold-start with a tight prior on a fresh model
  often comes in at 50-90% on the first try.

### Decision tree

| W1 raw coverage | What it means | Action |
|---|---|---|
| ≥ 70% | Cold-start is healthy; bridge windows will be even better | Proceed to full rolling run |
| 50%–70% | Cold-start is marginal; full run might still hit ≥ 70% mean | Proceed to full rolling run with a checkpoint to re-evaluate |
| < 50% | Likely particle collapse or model misspecification | **STOP and ask.** Don't autonomously crank N_PF or cap log-weights — the next bug class is upstream (model code) |

The < 50% case is rare and usually means a sim/est consistency issue
that the psim §1.4 tests should have caught. If you're seeing it,
the workflow gate failed somewhere — go back and tighten the §1.4
tests.

## Full rolling run

```bash
PYTHONPATH=. nohup python -m drivers.<model>.rolling --seed 42 \
    > /tmp/<model>_full.log 2>&1 &
```

Estimated wall time:
- Cold-start window 1: ~10 min for fsa_high_res-class models
  (96 bins/window); ~10-15 min for SWAT-class (288 bins/window).
- Bridge windows 2-N: ~3-5 min each.
- Full 27-window run: 1.5-4 hours.

Launch in background so you can keep working. Monitor with:

```bash
tail -f /tmp/<model>_full.log
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```

### When the run finishes

The driver prints a summary block:

```
======================================================================
  SUMMARY
======================================================================
  Windows:              27
  Mean coverage (raw):  XX.X%
  Mean coverage (inf):  XX.X%
  PASS (>=70%):         XX/27
  Total SMC time:       X.XXh
  Output:               outputs/<model>_rolling/<scenario>_N256_s42/
======================================================================
```

**Decision gate**: mean coverage < 70% or PASS rate < 70% means
something is wrong; stop and investigate. Don't ship a "we'll improve
it later" result — by the time later arrives, no one remembers the
context.

## Writing up

The output directory should contain:

```
outputs/<model>_rolling/<scenario>_N256_s42/
├── driver_config.json          # the frozen dataclass + CLI args dump
├── rolling_checkpoint.json     # per-window posterior summaries
├── <model>_channels.png        # 4-panel input diagnostic
├── parameter_tracking.png      # generic per-param posterior tracking
├── coverage_and_timing.png     # generic coverage + wall-time per window
└── result.md                   # YOU WRITE THIS
```

### `result.md` template

```markdown
# <Model> rolling-window estimation: <X.X>% mean coverage / <X> of 27 PASS

**Date:** YYYY-MM-DD
**Run:** `--seed 42` (defaults from <Model>RollingConfig)
**Output:** `outputs/<model>_rolling/<scenario>_N256_s42/`
**Driver commit:** `<commit>` (Phase A scaffolding + ...)
**Artifact:** `~/Repos/Python-Model-Scenario-Simulation/outputs/<model>/<scenario>/`
  produced by psim commit `<psim_commit>`.

## Summary

| Metric | Value |
|--------|------:|
| Windows | N |
| Mean coverage (raw) | XX.X% |
| Mean coverage (data-informed) | XX.X% |
| PASS rate (≥70%) | X / N |
| Min window coverage | XX.X% (W?) |
| Max window coverage | 100% (? windows) |
| Wall-clock | X.XX h |

## What was reproduced

The first end-to-end SMC² rolling-window estimation for `<model>`,
exercising the full three-repo workflow: model defs (public dev) →
validated scenario (psim) → estimation (this repo).

## Per-window detail

(extracted from rolling_checkpoint.json or the run log)

## Notes / anomalies

(any windows that came in low; uninformative misses; tempering-level
distribution; runtime per window — anything a future reader would
want to know)
```

### Update `HANDOFF.md`

Add a dated section at the bottom describing what landed:

```markdown
## YYYY-MM-DD: <model> through SMC²

`<model>` is now wired into the SMC² rolling-window framework via
`drivers/<model>/`. First reproduction:

  - Mean coverage (raw / informed): XX.X% / XX.X%
  - PASS rate: X / N

Modularity discipline preserved:
  - `drivers/fsa_high_res_rolling.py` regression: bit-identical.
  - `drivers/<model>/` is a fully-isolated package.
  - `smc2bj/` changes (if any): <list> — defensive only, generic.

Next: ...
```

### Update psim's `docs/ARCHITECTURE.md` compatibility table

```
| psim version | scenario schema | tested public-dev commit | tested SMC² commit |
| 0.1.X        | 1.0             | <commit>                 | <SMC² commit>       |
```

This row records that this combination was tested end-to-end. Future
versions should update it as the test target changes.

## When to commit + tag

After the result is written and HANDOFF is updated:

```bash
git add drivers/<model>/ tests/test_<model>_rolling_imports.py \
        outputs/<model>_rolling/<scenario>_*/result.md \
        HANDOFF.md \
        smc2bj/<any>/<defensive_fixes>.py    # if any
git commit -m "..."
git tag -a <model>-v0.1 -m "First reproduction of <model> via SMC²"
git push origin main --tags
```

The tag is for traceability; the SMC² repo is private but the tags
make it easy to find a known-good commit.
