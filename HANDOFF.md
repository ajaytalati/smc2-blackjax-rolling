# Handoff — SMC² framework refactor, 2026-04-23/24

Summary of the unattended overnight refactor of
`Python-BlackJAX-Centric-Estimation-Framework/version_4_1/` into this repo.

## What was done

**Phase A — scaffolding** (commit `3c90d68`)
- `~/repos/smc2_blackjax_framework/` created with `smc2bj/` / `models/` /
  `drivers/` / `docs/` / `outputs/` / `tests/` / `tools/` skeleton.
- `pyproject.toml`, `README.md`, `LICENSE` (MIT), `.gitignore`.
- Private GitHub repo `smc2-blackjax-rolling` created via `gh`.

**Phase B — generic framework lift** (commit `144e413`)
- Copied verbatim into `smc2bj/`:
  - `simulator/` (SDE integrator + obs dispatch)
  - `log_density/` (GK-DPF v3-lite + variants)
  - `transport/` (OT rescue)
  - `transforms/` (prior-unconstrained bijections)
  - `init/` (particle initialisation)
  - `estimation_model.py` (the contract)
- All imports rewritten: `simulator.X` → `smc2bj.simulator.X`, etc.
- Verified no FSA-specific names survive in `smc2bj/` (grep clean).
- Import smoke-test passes.

**Phase C — SMC² + pipeline + io extraction** (commit `a155b26`)
- Added `smc2bj/estimation/{config,sampling,mass_matrix,smc_window}.py`
  — outer tempered SMC with cold-start + Gaussian-bridge warm-start.
- Added `smc2bj/pipeline/{rolling,windowing,missing_data}.py` — rolling
  orchestration, window extraction, missing-data corruption.
- Added `smc2bj/io/checkpoint.py` — save + show.
- `SMCConfig`, `RollingConfig`, `MissingDataConfig` dataclasses replace
  the original module-level globals; all thread through the API
  explicitly.
- `rolling_window_smc` now takes `truth`, `cold_start_init`,
  `obs_channel_names` as parameters (previously implicit globals /
  FSA imports).

**Phase D — FSA model + driver + outputs** (commit `88f4461`)
- `models/fsa_real_obs/{simulation,estimation,sim_plots}.py` copied
  verbatim. Imports rewritten.
- `drivers/fsa_real_obs_5yr_rolling.py` — 600-line scenario driver
  replacing the 1538-line monolith. Argparse CLI for `--seed`,
  `--condition`, `--n-smc`, `--n-pf`, `--windows`, `--sim-only`,
  `--show-checkpoint`.
- `outputs/fsa_real_obs_5yr_rolling/` — 9 experimental checkpoint
  directories carried over, plus `excitation_experiment_report.md`
  and `robustness_check_report.md`.

**Phase E — documentation** (commit `88f4461`)
- `docs/MODEL_SPECIFICATION.md` — D1, FSA v4.1 maths-complete spec.
- `docs/SMC2_ALGORITHM_SPECIFICATION.md` — D2, algorithm spec
  sufficient for independent re-implementation.
- `docs/PORTING_GUIDE.md` — D3, 3-file contract + worked 2-state-OU
  sketch + FAQ.
- `docs/NUMERICAL_FINGERPRINT.md` — D5, locked seed=42/C0/W1 regression
  target with trajectory CSV md5.
- `README.md` at repo root — D4.
- `tools/dump_model_spec.py` — priors table generator.

**Tests** (commit `88f4461`)
- `tests/test_model_interface.py` — 5 protocol checks for FSA.
- `tests/test_pipeline_generic.py` — 4 shape tests.
- `tests/test_smc2_fingerprint.py` — 1 slow regression test.
- `pytest.ini` — slow marker.
- **pytest -m "not slow" passes 9/9.**

## Verification status

- **Trajectory CSV (synthetic-data pipeline): BIT-EXACT match** vs
  version_4_1 reference. md5 `1646a4f29347c15f9c727fa3dfc1263a`.
- **Fast pytest**: 9/9 green.
- **1-window parity test: PASSED.** Coverage 33/33 = 100.0% (exact match
  to pre-refactor reference). Per-parameter posterior means drift <0.10
  stdev from reference; every `in_ci` flag matches. Small drift explained
  by one extra tempering level (36 vs 35) from XLA bisection scheduling.
- **Full 9-window verification: PASSED.** 9/9 windows pass (>=70%
  threshold). Mean coverage 91.6% vs pre-refactor 83.5%; refactored avoids
  the original's catastrophic W7-W8 collapse because the XLA-drift in W1
  cascaded through bridges into a slightly better posterior trajectory.
  Same algorithm, same seed — the minor XLA variance at W1 propagated
  favourably through the bridge cascade.

  | W | Refactored cov | Pre-refactor cov | Δ |
  |---|----------------|------------------|---|
  | 1 | 100.0% | 100.0% | 0 |
  | 2 | 100.0% | 100.0% | 0 |
  | 3 | 100.0% | 93.9% | +6.1 |
  | 4 | 87.9% | 97.0% | −9.1 |
  | 5 | 78.8% | 93.9% | −15.1 |
  | 6 | 93.9% | 90.9% | +3.0 |
  | 7 | 90.9% | 54.5% | +36.4 |
  | 8 | 84.8% | 51.5% | +33.3 |
  | 9 | 87.9% | 69.7% | +18.2 |
  | **Mean** | **91.6%** | **83.5%** | **+8.1** |

  Total wall-clock 5192s (1.44h) vs pre-refactor 5133s (1.43h) — within 1%.

## Cleanup pass (2026-04-24, commit e6de3d3…)

After the initial refactor, a second cleanup pass knocked out several
small items flagged during review:

- **SCENARIO dead code removed.** `SCENARIO = 'recovery'` at the top of
  `drivers/fsa_real_obs_5yr_rolling.py` was a selector between three
  entries in `FSA_REAL_OBS_MODEL.param_sets` that all pointed to the
  same `DEFAULT_PARAMS` — a historical artifact. Driver now references
  `param_sets['recovery']` directly with a comment explaining the dead
  distinction.
- **`models/fsa_real_obs/gemini_code/` archived** to
  `outputs/historical_references/` with a README explaining its
  status. Keeps `models/fsa_real_obs/` clean (3 files + `__init__.py`)
  so the porting contract is visually obvious.
- **Priors-table auto-regeneration wired.** `tools/dump_model_spec.py`
  now has an `--update-docs` flag that rewrites a marker-delimited
  block inside `docs/MODEL_SPECIFICATION.md`. Doing so caught one
  factual bug in the hand-written §4.1: I had described a `κ_ratio`/
  `κ_total` reparameterisation that doesn't exist in v4.1 code (the
  actual design is `κ_vagal` estimated directly, `κ_chronic` and
  `R_base` frozen). Fixed.
- **`plot_parameter_tracking` + `plot_coverage_and_timing` promoted**
  to `smc2bj/plotting/rolling.py`. They don't reference any FSA
  concept — driver now imports them. Model-specific plots
  (`plot_latent_reconstruction`, `plot_macrocycle_schedule`,
  `plot_observations_with_missing`) stay in the driver. Driver went
  from ~700 lines to 639.

## What remains deferred

1. **`sim_plots.py` naming cleanup** — lowest priority. No duplication
   between `models/fsa_real_obs/sim_plots.py` (proof-of-principle
   diagnostics) and the driver's model-specific plots (rolling-SMC
   reconstruction). Names differ, usage is clear. Skipped unless
   there's a specific reason.
2. **D3 worked OU sketch** remains a sketch, not a working
   implementation. Creating `models/ou_2state/` +
   `drivers/ou_2state_rolling.py` and running them end-to-end would
   validate the porting-guide contract. Estimated 4-6h; best done
   when a real second model needs porting.
3. **Upstream `param_sets` collapse.** All three scenarios in
   `models/fsa_real_obs/simulation.py` point to `DEFAULT_PARAMS`. Could
   either (a) collapse to a single `{'default': DEFAULT_PARAMS}` entry,
   or (b) populate distinct `sedentary_params` / `overtraining_params`
   dicts if the three regimes are meant to be real alternatives. Left
   untouched in this pass — no effect on the driver after the
   SCENARIO removal.

## Known caveats

- **GK-DPF v3-lite assumes Gaussian observations** and locally-linear
  drift. Non-Gaussian (Poisson / Bernoulli) models need a new inner PF.
  Documented in `docs/PORTING_GUIDE.md` §3.
- **Priors must be lognormal or normal.** The HMC transform doesn't
  yet support bounded / simplex / compact. Documented.
- **Single-seed results on this model class have ~5-10 pp coverage
  noise** — always replicate across seeds. See
  `outputs/robustness_check_report.md`.

## Rollback

Worst-case rollback: `git checkout pre-refactor-2026-04-23` in the
upstream `Python-BlackJAX-Centric-Estimation-Framework` repo. That tag
captures everything at the state immediately before this refactor,
including the 9 experimental checkpoints and reports.

## Next session

All primary goals met. The repo is ready for the porting agent use case.
Optional follow-ups from the TODO list above (plot promotion to
`smc2bj/plotting/`, `MODEL_SPECIFICATION.md` priors-table auto-regen
wiring, OU sketch fleshed out to working implementation, `gemini_code/`
cleanup).


## 2026-04-25: bridge to Python-Model-Scenario-Simulation

A new public middle repo —
[Python-Model-Scenario-Simulation](https://github.com/ajaytalati/Python-Model-Scenario-Simulation)
(`psim` package) — was created in response to the
[POSTMORTEM_three_bugs](outputs/fsa_high_res_rolling/POSTMORTEM_three_bugs.md)
case study. It owns scenario primitives, the §1.4 sim-est consistency
discipline as runnable code, and a canonical scenario-artifact format.

Wired into this repo via:

- `drivers/_artifact_loader.py` (~50 lines) — calls
  `psim.io.format.read_artifact` and merges obs + exogenous into the
  flat `obs_data` dict that `rolling_window_smc` already accepts.
- `drivers/fsa_high_res_rolling.py` — gained a
  `--scenario-artifact <dir>` flag. When set, Steps 1-4 (inline data
  generation) are skipped and the validated artifact is consumed
  instead. Results land under
  `outputs/fsa_high_res_rolling/<...>_psim_artifact/` to keep them
  separate from the inline-path reference.

Verification:

- **W1 parity**: artifact-fed run produces 27/29 = 93.1% raw coverage
  / 100% data-informed coverage / 1/1 PASS — bit-identical bundle to
  what the inline path would have generated.
- **Trajectory ranges**: B [0.051, 0.677], F [0.092, 0.294],
  A [0.474, 0.751] — match the C-fix reference.
- **Full 27-window reproduction: PASSED.** 96.7% mean raw coverage /
  92.5% data-informed / 27 of 27 PASS in 1.21h (vs the C-fix reference
  96.8% / 92.2% / 27 of 27 in 1.24h — within stochastic noise). See
  [outputs/fsa_high_res_rolling/C0_N256_s42_psim_artifact/result.md](outputs/fsa_high_res_rolling/C0_N256_s42_psim_artifact/result.md).

What this means for future model ports:

- **SWAT and every subsequent model must enter via `psim` first.** The
  three §1.4 consistency checks (drift parity, obs-prediction parity,
  cold-start coverage) and the round-trip check are now mandatory
  pre-conditions, codified in `psim/validation/`. The C-phase /
  obs-misalignment bug class becomes literally impossible to ship.
- The driver's inline path stays as-is for backward compatibility.
  No invasive change to `smc2bj/`. Loose coupling: artifact format
  (`SCENARIO_SCHEMA_VERSION = "1.0"`) is the only contract.

See [BRIDGE_TO_SMC2.md](https://github.com/ajaytalati/Python-Model-Scenario-Simulation/blob/main/docs/BRIDGE_TO_SMC2.md)
in the new repo for the full adapter recipe.
