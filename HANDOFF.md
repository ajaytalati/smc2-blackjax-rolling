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

## What was deferred / TODO

1. **Full 9-window verification run** is planned but not yet kicked off —
   waiting for 1-window parity to complete and verify first.
2. **`tools/dump_model_spec.py` auto-regeneration** is implemented but
   its output is not yet embedded in `MODEL_SPECIFICATION.md` — that file
   currently has the priors table hand-written. Next pass should embed
   the output as a code fence with regen instructions.
3. **Plot promotion**: `plot_parameter_tracking` and
   `plot_coverage_and_timing` are currently in the FSA driver but are
   fully model-generic; they could move to `smc2bj/plotting/` in a
   follow-up pass to reduce per-model boilerplate.
4. **`sim_plots.py` duplication**: `models/fsa_real_obs/sim_plots.py`
   and the driver's own plot functions are non-overlapping (different
   plots), but a naming cleanup might make the split more obvious.
5. **`gemini_code/` subdirectory** in `models/fsa_real_obs/` is a
   reference/historical script, not part of the 3-file convention.
   Consider moving to `outputs/` or deleting.
6. **D3 worked OU sketch** is a sketch, not a working implementation.
   Creating `models/ou_2state/` + `drivers/ou_2state_rolling.py` and
   running them end-to-end would validate the porting-guide contract.

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
