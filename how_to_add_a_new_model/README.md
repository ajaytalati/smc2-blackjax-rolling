# How to add a new model to the SMC² repo

This guide is the **first and definitive resource** for adding a new
model to the SMC² rolling-window estimation framework. Follow it
end-to-end and you'll produce: a per-model driver package, the
modularity-protecting tests, and a reproduced rolling-window result
md — all matching the conventions of the existing models.

## Audience

A **human engineer** or **coding agent** porting a new model to SMC²
estimation, who has *already* completed the prior two stages of the
three-repo workflow:

1. **Public dev repo** ([Python-Model-Development-Simulation](https://github.com/ajaytalati/Python-Model-Development-Simulation))
   has the model under `version_1/models/<model>/` with the 3-file
   convention (`simulation.py`, `estimation.py`, `sim_plots.py`),
   exporting `<MODEL>_MODEL` and `<MODEL>_ESTIMATION`.

2. **psim** ([Python-Model-Scenario-Simulation](https://github.com/ajaytalati/Python-Model-Scenario-Simulation))
   has the model's scenarios under `psim/scenarios/presets/` plus the
   3 §1.4 consistency tests (`test_consistency_<model>.py`,
   `test_scenario_<model>.py`, `test_round_trip_<model>.py`). At
   least one scenario has been packaged into an artifact under
   `outputs/<model>/<scenario>/` with `validation/report.json` showing
   `all_passed: true`.

If either of these isn't done, **stop and finish them first**. The
SMC² port is the third leg of a tripod; the first two legs need to
be solid before the third can carry weight.

## The three-repo workflow

```
Python-Model-Development-Simulation       Python-Model-Scenario-Simulation
(public — defines models)                 (public — validates scenarios)
              │                                          │
              │  3-file model + TESTING.md               │  4 §1.4 checks
              │  + user doc                              │  + packaged artifact
              ▼                                          ▼
                                  ┌──────────────────────────┐
                                  │  smc2-blackjax-rolling   │  ◀── YOU ARE HERE
                                  │  (private — SMC² estim.) │
                                  │                          │
                                  │  per-model driver pkg    │
                                  │  + rolling result md     │
                                  └──────────────────────────┘
```

Each repo's discipline carries forward into the next. The SMC² port
is the easiest of the three because the hard work (model definition +
validation) was done upstream — but the modularity invariants are
strictest here, because this is where models cross-pollute the most
easily (one big GPU, multiple drivers sharing infrastructure).

## Prerequisites checklist

Before starting, confirm:

- [ ] Public dev repo cloned at `~/Repos/Python-Model-Development-Simulation`
      (or set the path explicitly in your driver).
- [ ] psim cloned at `~/Repos/Python-Model-Scenario-Simulation`.
- [ ] At least one packaged artifact at
      `~/Repos/Python-Model-Scenario-Simulation/outputs/<model>/<scenario>/`
      with `validation/report.json:all_passed=true`.
- [ ] All 3 psim consistency tests for your model pass:
      `pytest -k <model> tests/` from the psim root.
- [ ] Your model's `EstimationModel` exposes the SMC²-required callbacks:
      `propagate_fn`, `diffusion_fn`, `obs_log_weight_fn`, `align_obs_fn`,
      `shard_init_fn` (and `obs_log_prob_fn` for diagnostics).

## The 5-step add-a-new-model checklist

Tick these off as you go. Each step maps to a section in the numbered
docs.

- [ ] **Step 1 — Driver package.** Create `drivers/<model>/` with
      `config.py`, `plots.py`, `rolling.py`, `__init__.py`. All
      model-specific knobs in a frozen dataclass.
      → [01_driver_package.md](01_driver_package.md)
- [ ] **Step 2 — Modularity tests.** Add
      `tests/test_<model>_rolling_imports.py` with the 3-4 invariant
      tests. They run in milliseconds, no GPU.
      → [02_modularity_tests.md](02_modularity_tests.md)
- [ ] **Step 3 — fsa_high_res regression** (or whichever model is
      currently the production reference). Run a 1-window parity test;
      it must reproduce the existing result bit-identically. **If it
      doesn't, STOP** — your additions broke something.
      → [03_running_and_results.md](03_running_and_results.md#pre-run-gates)
- [ ] **Step 4 — 1-window cold-start for your model.** Confirms the
      inner-PF accepts your model's obs structure (mixed-likelihood,
      sparse channels, etc). Coverage target: > 50% raw as a sanity
      floor; actual production target ≥ 70%.
      → [03_running_and_results.md](03_running_and_results.md)
- [ ] **Step 5 — Full rolling run + result.md + HANDOFF + compat row.**
      The end-to-end reproduction document.
      → [03_running_and_results.md](03_running_and_results.md#writing-up)

## Repo layout (post-port)

After the port, the SMC² repo looks like:

```
smc2-blackjax-rolling/
├── smc2bj/                            # generic framework — DO NOT TOUCH
├── models/                            # namespace pkg shared with public dev
│   └── fsa_real_obs/                  # only model with SMC²-specific edits
├── drivers/
│   ├── _artifact_loader.py            # generic — DO NOT TOUCH
│   ├── fsa_high_res_rolling.py        # legacy flat-script driver — DO NOT TOUCH
│   ├── fsa_real_obs_5yr_rolling.py
│   └── <your_model>/                  # NEW — fully isolated
│       ├── __init__.py
│       ├── config.py
│       ├── plots.py
│       └── rolling.py
├── tests/
│   ├── conftest.py                    # public-dev path injection
│   ├── test_high_res_fsa.py
│   └── test_<your_model>_rolling_imports.py    # NEW
├── how_to_add_a_new_model/            # this guide
└── outputs/
    ├── fsa_high_res_rolling/
    └── <your_model>_rolling/                   # NEW
```

## The modularity invariants

The architecture this guide enforces:

1. **`smc2bj/` is generic.** Any model-specific code that creeps in
   here is a bug. The framework should never know which model is
   running; it just calls the model's callbacks.

2. **Per-model code lives in `drivers/<model>/`.** This is where the
   scenario constants, particle counts, channel names, plot funcs,
   and CLI all live. No model touches another model's package.

3. **Cross-model isolation is tested.** The modularity tests
   (Step 2 above) include `test_<model>_driver_imports_dont_touch_<other_model>`
   so that any accidental cross-import fails in milliseconds rather
   than corrupting a 4-hour SMC² run.

4. **The artifact format is the only contract.** Models communicate
   with the rest of the system via the
   [`SCENARIO_SCHEMA_VERSION = "1.0"`](https://github.com/ajaytalati/Python-Model-Scenario-Simulation/blob/main/docs/SCENARIO_FORMAT.md)
   artifact format from psim — not by importing each other's code.

If you find yourself wanting to break an invariant, **stop and ask** —
the more likely answer is "factor the shared bit into smc2bj/ as a
generic helper" rather than "add a model-specific shortcut."

## The guide files

| File | Contents |
|:---|:---|
| [README.md](README.md) | You are here — orientation, prerequisites, checklist. |
| [01_driver_package.md](01_driver_package.md) | The 3 files in `drivers/<model>/`: config, plots, rolling. |
| [02_modularity_tests.md](02_modularity_tests.md) | The 3-4 invariant tests that protect the architecture. |
| [03_running_and_results.md](03_running_and_results.md) | Pre-run gates, 1-window dry run, full rolling run, result md. |
| [worked_example_swat.md](worked_example_swat.md) | Line-by-line SWAT walkthrough referencing real files. |

Read `README.md` (this file) first, then `worked_example_swat.md` to
see what a finished port looks like. Use `01`–`03` as references
while you build your own.
