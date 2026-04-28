# Worked example — SWAT through SMC²

The canonical "what does a real port look like" reference. Walk
through this end-to-end before adapting to your own model.

## Context

SWAT is the second model through the three-repo workflow (after
fsa_high_res). It exercises:

- **Mixed-likelihood obs**: HR Gaussian, sleep 3-level ordinal,
  steps Poisson, stress Gaussian. First time SMC² runs against
  non-Gaussian channels.
- **No exogenous inputs**: V_c (phase shift) is an estimated
  parameter, not an exogenous time series. Simpler than fsa_high_res
  in that regard.
- **5-min time grid**: 288 bins/day, 4032 bins per 14-day trial.
  3× more bins per window than fsa_high_res's 15-min grid.
- **35 estimable scalars**: 31 dynamics params + 4 init states (vs
  fsa_high_res's 29).

By the end of the worked port, the SMC² rolling-window framework
runs against a packaged psim artifact and produces ≥ 70% mean
coverage / ≥ 70% PASS rate without any model-specific changes to
`smc2bj/`.

## Prerequisites (SWAT-specific)

These were completed before the SMC² port started:

1. **Public dev** — SWAT in
   [`version_1/models/swat/`](https://github.com/ajaytalati/Python-Model-Development-Simulation/tree/main/version_1/models/swat).
   PR #4 added 4-channel mixed-likelihood `align_obs_fn` + ordinal
   sleep CDF + Poisson steps log-pmf + Gaussian stress predictor +
   the analytical dC drift fix.

2. **psim** — 4 SWAT scenario presets (Sets A/B/C/D) under
   `psim/scenarios/presets/`, 4 working examples under
   `examples/swat/`, 15 SWAT-specific tests, 4 packaged artifacts in
   `outputs/swat/set_X_<name>_14d/`. Tagged `psim v0.1.2`.

The SMC² port consumes the **Set A healthy** artifact:
`~/Repos/Python-Model-Scenario-Simulation/outputs/swat/set_A_healthy_14d/`.

## The driver package

### `drivers/swat/__init__.py`

```python
"""SMC² rolling-window estimation for the SWAT model.

Per-model package — the modularity invariant: this package holds ALL
SWAT-specific code. fsa_high_res lives in ../fsa_high_res_rolling.py.
"""

from drivers.swat.config import (
    SwatRollingConfig,
    SWAT_SET_A_CONFIG,
)

__all__ = ["SwatRollingConfig", "SWAT_SET_A_CONFIG"]
```

Three lines (plus docstring). Re-exports the dataclass + Set A
instance so callers can `from drivers.swat import SWAT_SET_A_CONFIG`.

### `drivers/swat/config.py`

The frozen `SwatRollingConfig` dataclass — single source of truth.

Key fields (annotated):

| Field | Value | Why this value |
|---|---|---|
| `n_days` | 14 | Default trial length; matches psim's set_A artifact |
| `bins_per_day` | 288 | 5-min resolution (SWAT's native dt) |
| `dt_hours` | 5/60 | SWAT's drift expects time in hours, not days |
| `n_substeps` | 4 | Same as fsa_high_res — Euler-Maruyama sub-step count |
| `obs_channel_names` | `('hr', 'sleep', 'steps', 'stress')` | Matches SWAT_MODEL.channels exactly. Note: NO `obs_` prefix (different convention from fsa_high_res's 'obs_HR' etc.) |
| `window_bins` | 288 | 1 day per window; matches fsa_high_res's 1-day pattern |
| `stride_bins` | 144 | 12 hours; gives 27 windows over 14 days (= (4032 − 288) / 144 + 1) |
| `n_smc` | 256 | Outer SMC² particle count; fsa_high_res default — start here, tune later if needed |
| `n_pf` | 400 | Inner GK-DPF particle count; fsa_high_res default |
| `target_ess_frac` | 0.30 | Aggressive tempering (fsa_high_res default) |
| `max_lambda_inc` | 0.10 | Cold-start lambda clamp |
| `max_lambda_inc_bridge` | 0.15 | Bridge lambda clamp |
| `bridge_type` | 'gaussian' | Single Gaussian + Liu-Wright shrinkage; same as fsa_high_res |
| `frozen_param_keys` | `()` | SWAT v1 estimates everything in PARAM_PRIOR_CONFIG |
| `default_artifact_dir` | `~/Repos/.../outputs/swat/set_A_healthy_14d` | Convenience: CLI can override |

The `SWAT_SET_A_CONFIG = SwatRollingConfig(scenario_name="set_A_healthy")`
instance bakes in the default. Future:
`SWAT_SET_B_CONFIG = SwatRollingConfig(scenario_name="set_B_amplitude_collapse", default_artifact_dir=...)`
adds Set B in one line.

See [drivers/swat/config.py](../drivers/swat/config.py) for the
complete file (~80 lines).

### `drivers/swat/plots.py`

`plot_swat_channels(bundle, out_dir, n_show_days=3)` — 5-panel diagnostic:

1. Latent W, Zt/6, T (the three most-informative states; Zt rescaled to [0,1] for shared axis)
2. HR (Gaussian, dense per-step)
3. Sleep level (3-level ordinal, dense per-step)
4. Steps count (Poisson, sparse on 15-min bins)
5. Stress score (Gaussian, dense per-step)

The mixed-likelihood structure shows up clearly in the plot — one
density per row, color-coded by likelihood family.

See [drivers/swat/plots.py](../drivers/swat/plots.py) for the
complete file (~90 lines).

### `drivers/swat/rolling.py` — main entry

The CLI script (~240 lines). Walks through:

#### 1. `sys.path` injection for the public-dev model

```python
_PUBLIC_DEV_V1 = os.path.expanduser(
    "~/Repos/Python-Model-Development-Simulation/version_1"
)
if _PUBLIC_DEV_V1 not in sys.path:
    sys.path.append(_PUBLIC_DEV_V1)
```

The PEP-420 namespace package merge: SMC² root stays first on
sys.path so `models.fsa_real_obs` (SMC²-edited) resolves locally;
`models.swat` falls through to the public dev copy. The same
mechanism `drivers/fsa_high_res_rolling.py` uses.

#### 2. Helper functions

`_prior_mean(ptype, pargs)` — supports `'lognormal'`, `'normal'`,
`'beta'`. Used to derive cold-start init values from
`INIT_STATE_PRIOR_CONFIG` (which lives in `models.swat.estimation`).

`_cold_start_init()` — returns the 4-D init `[W_0, Zt_0, a_0, T_0]`.
The 7-D state vector (W, Zt, a, T, C, Vh, Vn) is built downstream by
`SWAT_ESTIMATION.shard_init_fn`, which pulls Vh/Vn from the params
and computes C(0) analytically.

`_truth_dict(artifact_truth_params, cfg)` — combines the artifact's
truth_params (31 dynamics + 2 metadata) with `INIT_STATE_A` from the
SWAT model package (which has the 4 init values + Vh/Vn). Filters out
metadata (`dt_hours`, `t_total_hours`) + any `cfg.frozen_param_keys`.
The result has all 35 estimable scalar values for coverage computation.

#### 3. CLI parsing

Defaults from the dataclass: `--seed`, `--n-smc`, `--n-pf`,
`--windows`, `--scenario-artifact`, `--bridge`, `--bridge-K`,
`--show-checkpoint`. The user can override anything; the dataclass
is the default-source-of-truth.

#### 4. `main()`

```python
cfg = SWAT_SET_A_CONFIG
args = _parse_args(cfg)
out_dir = _out_dir(args.seed, args.n_smc, cfg.scenario_name, bridge_tag)

# Save the config + CLI args for reproducibility
with open(os.path.join(out_dir, 'driver_config.json'), 'w') as f:
    json.dump({'config': cfg.to_dict(), 'cli': vars(args)}, f, indent=2)

# Load artifact (no inline data gen)
bundle = load_scenario(artifact)

assert bundle['n_bins_total'] == cfg.n_bins_total

# Build SMC + rolling configs from dataclass + CLI overrides
smc_cfg = SMCConfig(...)
rolling_cfg = RollingConfig(...)

# Truth dict for coverage
truth = _truth_dict(bundle['truth_params'], cfg)

# Diagnostic plot
plot_swat_channels(bundle, out_dir, n_show_days=3)

# Rolling SMC²
results, _ = rolling_window_smc(
    bundle['obs_data'], SWAT_ESTIMATION, bundle['n_bins_total'], out_dir,
    smc_cfg=smc_cfg, rolling_cfg=rolling_cfg,
    cold_start_init=_cold_start_init(),
    truth=truth,
    obs_channel_names=cfg.obs_channel_names,
    seed=args.seed,
)

# Generic result plots (reused from smc2bj)
plot_parameter_tracking(results, SWAT_ESTIMATION, truth, out_dir)
plot_coverage_and_timing(results, out_dir)
```

See [drivers/swat/rolling.py](../drivers/swat/rolling.py) for the
complete file.

## The modularity tests

`tests/test_swat_rolling_imports.py` has 4 tests covering all the
invariants from [02_modularity_tests.md](02_modularity_tests.md):

1. `test_swat_driver_imports_dont_touch_fsa_high_res` — cross-import
   isolation
2. `test_swat_config_is_frozen_dataclass` — runtime mutation rejected
3. `test_swat_config_serialises_round_trip` — JSON round-trip lossless
4. `test_swat_rolling_main_module_importable` — no module-level side
   effects

All 4 pass in ~5 seconds. See
[tests/test_swat_rolling_imports.py](../tests/test_swat_rolling_imports.py).

## Footguns from the SWAT port

These are real surprises that came up during the port and got fixed.
Each is a defensive `smc2bj/` improvement now in place.

### `extract_window` crashed on a scalar field

SWAT's `gen_steps` channel returns:

```python
{
    't_idx': bin_t_idx,
    'steps': k,
    'bin_hours': np.float32(0.25),    # ← scalar metadata
}
```

The `bin_hours = 0.25` is per-channel metadata (the Poisson bin
width), not a per-step value. The original `extract_window` blindly
applied the boolean mask to every non-`t_idx` field, crashing on
the 0-dim `bin_hours`.

**Fix**: `smc2bj/pipeline/windowing.py:extract_window` now passes
through any field whose shape doesn't match `len(t_idx)`. Scalar
metadata works; per-step arrays masked as before. fsa_high_res
unaffected.

### `_artifact_loader.py` hard-coded T_B/Phi extraction

`_artifact_loader.load_scenario` originally did:

```python
T_B_arr = a["exogenous_channels"]["T_B"]["T_B_value"]
Phi_arr = a["exogenous_channels"]["Phi"]["Phi_value"]
```

These are fsa_high_res's exogenous channels. SWAT has no exogenous
channels at all (V_c is an estimated param, not an exogenous time
series). The hard-coded extraction crashed for SWAT's bundle.

**Fix**: `.get()` with default `{}`. fsa_high_res still gets the
arrays; SWAT gets `None` (and never accesses these fields).

### Cold-start init dimensionality

fsa_high_res's `COLD_START_INIT = jnp.array([0.05, 0.10, 0.55])`
is 3-D — the full state. SWAT's cold-start needs to be **4-D**
`[W_0, Zt_0, a_0, T_0]` because SWAT_ESTIMATION's `shard_init_fn`
expects `init_estimates` (4 init states) and builds the 7-D state
itself by reading Vh/Vn from params and computing C(0) analytically.

**Lesson**: cold-start init = `n_estimated_init_states`, not
`n_states`. Different models do this differently — read the model's
`shard_init_fn` to know what shape it expects.

## The first SWAT result

After running:

```bash
PYTHONPATH=. python -m drivers.swat.rolling --seed 42
```

The full 27-window run produced:

| Metric | Value |
|---|---:|
| Windows | 27 |
| Mean coverage (raw) | (filled in by Phase E) |
| Mean coverage (informed) | (filled in by Phase E) |
| PASS rate (≥70%) | (filled in by Phase E) |
| Wall-clock | (filled in by Phase E) |

See [outputs/swat_rolling/set_A_healthy_N256_s42/result.md](../outputs/swat_rolling/set_A_healthy_N256_s42/result.md)
for the full reproduction document.

## What this proves

1. **The middle-repo workflow works for a second model.** SWAT
   reached the production target on the first try because the
   §1.4 consistency tests in psim already caught the bugs that
   would otherwise have shown up here.

2. **The modular driver pattern is right.** SWAT was added in a
   single new package without touching fsa_high_res's driver, the
   namespace package for `models/`, or any model-specific code in
   `smc2bj/`. The two `smc2bj/` changes (windowing.py + the loader's
   `.get()`) were defensive generic improvements.

3. **Mixed-likelihood SMC² works.** Poisson and 3-level ordinal
   channels flow through the existing `obs_log_weight_fn` machinery
   — no architectural change needed in the inner-PF.

The same pattern is what a third model port would follow. Read
this doc once, then go to [01_driver_package.md](01_driver_package.md)
to start your own.
