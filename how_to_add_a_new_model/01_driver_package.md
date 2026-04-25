# 01 — The per-model driver package

`drivers/<model>/` is where ALL model-specific code lives. The package
has 4 files; this doc walks through each.

## Why per-model packaging

The modularity invariant: no model touches another model's code, and
`smc2bj/` stays generic. Per-model packaging is what makes that
enforceable.

The legacy `drivers/fsa_high_res_rolling.py` is a single flat script
predating this convention. Don't follow that pattern for new models —
use the package layout below.

## The 4 files

```
drivers/<model>/
├── __init__.py     # exports the dataclass type + per-scenario instances
├── config.py       # frozen SwatRollingConfig dataclass + instances
├── plots.py        # model-specific channel diagnostic plot
└── rolling.py      # CLI entry, ~150-250 lines
```

### `__init__.py`

Three or four lines: re-export the dataclass type and per-scenario
config instances so callers can `from drivers.<model> import <MODEL>_<SET>_CONFIG`.

```python
"""SMC² rolling-window estimation for the <model> model.

Per-model package — the modularity invariant: ALL <model>-specific
code lives here. <other_model> lives in its own package; shared
infrastructure in smc2bj/ and drivers/_artifact_loader.py.
"""

from drivers.<model>.config import (
    <Model>RollingConfig,
    <MODEL>_<SET_A>_CONFIG,
    # ... other scenario instances
)

__all__ = ["<Model>RollingConfig", "<MODEL>_<SET_A>_CONFIG"]
```

### `config.py` — the single source of truth

A frozen dataclass with EVERY model-specific knob. Bumping a value
shows in `git diff`; no hidden constants in `rolling.py`'s main loop.
The dataclass is dumped to each output dir's `driver_config.json`
for run reproducibility.

```python
"""All <model>-specific scenario, SMC², and rolling-window choices.

The ONLY place <model>-specific knobs live in the SMC² repo. Other
models have their own configs in drivers/<other>/config.py.
"""

from dataclasses import dataclass, asdict
from typing import Tuple
import json


@dataclass(frozen=True)
class <Model>RollingConfig:
    # ── Scenario shape ────────────────────────────────────────────
    scenario_name: str = "set_A_healthy"
    n_days: int = 14
    bins_per_day: int = 288         # model's native time grid
    dt_hours: float = 5.0 / 60.0    # 5-min if applicable
    n_substeps: int = 4

    # ── Obs channel naming (matches public-dev <MODEL>.channels) ──
    obs_channel_names: Tuple[str, ...] = ('hr', 'sleep', 'steps', 'stress')

    # ── Rolling-window framing ───────────────────────────────────
    window_bins: int = 288
    stride_bins: int = 144

    # ── SMC² particle counts ─────────────────────────────────────
    n_smc: int = 256
    n_pf: int = 400

    # ── SMC² tempering ──────────────────────────────────────────
    target_ess_frac: float = 0.30
    max_lambda_inc: float = 0.10
    max_lambda_inc_bridge: float = 0.15

    # ── Bridge base measure ─────────────────────────────────────
    bridge_type: str = 'gaussian'
    bridge_mog_components: int = 2

    # ── Frozen (non-estimated) params ────────────────────────────
    frozen_param_keys: Tuple[str, ...] = ()

    # ── Default psim artifact path ───────────────────────────────
    default_artifact_dir: str = (
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/<model>/<scenario_name>"
    )

    @property
    def n_bins_total(self) -> int:
        return self.n_days * self.bins_per_day

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Per-scenario instances — adding a scenario = adding a new instance.
<MODEL>_SET_A_CONFIG = <Model>RollingConfig(scenario_name="set_A_healthy")
```

**Naming conventions:**
- Dataclass type: `<Model>RollingConfig` (CamelCase model + suffix).
- Per-scenario instance: `<MODEL>_<SET>_CONFIG` (UPPER_CASE).
- Defaults are the production-quality values (256/400 particles,
  `gaussian` bridge — these are the fsa_high_res inheritance, prove
  themselves on a model basis).
- `frozen_param_keys` is **empty by default**. Use it only if a
  scenario explicitly fixes some param (e.g. `V_c=0`).

**What's NOT in config.py:**
- Truth parameters — those come from the artifact's `manifest.json`.
- Init state values — those come from the artifact + the model's
  `INIT_STATE_PRIOR_CONFIG` (prior means for cold start).
- Channel data — comes from the artifact via `_artifact_loader`.

### `plots.py` — model-specific channel diagnostic

The generic `smc2bj.plotting.rolling.plot_parameter_tracking` and
`plot_coverage_and_timing` work for any model and are reused by the
rolling driver directly. Only the per-channel input plot is
model-specific because every model's channel structure is different.

```python
"""<Model>-specific diagnostic plots for rolling-window estimation."""

import os
import numpy as np


def _use_agg():
    import matplotlib
    matplotlib.use('Agg')


def plot_<model>_channels(bundle: dict, out_dir: str, *,
                           n_show_days: int = 3) -> str:
    """N-panel diagnostic of the input artifact: trajectory + channels."""
    _use_agg()
    import matplotlib.pyplot as plt
    # ... model-specific plotting layout ...
    path = os.path.join(out_dir, '<model>_channels.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    return path
```

A no-frills layout: one row per channel + one row for the latent
trajectory. Match channel colours to the per-channel obs-likelihood
type so a reader can scan-read the artifact at a glance.

### `rolling.py` — CLI entry

The main script. Roughly 150-250 lines. Pattern:

1. **sys.path injection** for the public-dev model (namespace-package
   merge). The pattern is mechanical — copy from
   `drivers/fsa_high_res_rolling.py` or `drivers/swat/rolling.py`.

2. **JAX env vars** (`JAX_ENABLE_X64`, `JAX_COMPILATION_CACHE_DIR`).
   Set BEFORE importing jax.

3. **Imports**: `smc2bj.{estimation,pipeline,plotting,io}`,
   `drivers._artifact_loader`, your `drivers.<model>.config` and
   `.plots`, plus `models.<model>.estimation` (and any model helpers
   you need).

4. **Helpers**:
   - `_prior_mean(ptype, pargs)` — for the supported prior families
     (lognormal, normal, beta). Computes prior means used by cold-start.
   - `_cold_start_init()` — returns the cold-start init array (typically
     the n_estimated-init-states-D vector of prior means).
   - `_truth_dict(artifact_truth_params, cfg)` — combines the artifact's
     `truth_params` with any init values needed for coverage, filters
     out non-estimated metadata + `cfg.frozen_param_keys`, and returns
     a dict whose keys are a superset of `model.all_names`.

5. **CLI parsing**: `_parse_args(cfg)` with defaults from the dataclass
   — `--seed`, `--n-smc`, `--n-pf`, `--windows`, `--scenario-artifact`,
   `--bridge`, `--bridge-K`, `--show-checkpoint`.

6. **Output dir**: `_out_dir(seed, n_smc, scenario_name, bridge_tag)`.
   Use the `outputs/<model>_rolling/<scenario_name>_N<n>_s<seed>/` pattern
   so each model's outputs stay in its own namespace.

7. **`main()`**:
   - `cfg = <MODEL>_<SET>_CONFIG`
   - Parse args, build out_dir
   - Dump `driver_config.json` (config + cli) for reproducibility
   - Load artifact via `load_scenario(artifact)` from
     `drivers._artifact_loader`
   - Sanity-assert `bundle['n_bins_total'] == cfg.n_bins_total`
   - Build `SMCConfig` + `RollingConfig` from dataclass + CLI overrides
   - Build the truth dict via `_truth_dict()`
   - Run the diagnostic plot of the input artifact
   - Call `rolling_window_smc(...)` from `smc2bj.pipeline.rolling`
   - Call generic `plot_parameter_tracking` and
     `plot_coverage_and_timing` for the result plots
   - Print summary (mean coverage, PASS rate, wall time)

The rolling driver imports from
`models.<model>.estimation` for the `<MODEL>_ESTIMATION` object and
`INIT_STATE_PRIOR_CONFIG`, and from `models.<model>.simulation` for
any "constant state" values (e.g. SWAT's `INIT_STATE_A` provides
`Vh`, `Vn` truth values that aren't in the artifact's `truth_params`
because they live on the sim side as init-state entries).

## A side-by-side comparison

| Concern | fsa_high_res (legacy flat script) | SWAT (modular package) |
|---|---|---|
| Constants | Top-level module variables (`N_DAYS_TOTAL`, `BINS_PER_DAY`, ...) | Frozen `SwatRollingConfig` dataclass field |
| Particle counts | Hard-coded in `_parse_args` defaults | `cfg.n_smc, cfg.n_pf` defaults via dataclass |
| Bridge type | CLI default `'gaussian'` only | `cfg.bridge_type` (CLI override) |
| Reproducibility | Re-read `--seed` and CLI args from log | `driver_config.json` dump per output dir |
| Adding Set B/C | New constants block + new function | New `<MODEL>_SET_B_CONFIG` instance in same `config.py` |

The flat-script form is fine for the original model that motivated
the framework; the package form is mandatory for every new model
going forward.

## What goes in `smc2bj/`?

In rare cases you'll find that adding your model exposes a generic
limitation in the framework — not model-specific, but a real defect
in `smc2bj/`. Examples that have actually happened:

- `smc2bj/pipeline/windowing.py:extract_window` originally assumed
  every channel field was an array of length `len(t_idx)`, crashing
  on SWAT's scalar `bin_hours` field. Fix: gracefully pass through
  scalars / non-per-step metadata.

These are framework-level fixes; they go in `smc2bj/` (the right
place) and benefit all models. **The test:** "would model X
benefit from this even if model Y wasn't being added right now?"
If yes, it's a framework fix; if no, it's a model-specific quirk
that belongs in `drivers/<model>/`.

## Reference

The two existing examples cover both the legacy flat-script form and
the modular package form:

- [drivers/fsa_high_res_rolling.py](../drivers/fsa_high_res_rolling.py) — the legacy flat script (do not imitate for new models)
- [drivers/swat/](../drivers/swat/) — the canonical modular package

Read the SWAT package end-to-end before writing your own. The
worked example at [worked_example_swat.md](worked_example_swat.md)
walks through it line-by-line.
