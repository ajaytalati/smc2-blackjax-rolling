"""All SWAT-specific scenario, SMC², and rolling-window choices.

Frozen dataclass instances per scenario. To bump a hyperparameter,
edit the dataclass and rerun — git diff is the audit trail. The
config is dumped to each output dir's ``driver_config.json`` so any
run is reproducible from its saved config.

The ONLY place SWAT-specific knobs live in the SMC² repo. fsa_high_res
has its own choices in ``../fsa_high_res_rolling.py`` (a flat script
predating this dataclass convention). Adding a new model = adding a
new ``drivers/<model>/config.py`` with its own dataclass.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class SwatRollingConfig:
    """Frozen knob-set for a SWAT rolling-window estimation run."""

    # ── Scenario shape ────────────────────────────────────────────
    scenario_name: str = "set_A_healthy"
    n_days: int = 14
    bins_per_day: int = 288         # 5-min resolution
    dt_hours: float = 5.0 / 60.0
    n_substeps: int = 4

    # ── Obs channel naming (matches public-dev SWAT_MODEL.channels)
    obs_channel_names: Tuple[str, ...] = ('hr', 'sleep', 'steps', 'stress')

    # ── Rolling-window framing ───────────────────────────────────
    window_bins: int = 288          # 1 day at 5-min
    stride_bins: int = 144          # 12 hours

    # ── SMC² particle counts (start with fsa_high_res defaults) ──
    n_smc: int = 256
    n_pf: int = 400

    # ── SMC² tempering ──────────────────────────────────────────
    target_ess_frac: float = 0.30
    max_lambda_inc: float = 0.10
    max_lambda_inc_bridge: float = 0.15

    # ── Bridge base measure ─────────────────────────────────────
    bridge_type: str = 'gaussian'      # or 'mog' or 'schrodinger_follmer'
    bridge_mog_components: int = 2
    sf_blend: float = 0.5              # Schrödinger-Föllmer t in [0,1]
    sf_entropy_reg: float = 0.0        # SF entropic regularisation

    # ── Frozen (non-estimated) params ────────────────────────────
    # Empty for SWAT v1 — everything in PARAM_PRIOR_CONFIG is estimated.
    # If a future scenario freezes V_c=0, it goes here.
    frozen_param_keys: Tuple[str, ...] = ()

    # ── Default psim artifact path ───────────────────────────────
    default_artifact_dir: str = (
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/swat/set_A_healthy_14d"
    )

    @property
    def n_bins_total(self) -> int:
        return self.n_days * self.bins_per_day

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Per-scenario instances ─────────────────────────────────────────
# Adding a scenario = adding a new instance here.

SWAT_SET_A_CONFIG = SwatRollingConfig(scenario_name="set_A_healthy")

# Future: SWAT_SET_B_CONFIG, SWAT_SET_C_CONFIG, SWAT_SET_D_CONFIG —
# each with set-specific tunings (e.g. tighter priors for B, longer
# trial for C). v0.1 ships only Set A.


__all__ = ["SwatRollingConfig", "SWAT_SET_A_CONFIG"]
