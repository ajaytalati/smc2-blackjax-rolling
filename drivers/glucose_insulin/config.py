"""All glucose_insulin SMC²-rolling-window choices.

Frozen dataclass instances per scenario. To bump a hyperparameter,
edit the dataclass and rerun — git diff is the audit trail. The
config is dumped to each output dir's ``driver_config.json`` so any
run is reproducible from its saved config.

Defaults: SF Path B-fixed bridge per
``outputs/SF_BEST_PRACTICE_3_models.md``. glucose_insulin is the
4th model exercising this bridge after SWAT, fsa_high_res, SIR.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class GiRollingConfig:
    """Frozen knob-set for a glucose_insulin rolling-window estimation run."""

    # ── Scenario shape ────────────────────────────────────────────
    # Defaults match Set A (healthy, Bergman 1979 paper-parity).
    scenario_name: str = "set_A_healthy"
    n_days: int = 1
    bins_per_day: int = 288             # 5-min CGM cadence
    dt_hours: float = 5.0 / 60.0
    n_substeps: int = 4

    # ── Obs channel naming ───────────────────────────────────────
    obs_channel_names: Tuple[str, ...] = ('cgm', 'meal_carbs')

    # ── Rolling-window framing ───────────────────────────────────
    # 6-hour window (72 bins), 3-hour stride (36 bins) → 7 windows / day.
    # Each window contains a full meal-response cycle (rise + return).
    window_bins: int = 72
    stride_bins: int = 36

    # ── SMC² particle counts ─────────────────────────────────────
    n_smc: int = 256
    n_pf: int = 400

    # ── SMC² tempering ───────────────────────────────────────────
    target_ess_frac: float = 0.30
    max_lambda_inc: float = 0.10
    max_lambda_inc_bridge: float = 0.15

    # ── Bridge base measure ─────────────────────────────────────
    # Default: SF Path B-fixed (annealed q1, q0_cov, blend=0.7, n_mh=5).
    # Override to 'gaussian' for the regression-baseline comparison run.
    bridge_type: str = 'schrodinger_follmer'
    bridge_mog_components: int = 2
    sf_blend: float = 0.7
    sf_entropy_reg: float = 0.0
    sf_q1_mode: str = 'annealed'
    sf_annealed_n_stages: int = 3
    sf_annealed_n_mh_steps: int = 5
    sf_annealed_proposal_scale: float = 0.4
    sf_use_q0_cov: bool = True

    # ── Frozen (non-estimated) params ────────────────────────────
    # The Bergman model's frozen params (Ib, V_G, V_I, BW, T_X, T_I,
    # n_beta, h_beta) are handled by the EstimationModel factory
    # ``make_glucose_insulin_estimation`` — not listed here.
    frozen_param_keys: Tuple[str, ...] = ()

    # ── Default psim artifact path ───────────────────────────────
    default_artifact_dir: str = (
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/glucose_insulin/set_A_healthy_24h"
    )

    @property
    def n_bins_total(self) -> int:
        return self.n_days * self.bins_per_day

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Per-scenario instances. Adding a scenario = adding a new instance.

GI_SET_A_CONFIG = GiRollingConfig(
    scenario_name="set_A_healthy",
    default_artifact_dir=(
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/glucose_insulin/set_A_healthy_24h"
    ),
)

GI_SET_B_CONFIG = GiRollingConfig(
    scenario_name="set_B_insulin_resistance",
    default_artifact_dir=(
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/glucose_insulin/set_B_insulin_resistance_24h"
    ),
)

GI_SET_C_CONFIG = GiRollingConfig(
    scenario_name="set_C_t1d_no_insulin",
    default_artifact_dir=(
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/glucose_insulin/set_C_t1d_no_insulin_24h"
    ),
)

GI_SET_D_CONFIG = GiRollingConfig(
    scenario_name="set_D_t1d_open_loop",
    default_artifact_dir=(
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/glucose_insulin/set_D_t1d_open_loop_24h"
    ),
)


__all__ = [
    "GiRollingConfig",
    "GI_SET_A_CONFIG",
    "GI_SET_B_CONFIG",
    "GI_SET_C_CONFIG",
    "GI_SET_D_CONFIG",
]
