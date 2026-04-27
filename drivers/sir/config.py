"""All SIR-specific scenario, SMC², and rolling-window choices.

Frozen dataclass instances per scenario. To bump a hyperparameter,
edit the dataclass and rerun — git diff is the audit trail. The
config is dumped to each output dir's ``driver_config.json`` so any
run is reproducible from its saved config.

The ONLY place SIR-specific knobs live in the SMC² repo. swat has its
own choices in ``../swat/config.py``; fsa_high_res in
``../fsa_high_res_rolling.py`` (a flat script predating this dataclass
convention). Adding a new model = adding a new
``drivers/<model>/config.py`` with its own dataclass.

Defaults: SF Path B-fixed bridge per the
``outputs/SF_BEST_PRACTICE_2_models.md`` recommendation. SIR is the
third model exercising this bridge after SWAT and fsa_high_res.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class SirRollingConfig:
    """Frozen knob-set for a SIR rolling-window estimation run."""

    # ── Scenario shape ────────────────────────────────────────────
    # Defaults match Set A (Anderson & May 1978 boarding-school flu;
    # the paper-parity benchmark). Override via per-scenario instances.
    scenario_name: str = "set_A_boarding_school"
    n_days: int = 14
    bins_per_day: int = 24          # hourly resolution
    dt_hours: float = 1.0
    n_substeps: int = 4
    population_N: float = 763.0     # frozen on the est side via DEFAULT_FROZEN_PARAMS

    # ── Obs channel naming (matches public-dev SIR_MODEL.channels) ─
    obs_channel_names: Tuple[str, ...] = ('cases', 'serology')

    # ── Rolling-window framing ───────────────────────────────────
    # 1 day = 24 bins, 12-hour stride = 12 bins. Short outbreak (14 d)
    # gives ~26 windows. For longer scenarios B/C/D the per-instance
    # configs below override.
    window_bins: int = 168          # 7 days
    stride_bins: int = 84           # 3.5 days

    # ── SMC² particle counts ──────────────────────────────────────
    # Started at N_SMC=128, N_PF=200 (proportional to SIR's 7-D vs SWAT's
    # 35-D), but empirically the smaller counts under-resolve SIR's sharp
    # Poisson-cases likelihood: Set A jumped from 19% mean coverage at
    # 128/200 to 42.9% at 256/400. Keeping SWAT's defaults for safety.
    n_smc: int = 256
    n_pf: int = 400

    # ── SMC² tempering (inherits SWAT's well-tuned values) ───────
    target_ess_frac: float = 0.30
    max_lambda_inc: float = 0.10
    max_lambda_inc_bridge: float = 0.15

    # ── Bridge base measure ─────────────────────────────────────
    # Default: SF Path B-fixed per outputs/SF_BEST_PRACTICE_2_models.md.
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
    # SIR's frozen N and v are handled by the EstimationModel factory
    # ``make_sir_estimation`` in models.sir.estimation, not by listing
    # keys here. Empty.
    frozen_param_keys: Tuple[str, ...] = ()

    # ── Default psim artifact path ───────────────────────────────
    default_artifact_dir: str = (
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/sir/set_A_boarding_school_14d"
    )

    @property
    def n_bins_total(self) -> int:
        return self.n_days * self.bins_per_day

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ── Per-scenario instances ─────────────────────────────────────────

# Set A — Anderson & May 1978 boarding-school flu (paper-parity, 14 d, N=763).
# Window = 7 d, stride = 3.5 d → first window covers most of the outbreak,
# subsequent windows shift toward the tail.
SIR_SET_A_CONFIG = SirRollingConfig(
    scenario_name="set_A_boarding_school",
    n_days=14,
    population_N=763.0,
    window_bins=7 * 24,
    stride_bins=int(3.5 * 24),
    default_artifact_dir=(
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/sir/set_A_boarding_school_14d"
    ),
)

# Set B — small community outbreak (60 d, R₀=2.5, N=10000).
# Window = 14 d, stride = 7 d → ~7 windows over the outbreak.
SIR_SET_B_CONFIG = SirRollingConfig(
    scenario_name="set_B_small_outbreak",
    n_days=60,
    population_N=10000.0,
    window_bins=14 * 24,
    stride_bins=7 * 24,
    default_artifact_dir=(
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/sir/set_B_small_outbreak_60d"
    ),
)

# Set C — large community outbreak (90 d, R₀=4.0, N=10000).
SIR_SET_C_CONFIG = SirRollingConfig(
    scenario_name="set_C_large_outbreak",
    n_days=90,
    population_N=10000.0,
    window_bins=14 * 24,
    stride_bins=7 * 24,
    default_artifact_dir=(
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/sir/set_C_large_outbreak_90d"
    ),
)

# Set D — vaccination intervention (90 d, R₀=3.0, v=0.02/day, N=10000).
# Sets up the closed-loop control benchmark for the next-stage Phase 5 work.
SIR_SET_D_CONFIG = SirRollingConfig(
    scenario_name="set_D_vaccination",
    n_days=90,
    population_N=10000.0,
    window_bins=14 * 24,
    stride_bins=7 * 24,
    default_artifact_dir=(
        "~/Repos/Python-Model-Scenario-Simulation/"
        "outputs/sir/set_D_vaccination_90d"
    ),
)


__all__ = [
    "SirRollingConfig",
    "SIR_SET_A_CONFIG",
    "SIR_SET_B_CONFIG",
    "SIR_SET_C_CONFIG",
    "SIR_SET_D_CONFIG",
]
