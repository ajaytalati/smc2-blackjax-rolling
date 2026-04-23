"""Config dataclasses for SMC² + rolling-window estimation.

Replaces the tangle of module-level globals in the original monolithic driver
with explicit, plain-Python config objects that pass through the API.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SMCConfig:
    """Config for the outer tempered-SMC over parameters, plus the inner PF."""

    # Outer SMC
    n_smc_particles: int = 256
    target_ess_frac: float = 0.5
    num_mcmc_steps: int = 5          # cold-start HMC moves per tempering level
    max_lambda_inc: float = 0.05     # cold-start lambda clamp

    # Bridge (warm-start)
    num_mcmc_steps_bridge: int = 3
    max_lambda_inc_bridge: float = 0.10

    # HMC kernel
    hmc_step_size: float = 0.025
    hmc_num_leapfrog: int = 8

    # Inner PF
    n_pf_particles: int = 400
    bandwidth_scale: float = 1.0

    # Optimal-transport rescue
    ot_ess_frac: float = 0.05
    ot_temperature: float = 5.0
    ot_max_weight: float = 0.01
    ot_rank: int = 5
    ot_n_iter: int = 2
    ot_epsilon: float = 0.5


@dataclass
class RollingConfig:
    """Rolling-window framing."""

    window_days: int = 120
    stride_days: int = 30
    dt: float = 1.0
    n_substeps: int = 10             # per-day sub-steps for SDE accuracy
    max_windows: int | None = None   # truncate if set


@dataclass
class MissingDataConfig:
    """Example missing-data corruption model.

    Models three gap patterns typical of consumer wearables + endurance-sport
    training logs:
      1. Rest days (weekly): mask active-measurement channels
      2. Random per-channel dropout: mask passive-measurement channels
      3. Continuous broken-watch gap: mask all channels

    This is an opinionated default. Adapt the parameters (or replace the
    function) for other sensor setups.
    """

    dropout_rate: float = 0.15
    broken_watch_days: int = 14
    rest_days_per_week: Tuple[int, int] = (2, 3)  # min, max

    # Channel groupings — depend on the observation model
    active_channels: Tuple[str, ...] = ()   # masked on rest days
    passive_channels: Tuple[str, ...] = ()  # subject to random dropout
    all_obs_channels: Tuple[str, ...] = ()  # masked during broken-watch gap
