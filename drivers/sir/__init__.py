"""drivers.sir — SIR-specific scenario driver for SMC² rolling-window estimation.

The canonical basic test model. See ``config.py`` for the per-scenario
``SirRollingConfig`` instances (Sets A/B/C/D), ``rolling.py`` for the CLI,
and ``plots.py`` for the model-specific input diagnostic.

Default bridge: SF Path B-fixed per outputs/SF_BEST_PRACTICE_2_models.md.
"""

from .config import (  # noqa: F401
    SirRollingConfig,
    SIR_SET_A_CONFIG,
    SIR_SET_B_CONFIG,
    SIR_SET_C_CONFIG,
    SIR_SET_D_CONFIG,
)
