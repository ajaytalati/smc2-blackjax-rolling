"""drivers.glucose_insulin — SMC² rolling-window driver for the Bergman model.

The canonical basic test model. See ``config.py`` for per-scenario
``GiRollingConfig`` instances (Sets A/B/C/D), ``rolling.py`` for the CLI,
``plots.py`` for the model-specific input diagnostic.

Default bridge: SF Path B-fixed per outputs/SF_BEST_PRACTICE_3_models.md.
"""

from .config import (  # noqa: F401
    GiRollingConfig,
    GI_SET_A_CONFIG,
    GI_SET_B_CONFIG,
    GI_SET_C_CONFIG,
    GI_SET_D_CONFIG,
)
