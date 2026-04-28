"""SMC² rolling-window estimation for the SWAT model.

Per-model package — the modularity invariant: this package holds ALL
SWAT-specific code. fsa_high_res lives in ``../fsa_high_res_rolling.py``
(parallel module, fully independent). Shared infrastructure lives in
``smc2bj/`` (generic) and ``../_artifact_loader.py`` (generic).

Public entries:
  - SWAT_SET_A_CONFIG : the frozen dataclass with all SWAT-specific
                         scenario + SMC + rolling defaults.
  - SwatRollingConfig : the dataclass type.

Run via:
  PYTHONPATH=. python -m drivers.swat.rolling --seed 42 --windows 1
"""

from drivers.swat.config import (
    SwatRollingConfig,
    SWAT_SET_A_CONFIG,
)

__all__ = ["SwatRollingConfig", "SWAT_SET_A_CONFIG"]
