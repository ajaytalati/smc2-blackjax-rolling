"""simulator — Model-Agnostic SDE Simulation Framework.

Date:    16 April 2026
Version: 2.0

Generic framework only — no model-specific code lives here.
Model definitions (SDEModel objects) live in models/<n>/simulation.py.
"""

import os as _os
import sys as _sys

# The generic sde_*.py files use bare imports (from sde_model import ...)
# matching the user's original standalone package convention.  For these
# to resolve when the simulator is imported as a package, its directory
# must be on sys.path.
_SIM_DIR = _os.path.dirname(_os.path.abspath(__file__))
if _SIM_DIR not in _sys.path:
    _sys.path.insert(0, _SIM_DIR)

from smc2bj.simulator.sde_model import SDEModel, StateSpec, ChannelSpec
from smc2bj.simulator.sde_observations import generate_all_channels

__all__ = ['SDEModel', 'StateSpec', 'ChannelSpec', 'generate_all_channels']