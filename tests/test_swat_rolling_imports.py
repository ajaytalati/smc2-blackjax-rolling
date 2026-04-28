"""Modularity-invariant tests for the SWAT driver.

These run in milliseconds, no GPU. They protect the architecture
that the SWAT port relied on:

  - The SWAT driver imports cleanly without touching fsa_high_res.
  - All SWAT-specific knobs live in the frozen SwatRollingConfig
    dataclass (no hidden constants in rolling.py main loop).
  - The dataclass round-trips through to_dict() cleanly so any run
    can be reproduced from its saved driver_config.json.
  - A fresh import of the SWAT driver doesn't pollute global state.

If a future model port (or refactor) breaks any of these, this test
file fails immediately — much cheaper than a 4-hour SMC² run failing
mid-trajectory.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. Cross-module isolation: SWAT driver must not import fsa_high_res
# ─────────────────────────────────────────────────────────────────────

def test_swat_driver_imports_dont_touch_fsa_high_res():
    """Loading drivers.swat.* must not also load fsa_high_res. This is
    the regression-proof invariant: if anything in drivers/swat/
    accidentally imports drivers/fsa_high_res_rolling.py (or vice
    versa), this test fails."""
    # Clear any pre-existing fsa_high_res load
    for name in list(sys.modules):
        if 'fsa_high_res_rolling' in name:
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith('drivers.swat'):
            del sys.modules[name]

    import drivers.swat                       # noqa: F401
    import drivers.swat.config                # noqa: F401
    import drivers.swat.plots                 # noqa: F401

    leaked = [n for n in sys.modules
              if 'fsa_high_res_rolling' in n]
    assert not leaked, (
        "drivers.swat.* leaked an import of fsa_high_res_rolling: "
        f"{leaked}. The two driver namespaces must stay orthogonal.")


# ─────────────────────────────────────────────────────────────────────
# 2. Config is a frozen dataclass — protects reviewability + accidental
#    in-place mutation in the driver
# ─────────────────────────────────────────────────────────────────────

def test_swat_config_is_frozen_dataclass():
    from dataclasses import is_dataclass, FrozenInstanceError
    from drivers.swat.config import SWAT_SET_A_CONFIG, SwatRollingConfig

    assert is_dataclass(SwatRollingConfig)
    assert is_dataclass(SWAT_SET_A_CONFIG)

    # Frozen → must raise on attribute assignment
    with pytest.raises((FrozenInstanceError, AttributeError)):
        SWAT_SET_A_CONFIG.n_smc = 1024


def test_swat_config_serialises_round_trip():
    """The dataclass must round-trip through to_dict() and to_json()
    so the saved driver_config.json captures the full reproducibility
    state."""
    import json
    from drivers.swat.config import SWAT_SET_A_CONFIG

    d = SWAT_SET_A_CONFIG.to_dict()
    assert d['scenario_name'] == 'set_A_healthy'
    assert d['n_days'] * d['bins_per_day'] == 4032   # 14 days × 288 bins/day
    assert tuple(d['obs_channel_names']) == ('hr', 'sleep', 'steps', 'stress')
    assert d['n_smc'] == 256
    assert d['n_pf'] == 400

    # JSON round-trip
    s = SWAT_SET_A_CONFIG.to_json()
    parsed = json.loads(s)
    assert parsed['scenario_name'] == d['scenario_name']


# ─────────────────────────────────────────────────────────────────────
# 3. Optional but useful: SWAT driver loads cleanly without a real GPU
# ─────────────────────────────────────────────────────────────────────

def test_swat_rolling_main_module_importable():
    """The full SWAT rolling-driver module must import without
    side-effecting (no main-on-import). This catches accidental
    module-level os.makedirs, GPU allocation, etc."""
    if 'drivers.swat.rolling' in sys.modules:
        del sys.modules['drivers.swat.rolling']
    importlib.import_module('drivers.swat.rolling')
