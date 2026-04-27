"""Modularity-invariant tests for the SIR driver.

Mirrors ``tests/test_swat_rolling_imports.py``. Runs in milliseconds, no GPU.

These protect the architecture that the SIR port relies on:

  - The SIR driver imports cleanly without touching swat or fsa_high_res.
  - All SIR-specific knobs live in the frozen SirRollingConfig dataclass
    (no hidden constants in rolling.py).
  - The dataclass round-trips through to_dict() / to_json() cleanly.
  - A fresh import of the SIR driver doesn't pollute global state.
  - Per-scenario instances (Set A/B/C/D) all expose the right shape.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ─────────────────────────────────────────────────────────────────────
# 1. Cross-module isolation: SIR must not pull in swat or fsa_high_res
# ─────────────────────────────────────────────────────────────────────

def test_sir_driver_imports_dont_touch_swat():
    """Loading drivers.sir.* must not also load drivers.swat.*."""
    for name in list(sys.modules):
        if name.startswith('drivers.swat') or name.startswith('drivers.sir'):
            del sys.modules[name]

    import drivers.sir            # noqa: F401
    import drivers.sir.config     # noqa: F401
    import drivers.sir.plots      # noqa: F401

    leaked_swat = [n for n in sys.modules if n.startswith('drivers.swat')]
    assert not leaked_swat, (
        f"drivers.sir.* leaked an import of drivers.swat: {leaked_swat}.")


def test_sir_driver_imports_dont_touch_fsa_high_res():
    """Loading drivers.sir.* must not also load fsa_high_res_rolling."""
    for name in list(sys.modules):
        if 'fsa_high_res_rolling' in name or name.startswith('drivers.sir'):
            del sys.modules[name]

    import drivers.sir            # noqa: F401
    import drivers.sir.config     # noqa: F401
    import drivers.sir.plots      # noqa: F401

    leaked = [n for n in sys.modules if 'fsa_high_res_rolling' in n]
    assert not leaked, (
        f"drivers.sir.* leaked an import of fsa_high_res_rolling: {leaked}.")


# ─────────────────────────────────────────────────────────────────────
# 2. Config is a frozen dataclass
# ─────────────────────────────────────────────────────────────────────

def test_sir_config_is_frozen_dataclass():
    from dataclasses import is_dataclass, FrozenInstanceError
    from drivers.sir.config import (
        SirRollingConfig, SIR_SET_A_CONFIG, SIR_SET_B_CONFIG,
        SIR_SET_C_CONFIG, SIR_SET_D_CONFIG,
    )

    assert is_dataclass(SirRollingConfig)
    for cfg in (SIR_SET_A_CONFIG, SIR_SET_B_CONFIG,
                 SIR_SET_C_CONFIG, SIR_SET_D_CONFIG):
        assert is_dataclass(cfg)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.n_smc = 9999


# ─────────────────────────────────────────────────────────────────────
# 3. JSON / dict round-trip
# ─────────────────────────────────────────────────────────────────────

def test_sir_config_serialises_round_trip():
    """to_dict() / to_json() must capture the full reproducibility state."""
    import json
    from drivers.sir.config import SIR_SET_A_CONFIG

    d = SIR_SET_A_CONFIG.to_dict()
    assert d['scenario_name'] == 'set_A_boarding_school'
    assert d['n_days'] == 14
    assert d['bins_per_day'] == 24
    assert d['n_days'] * d['bins_per_day'] == 336
    assert tuple(d['obs_channel_names']) == ('cases', 'serology')
    # Default bridge is SF Path B-fixed per the recommendation.
    assert d['bridge_type'] == 'schrodinger_follmer'
    assert d['sf_q1_mode'] == 'annealed'
    assert d['sf_use_q0_cov'] is True
    assert d['sf_blend'] == 0.7
    assert d['sf_annealed_n_mh_steps'] == 5

    # JSON round-trip
    s = SIR_SET_A_CONFIG.to_json()
    parsed = json.loads(s)
    assert parsed['scenario_name'] == d['scenario_name']


# ─────────────────────────────────────────────────────────────────────
# 4. Per-scenario instances
# ─────────────────────────────────────────────────────────────────────

def test_sir_set_distinguishing_values():
    """Each per-set config carries the right N + duration."""
    from drivers.sir.config import (
        SIR_SET_A_CONFIG, SIR_SET_B_CONFIG, SIR_SET_C_CONFIG, SIR_SET_D_CONFIG,
    )

    assert SIR_SET_A_CONFIG.population_N == 763.0
    assert SIR_SET_A_CONFIG.n_days == 14
    assert SIR_SET_B_CONFIG.population_N == 10000.0
    assert SIR_SET_B_CONFIG.n_days == 60
    assert SIR_SET_C_CONFIG.population_N == 10000.0
    assert SIR_SET_C_CONFIG.n_days == 90
    assert SIR_SET_D_CONFIG.population_N == 10000.0
    assert SIR_SET_D_CONFIG.n_days == 90


# ─────────────────────────────────────────────────────────────────────
# 5. Full driver module importable (no side effects)
# ─────────────────────────────────────────────────────────────────────

def test_sir_rolling_main_module_importable():
    """The full SIR rolling-driver module must import without side effects."""
    if 'drivers.sir.rolling' in sys.modules:
        del sys.modules['drivers.sir.rolling']
    importlib.import_module('drivers.sir.rolling')
