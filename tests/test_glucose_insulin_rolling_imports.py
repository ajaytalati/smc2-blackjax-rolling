"""Modularity-invariant tests for the glucose_insulin SMC² driver.

Mirrors test_swat_rolling_imports.py. Runs in milliseconds, no GPU.
Protects:
  - drivers.glucose_insulin imports cleanly without touching swat / fsa_high_res
  - GiRollingConfig is a frozen dataclass (no hidden constants in rolling.py)
  - JSON / dict round-trip works (driver_config.json reproducibility)
  - Per-scenario instances (Sets A/B/C/D) all expose the right shape
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_gi_driver_imports_dont_touch_swat():
    """drivers.glucose_insulin.* must not load drivers.swat.*."""
    for name in list(sys.modules):
        if name.startswith('drivers.swat') or name.startswith('drivers.glucose_insulin'):
            del sys.modules[name]

    import drivers.glucose_insulin            # noqa: F401
    import drivers.glucose_insulin.config     # noqa: F401
    import drivers.glucose_insulin.plots      # noqa: F401

    leaked = [n for n in sys.modules if n.startswith('drivers.swat')]
    assert not leaked, (
        f"drivers.glucose_insulin.* leaked an import of drivers.swat: {leaked}.")


def test_gi_driver_imports_dont_touch_fsa_high_res():
    """drivers.glucose_insulin.* must not load fsa_high_res_rolling."""
    for name in list(sys.modules):
        if 'fsa_high_res_rolling' in name or name.startswith('drivers.glucose_insulin'):
            del sys.modules[name]

    import drivers.glucose_insulin            # noqa: F401
    import drivers.glucose_insulin.config     # noqa: F401
    import drivers.glucose_insulin.plots      # noqa: F401

    leaked = [n for n in sys.modules if 'fsa_high_res_rolling' in n]
    assert not leaked, (
        f"drivers.glucose_insulin.* leaked import of fsa_high_res_rolling: {leaked}.")


def test_gi_config_is_frozen_dataclass():
    from dataclasses import is_dataclass, FrozenInstanceError
    from drivers.glucose_insulin.config import (
        GiRollingConfig, GI_SET_A_CONFIG, GI_SET_B_CONFIG,
        GI_SET_C_CONFIG, GI_SET_D_CONFIG,
    )

    assert is_dataclass(GiRollingConfig)
    for cfg in (GI_SET_A_CONFIG, GI_SET_B_CONFIG,
                 GI_SET_C_CONFIG, GI_SET_D_CONFIG):
        assert is_dataclass(cfg)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            cfg.n_smc = 9999


def test_gi_config_serialises_round_trip():
    """to_dict() / to_json() must capture the full reproducibility state."""
    import json
    from drivers.glucose_insulin.config import GI_SET_A_CONFIG

    d = GI_SET_A_CONFIG.to_dict()
    assert d['scenario_name'] == 'set_A_healthy'
    assert d['n_days'] == 1
    assert d['bins_per_day'] == 288
    assert d['n_days'] * d['bins_per_day'] == 288
    assert tuple(d['obs_channel_names']) == ('cgm', 'meal_carbs')
    # Default bridge is SF Path B-fixed
    assert d['bridge_type'] == 'schrodinger_follmer'
    assert d['sf_q1_mode'] == 'annealed'
    assert d['sf_use_q0_cov'] is True
    assert d['sf_blend'] == 0.7
    assert d['sf_annealed_n_mh_steps'] == 5

    s = GI_SET_A_CONFIG.to_json()
    parsed = json.loads(s)
    assert parsed['scenario_name'] == d['scenario_name']


def test_gi_set_distinguishing_values():
    """Each per-set config carries the right scenario_name and shared structure."""
    from drivers.glucose_insulin.config import (
        GI_SET_A_CONFIG, GI_SET_B_CONFIG, GI_SET_C_CONFIG, GI_SET_D_CONFIG,
    )

    assert GI_SET_A_CONFIG.scenario_name == 'set_A_healthy'
    assert GI_SET_B_CONFIG.scenario_name == 'set_B_insulin_resistance'
    assert GI_SET_C_CONFIG.scenario_name == 'set_C_t1d_no_insulin'
    assert GI_SET_D_CONFIG.scenario_name == 'set_D_t1d_open_loop'

    # All 4 share the same windowing defaults
    for cfg in (GI_SET_A_CONFIG, GI_SET_B_CONFIG,
                 GI_SET_C_CONFIG, GI_SET_D_CONFIG):
        assert cfg.window_bins == 72
        assert cfg.stride_bins == 36
        assert cfg.bins_per_day == 288


def test_gi_rolling_main_module_importable():
    """The full rolling-driver module must import without side effects."""
    if 'drivers.glucose_insulin.rolling' in sys.modules:
        del sys.modules['drivers.glucose_insulin.rolling']
    importlib.import_module('drivers.glucose_insulin.rolling')
