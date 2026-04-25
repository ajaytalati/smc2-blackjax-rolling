# 02 — The modularity-protecting tests

A small set of tests that catch architectural drift in milliseconds.
No GPU. They protect the per-model isolation that makes this repo
maintainable across many models.

## Why these tests exist

The SMC² repo's biggest risk is **model X silently breaking model
Y** when both are being added or maintained at the same time. The
ways this happens:

1. **Cross-imports.** Model Y's driver accidentally imports Model X's
   driver (e.g. via "I'll just copy this helper from there"). Now
   any module-level state in X leaks into Y.
2. **Shared mutable config.** Two drivers grow to share a config
   object that one of them mutates at runtime.
3. **Side effects on import.** Loading model Y's driver triggers GPU
   allocation or filesystem changes — making the test suite itself
   slow and order-dependent.
4. **Schema drift in the driver_config.json.** A dataclass field is
   renamed but the JSON dump still uses the old name; reproducibility
   breaks silently.

These are catastrophic bugs in production (a 4-hour SMC² run that
fails halfway through) but cheap to catch at test time. The 4 tests
below cost ~20 ms each.

## The 4 tests

Add as `tests/test_<model>_rolling_imports.py`:

### Test 1 — Cross-module isolation

Importing your model's driver must NOT also import any other model's
driver.

```python
import sys

def test_<model>_driver_imports_dont_touch_<other_model>():
    """Loading drivers.<model>.* must not also load <other_model>'s
    flat script."""
    # Clean any pre-existing imports
    for name in list(sys.modules):
        if '<other_model>_rolling' in name:
            del sys.modules[name]
    for name in list(sys.modules):
        if name.startswith('drivers.<model>'):
            del sys.modules[name]

    import drivers.<model>                       # noqa: F401
    import drivers.<model>.config                # noqa: F401
    import drivers.<model>.plots                 # noqa: F401

    leaked = [n for n in sys.modules
              if '<other_model>_rolling' in n]
    assert not leaked, (
        "drivers.<model>.* leaked an import of <other_model>_rolling: "
        f"{leaked}. The two driver namespaces must stay orthogonal.")
```

In practice, swap `<other_model>` for `fsa_high_res` (the existing
production driver). When more models land, this test checks against
all of them.

### Test 2 — Frozen dataclass

The config dataclass must be `@dataclass(frozen=True)` so that
runtime mutation raises immediately.

```python
import pytest
from dataclasses import is_dataclass, FrozenInstanceError

def test_<model>_config_is_frozen_dataclass():
    from drivers.<model>.config import <MODEL>_<SET>_CONFIG, <Model>RollingConfig

    assert is_dataclass(<Model>RollingConfig)
    assert is_dataclass(<MODEL>_<SET>_CONFIG)

    with pytest.raises((FrozenInstanceError, AttributeError)):
        <MODEL>_<SET>_CONFIG.n_smc = 1024
```

Why frozen: a knob that can be mutated at runtime is no longer the
single source of truth — a downstream caller can change it after the
`driver_config.json` dump, and the saved config no longer matches
the run.

### Test 3 — Dataclass round-trips through JSON

The dataclass must produce a `to_dict()` and `to_json()` representation
that can be losslessly re-parsed. This catches the schema-drift
failure mode (rename a field but forget to update `to_dict`).

```python
import json

def test_<model>_config_serialises_round_trip():
    from drivers.<model>.config import <MODEL>_<SET>_CONFIG

    d = <MODEL>_<SET>_CONFIG.to_dict()
    # Sanity-assert your model's expected values
    assert d['scenario_name'] == '<set_name>'
    assert d['n_days'] * d['bins_per_day'] == <expected_bin_count>
    assert tuple(d['obs_channel_names']) == ('hr', 'sleep', ...)

    s = <MODEL>_<SET>_CONFIG.to_json()
    parsed = json.loads(s)
    assert parsed['scenario_name'] == d['scenario_name']
```

### Test 4 — Driver imports clean (no side effects)

The full rolling-driver module imports without allocating GPU memory,
opening files, or running any setup code. This catches the "I added
a print at module level" or "I created the output dir at import time"
class of bugs that breaks parallel tests.

```python
import importlib
import sys

def test_<model>_rolling_main_module_importable():
    if 'drivers.<model>.rolling' in sys.modules:
        del sys.modules['drivers.<model>.rolling']
    importlib.import_module('drivers.<model>.rolling')
```

If this test starts allocating GPU memory or printing, the driver
has a side effect at module level. Move it into `main()`.

## Running the tests

```bash
PYTHONPATH=. python -m pytest tests/test_<model>_rolling_imports.py \
    tests/test_high_res_fsa.py -v
```

All tests must pass. Wall: ~5 seconds total (most of which is JAX
import).

## When a test fails

| Failure | Likely cause |
|---|---|
| `test_<model>_driver_imports_dont_touch_*` | Some `drivers/<model>/*.py` has a `from drivers.<other>...` line. Find it and refactor — usually you can `from smc2bj...` for the same helper. |
| `test_<model>_config_is_frozen_dataclass` | You forgot `@dataclass(frozen=True)` on the class definition. Add it. |
| `test_<model>_config_serialises_round_trip` | A field rename or type change broke `to_dict()`. Either fix the field or update the test's expected values. |
| `test_<model>_rolling_main_module_importable` | Your `rolling.py` runs something at import time (GPU init, file IO, or a stray `print`). Move it into `main()`. |

## Adding tests for future invariants

When you find a new failure mode (e.g. "config A and config B have
the same scenario_name, leading to output dir collision"), add a
test for it here. The test file is the live audit trail of what
modularity invariants we know about.

## Reference

- [tests/test_swat_rolling_imports.py](../tests/test_swat_rolling_imports.py)
  — the canonical example, 4 tests covering all the above.
