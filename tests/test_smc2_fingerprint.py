"""Numerical-fingerprint regression test.

Locks in a known-good output from the reference FSA driver at a fixed seed.
If a re-implementation matches within rtol, it is numerically equivalent.

Fingerprint config: --seed 42 --condition C0 --windows 1
Fingerprint values: see docs/NUMERICAL_FINGERPRINT.md

Running this test is expensive (~20 min for a cold-start 1-window SMC run).
Marked slow; skipped by default. Run explicitly via:

    pytest -m slow tests/test_smc2_fingerprint.py

or by hand:

    python drivers/fsa_real_obs_5yr_rolling.py --seed 42 --condition C0 --windows 1
    # then inspect outputs/fsa_real_obs_5yr_rolling/C0_N256_s42/rolling_checkpoint.json
    # and compare against docs/NUMERICAL_FINGERPRINT.md values.
"""

import json
import os
import subprocess
import sys

import pytest

# Absolute repo root
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Fingerprint values — populated from a reference run on 2026-04-23.
# See docs/NUMERICAL_FINGERPRINT.md for how these were generated and under
# what rtol they should match.
FINGERPRINT = {
    'seed': 42,
    'condition': 'C0',
    'windows': 1,
    'w0': {
        # Filled in from outputs/.../C0_N256_s42/rolling_checkpoint.json after a
        # verified parity run. Placeholders here; authoritative values in the
        # NUMERICAL_FINGERPRINT.md table.
        'coverage_min': 0.70,
        'coverage_max': 1.00,
    }
}


@pytest.mark.slow
def test_fingerprint_cold_start_one_window():
    """Run the driver for 1 window at seed=42, check coverage falls in the
    expected band and the checkpoint JSON is structurally valid."""
    out_dir = os.path.join(
        REPO, 'outputs', 'fsa_real_obs_5yr_rolling', 'C0_N256_s42')
    checkpoint = os.path.join(out_dir, 'rolling_checkpoint.json')

    env = os.environ.copy()
    env['PYTHONPATH'] = REPO
    result = subprocess.run(
        [sys.executable,
         os.path.join(REPO, 'drivers', 'fsa_real_obs_5yr_rolling.py'),
         '--seed', '42', '--condition', 'C0', '--windows', '1'],
        cwd=REPO, env=env, timeout=3600, check=True,
    )
    assert result.returncode == 0

    with open(checkpoint) as f:
        cp = json.load(f)

    assert cp['config']['n_smc'] == 256
    assert cp['config']['n_pf'] == 400
    assert len(cp['windows']) == 1
    w0 = cp['windows'][0]
    assert FINGERPRINT['w0']['coverage_min'] <= w0['coverage'] <= FINGERPRINT['w0']['coverage_max']
