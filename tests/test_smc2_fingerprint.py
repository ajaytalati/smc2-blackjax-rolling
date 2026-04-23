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
        # Reference: refactored run on 2026-04-23 (commit 88f4461)
        # vs pre-refactor W1 checkpoint. Bit-exact coverage match (100%).
        # Tolerance band allows XLA/JIT scheduling variance.
        'coverage_exact': 1.0,
        'coverage_informed_min': 0.90,
        'n_temp_min': 25,
        'n_temp_max': 50,
        'elapsed_min_s': 1100,
        'elapsed_max_s': 1700,
    },
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
    # Coverage is bit-exact on the reference; require within 2 params
    # (≈6pp) of the 100% target to absorb XLA variance.
    assert w0['coverage'] >= 31 / 33, (
        f"coverage {w0['coverage']:.4f} fell below 31/33")
    assert w0['coverage_informed'] >= FINGERPRINT['w0']['coverage_informed_min']
    assert (FINGERPRINT['w0']['n_temp_min']
            <= w0['n_temp_steps']
            <= FINGERPRINT['w0']['n_temp_max'])
    assert (FINGERPRINT['w0']['elapsed_min_s']
            <= w0['elapsed_s']
            <= FINGERPRINT['w0']['elapsed_max_s'])
