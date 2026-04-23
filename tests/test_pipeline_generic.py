"""Shape and behaviour tests for the generic pipeline utilities."""

import numpy as np
import pytest


def test_extract_window_reindexes_t_idx():
    from smc2bj.pipeline.windowing import extract_window
    obs = {
        'obs_x': {
            't_idx': np.array([0, 5, 15, 25, 35, 60]),
            'obs_value': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        }
    }
    win = extract_window(obs, start=10, end=40)
    assert list(win['obs_x']['t_idx']) == [5, 15, 25]   # 15,25,35 - 10
    assert list(win['obs_x']['obs_value']) == [3.0, 4.0, 5.0]


def test_missing_data_masks_each_channel_group():
    from smc2bj.pipeline.missing_data import apply_missing_data
    from smc2bj.estimation.config import MissingDataConfig

    n_days = 200
    t_idx = np.arange(n_days, dtype=np.int32)
    obs = {
        ch: {
            't_idx': t_idx.copy(),
            'obs_value': np.zeros(n_days, dtype=np.float32),
        }
        for ch in ('obs_A', 'obs_B', 'obs_C')
    }
    cfg = MissingDataConfig(
        dropout_rate=0.5,
        broken_watch_days=10,
        rest_days_per_week=(2, 3),
        active_channels=('obs_A',),
        passive_channels=('obs_B',),
        all_obs_channels=('obs_A', 'obs_B', 'obs_C'),
    )
    obs_masked = apply_missing_data(obs, n_days, cfg, seed=42, verbose=False)
    # All three channels should have strictly fewer entries after masking
    for ch in ('obs_A', 'obs_B', 'obs_C'):
        assert len(obs_masked[ch]['t_idx']) < n_days


def test_sample_from_prior_shape():
    import jax
    from smc2bj.estimation.sampling import sample_from_prior

    # Fake T_arr matching the (is_ln / is_norm) schema
    T_arr = {
        'is_ln':    np.array([1, 0, 1], dtype=np.float32),
        'is_norm':  np.array([0, 1, 0], dtype=np.float32),
        'ln_mu':    np.array([0.0, 0.0, 1.0], dtype=np.float32),
        'ln_sigma': np.array([0.5, 0.0, 0.3], dtype=np.float32),
        'n_mu':     np.array([0.0, 2.0, 0.0], dtype=np.float32),
        'n_sigma':  np.array([0.0, 0.5, 0.0], dtype=np.float32),
    }
    key = jax.random.PRNGKey(0)
    particles = sample_from_prior(100, T_arr, 3, key)
    assert particles.shape == (100, 3)


def test_estimate_mass_matrix_shape():
    import jax.numpy as jnp
    from smc2bj.estimation.mass_matrix import estimate_mass_matrix
    parts = jnp.ones((50, 7))
    mm = estimate_mass_matrix(parts)
    assert mm.shape == (1, 7)
