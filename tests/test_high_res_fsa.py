"""Protocol + simulation smoke tests for models/fsa_high_res/."""

import os

import numpy as np
import pytest


def test_high_res_fsa_model_loads():
    from models.fsa_high_res.estimation import HIGH_RES_FSA_ESTIMATION
    m = HIGH_RES_FSA_ESTIMATION
    assert m.n_dim == 29, f"expected 29 params, got {m.n_dim}"
    assert m.n_params == 29
    assert m.n_init_states == 0, "init states frozen, not estimated"
    assert m.name == 'fsa_high_res'


def test_high_res_fsa_callbacks_exist():
    from models.fsa_high_res.estimation import HIGH_RES_FSA_ESTIMATION
    m = HIGH_RES_FSA_ESTIMATION
    for fn_name in ('propagate_fn', 'diffusion_fn', 'obs_log_weight_fn',
                    'align_obs_fn', 'shard_init_fn'):
        assert callable(getattr(m, fn_name)), f"{fn_name} not callable"


def test_high_res_fsa_priors_well_formed():
    from models.fsa_high_res.estimation import HIGH_RES_FSA_ESTIMATION
    for name, (dist, args) in HIGH_RES_FSA_ESTIMATION.param_prior_config.items():
        assert dist in ('lognormal', 'normal'), f"{name}: bad dist {dist}"
        assert len(args) == 2


def test_high_res_fsa_frozen_params():
    from models.fsa_high_res.estimation import HIGH_RES_FSA_ESTIMATION
    frozen = HIGH_RES_FSA_ESTIMATION.frozen_params
    assert 'sigma_B' in frozen
    assert 'sigma_F' in frozen
    assert 'sigma_A' in frozen
    assert 'phi' in frozen
    assert frozen['phi'] == 0.0


def test_high_res_sdemodel_loads():
    from models.fsa_high_res.simulation import HIGH_RES_FSA_MODEL
    m = HIGH_RES_FSA_MODEL
    assert m.name == 'fsa_high_res'
    assert len(m.states) == 3
    # 4 obs channels + 2 exogenous
    obs_ch = [ch for ch in m.channels if ch.name.startswith('obs_')]
    assert len(obs_ch) == 4


def test_generate_phi_sub_daily_shape_and_zero_during_sleep():
    from models.fsa_high_res.simulation import (
        generate_phi_sub_daily, BINS_PER_DAY, DT_BIN_HOURS,
    )
    daily_phi = np.array([0.1, 0.1, 0.1])
    phi_arr = generate_phi_sub_daily(daily_phi, seed=42)
    assert phi_arr.shape == (3 * BINS_PER_DAY,)
    # First few bins (0:00-06:00) should be 0 (sleep hours)
    for k in range(int(6 / DT_BIN_HOURS)):   # bins for 0-6am
        assert phi_arr[k] == 0.0, f"bin {k} (hour {k*DT_BIN_HOURS}) should be 0"


def test_circadian_period_and_values():
    from models.fsa_high_res.simulation import circadian
    import numpy as np
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    C = circadian(t, phi=0.0)
    # midnight peak, noon trough
    assert abs(C[0] - 1.0) < 1e-6
    assert abs(C[2] - (-1.0)) < 1e-6
    # periodic
    assert abs(C[4] - 1.0) < 1e-6


def test_align_obs_fn_output_shapes():
    import jax.numpy as jnp
    from models.fsa_high_res.estimation import align_obs_fn
    n_bins = 288
    obs = {
        'obs_HR':     {'t_idx': np.array([10, 20]), 'obs_value': np.array([55.0, 60.0])},
        'obs_sleep':  {'t_idx': np.arange(n_bins),
                        'sleep_label': np.zeros(n_bins, dtype=np.int32)},
        'obs_stress': {'t_idx': np.array([40]), 'obs_value': np.array([25.0])},
        'obs_steps':  {'t_idx': np.array([50]), 'obs_value': np.array([500.0])},
        'T_B':        {'t_idx': np.arange(n_bins),
                        'T_B_value': np.full(n_bins, 0.6, dtype=np.float32)},
        'Phi':        {'t_idx': np.arange(n_bins),
                        'Phi_value': np.full(n_bins, 0.03, dtype=np.float32)},
    }
    grid = align_obs_fn(obs, n_bins, dt=1.0 / 96.0)
    for key in ('hr_value', 'hr_present', 'stress_value', 'stress_present',
                'log_steps_value', 'steps_present', 'sleep_label',
                'sleep_present', 'T_B', 'Phi', 'C', 'has_any_obs'):
        assert key in grid, f"missing key {key}"
        assert grid[key].shape == (n_bins,), f"{key} wrong shape {grid[key].shape}"
    # C should span [-1, 1] over 3 days
    assert abs(float(grid['C'][0]) - 1.0) < 1e-5
    # log(500+1) ≈ 6.22
    assert abs(float(grid['log_steps_value'][50]) - np.log(501.0)) < 1e-3


def test_propagate_fn_finite_and_jittable():
    import os
    os.environ.setdefault('JAX_ENABLE_X64', 'True')
    import jax
    import jax.numpy as jnp
    from models.fsa_high_res.estimation import (
        HIGH_RES_FSA_ESTIMATION, propagate_fn, align_obs_fn, diffusion_fn,
    )
    n = 288
    obs = {
        'obs_HR':     {'t_idx': np.array([10]), 'obs_value': np.array([55.0])},
        'obs_sleep':  {'t_idx': np.arange(n),
                        'sleep_label': np.zeros(n, dtype=np.int32)},
        'obs_stress': {'t_idx': np.array([40]), 'obs_value': np.array([25.0])},
        'obs_steps':  {'t_idx': np.array([50]), 'obs_value': np.array([500.0])},
        'T_B':        {'t_idx': np.arange(n),
                        'T_B_value': np.full(n, 0.6, dtype=np.float32)},
        'Phi':        {'t_idx': np.arange(n),
                        'Phi_value': np.full(n, 0.03, dtype=np.float32)},
    }
    grid = align_obs_fn(obs, n, dt=1.0 / 96.0)
    truth = HIGH_RES_FSA_ESTIMATION.get_init_theta_fn()
    params = jnp.array(truth, dtype=jnp.float64)
    y = jnp.array([0.05, 0.10, 0.01])
    sigma = diffusion_fn(params)
    noise = jnp.array([0.1, 0.05, 0.02])
    dt = 1.0 / 96.0 / 4.0
    y_new, lw = propagate_fn(y, 0.0, dt, params, grid, 10, sigma, noise, None)
    assert jnp.all(jnp.isfinite(y_new))
    assert jnp.isfinite(lw)
    # JIT
    fn = jax.jit(propagate_fn, static_argnums=())
    y_j, lw_j = fn(y, 0.0, dt, params, grid, 10, sigma, noise, None)
    assert jnp.all(jnp.isfinite(y_j))
    assert jnp.isfinite(lw_j)
