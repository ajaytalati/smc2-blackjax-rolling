"""Protocol test: does the FSA reference model satisfy the EstimationModel contract?

Minimal structural + callable check. Does not exercise numerics; for that see
test_smc2_fingerprint.py.
"""

import pytest


def test_fsa_model_loads():
    from models.fsa_real_obs.estimation import FSA_REAL_OBS_ESTIMATION
    m = FSA_REAL_OBS_ESTIMATION
    assert m.n_dim > 0
    assert m.n_params == len(m.param_prior_config)
    assert m.n_init_states == len(m.init_state_prior_config)
    assert len(m.all_names) == m.n_dim


def test_fsa_model_priors_well_formed():
    from models.fsa_real_obs.estimation import FSA_REAL_OBS_ESTIMATION
    for name, (dist, args) in FSA_REAL_OBS_ESTIMATION.param_prior_config.items():
        assert dist in ('lognormal', 'normal'), f"{name}: unsupported dist {dist}"
        assert len(args) == 2, f"{name}: prior args should be (mu, sigma)"


def test_fsa_model_callbacks_exist():
    from models.fsa_real_obs.estimation import FSA_REAL_OBS_ESTIMATION
    m = FSA_REAL_OBS_ESTIMATION
    assert callable(m.propagate_fn)
    assert callable(m.diffusion_fn)
    assert callable(m.obs_log_weight_fn)
    assert callable(m.align_obs_fn)
    assert callable(m.shard_init_fn)


def test_fsa_sdemodel_loads():
    from models.fsa_real_obs.simulation import FSA_REAL_OBS_MODEL
    m = FSA_REAL_OBS_MODEL
    assert m.name == 'fsa_real_obs'
    assert len(m.states) > 0
    assert len(m.channels) > 0
    assert 'recovery' in m.param_sets
    assert 'recovery' in m.init_states


def test_cold_start_init_shape():
    import numpy as np
    from models.fsa_real_obs.estimation import COLD_START_INIT
    arr = np.asarray(COLD_START_INIT)
    assert arr.shape == (3,), f"expected (3,) for FSA B/F/A, got {arr.shape}"
