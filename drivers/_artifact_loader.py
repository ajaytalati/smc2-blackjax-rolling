"""Load a Python-Model-Scenario-Simulation scenario artifact.

Bridges this repo to the validated-scenario format produced by
https://github.com/ajaytalati/Python-Model-Scenario-Simulation .

The loader returns a bundle in the same in-memory shape that
``drivers/fsa_high_res_rolling.py`` builds inline, so the rolling-window
driver downstream is unaware of where the data came from.
"""

from __future__ import annotations

import os
import sys

# Until psim is pip-installable from PyPI, point sys.path at the
# locally-cloned scenario-simulation checkout.
_PSIM_ROOT = os.path.expanduser("~/Repos/Python-Model-Scenario-Simulation")
if _PSIM_ROOT not in sys.path:
    sys.path.insert(0, _PSIM_ROOT)

from psim.io.format import read_artifact   # noqa: E402


def load_scenario(artifact_dir: str) -> dict:
    """Read a scenario artifact and merge obs + exogenous into one dict.

    Returns
    -------
    dict with keys:
        obs_data    : dict[channel_name -> {'t_idx', '<value_key>', ...}]
                      Includes both observation channels (obs_HR, obs_sleep,
                      obs_stress, obs_steps) AND exogenous channels (T_B, Phi,
                      C). The driver's downstream consumers (extract_window,
                      rolling_window_smc) treat these uniformly.
        trajectory  : (n_bins, n_state) float32 ndarray
        truth_params: dict[str -> float]  (simulator's keyword form)
        T_B_arr     : (n_bins,) float32   (convenience for plotters)
        Phi_arr     : (n_bins,) float32
        n_bins_total, dt_days, bins_per_day, seed, state_names,
        scenario_name, model_name, model_version
    """
    a = read_artifact(artifact_dir)
    m = a["manifest"]

    obs_data = dict(a["obs_channels"])
    obs_data.update(a["exogenous_channels"])

    T_B_arr = a["exogenous_channels"]["T_B"]["T_B_value"]
    Phi_arr = a["exogenous_channels"]["Phi"]["Phi_value"]

    return dict(
        obs_data=obs_data,
        trajectory=a["trajectory"],
        truth_params=dict(m.truth_params),
        T_B_arr=T_B_arr,
        Phi_arr=Phi_arr,
        n_bins_total=m.n_bins_total,
        dt_days=m.dt_days,
        bins_per_day=m.bins_per_day,
        seed=m.seed,
        state_names=list(m.state_names),
        scenario_name=m.scenario_name,
        model_name=m.model_name,
        model_version=m.model_version,
    )


__all__ = ["load_scenario"]
