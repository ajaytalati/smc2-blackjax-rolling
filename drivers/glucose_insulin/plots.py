"""glucose_insulin SMC²-driver diagnostic plot for the input artifact.

Generic param-tracking + coverage-and-timing plots are reused from
smc2bj.plotting.rolling. Only the per-channel input plot is model-
specific because glucose_insulin has unique channel structure
(CGM Gaussian + meal-carb Poisson + insulin schedule).
"""

from __future__ import annotations

import os

import numpy as np


def _use_agg():
    import matplotlib
    matplotlib.use('Agg')


def plot_glucose_insulin_channels(bundle: dict, out_dir: str) -> str:
    """4-panel diagnostic of the input artifact: G/X/I + CGM + carbs + insulin schedule."""
    _use_agg()
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    trajectory = np.asarray(bundle['trajectory'])
    bins_per_day = int(bundle['bins_per_day'])
    n_show = trajectory.shape[0]
    dt_h = float(bundle['dt_days'])    # native unit hours (misnomer name)
    t_h = (np.arange(n_show) * dt_h)

    obs = bundle['obs_data']
    truth_params = bundle.get('truth_params', {})
    Gb = float(truth_params.get('Gb', 90.0))

    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)

    # Panel 0: latent G (with CGM scatter overlaid)
    G = trajectory[:, 0]
    axes[0].plot(t_h, G, color='firebrick', lw=1.2, label='truth G(t)', alpha=0.85)
    cgm = obs.get('cgm', {})
    if cgm and 't_idx' in cgm:
        idx = np.asarray(cgm['t_idx']).astype(int)
        mask = idx < n_show
        xs = idx[mask] * dt_h
        ys = np.asarray(cgm.get('cgm_value', cgm.get('value', [])))[mask]
        axes[0].scatter(xs, ys, s=4, color='steelblue', alpha=0.5,
                        label='CGM (Gaussian, 5-min)')
    axes[0].axhline(Gb, color='k', ls='--', alpha=0.4, label=f"Gb={Gb:.0f}")
    axes[0].axhspan(70, 180, alpha=0.08, color='green', label='target range')
    axes[0].set_ylabel('G (mg/dL)')
    R0 = truth_params.get('p3', 0.0) / max(truth_params.get('p2', 1.0), 1e-6)
    axes[0].set_title(
        f"glucose_insulin input artifact / {bundle.get('scenario_name', '?')} — "
        f"SI = p₃/p₂ = {R0:.4f} /(hr·μU/mL)")
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.2)

    # Panel 1: latent X (remote insulin action)
    axes[1].plot(t_h, trajectory[:, 1], color='steelblue', lw=1.0)
    axes[1].set_ylabel('X (1/hr)')
    axes[1].grid(True, alpha=0.2)

    # Panel 2: latent I (plasma insulin)
    Ib = float(truth_params.get('Ib', 7.0))
    axes[2].plot(t_h, trajectory[:, 2], color='seagreen', lw=1.0)
    axes[2].axhline(Ib, color='k', ls='--', alpha=0.4, label=f"Ib={Ib:.1f}")
    axes[2].set_ylabel('I (μU/mL)')
    axes[2].legend(loc='upper right', fontsize=8)
    axes[2].grid(True, alpha=0.2)

    # Panel 3: meal carb counts (Poisson) + insulin schedule (if Set D)
    meals = obs.get('meal_carbs', {})
    if meals and 't_idx' in meals:
        idx = np.asarray(meals['t_idx']).astype(int)
        mask = idx < n_show
        xs = idx[mask] * dt_h
        ys = np.asarray(meals.get('carbs_g', []))[mask].astype(int)
        if len(ys) > 0:
            axes[3].bar(xs, ys, width=0.3, color='darkorange', alpha=0.7,
                         label='observed meal carbs (Poisson, g)')
            for tm in xs:
                axes[3].axvline(tm, color='k', alpha=0.2, lw=0.5)
    axes[3].set_xlabel('time (hours)')
    axes[3].set_ylabel('carbs (g)')
    axes[3].legend(loc='upper right', fontsize=8)
    axes[3].grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out_dir, 'glucose_insulin_channels.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    return path


__all__ = ["plot_glucose_insulin_channels"]
