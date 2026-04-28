"""SWAT-specific diagnostic plots for rolling-window estimation runs.

The generic ``smc2bj.plotting.rolling.plot_parameter_tracking`` and
``plot_coverage_and_timing`` work for any model and are reused by
``rolling.py`` directly (no SWAT-specific wrapper needed).

Only the per-channel input plot is model-specific because SWAT's 4
obs channels (HR Gaussian, sleep 3-level ordinal, steps Poisson,
stress Gaussian) have unique structure that no generic plotter
captures.
"""

from __future__ import annotations

import os

import numpy as np


def _use_agg():
    import matplotlib
    matplotlib.use('Agg')


def plot_swat_channels(bundle: dict, out_dir: str, *,
                        n_show_days: int = 3) -> str:
    """5-panel diagnostic of the input artifact: trajectory + 4 obs channels.

    Parameters
    ----------
    bundle : dict from ``drivers/_artifact_loader.load_scenario``
    out_dir : str
    n_show_days : int — clip the plot to the first N days for readability

    Returns
    -------
    path : str — the saved PNG path
    """
    _use_agg()
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    trajectory = np.asarray(bundle['trajectory'])
    bins_per_day = int(bundle['bins_per_day'])
    n_show = min(trajectory.shape[0], n_show_days * bins_per_day)
    dt_h = float(bundle['dt_days'])    # SWAT: dt_days is misnomer for hours
    t_hours = np.arange(n_show) * dt_h

    obs = bundle['obs_data']

    def _scatter_pad(ax, ch_name, value_key, color, ylabel,
                      cast_int=False, ylim=None, fallback_key=None):
        ch = obs.get(ch_name, {})
        if ch and 't_idx' in ch:
            idx = np.asarray(ch['t_idx']).astype(int)
            mask = idx < n_show
            xs = idx[mask] * dt_h
            key = value_key if value_key in ch else fallback_key
            if key is not None and key in ch:
                ys = np.asarray(ch[key])[mask]
                if cast_int:
                    ys = ys.astype(np.int32)
                ax.scatter(xs, ys, s=3, color=color, alpha=0.6)
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.2)

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

    # Panel 0: latent W, Zt, T (the three most-informative states)
    axes[0].plot(t_hours, trajectory[:n_show, 0], color='steelblue', lw=0.8, label='W')
    axes[0].plot(t_hours, trajectory[:n_show, 1] / 6.0, color='firebrick',
                 lw=0.8, label='Zt/6')
    axes[0].plot(t_hours, trajectory[:n_show, 3], color='darkgreen',
                 lw=0.8, label='T')
    axes[0].set_ylabel('latent (W, Zt/6, T)')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_title(
        f"SWAT input artifact / {bundle.get('scenario_name', '?')} — "
        f"first {n_show // bins_per_day} day(s)")

    # Panel 1: HR (Gaussian)
    _scatter_pad(axes[1], 'hr', 'hr_value', '#e74c3c', 'HR (bpm)')

    # Panel 2: sleep (3-level ordinal: 0=wake, 1=light+REM, 2=deep)
    _scatter_pad(axes[2], 'sleep', 'sleep_level', '#1abc9c', 'sleep level',
                  cast_int=True, ylim=(-0.2, 2.2))

    # Panel 3: steps (Poisson, sparse on 15-min bins)
    _scatter_pad(axes[3], 'steps', 'steps', '#3498db', 'steps / 15-min bin',
                  cast_int=True)

    # Panel 4: stress (Gaussian)
    _scatter_pad(axes[4], 'stress', 'stress_score', '#9b59b6', 'stress score')
    axes[4].set_xlabel('hours')

    plt.tight_layout()
    path = os.path.join(out_dir, 'swat_channels.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    return path


__all__ = ["plot_swat_channels"]
