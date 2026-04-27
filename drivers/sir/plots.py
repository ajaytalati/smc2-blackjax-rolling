"""SIR-specific diagnostic plots for rolling-window estimation runs.

The generic ``smc2bj.plotting.rolling.plot_parameter_tracking`` and
``plot_coverage_and_timing`` work for any model and are reused by
``rolling.py`` directly (no SIR-specific wrapper needed).

Only the per-channel input plot is model-specific. SIR has 2 obs
channels: Poisson daily case counts + Gaussian weekly serology survey.
"""

from __future__ import annotations

import os

import numpy as np


def _use_agg():
    import matplotlib
    matplotlib.use('Agg')


def plot_sir_channels(bundle: dict, out_dir: str, *,
                       n_show_days: int | None = None) -> str:
    """3-panel diagnostic of the input artifact: trajectory + cases + serology.

    Parameters
    ----------
    bundle : dict from ``drivers/_artifact_loader.load_scenario``
    out_dir : str
    n_show_days : int or None — clip the plot to the first N days for
        readability; None shows the full trial.

    Returns
    -------
    path : str — the saved PNG path
    """
    _use_agg()
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    trajectory = np.asarray(bundle['trajectory'])
    bins_per_day = int(bundle['bins_per_day'])
    n_total = trajectory.shape[0]
    n_show = n_total if n_show_days is None else min(n_total, n_show_days * bins_per_day)
    dt_h = float(bundle['dt_days'])    # arg name is misnomer; SIR's native unit is hours
    t_days = (np.arange(n_show) * dt_h) / 24.0

    obs = bundle['obs_data']
    truth_params = bundle.get('truth_params', {})
    N = float(truth_params.get('N', 763))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # Panel 0: latent S, I, R
    S = trajectory[:n_show, 0]
    I = trajectory[:n_show, 1]
    R = N - S - I
    axes[0].plot(t_days, S, color='steelblue', lw=1.2, label='S (susceptible)')
    axes[0].plot(t_days, I, color='firebrick', lw=1.2, label='I (infected)')
    axes[0].plot(t_days, R, color='seagreen', lw=1.2, label='R (recovered)')
    axes[0].set_ylabel('population')
    axes[0].legend(loc='right', fontsize=9)
    axes[0].grid(True, alpha=0.2)
    R0 = truth_params.get('beta', 0.0) / max(truth_params.get('gamma', 1.0), 1e-6)
    axes[0].set_title(
        f"SIR input artifact / {bundle.get('scenario_name', '?')} — "
        f"N={int(N)}, R₀={R0:.2f}, attack rate ≈ {R[-1]/N:.3f}")

    # Panel 1: cases (Poisson, daily bars)
    cases = obs.get('cases', {})
    if cases and 't_idx' in cases:
        idx = np.asarray(cases['t_idx']).astype(int)
        mask = idx < n_show
        xs = (idx[mask] * dt_h) / 24.0
        ys = np.asarray(cases.get('cases', cases.get('count', [])))[mask].astype(int)
        if len(ys) > 0:
            axes[1].bar(xs, ys, width=0.7, color='firebrick', alpha=0.7,
                         label='Poisson observed cases')
            # Truth expected: ρ β S I / N × 24h
            rho = truth_params.get('rho', 1.0)
            beta = truth_params.get('beta', 0.0)
            bin_h = float(cases.get('bin_hours', 24.0))
            bin_size = max(int(round(bin_h / dt_h)), 1)
            n_bins = n_show // bin_size
            if n_bins > 0:
                truth_expected_per_step = rho * beta * (S * I / N)
                truth_daily = (truth_expected_per_step[: n_bins * bin_size]
                                .reshape(n_bins, bin_size).mean(axis=1) * bin_h)
                # Convert bin start indices to days
                bin_t = (np.arange(n_bins) * bin_size + bin_size - 1) * dt_h / 24.0
                axes[1].plot(bin_t, truth_daily, color='black', lw=1.0,
                             label='E[cases | truth]')
    axes[1].set_ylabel('cases / day')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.2)
    axes[1].set_title('Observation channel 1 — Poisson daily case counts')

    # Panel 2: serology (Gaussian, weekly error-bars)
    sero = obs.get('serology', {})
    axes[2].plot(t_days, I / N, color='firebrick', lw=1.0,
                 label='truth I(t)/N', alpha=0.7)
    if sero and 't_idx' in sero:
        idx = np.asarray(sero['t_idx']).astype(int)
        mask = idx < n_show
        xs_d = (idx[mask] * dt_h) / 24.0
        ys = np.asarray(sero.get('prevalence', sero.get('value', [])))[mask]
        sigma_z = float(truth_params.get('sigma_z', 0.02))
        axes[2].errorbar(xs_d, ys, yerr=2.0 * sigma_z, fmt='o',
                         color='steelblue', capsize=3,
                         label='Gaussian serology survey (±2σ)')
    axes[2].set_xlabel('time (days)')
    axes[2].set_ylabel('prevalence')
    axes[2].legend(loc='upper right', fontsize=9)
    axes[2].grid(True, alpha=0.2)
    axes[2].set_title('Observation channel 2 — Gaussian weekly serology')

    plt.tight_layout()
    path = os.path.join(out_dir, 'sir_channels.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    return path


__all__ = ["plot_sir_channels"]
