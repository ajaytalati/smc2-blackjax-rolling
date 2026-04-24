"""Diagnostic plot for high_res_FSA proof-of-principle simulations.

Called by the generic simulator's plot_fn hook — shows a multi-day panel
of latent B/F/A, the circadian C(t) forcing, the exogenous Phi burst
schedule, and the four observed channels (HR, sleep, stress, steps) on
a shared hourly axis.
"""

import os

import numpy as np


def _use_agg():
    import matplotlib
    matplotlib.use('Agg')


def plot_fsa_high_res(trajectory, t_grid, channel_outputs, params, save_dir):
    """Per-model plot invoked by smc2bj.simulator.sde_solver.

    Parameters
    ----------
    trajectory : np.ndarray, shape (n_bins, 3)  — B, F, A
    t_grid     : np.ndarray, shape (n_bins,)   — time in days
    channel_outputs : dict of channel dicts (sleep_label, obs_value per channel)
    params     : dict of true parameters
    save_dir   : output directory
    """
    _use_agg()
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    n_bins = len(t_grid)
    # Show at most first 3 days to keep the plot readable
    n_show = min(n_bins, 3 * 96)
    t_hours = t_grid[:n_show] * 24.0

    B = trajectory[:n_show, 0]
    F = trajectory[:n_show, 1]
    A = trajectory[:n_show, 2]
    C = np.cos(2.0 * np.pi * t_grid[:n_show] + params.get('phi', 0.0))

    # Pull per-channel obs, padded by nans at non-observed bins
    def _pad(ch_name, value_key, n):
        out = np.full(n, np.nan)
        ch = channel_outputs.get(ch_name)
        if ch is None:
            return out
        t_idx = np.asarray(ch.get('t_idx', [])).astype(int)
        mask = t_idx < n
        idx = t_idx[mask]
        if value_key in ch:
            out[idx] = np.asarray(ch[value_key])[mask]
        elif 'sleep_label' in ch:
            out[idx] = np.asarray(ch['sleep_label'])[mask]
        return out

    hr = _pad('obs_HR', 'obs_value', n_show)
    sleep = _pad('obs_sleep', 'sleep_label', n_show)
    stress = _pad('obs_stress', 'obs_value', n_show)
    steps = _pad('obs_steps', 'obs_value', n_show)

    phi_input = None
    if 'Phi' in channel_outputs:
        phi_input = np.asarray(channel_outputs['Phi']['Phi_value'])[:n_show]

    fig, axes = plt.subplots(7, 1, figsize=(14, 14), sharex=True)

    axes[0].plot(t_hours, B, color='steelblue', lw=1.0, label='B')
    axes[0].plot(t_hours, F, color='firebrick', lw=1.0, label='F')
    axes[0].plot(t_hours, A, color='darkgreen', lw=1.0, label='A')
    axes[0].set_ylabel('latent')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.2)
    axes[0].set_title(f'FSA high-res proof-of-principle — first {n_show//96} day(s)')

    axes[1].plot(t_hours, C, color='purple', lw=0.8)
    axes[1].axhline(0, color='grey', ls=':', alpha=0.5)
    axes[1].set_ylabel('C(t) [circadian]')
    axes[1].grid(True, alpha=0.2)

    if phi_input is not None:
        axes[2].plot(t_hours, phi_input, color='darkorange', lw=0.8)
        axes[2].set_ylabel(r'$\Phi(t)$')
        axes[2].grid(True, alpha=0.2)

    axes[3].scatter(t_hours, hr, s=4, color='#e74c3c', alpha=0.6)
    axes[3].set_ylabel('HR (sleep only)')
    axes[3].grid(True, alpha=0.2)

    axes[4].scatter(t_hours, sleep, s=4, color='#1abc9c', alpha=0.6)
    axes[4].set_ylabel('sleep label')
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].grid(True, alpha=0.2)

    axes[5].scatter(t_hours, stress, s=4, color='#9b59b6', alpha=0.6)
    axes[5].set_ylabel('stress (wake only)')
    axes[5].grid(True, alpha=0.2)

    axes[6].scatter(t_hours, steps, s=4, color='#3498db', alpha=0.6)
    axes[6].set_ylabel('steps (wake only)')
    axes[6].set_xlabel('hours')
    axes[6].grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fsa_high_res_sim.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")
