"""models/fsa_real_obs/sim_plots.py — FSA Real-Obs Diagnostic Plots.

Date:    19 April 2026
Version: 1.0

Produces:
  1. latent_states.png   — B(t), F(t), A(t), mu(t), inputs (same as base FSA)
  2. landscape_evolution.png — V_A(A) at 5 snapshots (same as base FSA)
  3. observations.png    — 6-panel: RHR, I_norm, D_norm, Stress, Sleep, Timing
"""

import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_fsa_real_obs(trajectory, t_grid, channel_outputs, params, save_dir):
    """Entry point called by the simulator."""
    os.makedirs(save_dir, exist_ok=True)

    T_B_arr = _channel_arr(channel_outputs, 'T_B', 'T_B_value', len(t_grid))
    Phi_arr = _channel_arr(channel_outputs, 'Phi', 'Phi_value', len(t_grid))

    p1 = os.path.join(save_dir, "latent_states.png")
    _plot_latent(trajectory, t_grid, T_B_arr, Phi_arr, params, p1)
    print(f"  Plot: {p1}")

    p2 = os.path.join(save_dir, "landscape_evolution.png")
    _plot_landscape(trajectory, t_grid, params, p2)
    print(f"  Plot: {p2}")

    p3 = os.path.join(save_dir, "observations.png")
    _plot_observations(trajectory, t_grid, channel_outputs, params, p3)
    print(f"  Plot: {p3}")


# -------------------------------------------------------------------------
# Panel A — latent trajectories (same as base FSA)
# -------------------------------------------------------------------------

def _plot_latent(trajectory, t_grid, T_B_arr, Phi_arr, params, save_path):
    B = trajectory[:, 0]; F = trajectory[:, 1]; A = trajectory[:, 2]
    mu = _mu(B, F, params)
    A_star = np.where(mu > 0, np.sqrt(np.maximum(mu, 0) / params['eta']),
                      np.nan)

    fig, ((ax_BF, ax_A), (ax_mu, ax_in)) = plt.subplots(
        2, 2, figsize=(13, 8), sharex=True)

    ax_BF.plot(t_grid, B, color='steelblue', label='B (fitness)', lw=1.2)
    ax_BF.set_ylabel('B', color='steelblue')
    ax_BF.set_ylim(-0.05, 1.05)
    ax_BF.tick_params(axis='y', labelcolor='steelblue')
    ax_BF_r = ax_BF.twinx()
    ax_BF_r.plot(t_grid, F, color='firebrick', label='F (strain)', lw=1.2)
    ax_BF_r.set_ylabel('F', color='firebrick')
    ax_BF_r.tick_params(axis='y', labelcolor='firebrick')
    ax_BF.set_title('Fitness and Strain')
    ax_BF.grid(True, alpha=0.3)

    ax_A.plot(t_grid, A, color='darkgreen', lw=1.0, label='A')
    ax_A.plot(t_grid, A_star, color='darkorange', ls='--', lw=1.0,
              alpha=0.7, label=r'$A^* = \sqrt{\mu/\eta}$')
    ax_A.axhline(0, color='grey', alpha=0.4)
    ax_A.set_ylabel('A')
    ax_A.set_title('HPG Pulsatility Amplitude')
    ax_A.legend(loc='upper left', fontsize=9)
    ax_A.grid(True, alpha=0.3)

    ax_mu.plot(t_grid, mu, color='purple', lw=1.2, label=r'$\mu(B,F)$')
    ax_mu.axhline(0, color='black', ls='--', alpha=0.6)
    ax_mu.fill_between(t_grid, 0, mu, where=(mu < 0),
                       color='firebrick', alpha=0.15)
    ax_mu.fill_between(t_grid, 0, mu, where=(mu >= 0),
                       color='forestgreen', alpha=0.15)
    ax_mu.set_ylabel(r'$\mu(B,F)$')
    ax_mu.set_xlabel('Time (days)')
    ax_mu.set_title('Bifurcation Parameter')
    ax_mu.grid(True, alpha=0.3)

    ax_in.plot(t_grid, T_B_arr, color='navy', lw=1.3, drawstyle='steps-post',
               label=r'$T_B(t)$')
    ax_in_r = ax_in.twinx()
    ax_in_r.plot(t_grid, Phi_arr, color='darkorange', lw=1.3,
                 drawstyle='steps-post', label=r'$\Phi(t)$')
    ax_in.set_ylabel(r'$T_B$', color='navy')
    ax_in_r.set_ylabel(r'$\Phi$', color='darkorange')
    ax_in.tick_params(axis='y', labelcolor='navy')
    ax_in_r.tick_params(axis='y', labelcolor='darkorange')
    ax_in.set_xlabel('Time (days)')
    ax_in.set_title('Clinician Inputs')
    ax_in.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------
# Panel B — landscape V_A(A) snapshots (same as base FSA)
# -------------------------------------------------------------------------

def _plot_landscape(trajectory, t_grid, params, save_path):
    B = trajectory[:, 0]; F = trajectory[:, 1]
    n_snap = 5
    idxs = np.linspace(0, len(t_grid) - 1, n_snap).astype(int)
    A_range = np.linspace(0.0, 1.2, 400)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    cmap = plt.cm.viridis
    for i, idx in enumerate(idxs):
        mu_i = _mu_scalar(B[idx], F[idx], params)
        V = -0.5 * mu_i * A_range**2 + 0.25 * params['eta'] * A_range**4
        color = cmap(i / max(n_snap - 1, 1))
        ax.plot(A_range, V, color=color, lw=1.3,
                label=f't={t_grid[idx]:.1f}d  $\\mu$={mu_i:+.3f}')
    ax.axhline(0, color='grey', alpha=0.3)
    ax.set_xlabel('A')
    ax.set_ylabel(r'$V_A(A)$')
    ax.set_title('Landau Potential Evolution')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------
# Panel C — 6 observation channels
# -------------------------------------------------------------------------

_CHANNEL_DEFS = [
    ('obs_RHR',       'RHR (bpm)',         'Ch1: Resting Heart Rate',
     lambda B, F, A, p: p['R_base'] - p['kappa_vagal']*B + p['kappa_chronic']*F),
    ('obs_intensity', r'$I_{norm}$',       'Ch2: Performance Intensity',
     lambda B, F, A, p: p['I_base'] + p['c_B']*B - p['c_F']*F),
    ('obs_duration',  r'$D_{norm}$',       'Ch3: Duration',
     lambda B, F, A, p: p['D_base'] + p['d_B']*B - p['d_F']*F),
    ('obs_stress',    'Stress',            'Ch4: Daily Stress',
     lambda B, F, A, p: p['S_base'] - p['s_A']*A + p['s_F']*F),
    ('obs_sleep',     r'$Sleep_{norm}$',   'Ch5: Sleep Quality',
     lambda B, F, A, p: p['Sleep_base'] + p['sl_A']*A + p['sl_B']*B - p['sl_F']*F),
    ('obs_timing',    r'$Time_{logit}$',   'Ch6: Circadian Timing',
     lambda B, F, A, p: p['Time_base'] + p['t_A']*A - p['t_F']*F),
]


def _plot_observations(trajectory, t_grid, channel_outputs, params, save_path):
    B = trajectory[:, 0]; F = trajectory[:, 1]; A = trajectory[:, 2]
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    for i, (ch_name, ylabel, title, h_fn) in enumerate(_CHANNEL_DEFS):
        ax = axes[i]
        # True predicted signal (noiseless)
        true_signal = h_fn(B, F, A, params)
        ax.plot(t_grid, true_signal, color=colors[i], lw=1.0,
                alpha=0.8, label='true signal')

        # Noisy observations
        ch = channel_outputs.get(ch_name, {})
        if 't_idx' in ch and 'obs_value' in ch:
            idx = np.asarray(ch['t_idx'])
            y   = np.asarray(ch['obs_value'])
            ax.scatter(t_grid[idx], y, s=5, alpha=0.3, color='black',
                       label='obs', zorder=3)

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-2].set_xlabel('Time (days)')
    axes[-1].set_xlabel('Time (days)')
    fig.suptitle('FSA Real Observation Channels', fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _mu(B, F, p):
    return p['mu_0'] + p['mu_B'] * B - p['mu_F'] * F - p['mu_FF'] * F * F

def _mu_scalar(B, F, p):
    return float(p['mu_0'] + p['mu_B'] * B - p['mu_F'] * F - p['mu_FF'] * F * F)

def _channel_arr(channel_outputs, ch_name, value_key, default_len):
    ch = channel_outputs.get(ch_name, {})
    if value_key in ch:
        return np.asarray(ch[value_key])
    return np.zeros(default_len, dtype=np.float32)
