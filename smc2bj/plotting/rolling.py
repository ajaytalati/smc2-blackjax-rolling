"""Generic rolling-SMC diagnostic plots.

Not model-specific — rely only on the ``results`` list (as returned by
``smc2bj.pipeline.rolling.rolling_window_smc``), the ``model`` object's
``all_names`` / ``n_dim`` attributes, and the ``truth`` dict keyed by
parameter name.

For model-specific plots (e.g. FSA latent reconstruction with B/F/A
state semantics, FSA macrocycle schedule visualisation), keep those in
the scenario driver or in ``models/<name>/sim_plots.py``.
"""

from __future__ import annotations

import os

import numpy as np


def _use_agg():
    import matplotlib
    matplotlib.use('Agg')


def plot_parameter_tracking(results, model, truth, out_dir: str):
    """Posterior mean ± 90% CI across windows, one subplot per parameter.

    Green dashed line is the truth; blue band is the posterior 90% CI.
    Saves ``parameter_tracking.png`` in ``out_dir``.
    """
    _use_agg()
    import matplotlib.pyplot as plt

    mid_days = np.array([(r['start_day'] + r['end_day']) / 2
                         for r in results])
    nd = model.n_dim
    nc = 5
    nr = int(np.ceil(nd / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(nc * 3.5, nr * 2.5))
    axes = np.array(axes).flatten()

    for j, name in enumerate(model.all_names):
        ax = axes[j]
        means = np.array([r['stats'][name]['mean'] for r in results])
        q05s = np.array([r['stats'][name]['q05'] for r in results])
        q95s = np.array([r['stats'][name]['q95'] for r in results])
        ax.fill_between(mid_days, q05s, q95s, alpha=0.3, color='#6b7fd9')
        ax.plot(mid_days, means, color='#6b7fd9', lw=1.2)
        ax.axhline(truth[name], color='#2ca02c', lw=1.5, ls='--', alpha=0.8)
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
        if j >= nd - nc:
            ax.set_xlabel('Day', fontsize=7)
    for j in range(nd, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Parameter Tracking — Rolling Window SMC²\n'
                 '(blue: posterior mean ± 90% CI, green dashed: truth)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(out_dir, 'parameter_tracking.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")


def plot_coverage_and_timing(results, out_dir: str):
    """Per-window coverage, wall-clock, and tempering-level count.

    Saves ``coverage_and_timing.png`` in ``out_dir``.
    """
    _use_agg()
    import matplotlib.pyplot as plt

    n_wins = len(results)
    windows = np.arange(1, n_wins + 1)
    coverages = [r['coverage'] for r in results]
    timings = [r['elapsed_s'] for r in results]
    n_temps = [r['n_temp_steps'] for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax1.bar(windows, coverages, color='#6b7fd9', alpha=0.7)
    ax1.axhline(0.7, color='red', ls='--', alpha=0.5, label='70% threshold')
    ax1.set_ylabel('Coverage')
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)
    ax1.set_title('Parameter Recovery Coverage per Window')

    ax2.bar(windows, timings, color='#2ecc71', alpha=0.7)
    ax2.set_ylabel('Time (s)')
    ax2.set_title('SMC Wall Time per Window')

    ax3.bar(windows, n_temps, color='#f39c12', alpha=0.7)
    ax3.set_ylabel('# Temp Levels')
    ax3.set_xlabel('Window')
    ax3.set_title('Temperature Levels per Window')

    plt.tight_layout()
    path = os.path.join(out_dir, 'coverage_and_timing.png')
    plt.savefig(path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")
