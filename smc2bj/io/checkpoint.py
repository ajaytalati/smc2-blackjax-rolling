"""JSON checkpoint save/show for rolling SMC² runs."""

from __future__ import annotations

import json
import os
from typing import Dict

import numpy as np

from smc2bj.estimation.config import SMCConfig, RollingConfig


def save_checkpoint(results, out_dir: str, truth: Dict[str, float],
                    smc_cfg: SMCConfig, rolling_cfg: RollingConfig):
    """Write a JSON snapshot of all completed windows to ``out_dir``."""
    cp = {
        'truth': {k: float(v) for k, v in truth.items()},
        'config': {
            'n_smc': smc_cfg.n_smc_particles,
            'n_pf': smc_cfg.n_pf_particles,
            'window_days': rolling_cfg.window_days,
            'stride_days': rolling_cfg.stride_days,
            'warm_method': 'gaussian_base_bridge',
            'pf_version': 'v3-lite',
            'ot_max_weight': smc_cfg.ot_max_weight,
            'ot_ess_frac': smc_cfg.ot_ess_frac,
        },
        'windows': [],
    }
    for r in results:
        cp['windows'].append({
            'window': r['window'],
            'start_day': r['start_day'],
            'end_day': r['end_day'],
            'n_temp_steps': r['n_temp_steps'],
            'elapsed_s': r['elapsed_s'],
            'coverage': r['coverage'],
            'coverage_informed': r.get('coverage_informed', r['coverage']),
            'n_informed': r.get('n_informed', len(r['stats'])),
            'stats': {k: {kk: (bool(vv) if isinstance(vv, (bool, np.bool_))
                                else float(vv))
                           for kk, vv in v.items()}
                       for k, v in r['stats'].items()},
        })
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'rolling_checkpoint.json')
    with open(path, 'w') as f:
        json.dump(cp, f, indent=2)


def show_checkpoint(checkpoint_path: str):
    """Load and display a checkpoint in a compact human-readable form."""
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return

    with open(checkpoint_path) as f:
        cp = json.load(f)

    truth = cp['truth']
    windows = cp['windows']
    config = cp.get('config', {})

    print("=" * 110)
    print(f"  ROLLING WINDOW CHECKPOINT — {len(windows)} windows completed")
    print(f"  Config: N_SMC={config.get('n_smc','?')}, "
          f"K={config.get('n_pf','?')}, "
          f"window={config.get('window_days','?')}d, "
          f"stride={config.get('stride_days','?')}d, "
          f"warm={config.get('warm_method','?')}")
    print("=" * 110)

    for win in windows:
        w = win['window']
        stats = win['stats']
        n_in = sum(1 for s in stats.values() if s['in_ci'])
        n_tot = len(stats)
        cov = n_in / n_tot

        print(f"\n  Window {w+1}: days {win['start_day']}-{win['end_day']}  "
              f"| {win['n_temp_steps']} levels | {win['elapsed_s']:.0f}s | "
              f"coverage {n_in}/{n_tot} = {cov*100:.1f}%")
        print(f"  {'param':<16} {'true':>10} {'mean':>10} {'std':>10} "
              f"{'q05':>10} {'q95':>10} {'CI':>5}")
        print(f"  {'-'*75}")

        for name, s in stats.items():
            tv = truth.get(name, float('nan'))
            tag = 'in' if s['in_ci'] else 'OUT'
            print(f"  {name:<16} {tv:10.4f} {s['mean']:10.4f} "
                  f"{s['std']:10.4f} {s['q05']:10.4f} "
                  f"{s['q95']:10.4f} {tag:>5}")

    coverages = [w['coverage'] for w in windows]
    timings = [w['elapsed_s'] for w in windows]
    print(f"\n{'='*110}")
    print(f"  SUMMARY")
    print(f"  Windows completed: {len(windows)}")
    print(f"  Coverage:  mean={np.mean(coverages)*100:.1f}%  "
          f"min={np.min(coverages)*100:.1f}%  "
          f"max={np.max(coverages)*100:.1f}%")
    print(f"  Timing:    mean={np.mean(timings):.0f}s  "
          f"total={sum(timings)/3600:.1f}h")
    n_pass = sum(1 for c in coverages if c >= 0.7)
    print(f"  PASS (>=70%): {n_pass}/{len(windows)}")
    print(f"{'='*110}")
