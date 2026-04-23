"""Example missing-data corruption for wearable/training-log observation data.

Implements three gap patterns:
  1. Rest days (weekly):       mask ``cfg.active_channels``
  2. Random per-channel dropout: mask ``cfg.passive_channels``
  3. Continuous broken-watch gap: mask ``cfg.all_obs_channels``

This is an opinionated default appropriate for an endurance-sport data
stream with RHR/stress/sleep + intensity/duration/timing channels. To
adapt for a different sensor setup, either pass a different
``MissingDataConfig`` or replace this function entirely in your driver.
"""

import numpy as np

from smc2bj.estimation.config import MissingDataConfig


def apply_missing_data(obs_data, n_days: int, cfg: MissingDataConfig,
                       seed: int = 42, verbose: bool = True):
    """Apply rest-day / dropout / broken-watch corruption in place.

    Mutates and returns ``obs_data``.
    """
    rng = np.random.default_rng(seed)

    # 1. Rest days — mask active-measurement channels
    rest_mask = np.ones(n_days, dtype=bool)
    lo, hi = cfg.rest_days_per_week
    for week_start in range(0, n_days, 7):
        week_end = min(week_start + 7, n_days)
        week_len = week_end - week_start
        n_rest = rng.integers(lo, hi + 1)
        n_rest = min(n_rest, week_len)
        rest_days = (rng.choice(week_len, size=n_rest, replace=False)
                     + week_start)
        rest_mask[rest_days] = False

    for ch_name in cfg.active_channels:
        ch = obs_data[ch_name]
        idx = ch['t_idx']
        keep = rest_mask[idx]
        ch['t_idx'] = idx[keep]
        ch['obs_value'] = ch['obs_value'][keep]

    # 2. Random dropout on passive channels
    for ch_name in cfg.passive_channels:
        ch = obs_data[ch_name]
        idx = ch['t_idx']
        keep = rng.random(len(idx)) > cfg.dropout_rate
        ch['t_idx'] = idx[keep]
        ch['obs_value'] = ch['obs_value'][keep]

    # 3. Broken-watch gap — all observation channels
    gap_start = rng.integers(90, n_days - cfg.broken_watch_days - 90)
    gap_end = gap_start + cfg.broken_watch_days
    if verbose:
        print(f"  Broken watch gap: days {gap_start}-{gap_end}")

    for ch_name in cfg.all_obs_channels:
        ch = obs_data[ch_name]
        idx = ch['t_idx']
        keep = (idx < gap_start) | (idx >= gap_end)
        ch['t_idx'] = idx[keep]
        ch['obs_value'] = ch['obs_value'][keep]

    if verbose:
        for ch_name in cfg.all_obs_channels:
            n_obs = len(obs_data[ch_name]['t_idx'])
            pct = 100.0 * n_obs / n_days
            print(f"    {ch_name}: {n_obs}/{n_days} obs ({pct:.1f}%)")

    return obs_data
