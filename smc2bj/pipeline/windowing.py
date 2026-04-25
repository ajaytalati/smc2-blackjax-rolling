"""Window extraction: slice observations into [start, end) and re-index."""

import numpy as np


def extract_window(obs_data, start: int, end: int):
    """Extract and re-index observations for window [start, end).

    Parameters
    ----------
    obs_data : dict
        Per-channel dict ``{channel_name: {'t_idx': np.ndarray, ...}}``.
        Every channel entry must have a ``t_idx`` field. Other fields:
          - **Array fields with len == len(t_idx)** are masked alongside.
            This is the common case (obs_value, sleep_label, etc).
          - **Scalar / 0-dim fields** (e.g. SWAT's ``bin_hours = 0.25``)
            are passed through untouched. The estimator reads them as
            channel-wide metadata, not as per-step values.
          - **Any other-shape field** is also passed through untouched
            (defensive — extract_window doesn't know what to do with it).
    start, end : int
        Window bounds in the global timestep index.
    """
    window = {}
    for ch_name, ch_data in obs_data.items():
        t_idx = np.asarray(ch_data['t_idx'])
        mask = (t_idx >= start) & (t_idx < end)
        new_ch = {'t_idx': t_idx[mask] - start}
        n_t = len(t_idx)
        for key in ch_data:
            if key == 't_idx':
                continue
            val = ch_data[key]
            arr = np.asarray(val)
            if arr.ndim >= 1 and len(arr) == n_t:
                new_ch[key] = arr[mask]
            else:
                # Scalar or non-per-step metadata — pass through.
                new_ch[key] = val
        window[ch_name] = new_ch
    return window
