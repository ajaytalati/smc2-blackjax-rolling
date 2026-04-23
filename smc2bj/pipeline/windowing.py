"""Window extraction: slice observations into [start, end) and re-index."""


def extract_window(obs_data, start: int, end: int):
    """Extract and re-index observations for window [start, end).

    Parameters
    ----------
    obs_data : dict
        Per-channel dict ``{channel_name: {'t_idx': np.ndarray, ...}}``.
        Every channel entry must have a ``t_idx`` field; other fields
        are carried through untouched, sliced by the same boolean mask.
    start, end : int
        Window bounds in the global timestep index.
    """
    window = {}
    for ch_name, ch_data in obs_data.items():
        t_idx = ch_data['t_idx']
        mask = (t_idx >= start) & (t_idx < end)
        new_ch = {'t_idx': t_idx[mask] - start}
        for key in ch_data:
            if key != 't_idx':
                new_ch[key] = ch_data[key][mask]
        window[ch_name] = new_ch
    return window
