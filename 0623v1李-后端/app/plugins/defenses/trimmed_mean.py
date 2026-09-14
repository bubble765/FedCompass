"""Trimmed Mean: Remove extreme values before averaging."""
import numpy as np


def trimmed_mean_aggregate(client_params, weights, client_states=None, trim_ratio=0.1):
    """Trim top and bottom trim_ratio fraction of values per coordinate, then average."""
    stacked = np.stack(client_params)
    n = len(client_params)
    k = max(1, int(n * trim_ratio))

    sorted_stack = np.sort(stacked, axis=0)
    trimmed = sorted_stack[k:n - k] if n > 2 * k else sorted_stack
    aggregated = np.mean(trimmed, axis=0)
    return aggregated, {"method": "trimmed_mean", "trimmed": int(k * 2)}
