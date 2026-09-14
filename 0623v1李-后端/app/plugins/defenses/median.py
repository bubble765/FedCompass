"""Median: Coordinate-wise median aggregation, robust to outliers."""
import numpy as np


def median_aggregate(client_params, weights, client_states=None):
    """Coordinate-wise median of client parameters."""
    stacked = np.stack(client_params)
    aggregated = np.median(stacked, axis=0)
    return aggregated, {"method": "median"}
