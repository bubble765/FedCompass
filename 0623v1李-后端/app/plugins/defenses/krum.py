"""Krum: Byzantine-robust aggregation that selects the gradient closest to its neighbors."""
import numpy as np


def krum_aggregate(client_params, weights, client_states=None, num_to_keep=1):
    """Krum aggregation: select the parameter vector with smallest sum of distances to its closest n-f-2 neighbors."""
    n = len(client_params)
    if n <= 2:
        w = np.asarray(weights, dtype=np.float32)
        w = w / (w.sum() + 1e-8)
        return np.average(np.stack(client_params), axis=0, weights=w), {"method": "krum_fallback"}

    f = max(1, n // 4)  # Assume up to f Byzantine clients
    m = n - f - 2
    m = max(1, m)

    scores = np.zeros(n)
    for i in range(n):
        distances = [np.linalg.norm(client_params[i] - client_params[j]) for j in range(n) if j != i]
        distances.sort()
        scores[i] = sum(distances[:m])

    best_idx = np.argmin(scores)
    return client_params[best_idx].copy(), {
        "method": "krum",
        "selected_client": int(best_idx),
        "score": float(scores[best_idx]),
    }
