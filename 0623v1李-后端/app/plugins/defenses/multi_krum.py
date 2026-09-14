"""Multi-Krum robust aggregation."""
import numpy as np


def multi_krum_aggregate(client_params, weights, client_states=None, num_to_keep=5):
    n = len(client_params)
    if n <= 2:
        weights = _normalize(weights, n)
        return np.average(np.stack(client_params), axis=0, weights=weights), {"method": "multi_krum_fallback"}

    f = max(1, n // 4)
    neighbor_count = max(1, n - f - 2)
    scores = []
    for i in range(n):
        distances = [np.linalg.norm(client_params[i] - client_params[j]) for j in range(n) if j != i]
        distances.sort()
        scores.append(float(sum(distances[:neighbor_count])))

    keep = np.argsort(scores)[: max(1, min(int(num_to_keep), n))]
    kept_params = [client_params[int(i)] for i in keep]
    kept_weights = _normalize(np.asarray(weights, dtype=np.float32)[keep], len(keep))
    aggregated = np.average(np.stack(kept_params), axis=0, weights=kept_weights)
    return aggregated, {"method": "multi_krum", "selected": [int(i) for i in keep]}


def multi_krum_filter(client_updates, global_params, round_num, client_states=None, num_to_keep=5):
    if not client_updates:
        return [], [], {"method": "multi_krum", "scores": {}}
    params = [params for _, params in client_updates]
    n = len(params)
    if n <= 2:
        return [cid for cid, _ in client_updates], [], {"method": "multi_krum", "scores": {}}
    f = max(1, n // 4)
    neighbor_count = max(1, n - f - 2)
    scores = {}
    for idx, (cid, params_i) in enumerate(client_updates):
        distances = [np.linalg.norm(params_i - params_j) for j, params_j in enumerate(params) if j != idx]
        distances.sort()
        scores[cid] = float(sum(distances[:neighbor_count]))
    selected = [cid for cid, _ in sorted(scores.items(), key=lambda item: item[1])[: max(1, min(num_to_keep, n))]]
    filtered = [cid for cid, _ in client_updates if cid not in selected]
    return selected, filtered, {"method": "multi_krum", "scores": scores}


def _normalize(weights, n):
    total = float(np.sum(weights))
    if total <= 0 or not np.isfinite(total):
        return np.ones(n, dtype=np.float32) / max(n, 1)
    return np.asarray(weights, dtype=np.float32) / total
