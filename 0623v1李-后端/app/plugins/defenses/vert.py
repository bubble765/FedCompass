"""VERT: Vertical defense against model poisoning with trust scoring and client filtering."""
import numpy as np


def vert_filter(client_updates, global_params, round_num, client_states=None, threshold=0.8):
    """VERT defense: score clients by update consistency, filter low-trust clients."""
    trusted = []
    filtered = []
    scores = {}

    if len(client_updates) < 2:
        return [cid for cid, _ in client_updates], [], {"method": "vert", "scores": {}}

    for cid, params in client_updates:
        # Compute update magnitude
        update_norm = np.linalg.norm(params - global_params)
        if not np.isfinite(update_norm):
            update_norm = float("inf")

        # Compute consensus: how close is this update to others
        others = [p for cid2, p in client_updates if cid2 != cid]
        if others:
            mean_other = np.mean(np.stack(others), axis=0)
            consensus = 1.0 / (1.0 + np.linalg.norm(params - mean_other))
        else:
            consensus = 1.0
        if not np.isfinite(consensus):
            consensus = 0.0

        # Trust score: combines update magnitude reasonableness and consensus
        trust = consensus * min(1.0, 1.0 / (update_norm + 1e-8) * 0.1)
        if not np.isfinite(trust):
            trust = 0.0
        scores[cid] = float(trust)

        score_mean = float(np.mean(list(scores.values()))) if scores else 0.0
        if not np.isfinite(score_mean):
            score_mean = 0.0
        if trust >= threshold * score_mean:
            trusted.append(cid)
        else:
            filtered.append(cid)

    return trusted, filtered, {"method": "vert", "scores": scores, "threshold": threshold}


def vert_aggregate(client_params, weights, client_states=None, threshold=0.8):
    """VERT aggregation: only aggregate trusted clients."""
    # All clients are treated as potentially trusted; apply vert_filter first
    client_updates = list(enumerate(client_params))
    trusted_ids, filtered_ids, meta = vert_filter(client_updates, np.zeros_like(client_params[0]), 0, threshold=threshold)

    if len(trusted_ids) == 0:
        # Fall back to median if no trusted clients
        stacked = np.stack(client_params)
        return np.median(stacked, axis=0), {"method": "vert_fallback", "filtered": filtered_ids}

    trusted_params = [client_params[i] for i in trusted_ids]
    trusted_weights = np.array([weights[i] for i in trusted_ids])
    trusted_weights = trusted_weights / (trusted_weights.sum() + 1e-8)
    aggregated = np.average(np.stack(trusted_params), axis=0, weights=trusted_weights)

    return aggregated, {
        "method": "vert",
        "trusted": trusted_ids,
        "filtered": filtered_ids,
        **meta,
    }
