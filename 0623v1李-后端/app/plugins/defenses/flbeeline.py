"""FLBeeline: Lightweight large-scale model poisoning defense via geometric consistency."""
import numpy as np


def flbeeline_filter(client_updates, global_params, round_num, client_states=None, top_k=10):
    """FLBeeline: select top-K clients by historical gradient geometric consistency."""
    if len(client_updates) == 0:
        return [], [], {"method": "flbeeline", "scores": {}}

    # Compute update vectors
    update_vectors = {}
    for cid, params in client_updates:
        update_vectors[cid] = params - global_params

    # Compute geometric consistency: cosine similarity between each update and the mean
    if len(update_vectors) > 1:
        mean_update = np.mean(np.stack(list(update_vectors.values())), axis=0)
        mean_norm = np.linalg.norm(mean_update) + 1e-8
        if not np.isfinite(mean_norm):
            mean_norm = 1e-8

        scores = {}
        for cid, vec in update_vectors.items():
            vec_norm = np.linalg.norm(vec) + 1e-8
            if not np.isfinite(vec_norm):
                vec_norm = float("inf")
            cos_sim = np.dot(vec, mean_update) / (vec_norm * mean_norm)
            # Consistency score: high cosine similarity + reasonable magnitude
            score = cos_sim * min(1.0, mean_norm / vec_norm)
            if not np.isfinite(score):
                score = 0.0
            scores[cid] = float(score)
    else:
        scores = {cid: 1.0 for cid in update_vectors}

    # Sort by score descending
    sorted_clients = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    k = min(top_k, len(sorted_clients))
    trusted = [cid for cid, _ in sorted_clients[:k]]
    filtered = [cid for cid, _ in sorted_clients[k:]]

    return trusted, filtered, {"method": "flbeeline", "scores": dict(sorted_clients), "top_k": top_k}


def flbeeline_aggregate(client_params, weights, client_states=None, top_k=10):
    """FLBeeline aggregation: only aggregate top-K geometrically consistent clients."""
    client_updates = list(enumerate(client_params))
    trusted_ids, filtered_ids, meta = flbeeline_filter(
        client_updates, np.zeros_like(client_params[0]), 0, top_k=top_k
    )

    if len(trusted_ids) == 0:
        stacked = np.stack(client_params)
        return np.median(stacked, axis=0), {"method": "flbeeline_fallback"}

    trusted_params = [client_params[i] for i in trusted_ids]
    trusted_weights = np.array([weights[i] for i in trusted_ids])
    trusted_weights = trusted_weights / (trusted_weights.sum() + 1e-8)
    aggregated = np.average(np.stack(trusted_params), axis=0, weights=trusted_weights)

    return aggregated, {
        "method": "flbeeline",
        "trusted": trusted_ids,
        "filtered": filtered_ids,
        **meta,
    }
