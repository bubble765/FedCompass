"""FLDetector: Malicious client detection via historical update consistency."""
import numpy as np


def fldetector_filter(client_updates, global_params, round_num, client_states=None):
    """FLDetector: detect malicious clients by tracking update history consistency."""
    if client_states is None:
        client_states = {}

    if len(client_updates) < 2:
        return [cid for cid, _ in client_updates], [], {"method": "fldetector"}

    trusted = []
    filtered = []
    scores = {}

    for cid, params in client_updates:
        if cid not in client_states:
            client_states[cid] = {"history": []}

        update = params - global_params
        history = client_states[cid]["history"]

        if len(history) > 3:
            # Check consistency: is this update similar to historical updates?
            hist_mean = np.mean(np.stack(history[-3:]), axis=0)
            consistency = 1.0 / (1.0 + np.linalg.norm(update - hist_mean))
        else:
            consistency = 1.0
        if not np.isfinite(consistency):
            consistency = 0.0

        # Update magnitude check
        update_norm = np.linalg.norm(update)
        if not np.isfinite(update_norm):
            update_norm = float("inf")
        magnitude_score = min(1.0, 0.5 / (update_norm + 1e-8))

        score = 0.6 * consistency + 0.4 * magnitude_score
        if not np.isfinite(score):
            score = 0.0
        scores[cid] = float(score)

        # Keep history (max 10 entries)
        history.append(update.copy())
        if len(history) > 10:
            history.pop(0)

    # Dynamic threshold: mean - 1 std
    score_values = list(scores.values())
    threshold = np.mean(score_values) - 0.5 * np.std(score_values)
    if not np.isfinite(threshold):
        threshold = 0.0

    for cid, score in scores.items():
        if score >= threshold:
            trusted.append(cid)
        else:
            filtered.append(cid)

    return trusted, filtered, {"method": "fldetector", "scores": scores, "threshold": float(threshold)}


def fldetector_aggregate(client_params, weights, client_states=None):
    client_updates = list(enumerate(client_params))
    trusted_ids, filtered_ids, meta = fldetector_filter(client_updates, np.zeros_like(client_params[0]), 0, client_states)

    if len(trusted_ids) == 0:
        return np.median(np.stack(client_params), axis=0), {"method": "fldetector_fallback"}

    trusted_params = [client_params[i] for i in trusted_ids]
    trusted_weights = np.array([weights[i] for i in trusted_ids])
    trusted_weights = trusted_weights / (trusted_weights.sum() + 1e-8)
    aggregated = np.average(np.stack(trusted_params), axis=0, weights=trusted_weights)

    return aggregated, {
        "method": "fldetector",
        "trusted": trusted_ids,
        "filtered": filtered_ids,
        **meta,
    }
