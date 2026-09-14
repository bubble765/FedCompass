"""FLTrust-style trust scoring with a server reference update."""
import numpy as np


def fltrust_filter(client_updates, global_params, round_num, client_states=None, threshold=0.0):
    if not client_updates:
        return [], [], {"method": "fltrust", "scores": {}}
    updates = [(cid, params - global_params) for cid, params in client_updates]
    reference = np.mean(np.stack([update for _, update in updates]), axis=0)
    ref_norm = np.linalg.norm(reference) + 1e-8
    scores = {}
    trusted = []
    filtered = []
    for cid, update in updates:
        cosine = float(np.dot(update, reference) / ((np.linalg.norm(update) + 1e-8) * ref_norm))
        score = max(0.0, cosine)
        scores[cid] = score
        if score >= threshold:
            trusted.append(cid)
        else:
            filtered.append(cid)
    if not trusted:
        trusted = [cid for cid, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:1]]
        filtered = [cid for cid, _ in client_updates if cid not in trusted]
    return trusted, filtered, {"method": "fltrust", "scores": scores, "threshold": threshold}


def fltrust_aggregate(client_params, weights, client_states=None):
    global_ref = np.zeros_like(client_params[0])
    trusted, filtered, meta = fltrust_filter(list(enumerate(client_params)), global_ref, 0)
    trusted_params = [client_params[i] for i in trusted] or client_params
    trusted_weights = np.asarray([weights[i] for i in trusted], dtype=np.float32) if trusted else np.asarray(weights, dtype=np.float32)
    total = float(trusted_weights.sum())
    trusted_weights = trusted_weights / total if total > 0 else np.ones(len(trusted_params)) / len(trusted_params)
    aggregated = np.average(np.stack(trusted_params), axis=0, weights=trusted_weights)
    return aggregated, {"method": "fltrust", "trusted": trusted, "filtered": filtered, **meta}
