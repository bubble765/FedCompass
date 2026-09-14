"""FLAME-style clustering, clipping and noising defense."""
import numpy as np


def flame_filter(client_updates, global_params, round_num, client_states=None, threshold=0.0):
    if not client_updates:
        return [], [], {"method": "flame", "scores": {}}
    updates = [(cid, params - global_params) for cid, params in client_updates]
    mean_update = np.mean(np.stack([update for _, update in updates]), axis=0)
    mean_norm = np.linalg.norm(mean_update) + 1e-8
    scores = {}
    for cid, update in updates:
        score = float(np.dot(update, mean_update) / ((np.linalg.norm(update) + 1e-8) * mean_norm))
        scores[cid] = score if np.isfinite(score) else -1.0
    cutoff = float(np.median(list(scores.values())) - 0.25 * np.std(list(scores.values())))
    cutoff = max(cutoff, threshold)
    trusted = [cid for cid, score in scores.items() if score >= cutoff]
    filtered = [cid for cid, _ in client_updates if cid not in trusted]
    if not trusted:
        trusted = [max(scores, key=scores.get)]
        filtered = [cid for cid, _ in client_updates if cid not in trusted]
    return trusted, filtered, {"method": "flame", "scores": scores, "threshold": cutoff}


def flame_aggregate(client_params, weights, client_states=None, clip_quantile=0.8, noise_scale=1e-4):
    trusted, filtered, meta = flame_filter(list(enumerate(client_params)), np.zeros_like(client_params[0]), 0)
    trusted_params = [client_params[i] for i in trusted] or client_params
    stacked = np.stack(trusted_params)
    center = np.mean(stacked, axis=0)
    updates = stacked - center
    norms = np.linalg.norm(updates, axis=1)
    clip_norm = float(np.quantile(norms, clip_quantile)) if len(norms) else 0.0
    clipped = []
    for params, update, norm in zip(stacked, updates, norms):
        if clip_norm > 0 and norm > clip_norm:
            update = update / (norm + 1e-8) * clip_norm
        clipped.append(center + update)
    aggregated = np.mean(np.stack(clipped), axis=0)
    rng = np.random.default_rng(17)
    aggregated = aggregated + rng.normal(0.0, noise_scale, aggregated.shape)
    return aggregated.astype(np.float32), {"method": "flame", "trusted": trusted, "filtered": filtered, **meta}
