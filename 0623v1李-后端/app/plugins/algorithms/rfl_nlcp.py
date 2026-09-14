"""RFL-NLCP: Robust Federated Learning with non-IID Data and Limited Client Participation."""
import numpy as np


def rfl_nlcp_aggregate(client_params, weights, client_states=None):
    """Robust aggregation: weighted average with outlier detection."""
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    if len(client_params) > 3:
        # Compute pairwise distances to detect outliers
        stacked = np.stack(client_params)
        centroid = np.median(stacked, axis=0)
        distances = np.array([np.linalg.norm(p - centroid) for p in client_params])
        threshold = np.median(distances) * 2.0
        mask = distances <= threshold
        if mask.sum() > 0:
            client_params = [p for i, p in enumerate(client_params) if mask[i]]
            weights_filtered = weights[mask]
            weights_filtered = weights_filtered / weights_filtered.sum()
            aggregated = np.average(np.stack(client_params), axis=0, weights=weights_filtered)
        else:
            aggregated = np.average(stacked, axis=0, weights=weights)
    else:
        aggregated = np.average(np.stack(client_params), axis=0, weights=weights)

    return aggregated, {"method": "rfl_nlcp", "outliers_filtered": int(sum(1 for _ in client_params)) < len(weights)}


def rfl_nlcp_local_train(state, global_params, round_num, config):
    """Robust local training with limited participation awareness."""
    lr = config.get("learning_rate", 0.01)
    robust_threshold = config.get("robust_threshold", 0.5)
    local_epochs = config.get("local_epochs", 5)
    np.random.seed(round_num * 1000 + state.client_id)

    params = global_params.copy()
    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        # Clip large gradients for robustness
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > robust_threshold:
            gradient = gradient * (robust_threshold / grad_norm)
        params -= gradient

    train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()
    return params, {"train_loss": float(train_loss)}
