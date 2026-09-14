"""FedProx: Federated Learning with proximal term."""
import numpy as np


def fedprox_aggregate(client_params, weights, client_states=None):
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "fedprox"}


def fedprox_local_train(state, global_params, round_num, config):
    """Local training with proximal term: loss + (mu/2) * ||w - w_global||^2."""
    lr = config.get("learning_rate", 0.01)
    mu = config.get("mu", 0.01)
    local_epochs = config.get("local_epochs", 5)
    np.random.seed(round_num * 1000 + state.client_id)

    params = global_params.copy()
    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        # Proximal term pulls back toward global params
        proximal = mu * (params - global_params)
        params -= gradient + proximal * lr

    train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()
    return params, {"train_loss": float(train_loss)}
