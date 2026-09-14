"""FedAvg: Federated Averaging baseline. Standard weighted-average aggregation."""
import numpy as np


def fedavg_aggregate(client_params: list[np.ndarray], weights: np.ndarray,
                     client_states=None) -> tuple[np.ndarray, dict]:
    """Weighted average of client parameters."""
    weights = np.asarray(weights, dtype=np.float32)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(client_params)) / len(client_params)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "fedavg", "n_clients": len(client_params)}


def fedavg_local_train(state, global_params, round_num, config):
    """Standard SGD local training. Simulates gradient descent steps."""
    lr = config.get("learning_rate", 0.01)
    local_epochs = config.get("local_epochs", 5)
    np.random.seed(round_num * 1000 + state.client_id)

    # Simulate local training: params move toward a noisy optimum
    params = global_params.copy()
    for _ in range(local_epochs):
        # Simulated gradient: random direction with small magnitude
        gradient = np.random.randn(len(params)) * 0.01 * lr
        params -= gradient

    train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()
    return params, {"train_loss": float(train_loss)}
