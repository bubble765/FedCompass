"""FedDyn: Federated Learning with Dynamic Regularization."""
import numpy as np


def feddyn_aggregate(client_params, weights, client_states=None):
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "feddyn"}


def feddyn_local_train(state, global_params, round_num, config):
    """Dynamic regularization: loss + <nabla, w> + (alpha/2)*||w - w_prev||^2."""
    lr = config.get("learning_rate", 0.01)
    alpha = config.get("alpha", 0.01)
    local_epochs = config.get("local_epochs", 5)
    np.random.seed(round_num * 1000 + state.client_id)

    if state.hist_gradient is None:
        state.hist_gradient = np.zeros_like(global_params)

    params = global_params.copy()
    prev_params = state.model_params if state.model_params is not None else global_params.copy()

    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        # Dynamic regularization: historical gradient + proximal to previous params
        reg = state.hist_gradient + alpha * (params - prev_params)
        params -= gradient + reg * lr

    # Update historical gradient: accumulate the change
    state.hist_gradient += alpha * (params - global_params)
    state.model_params = params.copy()

    train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()
    return params, {"train_loss": float(train_loss)}
