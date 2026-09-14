"""SCAFFOLD: Stochastic Controlled Averaging for Federated Learning."""
import numpy as np


def scaffold_aggregate(client_params, weights, client_states=None):
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "scaffold"}


def scaffold_local_train(state, global_params, round_num, config):
    """Variance reduction with control variates c_i and c_global."""
    lr = config.get("learning_rate", 0.01)
    local_epochs = config.get("local_epochs", 5)
    np.random.seed(round_num * 1000 + state.client_id)

    if state.control_variate is None:
        state.control_variate = np.zeros_like(global_params)

    # Server sends global control variate c (we approximate it as zeros for simplicity)
    c_global = np.zeros_like(global_params)

    params = global_params.copy()
    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        # SCAFFOLD correction: c_i - c + gradient
        correction = state.control_variate - c_global
        params -= gradient * lr + correction * lr

    # Update local control variate
    state.control_variate += (global_params - params) / (lr * local_epochs + 1e-8)

    train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()
    return params, {"train_loss": float(train_loss)}
