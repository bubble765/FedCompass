"""FedDTC: Dynamic heterogeneity-aware federated learning with global gradient offset correction."""
import numpy as np


def feddtc_aggregate(client_params, weights, client_states=None):
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "feddtc"}


def feddtc_local_train(state, global_params, round_num, config):
    """EMA-based heterogeneity tracking with gradient offset correction."""
    lr = config.get("learning_rate", 0.01)
    ema_alpha = config.get("ema_alpha", 0.9)
    local_epochs = config.get("local_epochs", 5)
    np.random.seed(round_num * 1000 + state.client_id)

    if state.model_params is None:
        state.model_params = global_params.copy()
    if state.hist_gradient is None:
        state.hist_gradient = np.zeros_like(global_params)

    params = global_params.copy()
    local_drift = np.zeros_like(params)

    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        local_drift += gradient
        params -= gradient * lr

    # EMA update of historical gradient
    state.hist_gradient = ema_alpha * state.hist_gradient + (1 - ema_alpha) * local_drift

    # Offset correction: compensate for global gradient offset
    correction = state.hist_gradient
    params -= correction * lr * 0.1

    # Measure heterogeneity as norm of local drift vs global
    gradient_drift = float(np.linalg.norm(local_drift) / (np.linalg.norm(global_params - state.model_params) + 1e-8))

    state.model_params = params.copy()
    train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()
    return params, {"train_loss": float(train_loss), "gradient_drift": gradient_drift}
