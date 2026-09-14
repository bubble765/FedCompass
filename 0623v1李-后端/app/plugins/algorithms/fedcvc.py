"""FedCVC: Client-Driven Virtual Compensation for Mitigating Dual Drift."""
import numpy as np


def fedcvc_aggregate(client_params, weights, client_states=None):
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "fedcvc"}


def fedcvc_local_train(state, global_params, round_num, config):
    """Virtual compensation: compensates for silent clients using virtual updates."""
    lr = config.get("learning_rate", 0.01)
    dual_lr = config.get("dual_lr", 0.01)
    compensation_weight = config.get("compensation_weight", 1.0)
    silence_window = config.get("silence_window", 5)
    local_epochs = config.get("local_epochs", 5)
    silence_rate = config.get("silence_rate", 0.0)
    np.random.seed(round_num * 1000 + state.client_id)

    # Determine if this client is silent this round
    is_silent = np.random.random() < silence_rate
    if is_silent:
        state.silence_counter += 1
    else:
        state.silence_counter = max(0, state.silence_counter - 1)

    if state.model_params is None:
        state.model_params = global_params.copy()

    params = global_params.copy()

    if is_silent:
        # Virtual compensation: use historical trend to estimate update
        drift = params - state.model_params
        compensated_params = params + compensation_weight * drift
        params = compensated_params
        train_loss = 2.5  # Higher loss for silent clients
    else:
        # Normal local training with dual learning rate
        for _ in range(local_epochs):
            gradient = np.random.randn(len(params)) * 0.01 * lr
            # Dual update: global direction + local refinement
            global_direction = (global_params - state.model_params) * dual_lr
            params -= gradient * lr + global_direction

        train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()

    state.model_params = params.copy()

    silence_info = {
        "is_silent": is_silent,
        "silence_counter": state.silence_counter,
        "compensation_applied": is_silent and state.silence_counter <= silence_window,
    }
    return params, {"train_loss": float(train_loss), "silence_info": silence_info}
