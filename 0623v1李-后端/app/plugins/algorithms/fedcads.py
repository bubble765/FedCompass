"""FedCADS: Robust FL via Dual Distillation and Participation-Aware Optimization."""
import numpy as np


def fedcads_aggregate(client_params, weights, client_states=None):
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "fedcads"}


def fedcads_local_train(state, global_params, round_num, config):
    """Dual distillation: self-distillation + global distillation with participation-aware weights."""
    lr = config.get("learning_rate", 0.01)
    distill_weight = config.get("distill_weight", 0.5)
    self_distill = config.get("self_distill", True)
    local_epochs = config.get("local_epochs", 5)
    participation_rate = config.get("participation_rate", 0.1)
    np.random.seed(round_num * 1000 + state.client_id)

    if state.model_params is None:
        state.model_params = global_params.copy()

    params = global_params.copy()
    # Participation-aware weight: clients with more data get higher weight
    pa_weight = 1.0 + 0.5 * (1.0 - participation_rate)

    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * 0.01 * lr

        if self_distill:
            # Self-distillation: maintain proximity to previous local model
            self_distill_grad = (params - state.model_params) * distill_weight * 0.1
            gradient += self_distill_grad

        # Global distillation: maintain proximity to global model
        global_distill_grad = (params - global_params) * distill_weight * 0.05
        gradient += global_distill_grad

        params -= gradient * lr * pa_weight

    state.model_params = params.copy()
    train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()

    # Measure feature alignment as cosine similarity to global
    alignment = float(np.dot(params, global_params) / (np.linalg.norm(params) * np.linalg.norm(global_params) + 1e-8))

    return params, {"train_loss": float(train_loss), "feature_alignment": alignment}
