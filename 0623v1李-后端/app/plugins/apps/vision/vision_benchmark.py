"""Vision Benchmark: CIFAR/MNIST/AG News federated classification demo.

Provides mock local training and aggregation for vision classification tasks.
Supports IID, Dirichlet, and Pathological data splits.
"""
import numpy as np


def vision_local_train(state, global_params, round_num, config):
    """Simulated local training for vision classification.

    Incorporates split-type specific behavior:
    - IID: stable convergence
    - Dirichlet: increased client drift (heterogeneous data)
    - Pathological: some clients have extreme non-IID distribution
    """
    lr = config.get("learning_rate", 0.01)
    local_epochs = config.get("local_epochs", 5)
    split_type = config.get("split", "iid")
    participation_rate = config.get("participation_rate", 0.1)
    np.random.seed(round_num * 1000 + state.client_id)

    params = global_params.copy()

    # Split-type drift factor
    if split_type == "dirichlet":
        drift_factor = 1.0 + 0.3 * np.random.random()
        noise_scale = 0.015
    elif split_type == "pathological":
        drift_factor = 1.0 + 0.6 * np.random.random() if state.client_id % 2 == 0 else 0.6
        noise_scale = 0.02
    else:  # iid
        drift_factor = 1.0
        noise_scale = 0.008

    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * noise_scale * lr * drift_factor
        params -= gradient

    pa_penalty = 1.0 + 0.2 * (1.0 - participation_rate)
    train_loss = (2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()) * pa_penalty

    return params, {
        "train_loss": float(train_loss),
        "split_type": split_type,
        "drift_factor": float(drift_factor),
    }


def vision_aggregate(client_params, weights, client_states=None):
    """Weighted average aggregation for vision tasks."""
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "vision_fedavg"}
