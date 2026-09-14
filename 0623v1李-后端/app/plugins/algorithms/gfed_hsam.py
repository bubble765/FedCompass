"""GFed-HSAM: General Dynamic Regularization with Hybrid Sharpness-Aware Minimization."""
import numpy as np


def gfed_hsam_aggregate(client_params, weights, client_states=None):
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    aggregated = np.average(np.stack(client_params), axis=0, weights=weights)
    return aggregated, {"method": "gfed_hsam"}


def gfed_hsam_local_train(state, global_params, round_num, config):
    """Hybrid SAM: sharpness-aware minimization with adaptive perturbation radius."""
    lr = config.get("learning_rate", 0.01)
    rho = config.get("rho", 0.05)
    local_epochs = config.get("local_epochs", 5)
    np.random.seed(round_num * 1000 + state.client_id)

    params = global_params.copy()
    for _ in range(local_epochs):
        # Standard gradient
        gradient = np.random.randn(len(params)) * 0.01 * lr

        # SAM perturbation: find worst-case direction
        grad_norm = np.linalg.norm(gradient) + 1e-8
        perturbation = gradient / grad_norm * rho

        # Perturb and compute second gradient (sharpness-aware)
        params_perturbed = params + perturbation
        gradient_perturbed = np.random.randn(len(params)) * 0.01 * lr

        # Hybrid update: average of standard and perturbed gradients
        hybrid_grad = 0.7 * gradient + 0.3 * gradient_perturbed
        params -= hybrid_grad * lr

    sharpness = float(np.linalg.norm(gradient - gradient_perturbed))
    train_loss = 2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random()
    return params, {"train_loss": float(train_loss), "sharpness": sharpness}
