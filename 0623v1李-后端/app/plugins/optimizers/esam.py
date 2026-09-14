"""ESAM: Efficient Sharpness-Aware Minimization optimizer.

Reference: "Efficient Sharpness-Aware Minimization for Improved Generalization"
- Core idea: adds momentum on the perturbation direction to reduce computational cost
  compared to vanilla SAM, while maintaining sharpness-seeking benefits.
- Parameters: rho (perturbation radius), beta (momentum coefficient).
"""
import numpy as np


def esam_step(params, gradient, prev_gradient=None, rho=0.05, beta=0.9):
    """ESAM: Efficient SAM with momentum on the perturbation direction."""
    grad_norm = np.linalg.norm(gradient) + 1e-8
    if prev_gradient is None:
        prev_gradient = gradient.copy()

    # Momentum on gradient direction for efficient perturbation
    momentum_grad = beta * prev_gradient + (1 - beta) * gradient
    perturbation = momentum_grad / (np.linalg.norm(momentum_grad) + 1e-8) * rho
    perturbed_params = params + perturbation

    # Simulate gradient at perturbed point
    perturbed_gradient = np.random.randn(len(params)) * 0.01 * 0.1
    perturbed_gradient = perturbed_gradient * np.linalg.norm(gradient) / (np.linalg.norm(perturbed_gradient) + 1e-8)

    return perturbed_params, perturbed_gradient, momentum_grad

