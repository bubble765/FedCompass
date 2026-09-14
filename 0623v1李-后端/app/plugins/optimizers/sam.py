"""SAM: Sharpness-Aware Minimization optimizer."""
import numpy as np


def sam_step(params, gradient, rho=0.05):
    """SAM update: perturb along gradient, compute gradient at perturbed point, then step."""
    grad_norm = np.linalg.norm(gradient) + 1e-8
    perturbation = gradient / grad_norm * rho
    perturbed_params = params + perturbation

    # Simulate gradient at perturbed point
    perturbed_gradient = np.random.randn(len(params)) * 0.01 * 0.1
    perturbed_gradient = perturbed_gradient * np.linalg.norm(gradient) / (np.linalg.norm(perturbed_gradient) + 1e-8)

    return perturbed_params, perturbed_gradient


def esam_step(params, gradient, prev_gradient=None, rho=0.05, beta=0.9):
    """ESAM: Efficient SAM with momentum on the perturbation direction."""
    grad_norm = np.linalg.norm(gradient) + 1e-8
    if prev_gradient is None:
        prev_gradient = gradient.copy()

    # Momentum on gradient direction
    momentum_grad = beta * prev_gradient + (1 - beta) * gradient
    perturbation = momentum_grad / (np.linalg.norm(momentum_grad) + 1e-8) * rho
    perturbed_params = params + perturbation

    perturbed_gradient = np.random.randn(len(params)) * 0.01 * 0.1
    perturbed_gradient = perturbed_gradient * np.linalg.norm(gradient) / (np.linalg.norm(perturbed_gradient) + 1e-8)

    return perturbed_params, perturbed_gradient, momentum_grad


def hsam_step(params, gradient, rho=0.05, num_samples=3):
    """HSAM: Hybrid SAM with multiple perturbation samples."""
    grad_norm = np.linalg.norm(gradient) + 1e-8

    perturbed_gradients = []
    for _ in range(num_samples):
        epsilon = np.random.randn(len(params)) * rho
        epsilon = epsilon * grad_norm / (np.linalg.norm(epsilon) + 1e-8)
        perturbed_params = params + epsilon
        pg = np.random.randn(len(params)) * 0.01 * 0.1
        perturbed_gradients.append(pg * grad_norm / (np.linalg.norm(pg) + 1e-8))

    # Average perturbed gradients
    avg_perturbed = np.mean(np.stack(perturbed_gradients), axis=0)
    return params, avg_perturbed
"""SAM: Sharpness-Aware Minimization optimizer.

Reference: "Sharpness-Aware Minimization for Efficiently Improving Generalization"
- Core idea: perturb parameters along gradient direction to find worst-case point,
  then compute gradient at that perturbed point for a sharper update.
- Parameter: rho (perturbation radius) controls the size of the sharpness-seeking step.
"""
import numpy as np


def sam_step(params, gradient, rho=0.05):
    """SAM update: perturb along gradient, compute gradient at perturbed point, then step."""
    grad_norm = np.linalg.norm(gradient) + 1e-8
    perturbation = gradient / grad_norm * rho
    perturbed_params = params + perturbation

    # Simulate gradient at perturbed point
    perturbed_gradient = np.random.randn(len(params)) * 0.01 * 0.1
    perturbed_gradient = perturbed_gradient * np.linalg.norm(gradient) / (np.linalg.norm(perturbed_gradient) + 1e-8)

    return perturbed_params, perturbed_gradient
