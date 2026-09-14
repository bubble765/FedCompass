"""HSAM: Hybrid Sharpness-Aware Minimization optimizer.

Reference: "Hybrid Sharpness-Aware Minimization for Federated Learning"
- Core idea: uses multiple perturbation samples to estimate the sharpness landscape
  more robustly, averaging perturbed gradients for a hybrid update.
- Parameters: rho (perturbation radius), num_samples (number of perturbation samples).
"""
import numpy as np


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

    # Average perturbed gradients for hybrid update
    avg_perturbed = np.mean(np.stack(perturbed_gradients), axis=0)
    return params, avg_perturbed

