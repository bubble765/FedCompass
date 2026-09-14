"""Small attack transforms for smoke/demo experiments."""
import numpy as np


def gaussian_noise(state, params, global_params, round_num, noise_scale=0.15):
    rng = np.random.default_rng(round_num * 10000 + state.client_id)
    update = params - global_params
    scale = max(float(np.linalg.norm(update)) / max(update.size, 1), 1e-3) * noise_scale
    return params + rng.normal(0.0, scale, params.shape)


def model_replacement(state, params, global_params, round_num, scale_factor=3.0):
    update = params - global_params
    return global_params + update * scale_factor


def agr_byzantine(state, params, global_params, round_num):
    update = params - global_params
    return global_params - 1.5 * update


def alie(state, params, global_params, round_num, z_max=1.5):
    rng = np.random.default_rng(round_num * 7001 + state.client_id)
    update = params - global_params
    std = np.std(update) + 1e-6
    return params + rng.normal(z_max * std, 0.1 * std, params.shape)
