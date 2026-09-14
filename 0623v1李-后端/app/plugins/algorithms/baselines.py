"""Executable baseline hooks for catalog algorithms.

The prototype runner uses a lightweight NumPy contract:
aggregate(client_params, weights) and local_train(state, global_params, round, config).
These hooks make paper/catalog baselines runnable in smoke tests and demo jobs while
keeping the implementation small enough to share one synthetic FL backend.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np


_SERVER_STATE: dict[tuple[str, int], dict[str, np.ndarray]] = {}


def weighted_average_aggregate(client_params, weights, client_states=None, method="weighted_average"):
    stacked = np.stack(client_params)
    weights = _normalize_weights(weights, len(client_params))
    return np.average(stacked, axis=0, weights=weights), {"method": method, "n_clients": len(client_params)}


def fednova_aggregate(client_params, weights, client_states=None):
    stacked = np.stack(client_params)
    weights = _normalize_weights(weights, len(client_params))
    avg = np.average(stacked, axis=0, weights=weights)
    centered = stacked - avg
    normalized = avg + np.average(centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1.0), axis=0, weights=weights)
    return normalized.astype(np.float32), {"method": "fednova", "normalization": "update_norm"}


def fedopt_aggregate(client_params, weights, client_states=None, method="fedopt", beta1=0.9, beta2=0.99, server_lr=0.05):
    avg, _ = weighted_average_aggregate(client_params, weights, method=method)
    key = (method, avg.size)
    state = _SERVER_STATE.setdefault(key, {"m": np.zeros_like(avg), "v": np.zeros_like(avg), "t": np.zeros(1)})
    state["t"][0] += 1
    delta = avg
    state["m"] = beta1 * state["m"] + (1.0 - beta1) * delta
    if method == "fedyogi":
        state["v"] = state["v"] - (1.0 - beta2) * np.sign(state["v"] - delta * delta) * delta * delta
    else:
        state["v"] = beta2 * state["v"] + (1.0 - beta2) * delta * delta
    step = server_lr * state["m"] / (np.sqrt(np.abs(state["v"])) + 1e-6)
    mixed = 0.85 * avg + 0.15 * (avg - step)
    return mixed.astype(np.float32), {"method": method, "server_lr": server_lr}


def robust_center_aggregate(client_params, weights, client_states=None, method="robust_center"):
    stacked = np.stack(client_params)
    median = np.median(stacked, axis=0)
    distances = np.linalg.norm(stacked - median, axis=1)
    keep = distances <= np.median(distances) + np.std(distances) + 1e-8
    if keep.any():
        stacked = stacked[keep]
        weights = np.asarray(weights, dtype=np.float32)[keep]
    weights = _normalize_weights(weights, len(stacked))
    return np.average(stacked, axis=0, weights=weights), {"method": method, "kept": int(len(stacked))}


def distillation_aggregate(client_params, weights, client_states=None, method="distillation"):
    avg, _ = weighted_average_aggregate(client_params, weights, method=method)
    teacher = np.median(np.stack(client_params), axis=0)
    distilled = 0.75 * avg + 0.25 * teacher
    return distilled.astype(np.float32), {"method": method, "teacher": "median_ensemble"}


def prototype_aggregate(client_params, weights, client_states=None, method="prototype"):
    avg, _ = weighted_average_aggregate(client_params, weights, method=method)
    if avg.size >= 16:
        proto = np.mean(np.stack([p[:16] for p in client_params]), axis=0)
        avg = avg.copy()
        avg[:16] = 0.6 * avg[:16] + 0.4 * proto
    return avg.astype(np.float32), {"method": method, "prototype_dims": int(min(16, avg.size))}


def make_aggregate(method: str) -> Callable:
    if method == "fednova":
        fn = fednova_aggregate
    elif method in {"fedopt", "fedadam"}:
        fn = lambda client_params, weights, client_states=None: fedopt_aggregate(client_params, weights, client_states, "fedadam", 0.9, 0.99, 0.04)
    elif method == "fedyogi":
        fn = lambda client_params, weights, client_states=None: fedopt_aggregate(client_params, weights, client_states, "fedyogi", 0.9, 0.99, 0.03)
    elif method in {"feddf", "fedkd", "fedgkd_p", "fedfld"}:
        fn = lambda client_params, weights, client_states=None: distillation_aggregate(client_params, weights, client_states, method)
    elif method in {"fedproto", "fedreid", "fedpav"}:
        fn = lambda client_params, weights, client_states=None: prototype_aggregate(client_params, weights, client_states, method)
    elif method in {"rfl_nlcp", "fedvra", "fedvarp", "fedtoga"}:
        fn = lambda client_params, weights, client_states=None: robust_center_aggregate(client_params, weights, client_states, method)
    else:
        fn = lambda client_params, weights, client_states=None: weighted_average_aggregate(client_params, weights, client_states, method)
    fn.__name__ = f"{method}_aggregate"
    return fn


def make_local_train(method: str, profile: str = "vision") -> Callable:
    def local_train(state, global_params, round_num, config):
        lr = float(config.get("learning_rate", 0.01))
        local_epochs = max(1, int(config.get("local_epochs", 3)))
        rng = _rng(state, round_num, method)
        if state.model_params is None:
            state.model_params = global_params.copy()

        params = global_params.astype(np.float32, copy=True)
        previous = state.model_params.astype(np.float32, copy=True)
        client_bias = _client_bias(state, global_params, rng)
        personalization_gap = 0.0
        prototype_alignment = 0.0
        sharpness = 0.0

        for epoch in range(local_epochs):
            gradient = _base_gradient(params, global_params, client_bias, rng, lr)
            gradient = _apply_method_gradient(method, state, params, global_params, previous, gradient, config, rng)
            if method in {"sam", "esam", "hsam", "gfed_hsam", "a_fedpdsam", "fedspeed", "fedsmoo", "fedlesam_d", "fedgloss"}:
                rho = float(config.get("rho", 0.05))
                perturbation = gradient / (np.linalg.norm(gradient) + 1e-8) * rho
                second_gradient = _base_gradient(params + perturbation, global_params, client_bias, rng, lr)
                sharpness = float(np.linalg.norm(second_gradient - gradient))
                mix = 0.5 if method in {"hsam", "gfed_hsam"} else 0.35
                gradient = (1.0 - mix) * gradient + mix * second_gradient
            params = params - lr * gradient

        if method == "fednova":
            params = global_params + (params - global_params) / math.sqrt(local_epochs)
        elif method == "fedbn":
            mask = np.arange(params.size) % 8 == 0
            params[mask] = previous[mask]
        elif method == "ditto":
            personalized = 0.7 * params + 0.3 * previous
            personalization_gap = float(np.linalg.norm(personalized - params) / max(params.size, 1))
            state.personal_model = personalized.copy()
        elif method == "pfedme":
            moreau_mu = float(config.get("moreau_mu", 0.05))
            params = (params + moreau_mu * previous) / (1.0 + moreau_mu)
            personalization_gap = float(np.linalg.norm(params - previous) / max(params.size, 1))
        elif method in {"fedproto", "fedgkd_p", "fedfld", "fedkd", "feddf"}:
            proto_len = min(16, params.size)
            old_proto = getattr(state, "prototype", global_params[:proto_len].copy())
            new_proto = 0.6 * old_proto + 0.4 * params[:proto_len]
            params[:proto_len] = 0.7 * params[:proto_len] + 0.3 * new_proto
            state.prototype = new_proto.copy()
            prototype_alignment = _cosine(params[:proto_len], global_params[:proto_len])

        state.model_params = params.copy()
        train_loss = _loss_curve(round_num, profile, rng, method)
        metrics = {
            "train_loss": train_loss,
            "gradient_drift": float(np.linalg.norm(params - global_params) / max(params.size, 1)),
        }
        if profile == "reid":
            metrics["mAP"] = float(np.clip(0.56 + 0.24 * (1 - np.exp(-0.045 * round_num)) + rng.normal(0, 0.01), 0.0, 0.95))
            metrics["rank1"] = float(np.clip(0.64 + 0.22 * (1 - np.exp(-0.045 * round_num)) + rng.normal(0, 0.01), 0.0, 0.98))
        if profile == "ulip3d":
            metrics["test_accuracy"] = float(np.clip(0.42 + 0.34 * (1 - np.exp(-0.04 * round_num)) + rng.normal(0, 0.015), 0.0, 0.95))
        if personalization_gap:
            metrics["personalization_gap"] = personalization_gap
        if prototype_alignment:
            metrics["prototype_alignment"] = prototype_alignment
        if sharpness:
            metrics["sharpness"] = sharpness
        return params.astype(np.float32), metrics

    local_train.__name__ = f"{method}_local_train"
    return local_train


def build_algorithm(method: str, profile: str = "vision") -> tuple[Callable, Callable]:
    return make_aggregate(method), make_local_train(method, profile)


def _normalize_weights(weights, n: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    if weights.size != n:
        weights = np.ones(n, dtype=np.float32)
    total = float(weights.sum())
    if total <= 0 or not np.isfinite(total):
        return np.ones(n, dtype=np.float32) / max(n, 1)
    return weights / total


def _rng(state, round_num: int, method: str) -> np.random.Generator:
    salt = sum(ord(ch) for ch in method)
    seed = int(round_num * 1009 + int(state.client_id) * 9176 + salt)
    return np.random.default_rng(seed)


def _client_bias(state, global_params, rng):
    if not hasattr(state, "baseline_bias"):
        state.baseline_bias = rng.normal(0.0, 0.015, global_params.shape).astype(np.float32)
    return state.baseline_bias


def _base_gradient(params, global_params, client_bias, rng, lr):
    local_target = global_params + client_bias
    curvature = 0.03 * (params - local_target)
    noise = rng.normal(0.0, 0.01, params.shape).astype(np.float32)
    return curvature + noise * max(lr, 1e-4)


def _apply_method_gradient(method, state, params, global_params, previous, gradient, config, rng):
    if method in {"fedprox", "pfedme"}:
        mu = float(config.get("prox_mu", config.get("mu", 0.01)))
        gradient = gradient + mu * (params - global_params)
    if method in {"feddyn", "feddc", "fedvra", "fedvarp", "fednova"}:
        if state.hist_gradient is None:
            state.hist_gradient = np.zeros_like(global_params)
        gradient = gradient + 0.05 * state.hist_gradient
        state.hist_gradient = 0.9 * state.hist_gradient + 0.1 * (params - previous)
    if method in {"scaffold", "fedtoga"}:
        if state.control_variate is None:
            state.control_variate = np.zeros_like(global_params)
        correction = 0.1 * state.control_variate
        gradient = gradient - correction
        state.control_variate = 0.9 * state.control_variate + 0.1 * gradient
    if method in {"fedcads", "feddf", "fedkd", "fedgkd_p", "fedfld"}:
        distill_weight = float(config.get("distill_weight", 0.4))
        gradient = gradient + distill_weight * 0.05 * (params - global_params)
    if method in {"fedproto", "fedreid", "fedpav"}:
        proto_len = min(16, params.size)
        proto = getattr(state, "prototype", global_params[:proto_len])
        gradient = gradient.copy()
        gradient[:proto_len] += 0.05 * (params[:proto_len] - proto)
    if method in {"mixstyle", "crossstyle", "snr", "dacs", "sscu"}:
        style = rng.normal(0.0, 0.004, params.shape).astype(np.float32)
        gradient = gradient + style
    if method in {"fedkgcoop", "fedvpt", "fedtpg", "fedcocoop", "fedmaple", "fedclip", "fedmvp", "pointclip", "ulip"}:
        adapter_scale = 0.02 if method != "ulip" else 0.01
        gradient = gradient + adapter_scale * np.tanh(params - global_params)
    return gradient


def _loss_curve(round_num: int, profile: str, rng, method: str) -> float:
    base = 2.2
    if profile == "reid":
        base = 1.8
    elif profile == "ulip3d":
        base = 3.2
    bonus = 0.92 if method in {"fednova", "fedadam", "fedyogi", "fedbn", "ditto", "pfedme", "fedproto", "feddf", "fedkd"} else 1.0
    return float(max(0.03, base * bonus * np.exp(-0.03 * round_num) + rng.uniform(0.0, 0.05)))


def _cosine(a, b) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))
