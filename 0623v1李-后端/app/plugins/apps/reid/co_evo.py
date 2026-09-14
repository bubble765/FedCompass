"""CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID.

Mock implementation that simulates the CO-EVO training process for ReID domain generalization.
"""
import numpy as np


def co_evo_aggregate(client_params, weights, client_states=None):
    """CO-EVO aggregation: semantic anchor alignment + style diversification."""
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    # Domain-aware aggregation: each client represents a different domain
    stacked = np.stack(client_params)
    aggregated = np.average(stacked, axis=0, weights=weights)

    return aggregated, {
        "method": "co_evo",
        "n_domains": len(client_params),
        "style_diversity": float(np.std(stacked, axis=0).mean()),
    }


def co_evo_local_train(state, global_params, round_num, config):
    """CO-EVO local training with semantic anchor and style bank.

    Two-stage local update:
    1. Semantic anchoring: pull features toward global class anchors
    2. Style diversification: push domain-specific style bank away from others
    """
    lr = config.get("learning_rate", 0.001)
    lambda_anchor = config.get("lambda_anchor", 1.0)
    lambda_style = config.get("lambda_style", 0.1)
    local_epochs = config.get("local_epochs", 10)
    np.random.seed(round_num * 1000 + state.client_id)

    if state.model_params is None:
        state.model_params = global_params.copy()
    if state.hist_gradient is None:
        state.hist_gradient = np.zeros_like(global_params)

    params = global_params.copy()

    # Stage 1: Semantic anchoring - align features with global class anchors
    anchor_direction = global_params - state.model_params
    anchor_alignment = 0

    for _ in range(local_epochs // 2):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        # Pull toward global anchor
        gradient += lambda_anchor * anchor_direction * 0.01
        params -= gradient
        anchor_alignment = float(np.dot(params, global_params) / (
            np.linalg.norm(params) * np.linalg.norm(global_params) + 1e-8
        ))

    # Stage 2: Style diversification - push domain-specific features
    style_perturbation = np.random.randn(len(params)) * 0.005
    for _ in range(local_epochs // 2):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        # Push style away from mean
        gradient += lambda_style * style_perturbation
        params -= gradient

    state.model_params = params.copy()

    # ReID metrics: mAP and Rank-1 (simulated)
    base_map = 0.65 + 0.15 * (1 - np.exp(-0.05 * round_num))
    base_rank1 = 0.72 + 0.13 * (1 - np.exp(-0.05 * round_num))
    mAP = float(base_map + np.random.uniform(-0.02, 0.02))
    rank1 = float(base_rank1 + np.random.uniform(-0.02, 0.02))
    train_loss = float(2.0 * np.exp(-0.03 * round_num) + 0.1 * np.random.random())

    return params, {
        "train_loss": train_loss,
        "mAP": mAP,
        "rank1": rank1,
        "anchor_alignment": anchor_alignment,
        "style_diversity": float(np.linalg.norm(style_perturbation)),
    }


# ── Mock runner for ReID scenarios ──

def generate_reid_metrics(dataset="market1501", rounds=200):
    """Generate mock CO-EVO metrics for ReID experiments."""
    np.random.seed(hash(f"co_evo_{dataset}") % 2**31)
    history = []
    for r in range(1, rounds + 1):
        progress = r / rounds
        mAP = min(0.92, 0.55 + 0.37 * (1 - np.exp(-4 * progress)) + np.random.uniform(-0.02, 0.02))
        rank1 = min(0.95, 0.62 + 0.33 * (1 - np.exp(-4 * progress)) + np.random.uniform(-0.02, 0.02))
        loss = max(0.05, 3.0 * np.exp(-2.5 * progress) + np.random.uniform(-0.1, 0.1))
        history.append({
            "round": r,
            "train_loss": round(float(loss), 4),
            "mAP": round(float(mAP), 4),
            "rank1": round(float(rank1), 4),
        })
    return history

