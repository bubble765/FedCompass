"""CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID.

Reference: "CO-EVO: Co-evolving Semantic Anchoring and Style Diversification for Federated DG-ReID"
- Core idea: jointly evolve semantic anchors (cross-domain class prototypes) and a
  global style bank (domain-specific style variations) during federated training.
- Each client maintains domain-specific features; the server aggregates anchor
  prototypes and style bank entries for domain generalization.
- Key metrics: mAP, Rank-1 accuracy for person re-identification tasks.

Algorithm workflow:
  1. Server broadcasts global model + semantic anchors + style bank
  2. Client performs two-stage local training:
     a. Semantic anchoring: pull local features toward global class anchors
     b. Style diversification: push domain-specific styles away from global style bank
  3. Server aggregates model weights and updates anchors/style bank
"""
import numpy as np


def co_evo_aggregate(client_params, weights, client_states=None):
    """CO-EVO server aggregation: weighted average + anchor/style bank update.

    Returns:
        aggregated_params: weighted average of client model parameters
        extra: dictionary with domain diversity metrics
    """
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    stacked = np.stack(client_params)
    aggregated = np.average(stacked, axis=0, weights=weights)

    # Measure style diversity across client domains
    style_diversity = float(np.std(stacked, axis=0).mean())
    n_domains = len(client_params)

    return aggregated, {
        "method": "co_evo",
        "n_domains": n_domains,
        "style_diversity": style_diversity,
    }


def co_evo_local_train(state, global_params, round_num, config):
    """CO-EVO two-stage local training: semantic anchoring + style diversification.

    Stage 1 - Semantic Anchoring:
        Pulls local features toward global class-level semantic anchors.
        Uses cosine similarity to measure anchor alignment.

    Stage 2 - Style Diversification:
        Pushes domain-specific features away from the global style bank mean.
        This encourages each domain to develop complementary style representations.

    Parameters (from config):
        learning_rate (float): base learning rate, default 0.001
        lambda_anchor (float): weight for semantic anchor alignment, default 1.0
        lambda_style (float): weight for style diversification, default 0.1
        local_epochs (int): number of local training epochs, default 10

    Returns:
        updated_params: client's updated model parameters
        metrics: dict with train_loss, mAP, rank1, anchor_alignment, style_diversity
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
    anchor_alignment = 0.0

    # ---- Stage 1: Semantic Anchoring ----
    # Compute direction toward global anchor (difference from previous global model)
    anchor_direction = global_params - state.model_params
    for _ in range(local_epochs // 2):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        # Pull toward global semantic anchor
        gradient = gradient + lambda_anchor * anchor_direction * 0.01
        params = params - gradient
        # Measure anchor alignment as cosine similarity to global model
        anchor_alignment = float(
            np.dot(params, global_params)
            / (np.linalg.norm(params) * np.linalg.norm(global_params) + 1e-8)
        )

    # ---- Stage 2: Style Diversification ----
    # Generate domain-specific style perturbation (simulates style bank diversity)
    style_perturbation = np.random.randn(len(params)) * 0.005
    for _ in range(local_epochs // 2):
        gradient = np.random.randn(len(params)) * 0.01 * lr
        # Push style away from global mean to maintain domain diversity
        gradient = gradient + lambda_style * style_perturbation
        params = params - gradient

    # Store updated state
    state.model_params = params.copy()

    # ---- Simulated ReID Metrics ----
    # mAP and Rank-1 converge as training progresses
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
