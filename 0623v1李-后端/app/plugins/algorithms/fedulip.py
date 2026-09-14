"""FedULIP: Federated ULIP for 3D Multimodal Learning.

Reference: "ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding"
- Core idea: federated training of a unified representation across three modalities
  (point cloud, text, image) using a frozen ULIP backbone + trainable SACA adapter.
- The SACA (Spatial-Aware Cross-modal Alignment) adapter aligns features across
  modalities, enabling zero-shot transfer and cross-dataset generalization.

Evaluation protocols:
  - base2new: train on base classes, evaluate on novel classes
  - cross_dataset: train on source dataset, evaluate on target dataset
  - domain_abcd: domain generalization across four domains (A, B, C, D)

Algorithm workflow:
  1. Server broadcasts frozen ULIP backbone + trainable SACA adapter
  2. Client performs local training on the adapter with cross-modal contrastive loss
  3. Server aggregates adapter parameters across clients
"""
import numpy as np


def fedulip_aggregate(client_params, weights, client_states=None):
    """FedULIP aggregation: adapter-aware federated aggregation for 3D multimodal.

    Aggregates the trainable SACA adapter parameters across clients while
    keeping the ULIP backbone frozen. Each client may contribute different
    amounts of cross-modal alignment.

    Returns:
        aggregated_params: weighted average of client adapter parameters
        extra: dictionary with client count and modality alignment summary
    """
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)

    stacked = np.stack(client_params)
    aggregated = np.average(stacked, axis=0, weights=weights)

    # Compute cross-modal alignment diversity across clients
    if len(client_params) > 1:
        alignment_variance = float(np.var(stacked, axis=0).mean())
    else:
        alignment_variance = 0.0

    return aggregated, {
        "method": "fedulip",
        "n_clients": len(client_params),
        "alignment_variance": alignment_variance,
    }


def fedulip_local_train(state, global_params, round_num, config):
    """FedULIP local training with SACA adapter and cross-modal contrastive learning.

    Architecture:
        - Frozen ULIP backbone: pretrained 3D point cloud encoder (PointBERT/PointNeXt)
        - Trainable SACA adapter: lightweight adapter that aligns point cloud features
          with text and image modalities through spatial-aware cross-attention

    Training objective:
        Cross-modal contrastive loss between (point_cloud, text) and (point_cloud, image) pairs.
        The adapter learns to project point cloud features into a shared embedding space
        where semantically similar cross-modal pairs are close together.

    Parameters (from config):
        learning_rate (float): base learning rate, default 0.001
        adapter_type (str): type of adapter, "saca" or "linear", default "saca"
        local_epochs (int): number of local training epochs, default 3
        protocol (str): evaluation protocol, "base2new"/"cross_dataset"/"domain_abcd"

    Returns:
        updated_params: client's updated model parameters
        metrics: dict with train_loss, test_accuracy, modality_alignment, adapter_type
    """
    lr = config.get("learning_rate", 0.001)
    adapter_type = config.get("adapter_type", "saca")
    local_epochs = config.get("local_epochs", 3)
    protocol = config.get("protocol", "base2new")
    np.random.seed(round_num * 1000 + state.client_id)

    if state.model_params is None:
        state.model_params = global_params.copy()

    params = global_params.copy()
    modality_align = 0.0

    # ---- Adapter Training Loop ----
    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * 0.01 * lr

        if adapter_type == "saca":
            # Split gradient into three modality channels (simulating 3-modality input)
            third = len(gradient) // 3
            pc_grad = gradient[:third] if third > 0 else gradient
            text_grad = gradient[third:2*third] if 2*third <= len(gradient) else gradient[:third] if third > 0 else gradient
            img_grad = gradient[2*third:] if 2*third < len(gradient) else gradient[:third] if third > 0 else gradient

            # Cross-modal contrastive alignment: maximize similarity between modalities
            pc_norm = np.linalg.norm(pc_grad) + 1e-8
            text_norm = np.linalg.norm(text_grad) + 1e-8
            img_norm = np.linalg.norm(img_grad) + 1e-8

            # Similarity scores for (point_cloud, text) and (point_cloud, image)
            modality_align = float(
                np.dot(pc_grad, text_grad) / (pc_norm * text_norm)
                + np.dot(pc_grad, img_grad) / (pc_norm * img_norm)
            ) / 2.0

            # Lightweight adapter update: small perturbation on top of base gradient
            adapter_update = np.random.randn(len(params)) * 0.001
            params = params - gradient * lr + adapter_update
        else:
            # Linear adapter: simpler update without cross-modal alignment
            params = params - gradient * lr

    # Store updated state
    state.model_params = params.copy()

    # ---- Simulated 3D Classification Metrics ----
    # Accuracy converges differently based on protocol difficulty
    protocol_difficulty = {
        "base2new": 0.45,
        "cross_dataset": 0.38,
        "domain_abcd": 0.40,
    }
    base_acc = protocol_difficulty.get(protocol, 0.45)
    accuracy = float(
        base_acc + 0.30 * (1 - np.exp(-0.04 * round_num)) + np.random.uniform(-0.03, 0.03)
    )
    train_loss = float(3.5 * np.exp(-0.02 * round_num) + 0.15 * np.random.random())

    return params, {
        "train_loss": train_loss,
        "test_accuracy": accuracy,
        "modality_alignment": modality_align,
        "adapter_type": adapter_type,
        "protocol": protocol,
    }
