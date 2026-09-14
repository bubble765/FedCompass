"""FedULIP: Federated ULIP for 3D multimodal learning.

Mock implementation that simulates FedULIP training for point cloud/text/image multimodal tasks.
Supports base-to-new, cross-dataset, and domain ABCD evaluation protocols.
"""
import numpy as np


def fedulip_aggregate(client_params, weights, client_states=None):
    """FedULIP aggregation: adapter-aware federated aggregation for 3D multimodal."""
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / (weights.sum() + 1e-8)
    stacked = np.stack(client_params)
    aggregated = np.average(stacked, axis=0, weights=weights)
    return aggregated, {"method": "fedulip", "n_clients": len(client_params)}


def fedulip_local_train(state, global_params, round_num, config):
    """FedULIP local training with adapter/SACA mechanism.

    Key components:
    - Frozen ULIP backbone (pretrained 3D point cloud encoder)
    - Trainable SACA adapter for multimodal alignment
    - Cross-modal contrastive loss between point cloud, text, and image features
    """
    lr = config.get("learning_rate", 0.001)
    adapter_type = config.get("adapter_type", "saca")
    local_epochs = config.get("local_epochs", 3)
    np.random.seed(round_num * 1000 + state.client_id)

    if state.model_params is None:
        state.model_params = global_params.copy()

    params = global_params.copy()
    modality_align = 0

    for _ in range(local_epochs):
        gradient = np.random.randn(len(params)) * 0.01 * lr

        # SACA adapter: spatial-aware cross-modal alignment
        if adapter_type == "saca":
            # Simulate three-modality alignment: point cloud, text, image
            pc_grad = gradient[:len(gradient)//3]
            text_grad = gradient[len(gradient)//3:2*len(gradient)//3]
            img_grad = gradient[2*len(gradient)//3:]

            # Cross-modal contrastive: pull modalities together
            pc_norm = np.linalg.norm(pc_grad) + 1e-8
            text_norm = np.linalg.norm(text_grad) + 1e-8
            img_norm = np.linalg.norm(img_grad) + 1e-8

            modality_align = float(
                np.dot(pc_grad, text_grad) / (pc_norm * text_norm)
                + np.dot(pc_grad, img_grad) / (pc_norm * img_norm)
            ) / 2

            # Adapter update: small perturbation
            adapter_update = np.random.randn(len(params)) * 0.001
            params -= gradient * lr + adapter_update
        else:
            params -= gradient * lr

    state.model_params = params.copy()

    # 3D metrics: classification accuracy (simulated)
    base_acc = 0.45 + 0.30 * (1 - np.exp(-0.04 * round_num))
    accuracy = float(base_acc + np.random.uniform(-0.03, 0.03))
    train_loss = float(3.5 * np.exp(-0.02 * round_num) + 0.15 * np.random.random())

    return params, {
        "train_loss": train_loss,
        "test_accuracy": accuracy,
        "modality_alignment": modality_align,
        "adapter_type": adapter_type,
    }


# ── Mock runner for 3D multimodal scenarios ──

def generate_ulip3d_metrics(protocol="base2new", dataset="modelnet40", rounds=100):
    """Generate mock FedULIP metrics for 3D multimodal experiments.

    Protocols:
    - base2new: train on base classes, test on new classes
    - cross_dataset: train on one dataset, test on another
    - domain_abcd: domain generalization across A, B, C, D
    """
    np.random.seed(hash(f"fedulip_{protocol}_{dataset}") % 2**31)

    protocol_config = {
        "base2new": {"base_acc": 0.72, "new_acc": 0.55},
        "cross_dataset": {"source_acc": 0.68, "target_acc": 0.48},
        "domain_abcd": {"domain_a": 0.75, "domain_b": 0.65, "domain_c": 0.58, "domain_d": 0.52},
    }

    config = protocol_config.get(protocol, protocol_config["base2new"])
    history = []

    for r in range(1, rounds + 1):
        progress = r / rounds
        loss = max(0.05, 4.0 * np.exp(-2 * progress) + np.random.uniform(-0.15, 0.15))

        entry = {
            "round": r,
            "train_loss": round(float(loss), 4),
        }

        for metric_name, base_val in config.items():
            val = min(0.95, base_val * (1 - np.exp(-5 * progress)) * 1.1 + np.random.uniform(-0.03, 0.03))
            entry[metric_name] = round(float(val), 4)

        history.append(entry)

    return history

