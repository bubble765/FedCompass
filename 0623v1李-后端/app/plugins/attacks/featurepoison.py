"""FeaturePoison: Feature-level poisoning attack that manipulates prototype representations."""
import numpy as np


def featurepoison(state, params, global_params, round_num, poison_ratio=0.2):
    """Apply feature-level poisoning: perturb parameters to misalign prototypes."""
    np.random.seed(round_num * 10000 + state.client_id)

    should_poison = np.random.random() < poison_ratio

    if not should_poison:
        return params

    # Feature poisoning: add structured noise to mislead the global model
    update = params - global_params
    update_norm = np.linalg.norm(update) + 1e-8
    if not np.isfinite(update_norm) or update_norm <= 1e-8:
        return params

    # Poison: flip the update direction, but keep the perturbation bounded so
    # the real PyTorch training branch does not numerically explode.
    poisoned_update = -update * 1.2
    poisoned_params = global_params + poisoned_update

    # Add structured noise to specific dimensions (simulating prototype manipulation)
    dim = len(params)
    prototype_dims = np.random.choice(dim, size=max(1, dim // 10), replace=False)
    noise = np.random.randn(len(prototype_dims)) * min(update_norm * 0.35, 0.5)
    poisoned_params[prototype_dims] += noise

    poisoned_update = poisoned_params - global_params
    poisoned_norm = np.linalg.norm(poisoned_update)
    max_allowed_norm = max(update_norm * 2.0, 1.0)
    if not np.isfinite(poisoned_norm):
        return params
    if poisoned_norm > max_allowed_norm:
        poisoned_update = poisoned_update / (poisoned_norm + 1e-8) * max_allowed_norm
        poisoned_params = global_params + poisoned_update

    poisoned_params = np.nan_to_num(
        poisoned_params,
        nan=0.0,
        posinf=1e3,
        neginf=-1e3,
    )

    return poisoned_params


def featurepoison_prototype(protos, global_protos, poison_ratio=0.2):
    """Prototype-level poisoning: manipulate class prototypes to cause misclassification."""
    if not global_protos:
        return protos

    poisoned = dict(protos)
    labels = sorted(protos.keys())
    if len(labels) >= 2:
        for i in range(0, len(labels) - 1, 2):
            if np.random.random() < poison_ratio:
                poisoned[labels[i]], poisoned[labels[i + 1]] = (
                    protos[labels[i + 1]].copy(),
                    protos[labels[i]].copy(),
                )

    return poisoned
