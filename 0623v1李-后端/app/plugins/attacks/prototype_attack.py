"""PrototypeAttack: FedProto-style prototype poisoning attack."""
import numpy as np


def prototype_attack(state, params, global_params, round_num, poison_ratio=0.2):
    """Manipulate class prototypes by injecting adversarial patterns."""
    np.random.seed(round_num * 10000 + state.client_id)

    should_poison = np.random.random() < poison_ratio
    if not should_poison:
        return params

    # Prototype attack: create an adversarial update that shifts prototypes
    update = params - global_params
    update_norm = np.linalg.norm(update) + 1e-8

    # Generate adversarial direction orthogonal to the true gradient
    random_dir = np.random.randn(len(params))
    random_dir -= np.dot(random_dir, update) * update / (update_norm ** 2 + 1e-8)
    random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8)

    # Shift parameters along the adversarial direction
    poisoned = params + random_dir * update_norm * 5.0
    return poisoned
