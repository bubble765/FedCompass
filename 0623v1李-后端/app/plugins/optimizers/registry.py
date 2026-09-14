"""Optimizer registry: maps optimizer names to their SAM variants."""
from app.plugins.optimizers.sam import sam_step, esam_step, hsam_step


OPTIMIZER_REGISTRY = {
    "sgd": None,         # Standard SGD, no SAM
    "sam": sam_step,
    "esam": esam_step,
    "hsam": hsam_step,
}


def apply_optimizer(optimizer_name, params, gradient, **kwargs):
    """Apply the named optimizer step to params using gradient."""
    step_fn = OPTIMIZER_REGISTRY.get(optimizer_name)
    if step_fn is None:
        # Standard SGD
        return params - gradient * kwargs.get("lr", 0.01), gradient
    elif optimizer_name == "hsam":
        new_params, final_grad = step_fn(params, gradient, **kwargs)
        return new_params - final_grad * kwargs.get("lr", 0.01), final_grad
    elif optimizer_name == "esam":
        new_params, final_grad, _ = step_fn(params, gradient, **kwargs)
        return new_params - final_grad * kwargs.get("lr", 0.01), final_grad
    else:
        new_params, final_grad = step_fn(params, gradient, **kwargs)
        return new_params - final_grad * kwargs.get("lr", 0.01), final_grad
"""Optimizer registry: maps optimizer names to their SAM variants."""
from app.plugins.optimizers.sam import sam_step
from app.plugins.optimizers.esam import esam_step
from app.plugins.optimizers.hsam import hsam_step


OPTIMIZER_REGISTRY = {
    "sgd": None,         # Standard SGD, no SAM
    "sam": sam_step,
    "esam": esam_step,
    "hsam": hsam_step,
}


def apply_optimizer(optimizer_name, params, gradient, **kwargs):
    """Apply the named optimizer step to params using gradient."""
    step_fn = OPTIMIZER_REGISTRY.get(optimizer_name)
    if step_fn is None:
        # Standard SGD
        return params - gradient * kwargs.get("lr", 0.01), gradient
    elif optimizer_name == "hsam":
        new_params, final_grad = step_fn(params, gradient, **kwargs)
        return new_params - final_grad * kwargs.get("lr", 0.01), final_grad
    elif optimizer_name == "esam":
        new_params, final_grad, _ = step_fn(params, gradient, **kwargs)
        return new_params - final_grad * kwargs.get("lr", 0.01), final_grad
    else:
        new_params, final_grad = step_fn(params, gradient, **kwargs)
        return new_params - final_grad * kwargs.get("lr", 0.01), final_grad
