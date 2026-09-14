"""Real federated training helpers for CIFAR experiments."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from app.core.config import get_real_data_root

REAL_DATA_ROOT = get_real_data_root()
VISION_CLASS_COUNTS = {
    "mnist": 10,
    "fashion_mnist": 10,
    "emnist": 47,
    "femnist": 62,
    "cifar10": 10,
    "cifar100": 100,
    "tiny_imagenet": 200,
    "ag_news": 4,
}
REAL_CIFAR_DIRS = {
    "cifar10": REAL_DATA_ROOT / "cifar-10-batches-py",
    "cifar100": REAL_DATA_ROOT / "cifar-100-python",
}
_DATA_CACHE: dict[tuple, "PreparedFederatedDataset"] = {}


class SmallCifarCNN(nn.Module):
    """A lightweight CNN that keeps local CPU training reasonably fast."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


@dataclass
class PreparedFederatedDataset:
    train_dataset: torch.utils.data.Dataset
    test_dataset: torch.utils.data.Dataset
    client_indices: list[list[int]]
    num_classes: int
    data_source: str = "real"


class SyntheticVisionDataset(Dataset):
    """Backend-generated tensor benchmark used when public datasets are absent."""

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images.float().contiguous()
        self.labels = labels.long().contiguous()
        self.targets = self.labels.cpu().tolist()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return self.images[index], self.labels[index]


def choose_device(preferred: str | None = None) -> torch.device:
    preferred = (preferred or "").lower()
    if preferred == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if preferred != "cpu" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(network_id: str, num_classes: int) -> nn.Module:
    network_id = (network_id or "").lower()
    if network_id in {"", "small_cnn", "demo_cnn", "lenet5", "textcnn"}:
        return SmallCifarCNN(num_classes)
    if network_id == "resnet18":
        return models.resnet18(weights=None, num_classes=num_classes)
    if network_id == "mobilenetv2":
        return models.mobilenet_v2(weights=None, num_classes=num_classes)
    return SmallCifarCNN(num_classes)


def prepare_federated_dataset(
    dataset_name: str,
    split: str,
    num_clients: int,
    seed: int,
    data_source: str = "auto",
    train_samples: int | None = None,
    test_samples: int | None = None,
) -> PreparedFederatedDataset:
    dataset_name = (dataset_name or "cifar10").lower()
    requested_source = (data_source or "auto").lower()
    use_real_dataset = requested_source == "real" or (requested_source == "auto" and _has_real_vision_dataset(dataset_name))
    if use_real_dataset and not _has_real_vision_dataset(dataset_name):
        use_real_dataset = False

    resolved_source = "real" if use_real_dataset else "backend_synthetic"
    train_count = int(train_samples or _default_synthetic_train_samples(dataset_name, num_clients))
    test_count = int(test_samples or _default_synthetic_test_samples(dataset_name))
    cache_key = (dataset_name, split, num_clients, seed, resolved_source, train_count, test_count)
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if use_real_dataset:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            ]
        )
        if dataset_name == "cifar100":
            train_dataset = datasets.CIFAR100(root=str(REAL_DATA_ROOT), train=True, download=False, transform=transform)
            test_dataset = datasets.CIFAR100(root=str(REAL_DATA_ROOT), train=False, download=False, transform=transform)
            num_classes = 100
        else:
            train_dataset = datasets.CIFAR10(root=str(REAL_DATA_ROOT), train=True, download=False, transform=transform)
            test_dataset = datasets.CIFAR10(root=str(REAL_DATA_ROOT), train=False, download=False, transform=transform)
            num_classes = 10
        targets = np.asarray(train_dataset.targets)
    else:
        num_classes = VISION_CLASS_COUNTS.get(dataset_name, 10)
        train_dataset = _make_backend_synthetic_dataset(
            num_samples=train_count,
            num_classes=num_classes,
            seed=seed,
            dataset_name=dataset_name,
            train=True,
        )
        test_dataset = _make_backend_synthetic_dataset(
            num_samples=test_count,
            num_classes=num_classes,
            seed=seed + 10_000,
            dataset_name=dataset_name,
            train=False,
        )
        targets = np.asarray(train_dataset.targets)

    if split == "dirichlet":
        client_indices = _dirichlet_partition(targets, num_clients, seed)
    elif split == "pathological":
        client_indices = _pathological_partition(targets, num_clients, seed)
    else:
        client_indices = _iid_partition(len(train_dataset), num_clients, seed)

    prepared = PreparedFederatedDataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_indices=client_indices,
        num_classes=num_classes,
        data_source=resolved_source,
    )
    _DATA_CACHE[cache_key] = prepared
    return prepared


def state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def state_dict_to_vector(state_dict: dict[str, torch.Tensor]) -> tuple[np.ndarray, list[tuple[str, tuple[int, ...], int, np.dtype]]]:
    vector_parts: list[np.ndarray] = []
    spec: list[tuple[str, tuple[int, ...], int, np.dtype]] = []
    for key, tensor in state_dict.items():
        arr = tensor.detach().cpu().numpy()
        flat = arr.reshape(-1).astype(np.float32, copy=False)
        vector_parts.append(flat)
        spec.append((key, tuple(arr.shape), flat.size, arr.dtype))
    return np.concatenate(vector_parts, axis=0), spec


def vector_to_state_dict(
    vector: np.ndarray,
    spec: list[tuple[str, tuple[int, ...], int, np.dtype]],
    reference_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    rebuilt: dict[str, torch.Tensor] = {}
    cursor = 0
    for key, shape, size, dtype in spec:
        chunk = vector[cursor:cursor + size]
        cursor += size
        arr = chunk.reshape(shape).astype(dtype, copy=False)
        rebuilt[key] = torch.from_numpy(arr.copy()).to(reference_state[key].dtype)
    return rebuilt


def local_train(
    global_state: dict[str, torch.Tensor],
    dataset: torch.utils.data.Dataset,
    indices: list[int],
    config: dict,
    round_num: int,
    client_id: int,
    num_classes: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    device = choose_device(config.get("device"))
    torch.manual_seed(int(config.get("seed", 42)) + round_num * 1009 + client_id)
    rng = np.random.default_rng(int(config.get("seed", 42)) + round_num * 1009 + client_id)
    sampled_indices = _sample_client_indices(indices, int(config.get("train_sample_cap_per_client", 256)), rng)
    if not sampled_indices:
        return state_dict_to_cpu(global_state), {
            "train_loss": 0.0,
            "num_samples": 0,
            "gradient_drift": 0.0,
        }
    batch_size = max(int(config.get("batch_size", 64)), 1)
    local_epochs = max(int(config.get("local_epochs", 1)), 1)
    learning_rate = float(config.get("learning_rate", 0.01))
    weight_decay = float(config.get("weight_decay", 0.0))
    algorithm = str(config.get("algorithm", "fedavg")).lower()
    optimizer_name = str(config.get("optimizer", "sgd")).lower()
    prox_mu = float(config.get("prox_mu", 0.01))

    model = build_model(config.get("network", "resnet18"), num_classes).to(device)
    model.load_state_dict(global_state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    loader = DataLoader(Subset(dataset, sampled_indices), batch_size=batch_size, shuffle=True, num_workers=0)
    global_reference = {name: tensor.detach().to(device) for name, tensor in global_state.items()}
    teacher_model = _build_teacher_model(global_state, config, num_classes, device) if _uses_distillation(algorithm) else None

    total_loss = 0.0
    total_examples = 0
    local_steps = 0
    for _ in range(local_epochs):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if _uses_sam(algorithm, optimizer_name):
                loss = _sam_step(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    images=images,
                    labels=labels,
                    algorithm=algorithm,
                    config=config,
                    global_reference=global_reference,
                    teacher_model=teacher_model,
                    prox_mu=prox_mu,
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = _compute_supervised_loss(
                    model=model,
                    criterion=criterion,
                    images=images,
                    labels=labels,
                    algorithm=algorithm,
                    config=config,
                    global_reference=global_reference,
                    teacher_model=teacher_model,
                    prox_mu=prox_mu,
                )
                loss.backward()
                optimizer.step()

            batch_size_actual = int(labels.size(0))
            total_loss += float(loss.detach().item()) * batch_size_actual
            total_examples += batch_size_actual
            local_steps += 1

    updated_state = state_dict_to_cpu(model.state_dict())
    loss_value = total_loss / max(total_examples, 1)
    drift = compute_model_drift(global_state, updated_state)
    return updated_state, {
        "train_loss": float(loss_value),
        "num_samples": len(sampled_indices),
        "gradient_drift": float(drift),
        "local_steps": int(local_steps),
        "algorithm_strategy": _algorithm_strategy_name(algorithm, optimizer_name),
    }


def aggregate_state_dicts(
    client_states: list[dict[str, torch.Tensor]],
    weights: list[float],
    algorithm: str = "fedavg",
    server_state: dict | None = None,
    global_state: dict[str, torch.Tensor] | None = None,
    round_num: int = 1,
) -> dict[str, torch.Tensor]:
    total_weight = float(sum(weights)) or 1.0
    if sum(weights) <= 0:
        return state_dict_to_cpu(client_states[0])
    algorithm = (algorithm or "fedavg").lower()
    normalized = [float(weight) / total_weight for weight in weights]
    aggregated: dict[str, torch.Tensor] = {}
    for key in client_states[0]:
        if torch.is_floating_point(client_states[0][key]):
            if algorithm == "fedbn" and global_state is not None and _is_batchnorm_key(key):
                aggregated[key] = global_state[key].clone()
                continue
            stacked = torch.stack([state[key].float() * normalized[idx] for idx, state in enumerate(client_states)], dim=0)
            aggregated[key] = stacked.sum(dim=0).to(client_states[0][key].dtype)
        else:
            aggregated[key] = client_states[-1][key].clone()

    if algorithm in {"fedadam", "fedyogi"} and global_state is not None:
        aggregated = _apply_server_optimizer(
            target_state=aggregated,
            global_state=global_state,
            server_state=server_state if server_state is not None else {},
            algorithm=algorithm,
            round_num=round_num,
        )
    return aggregated


def _compute_supervised_loss(
    *,
    model: nn.Module,
    criterion: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    algorithm: str,
    config: dict,
    global_reference: dict[str, torch.Tensor],
    teacher_model: nn.Module | None,
    prox_mu: float,
) -> torch.Tensor:
    logits = model(images)
    loss = criterion(logits, labels)

    if algorithm in {"fedprox", "ditto", "pfedme", "feddyn", "fedcvc", "feddtc", "rfl_nlcp", "a_fedpd", "a_fedpdsam"}:
        coefficient = _proximal_coefficient(algorithm, config, prox_mu)
        prox_term = torch.zeros(1, device=images.device)
        for name, param in model.named_parameters():
            if name in global_reference and torch.is_floating_point(param):
                prox_term = prox_term + torch.mean((param - global_reference[name]) ** 2)
        loss = loss + coefficient * prox_term

    if algorithm in {"fedproto", "fedgkd_p"}:
        class_confidence = torch.softmax(logits, dim=1)
        prototype_compactness = torch.mean(class_confidence * (1.0 - class_confidence))
        loss = loss + 0.05 * prototype_compactness

    if teacher_model is not None:
        temperature = max(float(config.get("temperature", 2.0)), 1e-3)
        distill_weight = float(config.get("distill_weight", config.get("global_distill_weight", 0.4)))
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        kd_loss = F.kl_div(
            F.log_softmax(logits / temperature, dim=1),
            F.softmax(teacher_logits / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature * temperature)
        loss = loss + distill_weight * kd_loss

    return loss


def _sam_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    algorithm: str,
    config: dict,
    global_reference: dict[str, torch.Tensor],
    teacher_model: nn.Module | None,
    prox_mu: float,
) -> torch.Tensor:
    rho = float(config.get("rho", 0.05))
    optimizer.zero_grad(set_to_none=True)
    first_loss = _compute_supervised_loss(
        model=model,
        criterion=criterion,
        images=images,
        labels=labels,
        algorithm=algorithm,
        config=config,
        global_reference=global_reference,
        teacher_model=teacher_model,
        prox_mu=prox_mu,
    )
    first_loss.backward()
    grad_norm = _grad_norm(model)
    scale = rho / (grad_norm + 1e-12)
    perturbations: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue
            perturb = param.grad * scale
            param.add_(perturb)
            perturbations.append((param, perturb))

    optimizer.zero_grad(set_to_none=True)
    second_loss = _compute_supervised_loss(
        model=model,
        criterion=criterion,
        images=images,
        labels=labels,
        algorithm=algorithm,
        config=config,
        global_reference=global_reference,
        teacher_model=teacher_model,
        prox_mu=prox_mu,
    )
    second_loss.backward()
    with torch.no_grad():
        for param, perturb in perturbations:
            param.sub_(perturb)
    optimizer.step()
    return second_loss.detach()


def _grad_norm(model: nn.Module) -> torch.Tensor:
    norms = [torch.norm(param.grad.detach(), p=2) for param in model.parameters() if param.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.norm(torch.stack(norms), p=2)


def _build_teacher_model(
    global_state: dict[str, torch.Tensor],
    config: dict,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    teacher = build_model(config.get("network", "resnet18"), num_classes).to(device)
    teacher.load_state_dict(global_state)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def _uses_distillation(algorithm: str) -> bool:
    return algorithm in {"fedcads", "feddf", "fedkd", "fedgkd_p", "fedfld"}


def _uses_sam(algorithm: str, optimizer_name: str) -> bool:
    return optimizer_name in {"sam", "esam", "hsam"} or algorithm in {"gfed_hsam", "fedlesam_d", "a_fedpdsam", "fedgloss"}


def _proximal_coefficient(algorithm: str, config: dict, prox_mu: float) -> float:
    if algorithm == "ditto":
        return float(config.get("personalization_weight", 0.3)) * 0.02
    if algorithm == "pfedme":
        return float(config.get("moreau_mu", 0.05))
    if algorithm == "feddyn":
        return float(config.get("alpha", 0.01))
    if algorithm in {"fedcvc", "feddtc"}:
        return float(config.get("compensation_weight", 1.0)) * 0.01
    if algorithm == "rfl_nlcp":
        return 0.02
    return prox_mu


def _algorithm_strategy_name(algorithm: str, optimizer_name: str) -> str:
    if _uses_sam(algorithm, optimizer_name):
        return "sharpness_aware_local_update"
    if _uses_distillation(algorithm):
        return "global_teacher_distillation"
    if algorithm in {"fedprox", "ditto", "pfedme", "feddyn"}:
        return "proximal_regularized_local_update"
    if algorithm == "fedproto":
        return "prototype_compactness_regularization"
    return "cross_entropy_local_update"


def _is_batchnorm_key(key: str) -> bool:
    lowered = key.lower()
    return "bn" in lowered or "running_mean" in lowered or "running_var" in lowered or "num_batches_tracked" in lowered


def _apply_server_optimizer(
    *,
    target_state: dict[str, torch.Tensor],
    global_state: dict[str, torch.Tensor],
    server_state: dict,
    algorithm: str,
    round_num: int,
) -> dict[str, torch.Tensor]:
    beta1 = float(server_state.get("beta1", 0.9))
    beta2 = float(server_state.get("beta2", 0.99))
    server_lr = float(server_state.get("server_lr", 0.08 if algorithm == "fedadam" else 0.05))
    eps = float(server_state.get("eps", 1e-6))
    moments = server_state.setdefault("moments", {})
    variances = server_state.setdefault("variances", {})
    updated: dict[str, torch.Tensor] = {}

    for key, target_tensor in target_state.items():
        if not torch.is_floating_point(target_tensor):
            updated[key] = target_tensor.clone()
            continue
        delta = target_tensor.float() - global_state[key].float()
        m_prev = moments.get(key, torch.zeros_like(delta))
        v_prev = variances.get(key, torch.zeros_like(delta))
        m_t = beta1 * m_prev + (1.0 - beta1) * delta
        if algorithm == "fedyogi":
            v_t = v_prev - (1.0 - beta2) * torch.sign(v_prev - delta * delta) * delta * delta
        else:
            v_t = beta2 * v_prev + (1.0 - beta2) * delta * delta
        moments[key] = m_t.detach().clone()
        variances[key] = torch.clamp(v_t.detach().clone(), min=0.0)
        bias_correction_1 = 1.0 - beta1 ** max(round_num, 1)
        bias_correction_2 = 1.0 - beta2 ** max(round_num, 1)
        m_hat = m_t / max(bias_correction_1, eps)
        v_hat = variances[key] / max(bias_correction_2, eps)
        step = server_lr * m_hat / (torch.sqrt(v_hat) + eps)
        updated[key] = (global_state[key].float() + step).to(target_tensor.dtype)

    return updated


def evaluate_model(
    state_dict: dict[str, torch.Tensor],
    dataset: torch.utils.data.Dataset,
    config: dict,
    num_classes: int,
) -> dict:
    device = choose_device(config.get("device"))
    model = build_model(config.get("network", "resnet18"), num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_cap = int(config.get("eval_sample_cap", 1000))
    eval_indices = list(range(min(len(dataset), eval_cap)))
    loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=max(int(config.get("batch_size", 64)), 1),
        shuffle=False,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            batch_size_actual = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size_actual
            total_correct += int((preds == labels).sum().item())
            total_examples += batch_size_actual

    accuracy = total_correct / max(total_examples, 1)
    avg_loss = total_loss / max(total_examples, 1)
    return {
        "test_accuracy": float(accuracy),
        "eval_loss": float(avg_loss),
        "num_eval_samples": total_examples,
    }


def estimate_communication_cost_mb(state_dict: dict[str, torch.Tensor], n_clients: int) -> float:
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return round((total_bytes * n_clients) / (1024 * 1024), 4)


def compute_model_drift(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    total = 0.0
    count = 0
    for key in before:
        diff = (after[key].float() - before[key].float()).reshape(-1)
        total += float(torch.norm(diff, p=2).item())
        count += 1
    return total / max(count, 1)


def _has_real_vision_dataset(dataset_name: str) -> bool:
    real_dir = REAL_CIFAR_DIRS.get((dataset_name or "").lower())
    return bool(real_dir and real_dir.exists())


def _default_synthetic_train_samples(dataset_name: str, num_clients: int) -> int:
    num_classes = VISION_CLASS_COUNTS.get((dataset_name or "").lower(), 10)
    return max(num_clients * 48, num_classes * 24, 384)


def _default_synthetic_test_samples(dataset_name: str) -> int:
    num_classes = VISION_CLASS_COUNTS.get((dataset_name or "").lower(), 10)
    return max(num_classes * 12, 128)


def _make_backend_synthetic_dataset(
    *,
    num_samples: int,
    num_classes: int,
    seed: int,
    dataset_name: str,
    train: bool,
) -> SyntheticVisionDataset:
    images, labels = _generate_synthetic_vision_tensors(
        num_samples=max(int(num_samples), num_classes),
        num_classes=num_classes,
        seed=seed,
        dataset_name=dataset_name,
        train=train,
    )
    return SyntheticVisionDataset(images, labels)


def _generate_synthetic_vision_tensors(
    *,
    num_samples: int,
    num_classes: int,
    seed: int,
    dataset_name: str,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    channels, height, width = 3, 32, 32
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    yy = yy / max(height - 1, 1)
    xx = xx / max(width - 1, 1)
    prototypes = np.zeros((num_classes, channels, height, width), dtype=np.float32)
    dataset_shift = (sum(ord(ch) for ch in dataset_name) % 17) / 37.0
    for class_id in range(num_classes):
        frequency = 1.0 + (class_id % 7)
        phase = dataset_shift + class_id / max(num_classes, 1)
        wave = np.sin((xx * frequency + phase) * np.pi) + np.cos((yy * (frequency + 1.0) - phase) * np.pi)
        blob_x = ((class_id * 13) % width) / max(width - 1, 1)
        blob_y = ((class_id * 7) % height) / max(height - 1, 1)
        blob = np.exp(-((xx - blob_x) ** 2 + (yy - blob_y) ** 2) / 0.045)
        color = rng.normal(0.0, 0.12, size=(channels, 1, 1)).astype(np.float32)
        prototypes[class_id] = (0.55 * wave + 0.85 * blob)[None, :, :] + color

    labels = np.arange(num_samples, dtype=np.int64) % num_classes
    rng.shuffle(labels)
    noise_scale = 0.34 if train else 0.28
    images = prototypes[labels] + rng.normal(0.0, noise_scale, size=(num_samples, channels, height, width)).astype(np.float32)
    images = (images - images.mean(axis=(1, 2, 3), keepdims=True)) / (images.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
    return torch.from_numpy(images), torch.from_numpy(labels)


def _sample_client_indices(indices: list[int], sample_cap: int, rng: np.random.Generator) -> list[int]:
    if sample_cap <= 0 or len(indices) <= sample_cap:
        return list(indices)
    sampled = rng.choice(np.asarray(indices), size=sample_cap, replace=False).tolist()
    sampled.sort()
    return sampled


def _iid_partition(n_samples: int, num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)
    splits = np.array_split(shuffled, num_clients)
    return [split.astype(int).tolist() for split in splits]


def _dirichlet_partition(targets: np.ndarray, num_clients: int, seed: int, alpha: float = 0.5) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    num_classes = int(targets.max()) + 1
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for class_id in range(num_classes):
        class_indices = np.where(targets == class_id)[0]
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        cut_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        splits = np.split(class_indices, cut_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.astype(int).tolist())
    for indices in client_indices:
        indices.sort()
    return client_indices


def _pathological_partition(targets: np.ndarray, num_clients: int, seed: int, shards_per_client: int = 2) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    sorted_indices = np.argsort(targets)
    num_shards = num_clients * shards_per_client
    shard_size = math.ceil(len(sorted_indices) / num_shards)
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(num_shards)]
    shard_order = rng.permutation(len(shards))
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for client_id in range(num_clients):
        owned = shard_order[client_id * shards_per_client:(client_id + 1) * shards_per_client]
        for shard_idx in owned:
            client_indices[client_id].extend(shards[shard_idx].astype(int).tolist())
        client_indices[client_id].sort()
    return client_indices
