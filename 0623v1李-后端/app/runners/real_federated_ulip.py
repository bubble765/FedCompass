"""Real federated training helpers for lightweight ULIP3D-style experiments."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from app.core.config import get_real_data_root

REAL_DATA_ROOT = get_real_data_root()
_DATA_CACHE: dict[tuple[str, str, int, int], "PreparedFederatedULIPDataset"] = {}
ULIP_CLASS_COUNTS = {
    "modelnet40": 40,
    "scanobjectnn": 15,
    "shapenetcore": 55,
    "mvtec3d": 15,
    "mnist3d": 10,
    "3dimage": 10,
}


class PointCloudDataset(Dataset):
    def __init__(self, points: np.ndarray, labels: np.ndarray, augment: bool):
        self.points = points.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int):
        pts = self.points[index].copy()
        if self.augment:
            jitter = np.random.normal(0.0, 0.01, pts.shape).astype(np.float32)
            pts = pts + jitter
        pts = pts - pts.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(pts, axis=1).max()
        if scale > 0:
            pts = pts / scale
        return torch.from_numpy(pts), int(self.labels[index])


@dataclass
class PreparedFederatedULIPDataset:
    train_dataset: PointCloudDataset
    test_dataset: PointCloudDataset
    client_indices: list[list[int]]
    num_classes: int
    class_names: list[str]
    data_source: str = "real"


class TinyPointBERTLike(nn.Module):
    """A lightweight point encoder that is cheap enough for local CPU runs."""

    def __init__(self, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embedding_dim),
            nn.ReLU(inplace=True),
        )
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [B, N, 3]
        point_features = self.point_mlp(x)
        pooled = point_features.mean(dim=1)
        adapted = pooled + self.adapter(pooled)
        logits = self.classifier(adapted)
        return adapted, logits


def choose_device(preferred: str | None = None) -> torch.device:
    if (preferred or "").lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_ulip_model(network_id: str, num_classes: int, embedding_dim: int = 256) -> nn.Module:
    return TinyPointBERTLike(num_classes=num_classes, embedding_dim=embedding_dim)


def prepare_federated_ulip_dataset(
    dataset_name: str,
    split: str,
    num_clients: int,
    seed: int,
    data_source: str = "auto",
) -> PreparedFederatedULIPDataset:
    requested_source = (data_source or "auto").lower()
    cache_key = (dataset_name, split, num_clients, seed, requested_source)
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if requested_source == "backend_synthetic":
        train_points, train_labels, test_points, test_labels, class_names = _generate_synthetic_pointclouds(seed, dataset_name)
        resolved_source = "backend_synthetic"
    else:
        try:
            dataset_file = resolve_ulip_dataset_file(dataset_name)
        except FileNotFoundError:
            if requested_source == "real":
                raise
            train_points, train_labels, test_points, test_labels, class_names = _generate_synthetic_pointclouds(seed, dataset_name)
            resolved_source = "backend_synthetic"
        else:
            payload = np.load(dataset_file, allow_pickle=True)
            train_points = payload["train_points"].astype(np.float32)
            train_labels = payload["train_labels"].astype(np.int64)
            test_points = payload["test_points"].astype(np.float32)
            test_labels = payload["test_labels"].astype(np.int64)
            class_names = payload["class_names"].tolist()
            resolved_source = "real"

    train_dataset = PointCloudDataset(train_points, train_labels, augment=True)
    test_dataset = PointCloudDataset(test_points, test_labels, augment=False)

    if split == "dirichlet":
        client_indices = _dirichlet_partition(train_labels, num_clients, seed)
    else:
        client_indices = _iid_partition(len(train_dataset), num_clients, seed)

    prepared = PreparedFederatedULIPDataset(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_indices=client_indices,
        num_classes=len(class_names),
        class_names=class_names,
        data_source=resolved_source,
    )
    _DATA_CACHE[cache_key] = prepared
    return prepared


def resolve_ulip_dataset_file(dataset_name: str) -> Path:
    candidates = {
        "modelnet40": [
            REAL_DATA_ROOT / "modelnet40_mini.npz",
            REAL_DATA_ROOT / "modelnet40_mini" / "modelnet40_mini.npz",
        ],
        "scanobjectnn": [REAL_DATA_ROOT / "scanobjectnn_mini.npz"],
        "shapenetcore": [REAL_DATA_ROOT / "shapenetcore_mini.npz"],
        "mvtec3d": [REAL_DATA_ROOT / "mvtec3d_mini.npz"],
        "mnist3d": [REAL_DATA_ROOT / "mnist3d_mini.npz"],
        "3dimage": [REAL_DATA_ROOT / "3dimage_mini.npz"],
    }.get(dataset_name.lower(), [])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a prepared ULIP dataset for {dataset_name!r} under {REAL_DATA_ROOT}")


def _generate_synthetic_pointclouds(seed: int, dataset_name: str = "modelnet40") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(seed)
    num_classes = ULIP_CLASS_COUNTS.get((dataset_name or "").lower(), 10)
    train_per_class = 24
    test_per_class = 8
    num_points = 64
    class_names = [f"synthetic_shape_{idx:02d}" for idx in range(num_classes)]
    centers = rng.normal(0.0, 0.65, size=(num_classes, 3)).astype(np.float32)

    def build_split(per_class: int, noise: float) -> tuple[np.ndarray, np.ndarray]:
        clouds = []
        labels = []
        for class_id in range(num_classes):
            for sample_id in range(per_class):
                theta = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False, dtype=np.float32)
                radius = 0.35 + 0.04 * (class_id % 5)
                z = np.linspace(-0.45, 0.45, num_points, dtype=np.float32)
                base = np.stack(
                    [
                        radius * np.cos(theta + class_id * 0.11),
                        radius * np.sin(theta + sample_id * 0.07),
                        z * (0.6 + 0.08 * (class_id % 4)),
                    ],
                    axis=1,
                )
                rotation = _rotation_matrix(float((class_id * 13 + sample_id) % 360) / 180.0 * np.pi)
                cloud = base @ rotation.T + centers[class_id]
                cloud += rng.normal(0.0, noise, size=cloud.shape).astype(np.float32)
                clouds.append(cloud.astype(np.float32))
                labels.append(class_id)
        order = rng.permutation(len(clouds))
        return np.asarray(clouds, dtype=np.float32)[order], np.asarray(labels, dtype=np.int64)[order]

    train_points, train_labels = build_split(train_per_class, 0.035)
    test_points, test_labels = build_split(test_per_class, 0.03)
    return train_points, train_labels, test_points, test_labels, class_names


def _rotation_matrix(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def local_train_ulip(
    global_state: dict[str, torch.Tensor],
    dataset: PointCloudDataset,
    indices: list[int],
    config: dict,
    round_num: int,
    client_id: int,
    num_classes: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    device = choose_device(config.get("device"))
    rng = np.random.default_rng(int(config.get("seed", 42)) + round_num * 1459 + client_id)
    sampled_indices = _sample_client_indices(indices, int(config.get("train_sample_cap_per_client", 64)), rng)
    if not sampled_indices:
        return state_dict_to_cpu(global_state), {
            "train_loss": 0.0,
            "num_samples": 0,
            "gradient_drift": 0.0,
            "modality_alignment": 0.0,
        }

    batch_size = max(int(config.get("batch_size", 16)), 1)
    local_epochs = max(int(config.get("local_epochs", 1)), 1)
    learning_rate = float(config.get("learning_rate", 1e-3))
    embedding_dim = int(config.get("embedding_dim", 256))
    adapter_weight = float(config.get("adapter_weight", 0.1))

    model = build_ulip_model(config.get("network", "pointbert"), num_classes, embedding_dim).to(device)
    model.load_state_dict(global_state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loader = DataLoader(Subset(dataset, sampled_indices), batch_size=batch_size, shuffle=True, num_workers=0)

    total_loss = 0.0
    total_examples = 0
    alignment_scores: list[float] = []

    for _ in range(local_epochs):
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model(points)
            cls_loss = criterion(logits, labels)

            # Lightweight surrogate for cross-modal alignment: keep embeddings compact.
            align_loss = embeddings.std(dim=0).mean()
            loss = cls_loss + adapter_weight * align_loss
            loss.backward()
            optimizer.step()

            batch_examples = int(labels.size(0))
            total_loss += float(loss.detach().item()) * batch_examples
            total_examples += batch_examples
            alignment_scores.append(float(torch.exp(-align_loss.detach()).item()))

    updated_state = state_dict_to_cpu(model.state_dict())
    drift = compute_model_drift(global_state, updated_state)
    return updated_state, {
        "train_loss": total_loss / max(total_examples, 1),
        "num_samples": len(sampled_indices),
        "gradient_drift": float(drift),
        "modality_alignment": float(np.mean(alignment_scores)) if alignment_scores else 0.0,
    }


def aggregate_state_dicts(
    client_states: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    total_weight = float(sum(weights)) or 1.0
    normalized = [float(weight) / total_weight for weight in weights]
    aggregated: dict[str, torch.Tensor] = {}
    for key in client_states[0]:
        if torch.is_floating_point(client_states[0][key]):
            stacked = torch.stack([state[key].float() * normalized[idx] for idx, state in enumerate(client_states)], dim=0)
            aggregated[key] = stacked.sum(dim=0).to(client_states[0][key].dtype)
        else:
            aggregated[key] = client_states[-1][key].clone()
    return aggregated


def evaluate_ulip_model(
    state_dict: dict[str, torch.Tensor],
    dataset: PointCloudDataset,
    config: dict,
    num_classes: int,
) -> dict:
    device = choose_device(config.get("device"))
    model = build_ulip_model(config.get("network", "pointbert"), num_classes, int(config.get("embedding_dim", 256))).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_cap = int(config.get("eval_sample_cap", 800))
    eval_indices = list(range(min(len(dataset), eval_cap)))
    loader = DataLoader(Subset(dataset, eval_indices), batch_size=max(int(config.get("batch_size", 16)), 1), shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    alignments: list[float] = []

    with torch.no_grad():
        for points, labels in loader:
            points = points.to(device)
            labels = labels.to(device)
            embeddings, logits = model(points)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * int(labels.size(0))
            total_correct += int((preds == labels).sum().item())
            total_examples += int(labels.size(0))
            alignments.append(float(torch.exp(-embeddings.std(dim=0).mean()).item()))

    return {
        "eval_loss": total_loss / max(total_examples, 1),
        "test_accuracy": total_correct / max(total_examples, 1),
        "modality_alignment": float(np.mean(alignments)) if alignments else 0.0,
    }


def estimate_communication_cost_mb(state_dict: dict[str, torch.Tensor], num_clients: int) -> float:
    total_params = sum(int(tensor.numel()) for tensor in state_dict.values())
    bytes_per_client = total_params * 4
    return round(bytes_per_client * max(num_clients, 1) / (1024 * 1024), 4)


def compute_model_drift(reference_state: dict[str, torch.Tensor], updated_state: dict[str, torch.Tensor]) -> float:
    squared_sum = 0.0
    param_count = 0
    for key, ref_tensor in reference_state.items():
        upd_tensor = updated_state[key]
        diff = (upd_tensor.float() - ref_tensor.float()).reshape(-1)
        squared_sum += float(torch.sum(diff * diff).item())
        param_count += int(diff.numel())
    return float(np.sqrt(squared_sum / max(param_count, 1)))


def _sample_client_indices(indices: list[int], cap: int, rng: np.random.Generator) -> list[int]:
    if len(indices) <= cap:
        return list(indices)
    choice = rng.choice(np.asarray(indices), size=cap, replace=False)
    return choice.tolist()


def _iid_partition(num_samples: int, num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    chunks = np.array_split(perm, num_clients)
    return [chunk.astype(int).tolist() for chunk in chunks]


def _dirichlet_partition(targets: np.ndarray, num_clients: int, seed: int, alpha: float = 0.5) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    num_classes = int(targets.max()) + 1
    class_indices = [np.where(targets == cls)[0] for cls in range(num_classes)]
    client_buckets = [[] for _ in range(num_clients)]
    for indices in class_indices:
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        shards = np.split(indices, split_points)
        for client_id, shard in enumerate(shards):
            client_buckets[client_id].extend(int(idx) for idx in shard)
    return [sorted(bucket) for bucket in client_buckets]
