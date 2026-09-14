"""Real federated training helpers for ReID experiments."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

from app.core.config import get_real_data_root

REAL_DATA_ROOT = get_real_data_root()
_DATA_CACHE: dict[tuple[str, str, int, int], "PreparedFederatedReIDDataset"] = {}
_CAMERA_PATTERN = re.compile(r"c(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ReIDSample:
    image_path: Path
    pid: int
    camid: int
    domain_id: int


class ReIDImageDataset(Dataset):
    def __init__(self, samples: list[ReIDSample], transform: transforms.Compose):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        tensor = self.transform(image)
        return tensor, sample.pid, sample.camid


@dataclass
class PreparedFederatedReIDDataset:
    train_dataset: Dataset
    query_dataset: Dataset
    gallery_dataset: Dataset
    client_indices: list[list[int]]
    num_classes: int
    data_source: str = "real"


class TensorReIDDataset(Dataset):
    def __init__(self, images: torch.Tensor, pids: torch.Tensor, camids: torch.Tensor):
        self.images = images.float().contiguous()
        self.pids = pids.long().contiguous()
        self.camids = camids.long().contiguous()

    def __len__(self) -> int:
        return int(self.pids.numel())

    def __getitem__(self, index: int):
        return self.images[index], self.pids[index], self.camids[index]


class ReIDNet(nn.Module):
    """A small ReID wrapper with embedding and classification heads."""

    def __init__(self, network_id: str, num_classes: int, embedding_dim: int = 256):
        super().__init__()
        network_id = (network_id or "resnet50").lower()
        if network_id == "vgg16":
            backbone = models.vgg16(weights=None)
            feature_dim = 4096
            self.features = backbone.features
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            self.backbone_head = nn.Sequential(
                nn.Flatten(),
                backbone.classifier[0],
                nn.ReLU(inplace=True),
                backbone.classifier[3],
                nn.ReLU(inplace=True),
            )
        else:
            backbone = models.resnet50(weights=None)
            feature_dim = backbone.fc.in_features
            self.features = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.backbone_head = nn.Flatten()

        self.embedding = nn.Linear(feature_dim, embedding_dim)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        feats = self.pool(feats)
        feats = self.backbone_head(feats)
        embeddings = self.embedding(feats)
        normalized = self.bnneck(embeddings)
        logits = self.classifier(normalized)
        return normalized, logits


class TinyReIDNet(nn.Module):
    """Small ReID model for backend-generated demo tensors."""

    def __init__(self, num_classes: int, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.embedding = nn.Linear(48, embedding_dim)
        self.bnneck = nn.BatchNorm1d(embedding_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.features(x)
        embeddings = self.embedding(feats)
        normalized = self.bnneck(embeddings)
        logits = self.classifier(normalized)
        return normalized, logits


def choose_device(preferred: str | None = None) -> torch.device:
    if (preferred or "").lower() == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_reid_model(network_id: str, num_classes: int, embedding_dim: int = 256) -> nn.Module:
    if (network_id or "").lower() in {"tiny_reid", "small_cnn", "demo_reid"}:
        return TinyReIDNet(num_classes=num_classes, embedding_dim=embedding_dim)
    return ReIDNet(network_id=network_id, num_classes=num_classes, embedding_dim=embedding_dim)


def prepare_federated_reid_dataset(
    dataset_name: str,
    split: str,
    num_clients: int,
    seed: int,
    data_source: str = "auto",
) -> PreparedFederatedReIDDataset:
    requested_source = (data_source or "auto").lower()
    cache_key = (dataset_name, split, num_clients, seed, requested_source)
    cached = _DATA_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if requested_source == "backend_synthetic":
        prepared = _prepare_synthetic_reid_dataset(dataset_name, split, num_clients, seed)
        _DATA_CACHE[cache_key] = prepared
        return prepared

    try:
        dataset_root = resolve_reid_dataset_root(dataset_name)
    except FileNotFoundError:
        if requested_source == "real":
            raise
        prepared = _prepare_synthetic_reid_dataset(dataset_name, split, num_clients, seed)
        _DATA_CACHE[cache_key] = prepared
        return prepared

    train_samples = _load_split_samples(dataset_root / "train")
    if not train_samples:
        raise FileNotFoundError(f"No train samples found under {dataset_root / 'train'}")
    query_samples = _load_split_samples(dataset_root / "query")
    gallery_samples = _load_split_samples(dataset_root / "gallery")
    if not query_samples or not gallery_samples:
        query_samples, gallery_samples = _build_eval_splits_from_train(train_samples, seed)

    all_pids = sorted({sample.pid for sample in train_samples})
    pid_to_label = {pid: index for index, pid in enumerate(all_pids)}
    relabel = lambda sample: ReIDSample(
        image_path=sample.image_path,
        pid=pid_to_label[sample.pid],
        camid=sample.camid,
        domain_id=sample.domain_id,
    )
    train_samples = [relabel(sample) for sample in train_samples]
    query_samples = [relabel(sample) for sample in query_samples if sample.pid in pid_to_label]
    gallery_samples = [relabel(sample) for sample in gallery_samples if sample.pid in pid_to_label]
    if not query_samples or not gallery_samples:
        raise FileNotFoundError("ReID evaluation split is empty after relabeling.")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = ReIDImageDataset(train_samples, transform)
    query_dataset = ReIDImageDataset(query_samples, eval_transform)
    gallery_dataset = ReIDImageDataset(gallery_samples, eval_transform)
    if split == "domain_as_client":
        client_indices = _domain_as_client_partition(train_samples, num_clients, seed)
    else:
        client_indices = _iid_partition(len(train_samples), num_clients, seed)

    prepared = PreparedFederatedReIDDataset(
        train_dataset=train_dataset,
        query_dataset=query_dataset,
        gallery_dataset=gallery_dataset,
        client_indices=client_indices,
        num_classes=len(pid_to_label),
        data_source="real",
    )
    _DATA_CACHE[cache_key] = prepared
    return prepared


def _prepare_synthetic_reid_dataset(
    dataset_name: str,
    split: str,
    num_clients: int,
    seed: int,
) -> PreparedFederatedReIDDataset:
    num_ids_by_dataset = {
        "cuhk02": 18,
        "cuhk03": 20,
        "msmt17": 26,
        "market1501": 24,
    }
    num_ids = num_ids_by_dataset.get(dataset_name.lower(), 20)
    num_cameras = min(max(num_clients, 2), 6)
    train_images, train_pids, train_camids = _generate_reid_tensors(
        num_ids=num_ids,
        num_cameras=num_cameras,
        samples_per_id_per_camera=2,
        seed=seed,
        jitter=0.34,
    )
    query_images, query_pids, query_camids = _generate_reid_tensors(
        num_ids=num_ids,
        num_cameras=num_cameras,
        samples_per_id_per_camera=1,
        seed=seed + 1000,
        jitter=0.28,
    )
    gallery_images, gallery_pids, gallery_camids = _generate_reid_tensors(
        num_ids=num_ids,
        num_cameras=num_cameras,
        samples_per_id_per_camera=2,
        seed=seed + 2000,
        jitter=0.30,
    )

    train_dataset = TensorReIDDataset(train_images, train_pids, train_camids)
    query_dataset = TensorReIDDataset(query_images, query_pids, query_camids)
    gallery_dataset = TensorReIDDataset(gallery_images, gallery_pids, gallery_camids)
    if split == "domain_as_client":
        synthetic_samples = [
            ReIDSample(image_path=Path("."), pid=int(pid), camid=int(camid), domain_id=int(camid))
            for pid, camid in zip(train_pids.tolist(), train_camids.tolist())
        ]
        client_indices = _domain_as_client_partition(synthetic_samples, num_clients, seed)
    else:
        client_indices = _iid_partition(len(train_dataset), num_clients, seed)

    return PreparedFederatedReIDDataset(
        train_dataset=train_dataset,
        query_dataset=query_dataset,
        gallery_dataset=gallery_dataset,
        client_indices=client_indices,
        num_classes=num_ids,
        data_source="backend_synthetic",
    )


def _generate_reid_tensors(
    *,
    num_ids: int,
    num_cameras: int,
    samples_per_id_per_camera: int,
    seed: int,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    channels, height, width = 3, 64, 32
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    yy = yy / max(height - 1, 1)
    xx = xx / max(width - 1, 1)
    identity_patterns = []
    for pid in range(num_ids):
        torso = np.exp(-((xx - 0.5) ** 2 / 0.08 + (yy - 0.42) ** 2 / 0.18))
        stripe = np.sin((pid % 7 + 1) * np.pi * yy + pid * 0.17)
        bag = np.exp(-((xx - (0.25 + 0.04 * (pid % 5))) ** 2 + (yy - 0.55) ** 2) / 0.018)
        color = rng.normal(0.0, 0.18, size=(channels, 1, 1)).astype(np.float32)
        identity_patterns.append((0.8 * torso + 0.32 * stripe + 0.5 * bag)[None, :, :] + color)
    identity_patterns = np.asarray(identity_patterns, dtype=np.float32)
    camera_shifts = rng.normal(0.0, 0.16, size=(num_cameras, channels, 1, 1)).astype(np.float32)

    images = []
    pids = []
    camids = []
    for pid in range(num_ids):
        for camid in range(num_cameras):
            for _ in range(samples_per_id_per_camera):
                image = identity_patterns[pid] + camera_shifts[camid]
                image = image + rng.normal(0.0, jitter, size=(channels, height, width)).astype(np.float32)
                image = (image - image.mean()) / (image.std() + 1e-6)
                images.append(image)
                pids.append(pid)
                camids.append(camid)
    order = rng.permutation(len(images))
    images_np = np.asarray(images, dtype=np.float32)[order]
    pids_np = np.asarray(pids, dtype=np.int64)[order]
    camids_np = np.asarray(camids, dtype=np.int64)[order]
    return torch.from_numpy(images_np), torch.from_numpy(pids_np), torch.from_numpy(camids_np)


def resolve_reid_dataset_root(dataset_name: str) -> Path:
    dataset_name = dataset_name.lower()
    candidates = {
        "market1501": [
            REAL_DATA_ROOT / "market1501_mini",
            REAL_DATA_ROOT / "market1501",
            REAL_DATA_ROOT / "Market-1501-v15.09.15",
        ],
        "cuhk02": [
            REAL_DATA_ROOT / "cuhk02_mini",
            REAL_DATA_ROOT / "cuhk02",
            REAL_DATA_ROOT / "CUHK02",
        ],
        "cuhk03": [
            REAL_DATA_ROOT / "cuhk03_mini",
            REAL_DATA_ROOT / "cuhk03",
            REAL_DATA_ROOT / "CUHK03",
        ],
        "msmt17": [
            REAL_DATA_ROOT / "msmt17_mini",
            REAL_DATA_ROOT / "msmt17",
            REAL_DATA_ROOT / "MSMT17",
        ],
    }.get(dataset_name, [])
    for candidate in candidates:
        if (candidate / "train").exists() or (candidate / "bounding_box_train").exists():
            if (candidate / "bounding_box_train").exists():
                return _materialize_standard_reid_layout(candidate)
            return candidate
    raise FileNotFoundError(f"Could not find a prepared ReID dataset root for {dataset_name!r} under {REAL_DATA_ROOT}")


def _materialize_standard_reid_layout(source_root: Path) -> Path:
    materialized_root = source_root / "_fedcompass_layout"
    materialized_root.mkdir(exist_ok=True)
    split_map = {
        "bounding_box_train": "train",
        "query": "query",
        "bounding_box_test": "gallery",
    }
    for source_name, target_name in split_map.items():
        source_dir = source_root / source_name
        target_dir = materialized_root / target_name
        if target_dir.exists():
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(source_dir.glob("*.jpg")):
            pid = _parse_pid_from_name(image_path.name)
            if pid < 0:
                continue
            pid_dir = target_dir / f"{pid:04d}"
            pid_dir.mkdir(exist_ok=True)
            target_path = pid_dir / image_path.name
            if not target_path.exists():
                target_path.symlink_to(image_path.resolve())
    return materialized_root


def local_train_reid(
    global_state: dict[str, torch.Tensor],
    dataset: ReIDImageDataset,
    indices: list[int],
    config: dict,
    round_num: int,
    client_id: int,
    num_classes: int,
) -> tuple[dict[str, torch.Tensor], dict]:
    device = choose_device(config.get("device"))
    rng = np.random.default_rng(int(config.get("seed", 42)) + round_num * 1291 + client_id)
    sampled_indices = _sample_client_indices(indices, int(config.get("train_sample_cap_per_client", 16)), rng)
    if not sampled_indices:
        return state_dict_to_cpu(global_state), {
            "train_loss": 0.0,
            "num_samples": 0,
            "gradient_drift": 0.0,
        }

    batch_size = max(int(config.get("batch_size", 16)), 1)
    local_epochs = max(int(config.get("local_epochs", 1)), 1)
    learning_rate = float(config.get("learning_rate", 3.5e-4))
    embedding_dim = int(config.get("embedding_dim", 256))
    lambda_anchor = float(config.get("lambda_anchor", 1.0))
    lambda_style = float(config.get("lambda_style", 0.1))

    model = build_reid_model(config.get("network", "resnet50"), num_classes, embedding_dim).to(device)
    model.load_state_dict(global_state)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    loader = DataLoader(Subset(dataset, sampled_indices), batch_size=batch_size, shuffle=True, num_workers=0)

    total_loss = 0.0
    total_examples = 0
    for _ in range(local_epochs):
        for images, labels, camids in loader:
            images = images.to(device)
            labels = labels.to(device)
            camids = camids.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model(images)
            cls_loss = criterion(logits, labels)
            anchor_penalty = lambda_anchor * torch.mean((embeddings - embeddings.mean(dim=0, keepdim=True)) ** 2)
            camera_bias = camids.float().unsqueeze(1) / max(int(camids.max().item()) + 1, 1)
            style_penalty = lambda_style * torch.mean((embeddings - camera_bias) ** 2)
            loss = cls_loss + 0.05 * anchor_penalty + 0.02 * style_penalty
            loss.backward()
            optimizer.step()

            batch_size_actual = int(labels.size(0))
            total_loss += float(loss.detach().item()) * batch_size_actual
            total_examples += batch_size_actual

    updated_state = state_dict_to_cpu(model.state_dict())
    drift = compute_model_drift(global_state, updated_state)
    return updated_state, {
        "train_loss": total_loss / max(total_examples, 1),
        "num_samples": len(sampled_indices),
        "gradient_drift": float(drift),
    }


def evaluate_reid_model(
    state_dict: dict[str, torch.Tensor],
    query_dataset: ReIDImageDataset,
    gallery_dataset: ReIDImageDataset,
    config: dict,
    num_classes: int,
) -> dict:
    device = choose_device(config.get("device"))
    embedding_dim = int(config.get("embedding_dim", 256))
    model = build_reid_model(config.get("network", "resnet50"), num_classes, embedding_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    eval_cap = int(config.get("eval_sample_cap", 200))
    query_indices = list(range(min(len(query_dataset), eval_cap)))
    gallery_indices = list(range(min(len(gallery_dataset), max(eval_cap * 2, eval_cap))))
    query_loader = DataLoader(Subset(query_dataset, query_indices), batch_size=max(int(config.get("batch_size", 16)), 1), shuffle=False, num_workers=0)
    gallery_loader = DataLoader(Subset(gallery_dataset, gallery_indices), batch_size=max(int(config.get("batch_size", 16)), 1), shuffle=False, num_workers=0)

    query_features, query_pids, query_camids = _extract_features(model, query_loader, device)
    gallery_features, gallery_pids, gallery_camids = _extract_features(model, gallery_loader, device)
    if query_features.size == 0 or gallery_features.size == 0:
        return {
            "test_accuracy": 0.0,
            "map": 0.0,
            "rank1": 0.0,
            "eval_loss": 0.0,
        }

    distances = _pairwise_distance(query_features, gallery_features)
    map_score, rank1 = _compute_reid_metrics(distances, query_pids, gallery_pids, query_camids, gallery_camids)
    return {
        "test_accuracy": float(rank1),
        "map": float(map_score),
        "rank1": float(rank1),
        "eval_loss": float(max(0.0, 1.0 - rank1)),
    }


def state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state_dict.items()}


def compute_model_drift(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> float:
    total = 0.0
    count = 0
    for key in before:
        diff = (after[key].float() - before[key].float()).reshape(-1)
        total += float(torch.norm(diff, p=2).item())
        count += 1
    return total / max(count, 1)


def estimate_communication_cost_mb(state_dict: dict[str, torch.Tensor], n_clients: int) -> float:
    total_bytes = 0
    for tensor in state_dict.values():
        total_bytes += tensor.numel() * tensor.element_size()
    return round((total_bytes * n_clients) / (1024 * 1024), 4)


def _load_split_samples(split_dir: Path) -> list[ReIDSample]:
    if not split_dir.exists():
        return []
    samples: list[ReIDSample] = []
    pid_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    for domain_index, pid_dir in enumerate(pid_dirs):
        try:
            pid = int(pid_dir.name)
        except ValueError:
            continue
        for image_path in sorted(pid_dir.glob("*")):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            samples.append(
                ReIDSample(
                    image_path=image_path,
                    pid=pid,
                    camid=_parse_camid_from_name(image_path.name),
                    domain_id=domain_index,
                )
            )
    return samples


def _build_eval_splits_from_train(train_samples: list[ReIDSample], seed: int) -> tuple[list[ReIDSample], list[ReIDSample]]:
    grouped: dict[int, list[ReIDSample]] = {}
    for sample in train_samples:
        grouped.setdefault(sample.pid, []).append(sample)
    rng = np.random.default_rng(seed)
    query_samples: list[ReIDSample] = []
    gallery_samples: list[ReIDSample] = []
    for pid, samples in grouped.items():
        ordered = list(samples)
        rng.shuffle(ordered)
        split_point = max(1, len(ordered) // 3)
        query_samples.extend(ordered[:split_point])
        gallery_samples.extend(ordered[split_point:] or ordered[:split_point])
    return query_samples, gallery_samples


def _domain_as_client_partition(samples: list[ReIDSample], num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    domain_buckets: dict[int, list[int]] = {}
    for index, sample in enumerate(samples):
        domain_buckets.setdefault(sample.camid or sample.domain_id, []).append(index)
    for indices in domain_buckets.values():
        rng.shuffle(indices)

    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    cursor = 0
    for _, indices in sorted(domain_buckets.items()):
        for sample_index in indices:
            client_indices[cursor % num_clients].append(sample_index)
            cursor += 1

    leftovers = [index for index, bucket in enumerate(client_indices) if not bucket]
    if leftovers:
        flat_indices = [index for bucket in client_indices for index in bucket]
        for empty_client in leftovers:
            if not flat_indices:
                break
            client_indices[empty_client].append(flat_indices[empty_client % len(flat_indices)])

    for bucket in client_indices:
        bucket.sort()
    return client_indices


def _iid_partition(n_samples: int, num_clients: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)
    splits = np.array_split(shuffled, num_clients)
    return [split.astype(int).tolist() for split in splits]


def _sample_client_indices(indices: list[int], sample_cap: int, rng: np.random.Generator) -> list[int]:
    if sample_cap <= 0 or len(indices) <= sample_cap:
        return list(indices)
    sampled = rng.choice(np.asarray(indices), size=sample_cap, replace=False).tolist()
    sampled.sort()
    return sampled


def _extract_features(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    pids: list[np.ndarray] = []
    camids: list[np.ndarray] = []
    with torch.no_grad():
        for images, batch_pids, batch_camids in loader:
            images = images.to(device)
            embeddings, _ = model(images)
            normalized = torch.nn.functional.normalize(embeddings, dim=1)
            features.append(normalized.detach().cpu().numpy())
            pids.append(np.asarray(batch_pids))
            camids.append(np.asarray(batch_camids))
    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)
    return (
        np.concatenate(features, axis=0),
        np.concatenate(pids, axis=0),
        np.concatenate(camids, axis=0),
    )


def _pairwise_distance(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    query_norm = np.sum(query_features ** 2, axis=1, keepdims=True)
    gallery_norm = np.sum(gallery_features ** 2, axis=1, keepdims=True).T
    distances = query_norm + gallery_norm - 2.0 * np.matmul(query_features, gallery_features.T)
    return np.maximum(distances, 0.0)


def _compute_reid_metrics(
    distances: np.ndarray,
    query_pids: np.ndarray,
    gallery_pids: np.ndarray,
    query_camids: np.ndarray,
    gallery_camids: np.ndarray,
) -> tuple[float, float]:
    aps: list[float] = []
    correct_at_1 = 0
    valid_queries = 0
    for index in range(len(query_pids)):
        order = np.argsort(distances[index])
        remove = (gallery_pids[order] == query_pids[index]) & (gallery_camids[order] == query_camids[index])
        keep = ~remove
        ranked_pids = gallery_pids[order][keep]
        matches = (ranked_pids == query_pids[index]).astype(np.int32)
        if matches.sum() == 0:
            continue
        valid_queries += 1
        if matches[0] == 1:
            correct_at_1 += 1
        cumulative = np.cumsum(matches)
        precision = cumulative / (np.arange(len(matches)) + 1)
        aps.append(float(np.sum(precision * matches) / matches.sum()))

    if valid_queries == 0:
        return 0.0, 0.0
    return float(np.mean(aps)), float(correct_at_1 / valid_queries)


def _parse_pid_from_name(filename: str) -> int:
    try:
        return int(filename.split("_")[0])
    except (ValueError, IndexError):
        return -1


def _parse_camid_from_name(filename: str) -> int:
    match = _CAMERA_PATTERN.search(filename)
    if match:
        return int(match.group(1))
    return 0
