"""Service layer: modules, algorithms, references, experiments, results."""
import datetime
import hashlib
import json
import uuid
from pathlib import Path
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import text

from app.core.config import get_real_data_root, settings
from app.models.models import (
    Module, Algorithm, ModuleAlgorithm,
    ReferenceAsset, ModuleReference, AlgorithmReference,
    Dataset, Experiment, ExperimentMetric, ExperimentEvent,
    ImplementationStatus, ExperimentStatus, SecurityObservation, AgentTask,
)
from app.schemas.schemas import ExperimentConfig


# -- Module --



def _enrich_experiment(e):
    """Populate computed frontend-facing fields on experiment."""
    e.rounds = e.total_rounds
    config = e.config_json or {}
    e.participation_rate = config.get("participation_rate", 0.1)
    e.defense = e.defense or config.get("defense", "none")
    e.attack = e.attack or config.get("attack", "none")
    # best_accuracy stays None unless computed from metrics
def _enrich_module(m):
    """Populate frontend-facing fields from meta_json."""
    meta = getattr(m, "meta_json", None) or {}
    m.implementation_status = meta.get("implementation_status", _status_str(m.status))
    m.problem_focus = meta.get("problem_focus", m.description or "")
    m.io_contract = meta.get("io_contract", "输入为实验配置、客户端状态与算法参数；输出为模型更新、指标、日志与结果摘要。")
    m.frontend_views = meta.get("frontend_views", [])
    m.showcase_points = meta.get("showcase_points", [])
    m.backend_service = meta.get("backend_service", "")
    m.config_fields = meta.get("config_fields", [])
    m.backend_plugins = meta.get("backend_plugins", [])


def _status_str(status):
    """Convert enum status to string."""
    return status.value if hasattr(status, "value") else str(status)


TASK_TO_MODULE = {
    "vision_classification": "vision_benchmark",
    "text_classification": "participation_adaptation",
    "defense_demo": "robust_aggregation",
    "reid": "reid_generalization",
    "ulip3d": "multimodal_3d",
}

FRONTEND_TO_BACKEND_MODULE = {
    "core_engine": "core_engine",
    "participation_adaptation": "participation_adaptation",
    "optimization_control": "optimization_control",
    "distillation_alignment": "distillation_alignment",
    "robust_aggregation": "robust_aggregation",
    "trust_screening": "trust_screening",
    "representation_attack": "representation_attack",
    "vision_benchmark": "vision_benchmark",
    "reid_generalization": "reid_generalization",
    "multimodal_3d": "multimodal_3d",
    "experiment_toolkit": "experiment_toolkit",
}

REAL_DATA_ROOT = get_real_data_root()
REAL_DATASET_DIRS = {
    "cifar10": REAL_DATA_ROOT / "cifar-10-batches-py",
    "cifar100": REAL_DATA_ROOT / "cifar-100-python",
    "tiny_imagenet": REAL_DATA_ROOT / "tiny-imagenet-200",
    "mnist": REAL_DATA_ROOT / "mnist",
    "fashion_mnist": REAL_DATA_ROOT / "fashion_mnist",
    "emnist": REAL_DATA_ROOT / "emnist",
    "femnist": REAL_DATA_ROOT / "femnist",
    "market1501": REAL_DATA_ROOT / "market1501_mini",
    "cuhk02": REAL_DATA_ROOT / "cuhk02_mini",
    "msmt17": REAL_DATA_ROOT / "msmt17_mini",
    "cuhk03": REAL_DATA_ROOT / "cuhk03_mini",
    "modelnet40": REAL_DATA_ROOT / "modelnet40_mini" / "modelnet40_mini.npz",
    "scanobjectnn": REAL_DATA_ROOT / "scanobjectnn_mini.npz",
    "shapenetcore": REAL_DATA_ROOT / "shapenetcore_mini.npz",
    "mvtec3d": REAL_DATA_ROOT / "mvtec3d_mini.npz",
    "mnist3d": REAL_DATA_ROOT / "mnist3d_mini.npz",
    "3dimage": REAL_DATA_ROOT / "3dimage_mini.npz",
    "ag_news": REAL_DATA_ROOT / "ag_news",
}

CONFIG_CONSTRAINTS = {
    "core_engine": {
        "task_types": ["vision_classification", "reid", "ulip3d"],
        "datasets": ["cifar10", "cifar100", "mnist", "market1501", "modelnet40"],
        "splits": ["iid", "dirichlet", "pathological", "domain_as_client"],
        "protocols": ["base2new"],
        "networks": ["small_cnn", "resnet18", "mobilenetv2", "tiny_reid", "resnet50", "pointbert"],
        "algorithms": ["fedavg", "fedprox", "scaffold", "feddyn", "fednova", "fedadam", "fedyogi", "fedbn", "ditto", "pfedme", "fedproto", "feddf", "fedkd"],
        "optimizers": ["sgd", "sam", "esam", "hsam"],
        "defenses": ["none"],
        "attacks": ["none"],
        "paper_basis": ["FedAvg", "FedProx", "SCAFFOLD", "FedDyn", "FedOpt", "FedBN", "Ditto", "pFedMe", "FedProto", "FedDF", "FedKD"],
        "partition_params": {},
    },
    "participation_adaptation": {
        "task_types": ["vision_classification", "text_classification"],
        "datasets": ["mnist", "cifar10", "cifar100", "tiny_imagenet", "ag_news"],
        "splits": ["dirichlet", "pathological"],
        "protocols": ["limited_participation"],
        "networks": ["small_cnn", "resnet18", "mobilenetv2", "textcnn"],
        "algorithms": ["fedcvc", "feddtc", "rfl_nlcp", "fedavg", "fedprox", "scaffold", "feddyn", "a_fedpd", "feddc", "fedvra", "fedvarp", "fedspeed", "fedsmoo", "fedtoga", "gfed_hsam"],
        "optimizers": ["sgd"],
        "defenses": ["none"],
        "attacks": ["none"],
        "paper_basis": ["FedCVC", "RFL-NLCP", "FedDTC"],
        "partition_params": {"dirichlet_alpha": [0.1, 0.3, 0.6], "participation_ratio": [0.05, 0.1, 0.2, 0.5, 0.8, 1.0]},
    },
    "optimization_control": {
        "task_types": ["vision_classification"],
        "datasets": ["cifar10", "cifar100", "tiny_imagenet"],
        "splits": ["dirichlet"],
        "protocols": ["sam_comparison"],
        "networks": ["small_cnn", "resnet18", "mobilenetv2"],
        "algorithms": ["gfed_hsam", "sam", "esam", "hsam", "fedavg", "feddyn", "fedprox", "scaffold", "a_fedpd", "fedspeed", "fedsmoo", "fedlesam_d", "a_fedpdsam", "fedgloss", "fedvra", "fedvarp", "fedtoga"],
        "optimizers": ["sgd", "sam", "esam", "hsam"],
        "defenses": ["none"],
        "attacks": ["none"],
        "paper_basis": ["GFed-HSAM", "SAM", "ESAM", "FedCVC SAM comparison"],
        "partition_params": {"dirichlet_alpha": [0.1, 0.6]},
    },
    "distillation_alignment": {
        "task_types": ["vision_classification"],
        "datasets": ["cifar10", "cifar100"],
        "splits": ["dirichlet"],
        "protocols": ["low_participation_distillation"],
        "networks": ["small_cnn", "resnet18", "mobilenetv2"],
        "algorithms": ["fedcads", "fedavg", "scaffold", "feddyn", "feddc", "fedvra", "fedgkd_p", "a_fedpd", "fedfld", "feddf", "fedkd"],
        "optimizers": ["sgd"],
        "defenses": ["none"],
        "attacks": ["none"],
        "paper_basis": ["FedCADS"],
        "partition_params": {"participation_ratio": [0.05, 0.1, 0.2, 0.5, 0.8, 1.0], "local_epochs": [2, 5, 10, 20, 50]},
    },
    "robust_aggregation": {
        "task_types": ["defense_demo"],
        "datasets": ["mnist", "cifar10", "femnist"],
        "splits": ["dirichlet"],
        "protocols": ["model_poisoning"],
        "networks": ["small_cnn", "resnet18"],
        "algorithms": ["fedavg", "vert", "krum", "median", "trimmed_mean", "fldetector", "multi_krum", "fltrust", "flame"],
        "optimizers": ["sgd"],
        "defenses": ["vert", "krum", "multi_krum", "median", "trimmed_mean", "fldetector", "fltrust", "flame"],
        "attacks": ["none", "gn", "mr", "agr", "alie", "featurepoison", "fedproto_prototype_attack"],
        "paper_basis": ["FLDetector", "Krum", "Median/Trimmed Mean", "VERT"],
        "partition_params": {"non_iid_degree": [0.5]},
    },
    "trust_screening": {
        "task_types": ["defense_demo"],
        "datasets": ["mnist", "fashion_mnist", "emnist", "cifar10", "ag_news"],
        "splits": ["dirichlet"],
        "protocols": ["topk_trust_screening"],
        "networks": ["small_cnn", "resnet18", "textcnn"],
        "algorithms": ["flbeeline", "fedavg", "krum", "multi_krum", "median", "flame", "fltrust"],
        "optimizers": ["sgd"],
        "defenses": ["flbeeline", "krum", "multi_krum", "median", "flame", "fltrust"],
        "attacks": ["none", "gn", "mr", "agr", "alie"],
        "paper_basis": ["FLBeeline"],
        "partition_params": {"screening": ["top_k"]},
    },
    "representation_attack": {
        "task_types": ["defense_demo"],
        "datasets": ["cifar10"],
        "splits": ["iid", "dirichlet"],
        "protocols": ["feature_poisoning"],
        "networks": ["small_cnn", "resnet18"],
        "algorithms": ["featurepoison", "fedproto_prototype_attack", "fedavg", "fedproto"],
        "optimizers": ["sgd"],
        "defenses": ["none", "vert", "fltrust", "flame"],
        "attacks": ["featurepoison", "fedproto_prototype_attack"],
        "paper_basis": ["FeaturePoisonAttack code reference", "FedProto prototype attack"],
        "partition_params": {},
    },
    "vision_benchmark": {
        "task_types": ["vision_classification"],
        "datasets": ["mnist", "cifar10", "cifar100", "tiny_imagenet"],
        "splits": ["iid", "dirichlet", "pathological"],
        "protocols": ["visual_classification"],
        "networks": ["small_cnn", "resnet18", "mobilenetv2", "lenet5"],
        "algorithms": ["fedavg", "feddyn", "fedcads", "rfl_nlcp", "fedcvc", "feddtc", "gfed_hsam", "fedprox", "scaffold", "fednova", "fedadam", "fedyogi", "fedbn", "ditto", "pfedme", "fedproto", "feddf", "fedkd", "feddc", "fedvra", "fedvarp", "fedgkd_p", "a_fedpd", "fedfld", "fedspeed", "fedsmoo", "fedtoga"],
        "optimizers": ["sgd", "sam", "esam", "hsam"],
        "defenses": ["none"],
        "attacks": ["none"],
        "paper_basis": ["FedCVC", "FedCADS", "RFL-NLCP", "GFed-HSAM"],
        "partition_params": {"dirichlet_alpha": [0.1, 0.3, 0.6]},
    },
    "reid_generalization": {
        "task_types": ["reid"],
        "datasets": ["cuhk02", "cuhk03", "msmt17", "market1501"],
        "splits": ["domain_as_client"],
        "protocols": ["leave_one_domain_out", "multi_source_mixed_test", "source_domain_evaluation"],
        "networks": ["tiny_reid", "resnet50", "vgg16"],
        "algorithms": ["co_evo", "fedbn", "ditto", "pfedme", "fedproto", "scaffold", "moon", "fedprox", "mixstyle", "crossstyle", "fedreid", "fedpav", "snr", "dacs", "sscu"],
        "optimizers": ["sgd"],
        "defenses": ["none"],
        "attacks": ["none"],
        "paper_basis": ["CO-EVO"],
        "partition_params": {"protocols": ["Protocol I", "Protocol II", "Protocol III"]},
    },
    "multimodal_3d": {
        "task_types": ["ulip3d"],
        "datasets": ["modelnet40", "scanobjectnn", "shapenetcore", "mvtec3d", "mnist3d", "3dimage"],
        "splits": ["iid"],
        "protocols": ["base2new", "cross_dataset", "domain_abcd"],
        "networks": ["pointbert"],
        "algorithms": ["fedulip", "fedavg", "ulip", "pointclip", "fedkgcoop", "fedvpt", "fedtpg", "fedcocoop", "fedmaple", "fedclip", "fedmvp"],
        "optimizers": ["sgd"],
        "defenses": ["none"],
        "attacks": ["none"],
        "paper_basis": ["FedULIP", "ULIP"],
        "partition_params": {"protocols": ["base-to-new", "cross-dataset", "domain A/B/C/D"]},
    },
    "experiment_toolkit": {
        "task_types": ["vision_classification"],
        "datasets": ["cifar10"],
        "splits": ["iid"],
        "protocols": ["visual_classification"],
        "networks": ["small_cnn"],
        "algorithms": ["fedavg"],
        "optimizers": ["sgd"],
        "defenses": ["none"],
        "attacks": ["none"],
        "paper_basis": ["FedAvg smoke template"],
        "partition_params": {},
    },
}


CONFIG_OPTION_KEYS = (
    "task_types",
    "datasets",
    "splits",
    "protocols",
    "networks",
    "algorithms",
    "optimizers",
    "defenses",
    "attacks",
)

DEPRECATED_DATASET_IDS = {"dukemtmc", "shapenet"}


def get_experiment_config_options(module_id: str | None = None) -> dict:
    """Return paper-grounded config choices for a selected module."""
    normalized_module = FRONTEND_TO_BACKEND_MODULE.get(module_id or "", module_id or "vision_benchmark")
    if normalized_module not in CONFIG_CONSTRAINTS:
        normalized_module = "vision_benchmark"
    constraints = CONFIG_CONSTRAINTS[normalized_module]
    defaults = {
        "module_id": normalized_module,
        "task_type": constraints["task_types"][0],
        "dataset": constraints["datasets"][0],
        "split": constraints["splits"][0],
        "protocol": constraints.get("protocols", [""])[0],
        "network": constraints["networks"][0],
        "algorithm": constraints["algorithms"][0],
        "optimizer": "sgd" if "sgd" in constraints["optimizers"] else constraints["optimizers"][0],
        "defense": "none" if "none" in constraints["defenses"] else constraints["defenses"][0],
        "attack": "none" if "none" in constraints["attacks"] else constraints["attacks"][0],
    }
    return {
        "module_id": normalized_module,
        **{key: list(constraints.get(key, [])) for key in CONFIG_OPTION_KEYS},
        "defaults": defaults,
        "paper_basis": list(constraints.get("paper_basis", [])),
        "partition_params": constraints.get("partition_params", {}),
    }


def apply_experiment_config_constraints(resolved: dict) -> None:
    """Coerce an experiment config into the selected module's paper settings."""
    options = get_experiment_config_options(resolved.get("module_id"))
    defaults = options["defaults"]
    resolved["module_id"] = options["module_id"]
    resolved["backend_module_id"] = FRONTEND_TO_BACKEND_MODULE.get(options["module_id"], options["module_id"])

    fields = {
        "task_type": "task_types",
        "dataset": "datasets",
        "split": "splits",
        "protocol": "protocols",
        "network": "networks",
        "algorithm": "algorithms",
        "optimizer": "optimizers",
        "defense": "defenses",
        "attack": "attacks",
    }
    for field, option_key in fields.items():
        allowed = options.get(option_key, [])
        if allowed and resolved.get(field) not in allowed:
            resolved[field] = defaults[field]

    if resolved.get("dataset") == "ag_news" and "text_classification" in options.get("task_types", []):
        resolved["task_type"] = "text_classification"
        if "textcnn" in options.get("networks", []):
            resolved["network"] = "textcnn"
    elif resolved.get("task_type") == "text_classification" and "ag_news" in options.get("datasets", []):
        resolved["dataset"] = "ag_news"
        if "textcnn" in options.get("networks", []):
            resolved["network"] = "textcnn"

    if resolved.get("task_type") == "reid" and resolved.get("network") not in {"tiny_reid", "resnet50", "vgg16"}:
        resolved["network"] = "tiny_reid" if "tiny_reid" in options.get("networks", []) else defaults["network"]
    elif resolved.get("task_type") == "ulip3d":
        resolved["network"] = "pointbert"
    elif resolved.get("task_type") in {"vision_classification", "defense_demo"} and resolved.get("network") == "textcnn":
        resolved["network"] = defaults["network"]


def _reid_dataset_exists(dataset_id: str) -> tuple[bool, Path | None]:
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
    }.get(dataset_id, [])
    for candidate in candidates:
        if (candidate / "train").exists() or (candidate / "bounding_box_train").exists():
            return True, candidate
    return False, None


def resolve_experiment_config(config: ExperimentConfig) -> dict:
    """Normalize frontend config into a runner-friendly config dict."""
    resolved = config.model_dump()

    attack_aliases = {
        "feature_poison": "featurepoison",
    }
    defense_aliases = {
        "fl_beeline": "flbeeline",
    }
    algorithm_aliases = {
        "feature_poison": "featurepoison",
    }
    resolved["attack"] = attack_aliases.get(resolved.get("attack"), resolved.get("attack"))
    resolved["defense"] = defense_aliases.get(resolved.get("defense"), resolved.get("defense"))
    resolved["algorithm"] = algorithm_aliases.get(resolved.get("algorithm"), resolved.get("algorithm"))

    explicit_module = bool(resolved.get("module_id"))
    module_id = resolved.get("module_id") or TASK_TO_MODULE.get(resolved["task_type"], "vision_benchmark")
    resolved["module_id"] = module_id
    resolved["backend_module_id"] = FRONTEND_TO_BACKEND_MODULE.get(module_id, module_id)

    if not explicit_module:
        if resolved["algorithm"] == "fedcads":
            resolved["module_id"] = "distillation_alignment"
            resolved["backend_module_id"] = FRONTEND_TO_BACKEND_MODULE["distillation_alignment"]
        elif resolved["defense"] != "none":
            resolved["module_id"] = "robust_aggregation"
            resolved["backend_module_id"] = FRONTEND_TO_BACKEND_MODULE["robust_aggregation"]

    apply_experiment_config_constraints(resolved)

    if resolved["task_type"] == "text_classification" and resolved["dataset"] == "ag_news":
        resolved["network"] = "textcnn"
    elif resolved["task_type"] == "reid" and resolved["network"] not in {"tiny_reid", "resnet50", "vgg16"}:
        resolved["network"] = "tiny_reid"
    elif resolved["task_type"] == "ulip3d" and resolved["network"] != "pointbert":
        resolved["network"] = "pointbert"

    if resolved["task_type"] == "defense_demo":
        resolved["task_family"] = "vision_defense"
        if resolved.get("defense") == "none":
            resolved["defense"] = "vert"
    else:
        resolved["task_family"] = resolved["task_type"]

    if resolved["task_type"] in {"vision_classification", "text_classification", "defense_demo"}:
        resolved.setdefault("train_sample_cap_per_client", 256)
        resolved.setdefault("eval_sample_cap", 1000)
    elif resolved["task_type"] == "reid":
        resolved.setdefault("train_sample_cap_per_client", 8)
        resolved.setdefault("eval_sample_cap", 24)
        resolved.setdefault("embedding_dim", 128)
        resolved.setdefault("batch_size", min(int(resolved.get("batch_size", 16)), 8))
        resolved.setdefault("local_epochs", min(int(resolved.get("local_epochs", 2)), 2))
    elif resolved["task_type"] == "ulip3d":
        resolved.setdefault("train_sample_cap_per_client", 64)
        resolved.setdefault("eval_sample_cap", 800)
        resolved.setdefault("embedding_dim", 256)
        resolved.setdefault("batch_size", min(int(resolved.get("batch_size", 32)), 16))
        resolved.setdefault("local_epochs", min(int(resolved.get("local_epochs", 4)), 3))
        resolved.setdefault("adapter_type", "saca")
        resolved.setdefault("adapter_weight", 0.1)

    dataset_meta = get_dataset_runtime_meta(resolved["dataset"])
    requested_data_source = str(resolved.get("data_source") or "auto").lower()
    if requested_data_source in {"real", "backend_synthetic", "mock"}:
        resolved["data_source"] = requested_data_source
    else:
        resolved["data_source"] = dataset_meta["data_source"]
    if resolved["data_source"] == "real":
        resolved["training_mode"] = "real_federated"
    elif resolved["data_source"] == "backend_synthetic":
        resolved["training_mode"] = "pytorch_federated"
        _apply_backend_synthetic_caps(resolved)
    else:
        resolved["training_mode"] = "synthetic"

    resolved["runner_type"] = resolved.get("runner_type") or "fedcompass"
    try:
        monitor_interval = int(
            resolved.get(
                "runtime_agent_check_interval_rounds",
                getattr(settings, "runtime_agent_check_interval_rounds", 3),
            )
            or 3
        )
    except (TypeError, ValueError):
        monitor_interval = 3
    resolved["runtime_agent_check_interval_rounds"] = max(1, min(monitor_interval, 100))
    return resolved


def get_dataset_runtime_meta(dataset_id: str) -> dict:
    real_dir = REAL_DATASET_DIRS.get(dataset_id)
    if dataset_id in {"market1501", "cuhk02", "cuhk03", "msmt17"}:
        is_downloaded, detected_dir = _reid_dataset_exists(dataset_id)
        return {
            "is_downloaded": is_downloaded,
            "data_source": "real" if is_downloaded else "backend_synthetic",
            "local_path": str(detected_dir) if detected_dir else None,
        }

    is_downloaded = bool(real_dir and real_dir.exists())
    supports_backend_synthetic = dataset_id in {
        "mnist",
        "fashion_mnist",
        "emnist",
        "femnist",
        "cifar10",
        "cifar100",
        "tiny_imagenet",
        "ag_news",
        "modelnet40",
        "scanobjectnn",
        "shapenetcore",
        "mvtec3d",
        "mnist3d",
        "3dimage",
    }
    return {
        "is_downloaded": is_downloaded,
        "data_source": "real" if is_downloaded else ("backend_synthetic" if supports_backend_synthetic else "mock"),
        "local_path": str(real_dir) if is_downloaded else None,
    }


def _apply_backend_synthetic_caps(resolved: dict):
    profile = str(resolved.get("pytorch_profile", "demo")).lower()
    if profile not in {"demo", "smoke"}:
        return

    task_type = resolved.get("task_type")
    if task_type in {"vision_classification", "text_classification", "defense_demo"}:
        if task_type != "text_classification" and str(resolved.get("network", "")).lower() in {"resnet18", "mobilenetv2", "textcnn"}:
            resolved["network"] = "small_cnn"
        resolved["local_epochs"] = min(int(resolved.get("local_epochs", 1)), 1)
        resolved["batch_size"] = min(int(resolved.get("batch_size", 32)), 32)
        resolved["train_sample_cap_per_client"] = min(int(resolved.get("train_sample_cap_per_client", 32)), 32)
        resolved["eval_sample_cap"] = min(int(resolved.get("eval_sample_cap", 128)), 128)
        if int(resolved.get("synthetic_train_samples", 0) or 0) <= 0:
            resolved["synthetic_train_samples"] = max(int(resolved.get("num_clients", 30)) * 64, 384)
        if int(resolved.get("synthetic_test_samples", 0) or 0) <= 0:
            resolved["synthetic_test_samples"] = 160
        resolved.setdefault("device", "cpu")
    elif task_type == "reid":
        resolved["network"] = "tiny_reid"
        resolved["local_epochs"] = min(int(resolved.get("local_epochs", 1)), 1)
        resolved["batch_size"] = min(int(resolved.get("batch_size", 8)), 8)
        resolved["train_sample_cap_per_client"] = min(int(resolved.get("train_sample_cap_per_client", 8)), 8)
        resolved["eval_sample_cap"] = min(int(resolved.get("eval_sample_cap", 32)), 32)
        resolved["embedding_dim"] = min(int(resolved.get("embedding_dim", 128)), 128)
        resolved.setdefault("device", "cpu")
    elif task_type == "ulip3d":
        resolved["local_epochs"] = min(int(resolved.get("local_epochs", 1)), 1)
        resolved["batch_size"] = min(int(resolved.get("batch_size", 16)), 16)
        resolved["train_sample_cap_per_client"] = min(int(resolved.get("train_sample_cap_per_client", 32)), 32)
        resolved["eval_sample_cap"] = min(int(resolved.get("eval_sample_cap", 160)), 160)
        resolved["embedding_dim"] = min(int(resolved.get("embedding_dim", 128)), 128)
        resolved.setdefault("device", "cpu")


async def list_modules(db: AsyncSession) -> list[Module]:
    r = await db.execute(
        select(Module).options(selectinload(Module.algorithms))
    )
    modules = list(r.scalars().unique())
    for m in modules:
        _enrich_module(m)
        for alg in m.algorithms:
            if not hasattr(alg, "_module_ids") or not getattr(alg, "_module_ids", None):
                setattr(alg, "module_ids", [m.id])
    return modules


async def get_module(db: AsyncSession, module_id: str) -> Module | None:
    r = await db.execute(
        select(Module).options(selectinload(Module.algorithms)).where(Module.id == module_id)
    )
    m = r.scalars().unique().first()
    if m:
        _enrich_module(m)
        for alg in m.algorithms:
            if not hasattr(alg, "_module_ids") or not getattr(alg, "_module_ids", None):
                setattr(alg, "module_ids", [m.id])
    return m


# -- Algorithm --

async def list_algorithms(db: AsyncSession) -> list[Algorithm]:
    r = await db.execute(select(Algorithm).options(selectinload(Algorithm.modules)))
    algs = list(r.scalars().unique())
    for alg in algs:
        alg.module_ids = [m.id for m in alg.modules] if alg.modules else []
    return algs


async def get_algorithm(db: AsyncSession, algorithm_id: str) -> Algorithm | None:
    r = await db.execute(
        select(Algorithm).options(selectinload(Algorithm.modules)).where(Algorithm.id == algorithm_id)
    )
    alg = r.scalars().unique().first()
    if alg:
        alg.module_ids = [m.id for m in alg.modules] if alg.modules else []
    return alg


# -- Reference --

async def list_references(db: AsyncSession) -> list[ReferenceAsset]:
    r = await db.execute(
        select(ReferenceAsset).options(
            selectinload(ReferenceAsset.modules),
            selectinload(ReferenceAsset.algorithms),
        )
    )
    assets = list(r.scalars().unique())
    for a in assets:
        a.related_modules = [m.id for m in (a.modules or [])]
        a.related_algorithms = [alg.id for alg in (a.algorithms or [])]
    return assets


async def get_module_references(db: AsyncSession, module_id: str) -> list[ReferenceAsset]:
    r = await db.execute(
        select(ReferenceAsset)
        .join(ModuleReference)
        .where(ModuleReference.module_id == module_id)
    )
    return list(r.scalars())


async def get_algorithm_references(db: AsyncSession, algorithm_id: str) -> list[ReferenceAsset]:
    r = await db.execute(
        select(ReferenceAsset)
        .join(AlgorithmReference)
        .where(AlgorithmReference.algorithm_id == algorithm_id)
    )
    return list(r.scalars())


# -- Dataset --

async def list_datasets(db: AsyncSession) -> list[Dataset]:
    r = await db.execute(select(Dataset))
    datasets = [d for d in r.scalars() if d.id not in DEPRECATED_DATASET_IDS]
    for d in datasets:
        meta = get_dataset_runtime_meta(d.id)
        d.is_downloaded = meta["is_downloaded"]
        d.data_source = meta["data_source"]
        d.local_path = meta["local_path"]
    return datasets


# -- Experiment --


def _add_scenario_agent_task(db: AsyncSession, exp_id: str, resolved_config: dict):
    """Persist the existing configuration agent as the first workflow stage."""
    config_hash = hashlib.sha256(
        json.dumps(resolved_config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    db.add(
        AgentTask(
            id=f"agt_{uuid.uuid4().hex[:16]}",
            experiment_id=exp_id,
            agent_name="实验场景智能体",
            stage="scenario_configuration",
            status="completed",
            input_hash=config_hash,
            output_json={
                "agent_name": "实验场景智能体",
                "stage": "scenario_configuration",
                "status": "completed",
                "summary": {
                    "message": "配置字段已由场景解析、建议并经过后端约束归一化。",
                    "config_hash": config_hash,
                },
            },
            tool_calls_json=[
                {"tool": "catalog.read", "status": "ok"},
                {"tool": "config.resolve", "status": "ok"},
            ],
            provider="local_fallback",
            started_at=datetime.datetime.utcnow(),
            finished_at=datetime.datetime.utcnow(),
        )
    )
    db.add(
        ExperimentEvent(
            experiment_id=exp_id,
            event_type="agent_update",
            payload_json={
                "agent_name": "实验场景智能体",
                "stage": "scenario_configuration",
                "status": "completed",
                "tool_count": 2,
                "message": "配置草案已归一化，等待运行预检。",
            },
        )
    )

async def create_experiment(db: AsyncSession, config: ExperimentConfig) -> Experiment:
    resolved_config = resolve_experiment_config(config)
    exp = Experiment(
        id=f"exp_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        name=f"{config.algorithm} on {config.dataset}",
        task_type=config.task_type,
        dataset=config.dataset,
        algorithm=config.algorithm,
        optimizer=config.optimizer,
        defense=config.defense,
        attack=config.attack,
        status=ExperimentStatus.draft,
        total_rounds=config.rounds,
        config_json=resolved_config,
    )
    db.add(exp)
    _add_scenario_agent_task(db, exp.id, resolved_config)
    await db.commit()
    await db.refresh(exp)
    return exp


async def list_experiments(db: AsyncSession, status: str | None = None) -> list[Experiment]:
    stmt = select(Experiment).order_by(Experiment.created_at.desc())
    if status:
        stmt = stmt.where(Experiment.status == status)
    r = await db.execute(stmt)
    return list(r.scalars())


async def get_experiment(db: AsyncSession, exp_id: str) -> Experiment | None:
    r = await db.execute(select(Experiment).where(Experiment.id == exp_id))
    return r.scalar()


async def update_experiment_status(db: AsyncSession, exp_id: str, status: ExperimentStatus):
    exp = await get_experiment(db, exp_id)
    if exp:
        exp.status = status
        if status in (ExperimentStatus.completed, ExperimentStatus.failed, ExperimentStatus.stopped):
            exp.finished_at = datetime.datetime.utcnow()
        await db.commit()


async def update_experiment_round(db: AsyncSession, exp_id: str, round_num: int):
    exp = await get_experiment(db, exp_id)
    if exp:
        exp.current_round = round_num
        await db.commit()


async def clone_experiment(db: AsyncSession, exp_id: str) -> Experiment | None:
    orig = await get_experiment(db, exp_id)
    if not orig:
        return None
    new_exp = Experiment(
        id=f"exp_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        name=f"{orig.name} (clone)",
        task_type=orig.task_type,
        dataset=orig.dataset,
        algorithm=orig.algorithm,
        optimizer=orig.optimizer,
        defense=orig.defense,
        attack=orig.attack,
        status=ExperimentStatus.draft,
        total_rounds=orig.total_rounds,
        config_json=orig.config_json,
    )
    db.add(new_exp)
    _add_scenario_agent_task(db, new_exp.id, dict(orig.config_json or {}))
    await db.commit()
    await db.refresh(new_exp)
    return new_exp


# -- Metrics --

async def add_metric(db: AsyncSession, exp_id: str, round_num: int, metric_name: str, metric_value: float):
    m = ExperimentMetric(experiment_id=exp_id, round=round_num, metric_name=metric_name, metric_value=metric_value)
    db.add(m)
    await db.commit()


async def persist_round_update(
    db: AsyncSession,
    exp_id: str,
    round_num: int,
    metrics: dict[str, float],
    events: list[tuple[str, dict]],
):
    exp = await get_experiment(db, exp_id)
    if not exp:
        return

    for metric_name, metric_value in metrics.items():
        db.add(
            ExperimentMetric(
                experiment_id=exp_id,
                round=round_num,
                metric_name=metric_name,
                metric_value=float(metric_value),
            )
        )

    for event_type, payload in events:
        db.add(
            ExperimentEvent(
                experiment_id=exp_id,
                event_type=event_type,
                payload_json=payload,
            )
        )

    # Persist raw round evidence only.  The security-monitoring agent will
    # inspect this JSON snapshot with its LLM skill at the configured runtime
    # checkpoint; this hook must not pre-classify the round with local rules.
    from app.agents.security_signals import build_round_snapshot

    security_snapshot = build_round_snapshot(exp.config_json or {}, round_num, metrics, events)
    db.add(
        SecurityObservation(
            experiment_id=exp_id,
            round=round_num,
            category="fl_round_snapshot",
            severity="unreviewed",
            signal_json=security_snapshot,
            evidence_json=[],
            recommended_action="pending_llm_review",
        )
    )
    db.add(
        ExperimentEvent(
            experiment_id=exp_id,
            event_type="security_snapshot",
            payload_json={
                "round": round_num,
                "status": "pending_llm_review",
                "category": "fl_round_snapshot",
                "schema_version": security_snapshot["schema_version"],
                "message": "当前轮次运行状态已采集，等待安全监控智能体使用 LLM 检查。",
            },
        )
    )

    exp.current_round = max(int(exp.current_round or 0), int(round_num))
    await db.commit()


async def get_metrics(db: AsyncSession, exp_id: str) -> dict[str, list[dict]]:
    r = await db.execute(
        select(ExperimentMetric).where(ExperimentMetric.experiment_id == exp_id).order_by(ExperimentMetric.round)
    )
    rows = list(r.scalars())
    series: dict[str, list[dict]] = {}
    for row in rows:
        series.setdefault(row.metric_name, []).append({"round": row.round, "value": row.metric_value})
    return series


# -- Events --

async def add_event(db: AsyncSession, exp_id: str, event_type: str, payload: dict):
    e = ExperimentEvent(experiment_id=exp_id, event_type=event_type, payload_json=payload)
    db.add(e)
    await db.commit()
    return e


async def get_logs(db: AsyncSession, exp_id: str, limit: int = 200):
    r = await db.execute(
        select(ExperimentEvent)
        .where(ExperimentEvent.experiment_id == exp_id)
        .order_by(ExperimentEvent.created_at.desc())
        .limit(limit)
    )
    return list(r.scalars())


# -- Overview --

async def get_overview(db: AsyncSession) -> dict:
    total = (await db.execute(select(func.count(Experiment.id)))).scalar()
    running = (await db.execute(
        select(func.count(Experiment.id)).where(Experiment.status == ExperimentStatus.running)
    )).scalar()
    alg_count = (await db.execute(select(func.count(Algorithm.id)))).scalar()
    ds_count = (await db.execute(select(func.count(Dataset.id)))).scalar()
    defense_count = (await db.execute(
        select(func.count(Algorithm.id)).where(Algorithm.category == "defense")
    )).scalar()
    recent = (await db.execute(
        select(Experiment).order_by(Experiment.created_at.desc()).limit(5)
    )).scalars().all()

    recent_list = []
    for e in recent:
        status_val = e.status.value if hasattr(e.status, "value") else str(e.status)
        config = e.config_json or {}
        recent_list.append({
            "id": e.id,
            "name": e.name,
            "status": status_val,
            "algorithm": e.algorithm,
            "dataset": e.dataset,
            "current_round": e.current_round,
            "total_rounds": e.total_rounds,
            "rounds": e.total_rounds,
            "participation_rate": config.get("participation_rate", 0.1),
            "defense": config.get("defense", "none"),
            "attack": config.get("attack", "none"),
        })

    return {
        "total_experiments": total,
        "running_experiments": running,
        "registered_algorithms": alg_count,
        "registered_datasets": ds_count,
        "available_defenses": defense_count,
        "recent_experiments": recent_list,
    }
