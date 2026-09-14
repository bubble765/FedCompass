"""Turn natural-language experiment requests into validated configuration drafts.

The service is provider-agnostic. Until an LLM endpoint is configured, a small
deterministic parser keeps the UI and API usable for local integration tests.
LLM output is always treated as an untrusted candidate and passed through the
same backend configuration constraints used by manual form submissions.
"""

from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from app.core.config import settings
from app.schemas.schemas import (
    ConfigAssistantChange,
    ConfigAssistantNote,
    ConfigAssistantRequest,
    ConfigAssistantResponse,
    ExperimentConfig,
)
from app.services.services import (
    get_experiment_config_options,
    resolve_experiment_config,
)


class ConfigAssistantError(RuntimeError):
    """Raised when the assistant cannot produce a usable configuration draft."""


CONFIG_FIELDS = (
    "module_id",
    "task_type",
    "dataset",
    "split",
    "protocol",
    "network",
    "algorithm",
    "optimizer",
    "defense",
    "attack",
    "num_clients",
    "participation_rate",
    "rounds",
    "local_epochs",
    "batch_size",
    "learning_rate",
    "seed",
    "trust_threshold",
    "distill_weight",
    "global_distill_weight",
    "temperature",
)

ASSISTANT_REQUIRED_FIELDS = (
    "module_id",
    "task_type",
    "dataset",
    "split",
    "network",
    "algorithm",
    "optimizer",
    "num_clients",
    "participation_rate",
    "rounds",
    "local_epochs",
    "batch_size",
    "learning_rate",
    "seed",
)

DATASET_ALIASES = {
    "cifar-10": "cifar10",
    "cifar 10": "cifar10",
    "cifar10": "cifar10",
    "cifar-100": "cifar100",
    "cifar 100": "cifar100",
    "cifar100": "cifar100",
    "mnist": "mnist",
    "fashion-mnist": "fashion_mnist",
    "fashion mnist": "fashion_mnist",
    "fashionmnist": "fashion_mnist",
    "market-1501": "market1501",
    "market 1501": "market1501",
    "market1501": "market1501",
    "modelnet-40": "modelnet40",
    "modelnet 40": "modelnet40",
    "modelnet40": "modelnet40",
    "scanobjectnn": "scanobjectnn",
    "ag news": "ag_news",
    "ag_news": "ag_news",
    "tiny-imagenet": "tiny_imagenet",
    "tiny imagenet": "tiny_imagenet",
}

SUPPORTED_MODULE_IDS = (
    "core_engine",
    "participation_adaptation",
    "optimization_control",
    "distillation_alignment",
    "robust_aggregation",
    "trust_screening",
    "representation_attack",
    "vision_benchmark",
    "reid_generalization",
    "multimodal_3d",
    "experiment_toolkit",
)

ALGORITHM_ALIASES = {
    "fedavg": "fedavg",
    "fed avg": "fedavg",
    "fedprox": "fedprox",
    "fed prox": "fedprox",
    "feddyn": "feddyn",
    "fed dyn": "feddyn",
    "scaffold": "scaffold",
    "fedcvc": "fedcvc",
    "fed cvc": "fedcvc",
    "feddtc": "feddtc",
    "fed dtc": "feddtc",
    "rfl-nlcp": "rfl_nlcp",
    "rfl nlcp": "rfl_nlcp",
    "fedcads": "fedcads",
    "fed cads": "fedcads",
    "gfed-hsam": "gfed_hsam",
    "gfed hsam": "gfed_hsam",
    "co-evo": "co_evo",
    "co evo": "co_evo",
    "fedulip": "fedulip",
    "fed ulip": "fedulip",
    "fednova": "fednova",
    "fed nova": "fednova",
}

OPTIMIZER_ALIASES = {
    "sgd": "sgd",
    "sam": "sam",
    "esam": "esam",
    "e-sam": "esam",
    "hsam": "hsam",
    "h-sam": "hsam",
}

NETWORK_ALIASES = {
    "small cnn": "small_cnn",
    "small_cnn": "small_cnn",
    "resnet-18": "resnet18",
    "resnet18": "resnet18",
    "mobilenetv2": "mobilenetv2",
    "mobile net v2": "mobilenetv2",
    "lenet-5": "lenet5",
    "lenet5": "lenet5",
    "tiny reid": "tiny_reid",
    "tiny_reid": "tiny_reid",
    "resnet-50": "resnet50",
    "resnet50": "resnet50",
    "pointbert": "pointbert",
    "point bert": "pointbert",
    "textcnn": "textcnn",
}

PROTOCOL_ALIASES = {
    "limited_participation": "limited_participation",
    "limited participation": "limited_participation",
    "低参与协议": "limited_participation",
    "低参与蒸馏协议": "low_participation_distillation",
    "visual_classification": "visual_classification",
    "视觉分类协议": "visual_classification",
    "视觉分类评测": "visual_classification",
    "sam_comparison": "sam_comparison",
    "sam comparison": "sam_comparison",
    "sam 对比": "sam_comparison",
    "model_poisoning": "model_poisoning",
    "model poisoning": "model_poisoning",
    "模型投毒": "model_poisoning",
    "topk_trust_screening": "topk_trust_screening",
    "top-k trust screening": "topk_trust_screening",
    "top-k 可信筛选": "topk_trust_screening",
    "topk 可信筛选": "topk_trust_screening",
    "可信筛选": "topk_trust_screening",
    "feature_poisoning": "feature_poisoning",
    "feature poisoning": "feature_poisoning",
    "特征投毒协议": "feature_poisoning",
    "leave_one_domain_out": "leave_one_domain_out",
    "leave one domain out": "leave_one_domain_out",
    "留一域": "leave_one_domain_out",
    "multi_source_mixed_test": "multi_source_mixed_test",
    "multi-source mixed test": "multi_source_mixed_test",
    "多源混合测试": "multi_source_mixed_test",
    "source_domain_evaluation": "source_domain_evaluation",
    "source domain evaluation": "source_domain_evaluation",
    "源域评测": "source_domain_evaluation",
    "base2new": "base2new",
    "base-to-new": "base2new",
    "base to new": "base2new",
    "cross_dataset": "cross_dataset",
    "cross-dataset": "cross_dataset",
    "跨数据集": "cross_dataset",
    "domain_abcd": "domain_abcd",
    "domain a/b/c/d": "domain_abcd",
    "domain abcd": "domain_abcd",
}

DEFENSE_ALIASES = {
    "vert": "vert",
    "v-e-r-t": "vert",
    "krum": "krum",
    "multi krum": "multi_krum",
    "multi-krum": "multi_krum",
    "median": "median",
    "trimmed mean": "trimmed_mean",
    "flbeeline": "flbeeline",
    "fl beeline": "flbeeline",
    "fltrust": "fltrust",
    "fl trust": "fltrust",
    "flame": "flame",
}

ATTACK_ALIASES = {
    "featurepoison": "featurepoison",
    "feature poison": "featurepoison",
    "feature poisoning": "featurepoison",
    "gaussian noise": "gn",
    "高斯噪声": "gn",
    "model replacement": "mr",
    "模型替换": "mr",
    "agr": "agr",
    "alie": "alie",
}

MODULE_HINTS = {
    "participation_adaptation": ("参与率", "低参与", "客户端参与", "fedcvc", "feddtc", "rfl_nlcp"),
    "optimization_control": ("sam", "esam", "hsam", "尖锐", "sharpness"),
    "distillation_alignment": ("蒸馏", "distillation", "fedcads"),
    "robust_aggregation": ("防御", "投毒", "鲁棒", "恶意客户端", "krum", "vert", "fltrust", "flame"),
    "representation_attack": ("特征投毒", "feature poison", "featurepoison"),
    "reid_generalization": ("reid", "行人重识别", "域泛化", "market1501", "market-1501", "co-evo"),
    "multimodal_3d": ("3d", "点云", "point cloud", "modelnet", "fedulip", "pointbert"),
}

MODULE_ALIASES = {
    "core_engine": "core_engine",
    "训练编排核心模块": "core_engine",
    "训练编排核心": "core_engine",
    "participation_adaptation": "participation_adaptation",
    "低参与异构适配模块": "participation_adaptation",
    "低参与异构适配": "participation_adaptation",
    "optimization_control": "optimization_control",
    "联邦优化控制模块": "optimization_control",
    "优化控制模块": "optimization_control",
    "distillation_alignment": "distillation_alignment",
    "蒸馏对齐增强模块": "distillation_alignment",
    "蒸馏对齐模块": "distillation_alignment",
    "robust_aggregation": "robust_aggregation",
    "鲁棒聚合防护模块": "robust_aggregation",
    "鲁棒聚合模块": "robust_aggregation",
    "trust_screening": "trust_screening",
    "可信客户端筛选模块": "trust_screening",
    "可信筛选模块": "trust_screening",
    "representation_attack": "representation_attack",
    "特征投毒分析模块": "representation_attack",
    "特征投毒模块": "representation_attack",
    "vision_benchmark": "vision_benchmark",
    "视觉分类基准模块": "vision_benchmark",
    "视觉基准模块": "vision_benchmark",
    "reid_generalization": "reid_generalization",
    "reid 域泛化应用模块": "reid_generalization",
    "reid域泛化应用模块": "reid_generalization",
    "3d 多模态应用模块": "multimodal_3d",
    "3d多模态应用模块": "multimodal_3d",
    "multimodal_3d": "multimodal_3d",
    "experiment_toolkit": "experiment_toolkit",
    "实验配置与结果工具": "experiment_toolkit",
    "实验工具箱": "experiment_toolkit",
}

FIELD_LABELS = {
    "module_id": "功能模块",
    "task_type": "任务类型",
    "dataset": "数据集",
    "split": "数据划分",
    "protocol": "评测协议",
    "network": "网络选择",
    "algorithm": "算法方案",
    "optimizer": "优化器",
    "defense": "防御策略",
    "attack": "攻击方式",
    "num_clients": "客户端数量",
    "participation_rate": "参与率",
    "rounds": "训练轮数",
    "local_epochs": "本地 Epoch",
    "batch_size": "批大小",
    "learning_rate": "学习率",
    "seed": "随机种子",
    "trust_threshold": "可信筛选阈值",
    "distill_weight": "本地蒸馏权重",
    "global_distill_weight": "全局蒸馏权重",
    "temperature": "蒸馏温度",
}

SCENARIO_NUMERIC_DEFAULTS = {
    "reid_generalization": {
        "num_clients": 16,
        "participation_rate": 1.0,
        "rounds": 10,
        "local_epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.001,
        "seed": 42,
    },
    "multimodal_3d": {
        "num_clients": 10,
        "participation_rate": 1.0,
        "rounds": 5,
        "local_epochs": 1,
        "batch_size": 16,
        "learning_rate": 0.001,
        "seed": 5,
    },
    "robust_aggregation": {
        "num_clients": 30,
        "participation_rate": 0.5,
        "rounds": 10,
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "seed": 42,
        "trust_threshold": 0.8,
    },
    "trust_screening": {
        "num_clients": 30,
        "participation_rate": 0.5,
        "rounds": 10,
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "seed": 42,
        "trust_threshold": 0.8,
    },
    "distillation_alignment": {
        "num_clients": 30,
        "participation_rate": 0.2,
        "rounds": 10,
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "seed": 42,
        "distill_weight": 0.4,
        "global_distill_weight": 0.6,
        "temperature": 2.0,
    },
    "default": {
        "num_clients": 30,
        "participation_rate": 0.2,
        "rounds": 10,
        "local_epochs": 1,
        "batch_size": 32,
        "learning_rate": 0.01,
        "seed": 42,
    },
}

DETAILED_CONFIGURATION_FIELDS = {
    "num_clients",
    "participation_rate",
    "rounds",
    "local_epochs",
    "batch_size",
    "learning_rate",
    "seed",
    "trust_threshold",
    "distill_weight",
    "global_distill_weight",
    "temperature",
}


def _contains_chinese(value: str | None) -> bool:
    return bool(value and re.search(r"[\u4e00-\u9fff]", value))


def _normalize_note_language(note: ConfigAssistantNote) -> ConfigAssistantNote:
    if _contains_chinese(note.reason):
        return note
    label = FIELD_LABELS.get(note.field or "", "相关配置项")
    return ConfigAssistantNote(
        field=note.field,
        value=note.value,
        reason=f"本次输入未明确{label}，请确认当前值。",
    )


def _normalize_warning_language(warning: str) -> str:
    if _contains_chinese(warning):
        return warning
    return "模型返回了一条非中文提醒，请检查配置草案中的相关参数后再确认。"


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_alias(text: str, aliases: dict[str, str]) -> tuple[str | None, str | None]:
    for phrase, value in sorted(aliases.items(), key=lambda item: len(item[0]), reverse=True):
        if phrase in text:
            return value, phrase
    return None, None


def _number_after(text: str, patterns: list[str]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (TypeError, ValueError):
                continue
    return None


def _parse_percentage(text: str) -> float | None:
    value = _number_after(
        text,
        [
            r"(?:参与率|参与比例|participation(?:\s+rate)?)[^\d]{0,12}(\d+(?:\.\d+)?)\s*%?",
            r"(\d+(?:\.\d+)?)\s*%[^\n]{0,8}(?:参与|participation)",
        ],
    )
    if value is None:
        return None
    return value / 100 if value > 1 else value


def _parse_local_candidate(message: str, current: dict[str, Any], options: dict[str, Any]) -> tuple[dict[str, Any], set[str], list[ConfigAssistantNote], list[str]]:
    text = _compact_text(message)
    candidate: dict[str, Any] = {}
    mentioned: set[str] = set()
    assumptions: list[ConfigAssistantNote] = []
    warnings: list[str] = []

    dataset, dataset_phrase = _find_alias(text, DATASET_ALIASES)
    if dataset:
        candidate["dataset"] = dataset
        mentioned.add("dataset")

    algorithm, algorithm_phrase = _find_alias(text, ALGORITHM_ALIASES)
    if algorithm:
        candidate["algorithm"] = algorithm
        mentioned.add("algorithm")

    optimizer, _ = _find_alias(text, OPTIMIZER_ALIASES)
    if optimizer:
        candidate["optimizer"] = optimizer
        mentioned.add("optimizer")

    network, _ = _find_alias(text, NETWORK_ALIASES)
    if network:
        candidate["network"] = network
        mentioned.add("network")

    protocol, _ = _find_alias(text, PROTOCOL_ALIASES)
    if protocol:
        candidate["protocol"] = protocol
        mentioned.add("protocol")

    defense, _ = _find_alias(text, DEFENSE_ALIASES)
    if defense:
        candidate["defense"] = defense
        mentioned.add("defense")
    elif re.search(r"不(?:启用|使用|考虑|需要).{0,6}(?:防御|defense)|无防御", text):
        candidate["defense"] = "none"
        mentioned.add("defense")

    attack, _ = _find_alias(text, ATTACK_ALIASES)
    if attack:
        candidate["attack"] = attack
        mentioned.add("attack")
    elif re.search(r"不(?:启用|使用|考虑|需要).{0,6}(?:攻击|投毒|attack)|无攻击|不考虑攻击", text):
        candidate["attack"] = "none"
        mentioned.add("attack")

    split = None
    # A module name such as "低参与异构适配模块" does not specify a data split.
    if re.search(r"非\s*[- ]?iid|non[- ]?iid|非独立同分布", text):
        split = "dirichlet"
        assumptions.append(ConfigAssistantNote(field="split", value="dirichlet", reason="识别到非 IID/异构描述，默认采用 Dirichlet 划分。"))
    else:
        for phrase, value in (("dirichlet", "dirichlet"), ("病态", "pathological"), ("pathological", "pathological"), ("iid", "iid"), ("domain as client", "domain_as_client"), ("按域", "domain_as_client")):
            if phrase in text:
                split = value
                break
    if split:
        candidate["split"] = split
        mentioned.add("split")

    task_type = None
    task_type_explicit = False
    if re.search(r"行人重识别|\breid\b|域泛化", text):
        task_type = "reid"
        task_type_explicit = True
    elif re.search(r"点云|3d|三维|多模态", text):
        task_type = "ulip3d"
        task_type_explicit = True
    elif re.search(r"文本|自然语言|新闻分类|text", text):
        task_type = "text_classification"
        task_type_explicit = True
    elif re.search(r"攻防演示|防御演示|鲁棒防御|模型投毒|特征投毒", text):
        task_type = "defense_demo"
        task_type_explicit = True
    elif re.search(r"视觉分类|图像分类|图片分类|视觉任务|vision\s+classification|image\s+classification", text):
        task_type = "vision_classification"
        task_type_explicit = True
    elif dataset or algorithm:
        task_type = "vision_classification"
    if task_type:
        candidate["task_type"] = task_type
        if task_type_explicit:
            mentioned.add("task_type")

    explicit_module, _ = _find_alias(text, MODULE_ALIASES)
    module_id = explicit_module
    current_module = current.get("module_id")
    if not module_id and task_type == "reid":
        module_id = "reid_generalization"
    elif not module_id and task_type == "ulip3d":
        module_id = "multimodal_3d"
    elif not module_id and task_type == "text_classification":
        module_id = "participation_adaptation"
    elif not module_id:
        module_priority = (
            "representation_attack",
            "distillation_alignment",
            "robust_aggregation",
            "participation_adaptation",
            "optimization_control",
        )
        for candidate_module in module_priority:
            hints = MODULE_HINTS[candidate_module]
            if candidate_module == "robust_aggregation" and re.search(
                r"不(?:启用|使用|考虑|需要).{0,6}(?:防御|攻击|投毒)|无(?:防御|攻击)", text
            ):
                continue
            if any(hint in text for hint in hints):
                module_id = candidate_module
                break
    if module_id:
        candidate["module_id"] = module_id
        if explicit_module:
            mentioned.add("module_id")
    elif current_module:
        candidate["module_id"] = current_module

    numeric_patterns = {
        "num_clients": [r"(\d+)\s*(?:个)?\s*(?:客户端|clients?)", r"(?:客户端|clients?)[^\d]{0,10}(\d+)"],
        "rounds": [r"(\d+)\s*(?:轮|rounds?)", r"(?:训练轮数|rounds?)[^\d]{0,10}(\d+)"],
        "local_epochs": [r"(?:本地|local)[^\d]{0,12}(\d+)\s*(?:个)?\s*(?:epoch|轮次)?"],
        "batch_size": [r"(?:批大小|batch(?:\s+size)?)[^\d]{0,10}(\d+)"],
        "seed": [r"(?:随机种子|seed)[^\d]{0,10}(\d+)"],
        "learning_rate": [r"(?:学习率|learning\s+rate|lr)[^\d]{0,10}(0?\.\d+|\d+(?:\.\d+)?)"],
        "trust_threshold": [r"(?:可信(?:筛选)?阈值|trust\s+threshold)[^\d]{0,10}(0?\.\d+|\d+(?:\.\d+)?)"],
        "distill_weight": [r"(?:本地|local)\s*(?:蒸馏|distillation|distill)\s*(?:权重|weight)[^\d]{0,10}(0?\.\d+|\d+(?:\.\d+)?)", r"distill[_ ]weight[^\d]{0,10}(0?\.\d+|\d+(?:\.\d+)?)"],
        "global_distill_weight": [r"(?:全局|global)\s*(?:蒸馏|distillation|distill)\s*(?:权重|weight)[^\d]{0,10}(0?\.\d+|\d+(?:\.\d+)?)", r"global[_ ]distill[_ ]weight[^\d]{0,10}(0?\.\d+|\d+(?:\.\d+)?)"],
        "temperature": [r"(?:蒸馏温度|temperature)[^\d]{0,10}(0?\.\d+|\d+(?:\.\d+)?)"],
    }
    for field, patterns in numeric_patterns.items():
        value = _number_after(text, patterns)
        if value is not None:
            candidate[field] = int(value) if field in {"num_clients", "rounds", "local_epochs", "batch_size", "seed"} else value
            mentioned.add(field)
    participation_rate = _parse_percentage(text)
    if participation_rate is not None:
        candidate["participation_rate"] = participation_rate
        mentioned.add("participation_rate")

    selected_options = get_experiment_config_options(candidate.get("module_id") or options.get("module_id"))
    allowed = set(selected_options.get("algorithms", []))
    if algorithm and allowed and algorithm not in allowed:
        warnings.append(f"当前模块不支持算法 {algorithm}，后端将使用该模块的默认算法。")
    return candidate, mentioned, assumptions, warnings


def _is_scenario_request(mentioned: set[str]) -> bool:
    """Use recommendation mode for high-level descriptions without training numbers."""
    return not (mentioned & DETAILED_CONFIGURATION_FIELDS) and len(mentioned) <= 7


def _build_scenario_recommendation(
    parsed_candidate: dict[str, Any],
    current: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Complete a high-level scenario with values from the selected module recipe."""
    module_id = parsed_candidate.get("module_id") or current.get("module_id") or options.get("module_id")
    module_options = get_experiment_config_options(module_id)
    defaults = module_options.get("defaults", {})
    recommendation = {
        field: value
        for field, value in defaults.items()
        if field in CONFIG_FIELDS and value not in (None, "")
    }
    recommendation.update(
        {
            field: value
            for field, value in SCENARIO_NUMERIC_DEFAULTS.get(
                module_id,
                SCENARIO_NUMERIC_DEFAULTS["default"],
            ).items()
            if field in CONFIG_FIELDS
        }
    )

    task_type = parsed_candidate.get("task_type") or recommendation.get("task_type")
    if task_type == "reid":
        recommendation["split"] = "domain_as_client"
    elif task_type == "ulip3d":
        recommendation["split"] = "iid"
    elif task_type in {"vision_classification", "text_classification", "defense_demo"}:
        recommendation.setdefault("split", defaults.get("split", "iid"))

    # Explicitly parsed values have priority over scenario defaults.
    recommendation.update(
        {
            field: value
            for field, value in parsed_candidate.items()
            if field in CONFIG_FIELDS
        }
    )
    recommendation["module_id"] = module_options.get("module_id", module_id)
    return recommendation


def _build_prompt(request: ConfigAssistantRequest, options: dict[str, Any]) -> str:
    return (
        "你是实验场景配置智能体，负责把用户描述的联邦学习实验场景转换为可确认的完整配置建议。只输出一个 JSON 对象，不要输出 Markdown。"
        "JSON 必须包含 config、explicit_fields、suggested_fields、recommended_fields、mode；可选包含 assumptions、missing_fields、warnings、follow_up_question、confidence。"
        "当用户只描述任务、数据集、协议或研究目标时，使用 scenario_recommendation 模式，依据模块约束和常见可复现实验设置补全一整套配置。"
        "当用户已经开始逐项指定训练参数时，使用 detailed_configuration 模式；不要用默认值替代用户没有写出的字段。"
        "explicit_fields 只能列出用户需求中明确给出或直接表达的字段；suggested_fields 只能列出根据场景推断的字段；recommended_fields 列出本次建议完整配置中的字段。"
        "missing_fields 只列出无法依据场景安全建议、且需要用户补充的字段。场景模式下已经有合理建议的字段不要列入 missing_fields。"
        "不要把 current_config 中仅用于对比的值冒充用户输入，也不要生成可配置字段列表之外的字段。"
        "assumptions 只在字段缺失或存在不确定性时返回；warnings 和 follow_up_question 只在字段缺失或值不合理时返回。"
        "assumptions.reason、warnings 和 follow_up_question 必须使用简体中文；字段 ID、算法名和数据集 ID 可以保留英文。"
        "config 只能使用下列模块及其对应选项，不能编造算法、数据集或模块。"
        f"\n可配置字段：{json.dumps(CONFIG_FIELDS, ensure_ascii=False)}"
        f"\n合法模块及选项：{json.dumps(options, ensure_ascii=False)}"
        f"\n场景模式数值建议：{json.dumps(SCENARIO_NUMERIC_DEFAULTS, ensure_ascii=False)}"
        f"\n当前配置：{json.dumps(request.current_config, ensure_ascii=False)}"
        f"\n用户需求：{request.message}"
    )


def _validate_llm_base_url(base_url: str) -> str:
    """Validate a user-supplied OpenAI-compatible endpoint without logging it."""
    normalized = base_url.strip().rstrip("/")
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigAssistantError("模型服务地址必须是以 http:// 或 https:// 开头的完整地址")
    if parsed.username or parsed.password:
        raise ConfigAssistantError("模型服务地址不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        raise ConfigAssistantError("模型服务地址不能包含查询参数或片段")
    return normalized


def _resolve_llm_runtime(request: ConfigAssistantRequest) -> dict[str, str] | None:
    """Resolve one-shot UI settings, falling back to server environment settings.

    A supplied request config is validated strictly. When no request config is
    supplied, an incomplete environment configuration simply disables the LLM
    path and lets the deterministic parser handle the request.
    """
    supplied = request.llm
    if supplied is not None:
        provider = supplied.provider
        if provider == "local":
            # The browser intentionally does not expose local connection
            # details. Use the backend/container setting first, while still
            # accepting an explicit value for direct API clients.
            base_url = supplied.base_url.strip() or settings.llm_local_base_url.strip()
            model = supplied.model.strip() or settings.llm_local_model.strip()
            api_key = ""
        else:
            base_url = supplied.base_url.strip()
            model = supplied.model.strip()
            api_key = supplied.api_key.strip()
        if not base_url:
            label = "API Base URL" if provider == "api" else "本地模型服务地址"
            raise ConfigAssistantError(f"请填写{label}")
        if not model:
            raise ConfigAssistantError("请填写模型名称")
        if provider == "api" and not api_key:
            raise ConfigAssistantError("调用 API 模型时必须填写 API Key")
        return {
            "provider": provider,
            "base_url": _validate_llm_base_url(base_url),
            "api_key": api_key,
            "model": model,
        }

    provider = (settings.llm_provider or "api").strip().lower()
    if provider not in {"local", "api"}:
        provider = "api"
    if provider == "local":
        base_url = settings.llm_local_base_url.strip()
        model = settings.llm_local_model.strip()
        api_key = ""
    else:
        base_url = settings.llm_base_url.strip()
        model = settings.llm_model.strip()
        api_key = settings.llm_api_key.strip()

    if not base_url or not model or (provider == "api" and not api_key):
        return None
    try:
        base_url = _validate_llm_base_url(base_url)
    except ConfigAssistantError:
        return None
    return {
        "provider": provider,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }


def _call_llm_sync(prompt: str, runtime: dict[str, str]) -> dict[str, Any]:
    if not runtime:
        raise ConfigAssistantError("LLM provider is not configured")

    base_url = runtime["base_url"].rstrip("/")
    url = base_url if base_url.endswith("/chat/completions") else base_url + "/chat/completions"
    body = {
        "model": runtime["model"],
        "temperature": 0,
        "max_tokens": max(1, int(settings.llm_max_tokens)),
        "stream": False,
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    if runtime.get("provider") == "local":
        body["response_format"] = {"type": "json_object"}
    headers = {
        "Content-Type": "application/json",
    }
    if runtime.get("api_key"):
        headers["Authorization"] = f"Bearer {runtime['api_key']}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.llm_timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ConfigAssistantError("LLM provider request failed") from exc

    try:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        content = str(content).strip()
        # Some Qwen3-compatible servers include a reasoning block even when
        # the prompt asks for JSON only. Remove that block before parsing.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
            content = re.sub(r"\s*```$", "", content)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            # Be tolerant of a short natural-language prefix while still
            # requiring the first decodable value to be a JSON object.
            decoder = json.JSONDecoder()
            parsed = None
            for match in re.finditer(r"\{", content):
                try:
                    candidate, _ = decoder.raw_decode(content[match.start():])
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    parsed = candidate
                    break
            if parsed is None:
                raise
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ConfigAssistantError("LLM provider returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ConfigAssistantError("LLM provider returned a non-object JSON value")
    return parsed


def _safe_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    candidate = raw.get("config", raw)
    if not isinstance(candidate, dict):
        raise ConfigAssistantError("Assistant config must be a JSON object")
    return {key: value for key, value in candidate.items() if key in CONFIG_FIELDS}


def _normalize_config(candidate: dict[str, Any], current: dict[str, Any], module_id: str | None) -> dict[str, Any]:
    base = ExperimentConfig().model_dump()
    base.update({key: value for key, value in current.items() if key in CONFIG_FIELDS})
    if module_id:
        base["module_id"] = module_id
    base.update(candidate)
    try:
        return resolve_experiment_config(ExperimentConfig(**base))
    except (TypeError, ValueError) as exc:
        raise ConfigAssistantError("Assistant config failed backend validation") from exc


def _assistant_required_fields(config: dict[str, Any]) -> list[str]:
    """Return the fields currently rendered by the experiment configuration form."""
    module_options = get_experiment_config_options(config.get("module_id"))
    required_fields = list(ASSISTANT_REQUIRED_FIELDS)
    if module_options.get("protocols"):
        required_fields.insert(4, "protocol")

    defenses = module_options.get("defenses", [])
    attacks = module_options.get("attacks", [])
    show_defense_attack = any(value != "none" for value in defenses) or any(value != "none" for value in attacks)
    if show_defense_attack:
        required_fields.append("defense")
    if show_defense_attack and len(attacks) > 1:
        required_fields.append("attack")
    if config.get("defense") in {"vert", "flbeeline"}:
        required_fields.append("trust_threshold")
    if config.get("algorithm") == "fedcads" or config.get("module_id") == "distillation_alignment":
        required_fields.extend(["distill_weight", "global_distill_weight", "temperature"])
    return list(dict.fromkeys(required_fields))


def _build_response(
    normalized: dict[str, Any],
    current: dict[str, Any],
    mentioned: set[str],
    suggested: set[str],
    assumptions: list[ConfigAssistantNote],
    warnings: list[str],
    raw: dict[str, Any],
    provider: str,
    mode: str,
) -> ConfigAssistantResponse:
    required_fields = _assistant_required_fields(normalized)
    visible_fields = set(required_fields)
    explicit_fields = {field for field in mentioned if field in CONFIG_FIELDS}
    suggested_fields = {
        field for field in suggested if field in CONFIG_FIELDS and field not in explicit_fields
    }
    explicit_fields &= visible_fields
    suggested_fields &= visible_fields
    source_fields = explicit_fields | suggested_fields
    changes: list[ConfigAssistantChange] = []
    for field in CONFIG_FIELDS:
        old_value = current.get(field)
        new_value = normalized.get(field)
        if old_value != new_value and field in source_fields:
            changes.append(
                ConfigAssistantChange(
                    field=field,
                    old_value=old_value,
                    new_value=new_value,
                    reason=(
                        f"根据场景建议设置{FIELD_LABELS.get(field, field)}。"
                        if field in suggested_fields
                        else f"根据用户输入更新{FIELD_LABELS.get(field, field)}。"
                    ),
                    source="suggested" if field in suggested_fields else "explicit",
                )
            )

    for field in ("module_id", "task_type", "dataset", "algorithm"):
        if field not in source_fields and current.get(field) is not None:
            assumptions.append(ConfigAssistantNote(field=field, value=normalized.get(field), reason="用户未明确指定，沿用当前配置。"))

    missing = [
        field
        for field in required_fields
        if field not in (source_fields if mode == "scenario_recommendation" else explicit_fields)
    ]
    model_missing = raw.get("missing_fields", [])
    if isinstance(model_missing, list):
        missing.extend(
            str(field)
            for field in model_missing
            if field in CONFIG_FIELDS and field not in source_fields
        )
    missing = list(dict.fromkeys(missing))
    if missing:
        missing_labels = "、".join(FIELD_LABELS.get(field, field) for field in missing)
        assumptions.extend(
            ConfigAssistantNote(
                field=field,
                value=normalized.get(field),
                reason=f"本次输入未明确{FIELD_LABELS.get(field, field)}，暂保留当前表单值，请确认。",
            )
            for field in missing
        )
        if mode == "scenario_recommendation":
            warnings.append(f"以下配置项无法仅根据当前场景安全建议：{missing_labels}。")
        else:
            warnings.append(f"以下配置项未在本次输入中明确，未用助手结果覆盖：{missing_labels}。请在表单中补充或确认。")
    follow_up = raw.get("follow_up_question") if isinstance(raw.get("follow_up_question"), str) else None
    if not follow_up and missing:
        follow_up = f"请确认以下未明确的配置项：{missing_labels}。"
    raw_warnings = raw.get("warnings", [])
    if isinstance(raw_warnings, list):
        warnings.extend(str(item) for item in raw_warnings if item)
    assumptions = [_normalize_note_language(note) for note in assumptions]
    warnings = [_normalize_warning_language(warning) for warning in warnings]
    if not missing and not warnings:
        assumptions = []
        follow_up = None
    if follow_up and not _contains_chinese(follow_up):
        follow_up = f"请确认以下配置项：{missing_labels}。" if missing else "请确认配置草案中的相关参数。"
    default_confidence = (
        0.72
        if provider == "local_fallback"
        else (0.86 if mode == "scenario_recommendation" else 0.78)
    )
    confidence = raw.get("confidence", default_confidence)
    try:
        confidence = max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = 0.0
    visible_config = {field: normalized.get(field) for field in required_fields if field in normalized}
    recommended_fields = [
        field
        for field in CONFIG_FIELDS
        if field in normalized and field in required_fields
    ]
    return ConfigAssistantResponse(
        config=visible_config,
        changes=changes,
        assumptions=assumptions,
        missing_fields=missing,
        explicit_fields=[field for field in CONFIG_FIELDS if field in explicit_fields],
        suggested_fields=[field for field in CONFIG_FIELDS if field in suggested_fields],
        recommended_fields=recommended_fields,
        mode=mode,
        warnings=list(dict.fromkeys(warnings)),
        follow_up_question=follow_up,
        confidence=confidence,
        provider=provider,
    )


async def parse_config_request(request: ConfigAssistantRequest) -> ConfigAssistantResponse:
    if not request.message.strip():
        raise ConfigAssistantError("message cannot be empty")

    current = dict(request.current_config or {})
    module_id = request.module_id or current.get("module_id") or "vision_benchmark"
    options = get_experiment_config_options(module_id)
    option_catalog = {item: get_experiment_config_options(item) for item in SUPPORTED_MODULE_IDS}
    local_candidate, text_mentioned, local_assumptions, local_warnings = _parse_local_candidate(
        request.message,
        current,
        options,
    )
    mode = "scenario_recommendation" if _is_scenario_request(text_mentioned) else "detailed_configuration"
    scenario_candidate = (
        _build_scenario_recommendation(local_candidate, current, options)
        if mode == "scenario_recommendation"
        else {}
    )
    llm_runtime = _resolve_llm_runtime(request)
    provider = (
        "llm_local"
        if llm_runtime and llm_runtime["provider"] == "local"
        else "llm_api"
        if llm_runtime
        else "local_fallback"
    )
    raw: dict[str, Any]
    mentioned: set[str]
    suggested: set[str]
    assumptions: list[ConfigAssistantNote]
    warnings: list[str]

    try:
        raw = await asyncio.to_thread(
            _call_llm_sync,
            _build_prompt(request, option_catalog),
            llm_runtime,
        )
        model_candidate = _safe_candidate(raw)
        # The model may omit explicit_fields or return an empty list even when
        # config contains values copied from the user's message. Local parsing
        # is the authority for detecting user-provided fields; model output
        # cannot turn current_config into user input.
        mentioned = set(model_candidate) & set(text_mentioned)
        if mode == "scenario_recommendation":
            candidate = {**scenario_candidate, **model_candidate}
            suggested = set(candidate) - mentioned
        else:
            # Preserve values that the deterministic parser can identify when
            # a small model omits them from its JSON response. Model values
            # remain available for fields expressed through richer wording.
            candidate = {
                **model_candidate,
                **{
                    field: value
                    for field, value in local_candidate.items()
                    if field in text_mentioned
                },
            }
            suggested = set()
        assumptions = [
            ConfigAssistantNote.model_validate(item)
            for item in raw.get("assumptions", [])
            if isinstance(item, dict)
        ]
        warnings = [str(item) for item in raw.get("warnings", []) if item]
    except ConfigAssistantError as llm_error:
        provider = "local_fallback"
        mentioned = text_mentioned
        candidate = scenario_candidate if mode == "scenario_recommendation" else local_candidate
        suggested = set(candidate) - mentioned if mode == "scenario_recommendation" else set()
        assumptions = local_assumptions
        warnings = local_warnings
        raw = {"config": candidate}

    candidate = {field: value for field, value in candidate.items() if field in CONFIG_FIELDS}
    mentioned &= set(candidate)
    suggested &= set(candidate) - mentioned
    normalized = _normalize_config(candidate, current, module_id)
    for field, requested_value in candidate.items():
        normalized_value = normalized.get(field)
        if normalized_value != requested_value:
            warnings.append(
                f"{FIELD_LABELS.get(field, field)} {requested_value} 与当前模块约束不一致，已归一化为 {normalized_value}。"
            )
    return _build_response(
        normalized,
        current,
        mentioned,
        suggested,
        assumptions,
        warnings,
        raw,
        provider,
        mode,
    )
