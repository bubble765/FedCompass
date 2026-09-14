import datetime
from pydantic import BaseModel
from typing import Literal, Optional

from pydantic import Field


# Module
class ModuleOut(BaseModel):
    id: str
    name: str
    positioning: Optional[str] = None
    description: Optional[str] = None
    capabilities: list[str] = []
    primary_algorithms: list[str] = []
    baseline_algorithms: list[str] = []
    status: str
    algorithms: list["AlgorithmBrief"] = []

    # Frontend-facing extra fields (from meta_json)
    implementation_status: str = "planned"
    problem_focus: str = ""
    io_contract: str = ""
    frontend_views: list[str] = []
    showcase_points: list[str] = []
    backend_service: str = ""
    config_fields: list[str] = []
    backend_plugins: list[str] = []

    model_config = {"from_attributes": True}


class ModuleBrief(BaseModel):
    id: str
    name: str
    positioning: Optional[str] = None
    status: str
    tags: list[str] = []
    model_config = {"from_attributes": True}


# Algorithm
class AlgorithmOut(BaseModel):
    id: str
    name: str
    category: Optional[str] = None
    description: Optional[str] = None
    task_types: list[str] = []
    parameter_schema: dict = {}
    hooks: list[str] = []
    status: str
    module_ids: list[str] = []
    model_config = {"from_attributes": True}


class AlgorithmBrief(BaseModel):
    id: str
    name: str
    category: Optional[str] = None
    status: str
    description: Optional[str] = None
    task_types: list[str] = []
    parameter_schema: dict = {}
    hooks: list[str] = []
    module_ids: list[str] = []
    model_config = {"from_attributes": True}


# Reference
class ReferenceAssetOut(BaseModel):
    id: str
    asset_type: str
    title: str
    local_path: Optional[str] = None
    external_url: Optional[str] = None
    default_branch: Optional[str] = None
    key_files: list[str] = []
    status: str
    related_modules: list[str] = []
    related_algorithms: list[str] = []
    model_config = {"from_attributes": True}


# Dataset
class DatasetOut(BaseModel):
    id: str
    name: str
    task_type: Optional[str] = None
    split_options: list[str] = []
    description: Optional[str] = None
    is_downloaded: bool = False
    data_source: str = "mock"
    local_path: Optional[str] = None
    model_config = {"from_attributes": True}


# Experiment
class KeyMetrics(BaseModel):
    accuracy: float = 0.85
    map: float = 0.78
    rank1: float = 0.82
    attack_detection_accuracy: float = 0.88
    communication_cost: float = 128.0


class ExperimentConfig(BaseModel):
    task_type: str = "vision_classification"
    dataset: str = "cifar10"
    split: str = "iid"
    network: str = "resnet18"
    algorithm: str = "fedavg"
    optimizer: str = "sgd"
    defense: str = "none"
    attack: str = "none"
    num_clients: int = 30
    participation_rate: float = 0.1
    rounds: int = 10
    local_epochs: int = 5
    batch_size: int = 64
    learning_rate: float = 0.01
    seed: int = 42
    silence_rate: float = 0.0
    trust_threshold: float = 0.75
    distill_weight: float = 0.4
    global_distill_weight: float = 0.6
    temperature: float = 2.0
    lambda_anchor: float = 1.0
    lambda_style: float = 0.1
    adapter_type: str = "saca"
    protocol: str = "base2new"
    runner_type: str = "fedcompass"
    data_source: str = "auto"
    training_mode: str = "auto"
    pytorch_profile: str = "demo"
    module_id: str = ""
    train_sample_cap_per_client: int = 256
    eval_sample_cap: int = 1000
    synthetic_train_samples: int = 0
    synthetic_test_samples: int = 0
    prox_mu: float = 0.01
    embedding_dim: int = 256

    model_config = {"extra": "allow"}


class ConfigAssistantLLMConfig(BaseModel):
    """One-shot model connection settings supplied by the assistant UI.

    The API key is intentionally part of the request only. It is never returned
    in a response and is not persisted by the backend.
    """

    provider: Literal["local", "api"] = "local"
    base_url: str = Field(default="", max_length=2048)
    api_key: str = Field(default="", max_length=4096)
    model: str = Field(default="", max_length=256)

    model_config = {"extra": "forbid"}


class ConfigAssistantRequest(BaseModel):
    """Natural-language request for a validated experiment configuration draft."""

    message: str
    current_config: dict = {}
    module_id: Optional[str] = None
    llm: Optional[ConfigAssistantLLMConfig] = None


class ConfigAssistantChange(BaseModel):
    field: str
    old_value: object = None
    new_value: object = None
    reason: str = ""
    source: str = "explicit"


class ConfigAssistantNote(BaseModel):
    field: Optional[str] = None
    value: object = None
    reason: str = ""


class ConfigAssistantResponse(BaseModel):
    config: dict
    changes: list[ConfigAssistantChange] = []
    assumptions: list[ConfigAssistantNote] = []
    missing_fields: list[str] = []
    explicit_fields: list[str] = []
    suggested_fields: list[str] = []
    recommended_fields: list[str] = []
    mode: str = "detailed_configuration"
    warnings: list[str] = []
    follow_up_question: Optional[str] = None
    confidence: float = 0.0
    provider: str = "local_fallback"


class ExperimentOut(BaseModel):
    id: str
    name: str
    task_type: Optional[str] = None
    dataset: Optional[str] = None
    algorithm: Optional[str] = None
    optimizer: Optional[str] = None
    defense: Optional[str] = None
    attack: Optional[str] = None
    status: str
    current_round: int = 0
    total_rounds: int = 100
    rounds: int = 100
    num_clients: int = 100
    config_json: dict = {}
    created_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None
    model_config = {"from_attributes": True}


class ExperimentBrief(BaseModel):
    id: str
    name: str
    status: str
    algorithm: Optional[str] = None
    dataset: Optional[str] = None
    current_round: int = 0
    total_rounds: int = 100
    rounds: int = 100
    best_accuracy: Optional[float] = None
    participation_rate: Optional[float] = None
    defense: Optional[str] = None
    attack: Optional[str] = None
    created_at: Optional[datetime.datetime] = None
    model_config = {"from_attributes": True}


# Overview
class OverviewOut(BaseModel):
    total_experiments: int = 0
    running_experiments: int = 0
    algorithm_count: int = 0
    dataset_count: int = 0
    defense_count: int = 0
    key_metrics: KeyMetrics = KeyMetrics()
    recent_experiments: list[ExperimentBrief] = []


# Results
class CompareRequest(BaseModel):
    experiment_ids: list[str]


class CompareResult(BaseModel):
    experiment_id: str
    experiment_name: str
    algorithm: Optional[str] = None
    dataset: Optional[str] = None
    task_type: Optional[str] = None
    best_accuracy: Optional[float] = None
    final_accuracy: Optional[float] = None
    best_loss: Optional[float] = None
    communication_rounds: int = 0
    status: str
    attack: Optional[str] = "none"
    defense: Optional[str] = "none"
    participation_rate: Optional[float] = 0.1
    seed: Optional[int] = 42
    map: Optional[float] = 0.0
    rank1: Optional[float] = 0.0
    detection_accuracy: Optional[float] = 0.0
    history: list[dict] = []
    series: dict = {}
    data_source: Optional[str] = "mock"
    training_mode: Optional[str] = "synthetic"


# Multi-agent workflow
class AgentTaskOut(BaseModel):
    id: str
    agent_name: str
    stage: str
    status: str
    input_hash: str
    output: dict = {}
    tool_calls: list[dict] = []
    provider: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[datetime.datetime] = None
    finished_at: Optional[datetime.datetime] = None


class WorkflowOut(BaseModel):
    experiment_id: str
    experiment_status: str
    agents: list[AgentTaskOut] = []
    runtime_agent_check_interval_rounds: int = 3
    runtime_check_count: int = 0
    operations_report: Optional[dict] = None
    security_report: Optional[dict] = None
    result_analysis: Optional[dict] = None


class ResultAnalysisOut(BaseModel):
    experiment_id: str
    status: str
    claim_level: str
    analysis: dict = {}
    evidence_refs: list[dict] = []
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None
