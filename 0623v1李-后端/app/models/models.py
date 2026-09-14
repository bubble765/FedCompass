import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, JSON, Enum as SAEnum
from sqlalchemy.orm import relationship
import enum

from app.core.database import Base


class AssetType(str, enum.Enum):
    paper = "paper"
    external_code = "external_code"
    local_archive = "local_archive"
    key_file = "key_file"


class ImplementationStatus(str, enum.Enum):
    paper_only = "paper_only"
    external_code = "external_code"
    local_archive = "local_archive"
    mocked = "mocked"
    replay_ready = "replay_ready"
    implemented = "implemented"
    validated = "validated"


class ExperimentStatus(str, enum.Enum):
    draft = "draft"
    preflight_pending = "preflight_pending"
    preflight_completed = "preflight_completed"
    ready_to_start = "ready_to_start"
    pending = "pending"
    running = "running"
    completed = "completed"
    result_analyzing = "result_analyzing"
    analyzed = "analyzed"
    failed = "failed"
    stopped = "stopped"
    blocked = "blocked"
    invalid = "invalid"


class Module(Base):
    __tablename__ = "modules"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    positioning = Column(Text)
    description = Column(Text)
    capabilities = Column(JSON, default=list)
    primary_algorithms = Column(JSON, default=list)
    baseline_algorithms = Column(JSON, default=list)
    status = Column(SAEnum(ImplementationStatus), default=ImplementationStatus.paper_only)
    meta_json = Column(JSON, default=dict)  # frontend-facing: problem_focus, io_contract, frontend_views,
                                            # showcase_points, backend_service, config_fields, backend_plugins

    algorithms = relationship("Algorithm", secondary="module_algorithms", back_populates="modules")
    references = relationship("ReferenceAsset", secondary="module_references", back_populates="modules")


class Algorithm(Base):
    __tablename__ = "algorithms"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String)
    description = Column(Text)
    task_types = Column(JSON, default=list)
    parameter_schema = Column(JSON, default=dict)
    hooks = Column(JSON, default=list)
    status = Column(SAEnum(ImplementationStatus), default=ImplementationStatus.paper_only)

    modules = relationship("Module", secondary="module_algorithms", back_populates="algorithms")
    references = relationship("ReferenceAsset", secondary="algorithm_references", back_populates="algorithms")


class ModuleAlgorithm(Base):
    __tablename__ = "module_algorithms"
    module_id = Column(String, ForeignKey("modules.id"), primary_key=True)
    algorithm_id = Column(String, ForeignKey("algorithms.id"), primary_key=True)
    role = Column(String, default="primary")


class ReferenceAsset(Base):
    __tablename__ = "reference_assets"
    id = Column(String, primary_key=True)
    asset_type = Column(SAEnum(AssetType), nullable=False)
    title = Column(String, nullable=False)
    local_path = Column(String)
    external_url = Column(String)
    default_branch = Column(String)
    key_files = Column(JSON, default=list)
    metadata_json = Column(JSON, default=dict)
    status = Column(SAEnum(ImplementationStatus), default=ImplementationStatus.paper_only)

    modules = relationship("Module", secondary="module_references", back_populates="references")
    algorithms = relationship("Algorithm", secondary="algorithm_references", back_populates="references")


class ModuleReference(Base):
    __tablename__ = "module_references"
    module_id = Column(String, ForeignKey("modules.id"), primary_key=True)
    reference_asset_id = Column(String, ForeignKey("reference_assets.id"), primary_key=True)
    role = Column(String, default="paper")


class AlgorithmReference(Base):
    __tablename__ = "algorithm_references"
    algorithm_id = Column(String, ForeignKey("algorithms.id"), primary_key=True)
    reference_asset_id = Column(String, ForeignKey("reference_assets.id"), primary_key=True)
    role = Column(String, default="paper")


class Dataset(Base):
    __tablename__ = "datasets"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    task_type = Column(String)
    split_options = Column(JSON, default=list)
    description = Column(Text)


class Experiment(Base):
    __tablename__ = "experiments"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    task_type = Column(String)
    dataset = Column(String)
    algorithm = Column(String)
    optimizer = Column(String)
    defense = Column(String)
    attack = Column(String)
    status = Column(SAEnum(ExperimentStatus), default=ExperimentStatus.draft)
    current_round = Column(Integer, default=0)
    total_rounds = Column(Integer, default=100)
    config_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)


class ExperimentMetric(Base):
    __tablename__ = "experiment_metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    round = Column(Integer, nullable=False)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)


class ExperimentEvent(Base):
    __tablename__ = "experiment_events"
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    event_type = Column(String, nullable=False)
    payload_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class AgentTask(Base):
    """Auditable execution record for one backend agent task.

    The input is represented by a hash rather than persisted verbatim. Outputs
    are expected to be structured, redacted JSON so the workflow can be shown
    in the UI without exposing prompts, keys, or private training data.
    """

    __tablename__ = "agent_tasks"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    agent_name = Column(String, nullable=False)
    stage = Column(String, nullable=False)
    status = Column(String, nullable=False, default="queued")
    input_hash = Column(String, nullable=False)
    output_json = Column(JSON, default=dict)
    tool_calls_json = Column(JSON, default=list)
    provider = Column(String, nullable=True)
    model = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)


class OperationsReportRecord(Base):
    """Runtime/reproducibility report produced by 运行保障智能体."""

    __tablename__ = "operations_reports"
    id = Column(String, primary_key=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    stage = Column(String, nullable=False)
    status = Column(String, nullable=False)
    findings_json = Column(JSON, default=list)
    manifest_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class SecurityObservation(Base):
    """One explainable security observation from the passive monitor."""

    __tablename__ = "security_observations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String, ForeignKey("experiments.id"), nullable=False)
    round = Column(Integer, nullable=True)
    category = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    signal_json = Column(JSON, default=dict)
    evidence_json = Column(JSON, default=list)
    recommended_action = Column(String, nullable=False, default="continue")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ResultAnalysisRecord(Base):
    """Evidence-bound analysis for the current run, without historical comparison."""

    __tablename__ = "result_analyses"
    experiment_id = Column(String, ForeignKey("experiments.id"), primary_key=True)
    status = Column(String, nullable=False, default="pending")
    claim_level = Column(String, nullable=False, default="descriptive_only")
    analysis_json = Column(JSON, default=dict)
    evidence_refs_json = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
