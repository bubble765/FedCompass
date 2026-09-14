"""Experiments API."""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import ExperimentConfig, ExperimentOut, ExperimentBrief
from app.services import services
from app.runners.mock_runner import MockRunner
from app.runners.fedcompass_runner import FedCompassRunner
from app.models.models import ExperimentStatus
from app.orchestrator import workflow

router = APIRouter(prefix="/api", tags=["experiments"])

_runners: dict = {}


def _build_experiment_brief(e):
    """Build ExperimentBrief with computed frontend fields."""
    config = e.config_json or {}
    return ExperimentBrief(
        id=e.id,
        name=e.name,
        status=e.status.value if hasattr(e.status, "value") else str(e.status),
        algorithm=e.algorithm,
        dataset=e.dataset,
        current_round=e.current_round or 0,
        total_rounds=e.total_rounds or 100,
        rounds=e.total_rounds or 100,
        best_accuracy=None,
        participation_rate=config.get("participation_rate", 0.1),
        defense=e.defense or config.get("defense", "none"),
        attack=e.attack or config.get("attack", "none"),
        created_at=e.created_at,
    )


def _build_experiment_out(e):
    """Build ExperimentOut with computed frontend fields."""
    config = e.config_json or {}
    return ExperimentOut(
        id=e.id,
        name=e.name,
        task_type=e.task_type,
        dataset=e.dataset,
        algorithm=e.algorithm,
        optimizer=e.optimizer,
        defense=e.defense or config.get("defense", "none"),
        attack=e.attack or config.get("attack", "none"),
        status=e.status.value if hasattr(e.status, "value") else str(e.status),
        current_round=e.current_round or 0,
        total_rounds=e.total_rounds or 100,
        rounds=e.total_rounds or 100,
        num_clients=config.get("num_clients", 100),
        config_json=config,
        created_at=e.created_at,
        finished_at=e.finished_at,
    )


PRESET_TEMPLATES = [
    {
        "id": "low_participation", "name": "低参与率非IID场景",
        "description": "模拟低客户端参与率 + Dirichlet 异构划分，适合测试 FedAdapt 模块",
        "config": {
            "module_id": "participation_adaptation",
            "task_type": "vision_classification", "dataset": "cifar10", "split": "dirichlet",
            "network": "resnet18", "algorithm": "fedcvc", "optimizer": "sgd", "defense": "none", "attack": "none",
            "num_clients": 30, "participation_rate": 0.1,
            "rounds": 10, "local_epochs": 5, "batch_size": 64, "learning_rate": 0.01, "seed": 42,
            "runner_type": "fedcompass"
        }
    },
    {
        "id": "poison_defense", "name": "投毒攻击与防御对比",
        "description": "启用 FeaturePoison 攻击 + VERT 防御，测试 FedGuard 模块",
        "config": {
            "module_id": "robust_aggregation",
            "task_type": "defense_demo", "dataset": "cifar10", "split": "dirichlet",
            "network": "resnet18", "algorithm": "fedavg", "optimizer": "sgd", "defense": "vert", "attack": "featurepoison",
            "num_clients": 30, "participation_rate": 0.2,
            "rounds": 10, "local_epochs": 5, "batch_size": 64, "learning_rate": 0.01, "seed": 123,
            "runner_type": "fedcompass"
        }
    },
    {
        "id": "reid_baseline", "name": "ReID 域泛化基准",
        "description": "CO-EVO 在 Market1501 上的域泛化实验",
        "config": {
            "module_id": "reid_generalization",
            "task_type": "reid", "dataset": "market1501", "split": "domain_as_client",
            "network": "resnet50", "algorithm": "co_evo", "optimizer": "sgd", "defense": "none", "attack": "none",
            "num_clients": 30, "participation_rate": 0.5,
            "rounds": 10, "local_epochs": 2, "batch_size": 8, "learning_rate": 0.001, "seed": 42,
            "train_sample_cap_per_client": 8, "eval_sample_cap": 24, "embedding_dim": 128,
            "runner_type": "fedcompass"
        }
    },
    {
        "id": "fedulip_3d", "name": "FedULIP 3D 多模态",
        "description": "FedULIP 在 ModelNet40 上的联邦 3D 学习",
        "config": {
            "module_id": "multimodal_3d",
            "task_type": "ulip3d", "dataset": "modelnet40", "split": "iid",
            "network": "pointbert", "algorithm": "fedulip", "optimizer": "sgd", "defense": "none", "attack": "none",
            "num_clients": 30, "participation_rate": 0.5,
            "rounds": 10, "local_epochs": 3, "batch_size": 16, "learning_rate": 0.001, "seed": 42,
            "train_sample_cap_per_client": 64, "eval_sample_cap": 800, "embedding_dim": 256,
            "adapter_type": "saca", "adapter_weight": 0.1,
            "runner_type": "fedcompass"
        }
    },
]


@router.get("/experiments/templates")
async def list_templates():
    return PRESET_TEMPLATES


@router.post("/experiments", response_model=ExperimentOut)
async def create_experiment(config: ExperimentConfig, db: AsyncSession = Depends(get_db)):
    exp = await services.create_experiment(db, config)
    return _build_experiment_out(exp)


@router.get("/experiments", response_model=list[ExperimentBrief])
async def list_experiments(status: str | None = None, db: AsyncSession = Depends(get_db)):
    exps = await services.list_experiments(db, status)
    return [_build_experiment_brief(e) for e in exps]


@router.get("/experiments/{exp_id}", response_model=ExperimentOut)
async def get_experiment(exp_id: str, db: AsyncSession = Depends(get_db)):
    exp = await services.get_experiment(db, exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    return _build_experiment_out(exp)


@router.post("/experiments/{exp_id}/start")
async def start_experiment(exp_id: str, background_tasks: BackgroundTasks, db: AsyncSession = Depends(get_db)):
    exp = await services.get_experiment(db, exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if exp.status != ExperimentStatus.ready_to_start:
        raise HTTPException(400, "Experiment must pass preflight before it can start")

    await services.update_experiment_status(db, exp_id, ExperimentStatus.running)
    await services.add_event(
        db,
        exp_id,
        "workflow_stage",
        {"stage": "training", "status": "started", "message": "用户确认启动，Runner 已提交。"},
    )
    runner_type = (exp.config_json or {}).get("runner_type", "mock")
    if runner_type == "fedcompass":
        runner = FedCompassRunner(exp.id)
    else:
        runner = MockRunner(exp.id)
    _runners[exp.id] = runner
    background_tasks.add_task(workflow.execute_runner, runner)
    return {"status": "started", "runner_type": runner_type, "workflow_status": "running"}


@router.post("/experiments/{exp_id}/stop")
async def stop_experiment(exp_id: str, db: AsyncSession = Depends(get_db)):
    exp = await services.get_experiment(db, exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    runner = _runners.get(exp_id)
    if runner:
        runner.stop()
    await services.update_experiment_status(db, exp_id, ExperimentStatus.stopped)
    await services.add_event(
        db,
        exp_id,
        "stopped",
        {"round": exp.current_round or 0, "message": "Experiment stopped by user"},
    )
    return {"status": "stopped"}


@router.post("/experiments/{exp_id}/clone", response_model=ExperimentOut)
async def clone_experiment(exp_id: str, db: AsyncSession = Depends(get_db)):
    new_exp = await services.clone_experiment(db, exp_id)
    if not new_exp:
        raise HTTPException(404, "Experiment not found")
    return _build_experiment_out(new_exp)


@router.get("/experiments/{exp_id}/metrics")
async def get_experiment_metrics(exp_id: str, db: AsyncSession = Depends(get_db)):
    """Return experiment metrics in frontend-compatible format."""
    raw = await services.get_metrics(db, exp_id)
    history = []
    if "test_accuracy" in raw:
        rounds = [p["round"] for p in raw["test_accuracy"]]
        acc_vals = [p["value"] for p in raw["test_accuracy"]]
        loss_vals = [p["value"] for p in raw.get("train_loss", raw.get("test_accuracy", []))]
        map_vals = [p["value"] for p in raw.get("map", [])]
        rank1_vals = [p["value"] for p in raw.get("rank1", [])]
        det_vals = [p["value"] for p in raw.get("detection_accuracy", [])]
        comm_vals = [p["value"] for p in raw.get("communication_cost_mb", [])]
        drift_vals = [p["value"] for p in raw.get("gradient_drift", [])]
        part_vals = [p["value"] for p in raw.get("client_participation_rate", [])]
        for i, r in enumerate(rounds):
            history.append({
                "round": r,
                "train_loss": loss_vals[i] if i < len(loss_vals) else 0.5,
                "test_accuracy": acc_vals[i] if i < len(acc_vals) else 0.8,
                "map": map_vals[i] if i < len(map_vals) else 0,
                "rank1": rank1_vals[i] if i < len(rank1_vals) else 0,
                "detection_accuracy": det_vals[i] if i < len(det_vals) else 0,
                "communication_cost_mb": comm_vals[i] if i < len(comm_vals) else 0,
                "gradient_drift": drift_vals[i] if i < len(drift_vals) else 0,
                "client_participation_rate": part_vals[i] if i < len(part_vals) else 0,
                "timestamp": "",
            })

    events = list(reversed(await services.get_logs(db, exp_id, 1000)))
    participation = []
    defense_by_round = {}
    for ev in events:
        if ev.event_type == "client_participation":
            p = ev.payload_json or {}
            participation.append({
                "round": p.get("round", 0),
                "participating": p.get("participating", []),
                "malicious": p.get("malicious", []),
            })
        elif ev.event_type == "defense_detection":
            d = ev.payload_json or {}
            defense_by_round[d.get("round", 0)] = {
                "round": d.get("round", 0),
                "trusted_clients": d.get("trusted_clients", []),
                "filtered_clients": d.get("filtered_clients", []),
                "detection_accuracy": d.get("detection_accuracy", 0.0),
            }
        elif ev.event_type == "round_complete":
            d = ev.payload_json or {}
            if "trusted_clients" in d or "filtered_clients" in d:
                defense_by_round[d.get("round", 0)] = {
                    "round": d.get("round", 0),
                    "trusted_clients": d.get("trusted_clients", []),
                    "filtered_clients": d.get("filtered_clients", []),
                    "detection_accuracy": d.get("detection_accuracy", 0.0),
                }

    defense = [defense_by_round[round_num] for round_num in sorted(defense_by_round)]

    last5 = history[-5:]
    if len(last5) >= 2:
        ema_heterogeneity = round(
            sum(abs(last5[i]["test_accuracy"] - last5[i - 1]["test_accuracy"]) for i in range(1, len(last5)))
            / max(len(last5) - 1, 1),
            4,
        )
    else:
        recent_participation = raw.get("client_participation_rate", [])[-10:]
        ema_heterogeneity = round(
            (sum(point["value"] for point in recent_participation) / max(len(recent_participation), 1)) if recent_participation else 0.0,
            4,
        )
    gradient_drift = round(raw.get("gradient_drift", [])[-1]["value"], 4) if raw.get("gradient_drift") else 0.0
    return {
        "history": history,
        "participation": participation,
        "defense": defense,
        "drift": {
            "ema_heterogeneity": ema_heterogeneity,
            "gradient_drift": gradient_drift,
            "compensation": round(gradient_drift * 0.62, 4),
        },
    }


@router.get("/experiments/{exp_id}/logs")
async def get_experiment_logs(exp_id: str, limit: int = 200, db: AsyncSession = Depends(get_db)):
    logs = await services.get_logs(db, exp_id, limit)
    return [{"event_type": l.event_type, "payload": l.payload_json, "created_at": l.created_at.isoformat()} for l in reversed(logs)]
