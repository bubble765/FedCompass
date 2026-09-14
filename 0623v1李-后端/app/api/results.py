"""Results API."""
import csv
import io
import json
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import CompareRequest, CompareResult
from app.services import services

router = APIRouter(prefix="/api", tags=["results"])


def _build_history(metrics: dict[str, list[dict]]) -> list[dict]:
    rounds = []
    if metrics.get("test_accuracy"):
        rounds = [point["round"] for point in metrics["test_accuracy"]]
    elif metrics.get("train_loss"):
        rounds = [point["round"] for point in metrics["train_loss"]]

    history = []
    for idx, round_num in enumerate(rounds):
        history.append(
            {
                "round": round_num,
                "train_loss": metrics.get("train_loss", [{}] * len(rounds))[idx].get("value", 0.0) if idx < len(metrics.get("train_loss", [])) else 0.0,
                "test_accuracy": metrics.get("test_accuracy", [{}] * len(rounds))[idx].get("value", 0.0) if idx < len(metrics.get("test_accuracy", [])) else 0.0,
                "map": metrics.get("map", [{}] * len(rounds))[idx].get("value", 0.0) if idx < len(metrics.get("map", [])) else 0.0,
                "rank1": metrics.get("rank1", [{}] * len(rounds))[idx].get("value", 0.0) if idx < len(metrics.get("rank1", [])) else 0.0,
                "detection_accuracy": metrics.get("detection_accuracy", [{}] * len(rounds))[idx].get("value", 0.0) if idx < len(metrics.get("detection_accuracy", [])) else 0.0,
                "communication_cost_mb": metrics.get("communication_cost_mb", [{}] * len(rounds))[idx].get("value", 0.0) if idx < len(metrics.get("communication_cost_mb", [])) else 0.0,
                "gradient_drift": metrics.get("gradient_drift", [{}] * len(rounds))[idx].get("value", 0.0) if idx < len(metrics.get("gradient_drift", [])) else 0.0,
            }
        )
    return history


async def _resolve_result_source(db: AsyncSession, exp_id: str, config: dict) -> tuple[str, str]:
    data_source = config.get("data_source")
    training_mode = config.get("training_mode")
    if data_source and training_mode:
        return data_source, training_mode

    events = await services.get_logs(db, exp_id, 50)
    for event in events:
        payload = event.payload_json or {}
        event_data_source = payload.get("data_source")
        event_training_mode = payload.get("training_mode")
        if event_data_source or event_training_mode:
            return event_data_source or "mock", event_training_mode or "synthetic"

    return "mock", "synthetic"


def _is_backend_result_source(data_source: str | None) -> bool:
    return data_source in {"real", "backend_synthetic"}


async def _build_compare_result(db: AsyncSession, exp_id: str) -> CompareResult | None:
    exp = await services.get_experiment(db, exp_id)
    if not exp:
        return None
    metrics = await services.get_metrics(db, exp_id)
    accuracy_points = [p["value"] for p in metrics.get("test_accuracy", [])]
    loss_points = [p["value"] for p in metrics.get("train_loss", [])]
    map_points = [p["value"] for p in metrics.get("map", [])]
    rank1_points = [p["value"] for p in metrics.get("rank1", [])]
    detection_points = [p["value"] for p in metrics.get("detection_accuracy", [])]
    config = exp.config_json or {}

    best_acc = max(accuracy_points) if accuracy_points else None
    final_acc = accuracy_points[-1] if accuracy_points else None
    best_loss = min(loss_points) if loss_points else None
    map_val = max(map_points) if map_points else 0.0
    rank1_val = max(rank1_points) if rank1_points else 0.0
    detection_acc = max(detection_points) if detection_points else 0.0
    history = _build_history(metrics)
    data_source, training_mode = await _resolve_result_source(db, exp_id, config)

    return CompareResult(
        experiment_id=exp.id,
        experiment_name=exp.name,
        algorithm=exp.algorithm,
        dataset=exp.dataset,
        task_type=exp.task_type,
        best_accuracy=best_acc,
        final_accuracy=final_acc,
        best_loss=best_loss,
        communication_rounds=exp.current_round or exp.total_rounds,
        status=exp.status.value if hasattr(exp.status, "value") else str(exp.status),
        attack=config.get("attack", "none"),
        defense=config.get("defense", "none"),
        participation_rate=config.get("participation_rate", 0.1),
        seed=config.get("seed", 42),
        map=map_val,
        rank1=rank1_val,
        detection_accuracy=detection_acc,
        history=history,
        series={
            "accuracy": [point["test_accuracy"] for point in history],
            "loss": [point["train_loss"] for point in history],
            "robustness": [point["detection_accuracy"] for point in history],
            "rounds": [point["round"] for point in history],
        },
        data_source=data_source,
        training_mode=training_mode,
    )


@router.get("/results", response_model=list[CompareResult])
async def list_results(
    task_type: str | None = Query(None),
    algorithm: str | None = Query(None),
    dataset: str | None = Query(None),
    defense: str | None = Query(None),
    attack: str | None = Query(None),
    participation_rate: float | None = Query(None),
    seed: int | None = Query(None),
    include_mock: bool = Query(False),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    exps = await services.list_experiments(db)
    results = []
    for e in exps:
        if e.status.value not in ("completed", "running"):
            continue
        config = e.config_json or {}
        if task_type and e.task_type != task_type:
            continue
        if algorithm and e.algorithm != algorithm:
            continue
        if dataset and e.dataset != dataset:
            continue
        if defense and config.get("defense", "none") != defense:
            continue
        if attack and config.get("attack", "none") != attack:
            continue
        if participation_rate is not None and abs(config.get("participation_rate", 0.1) - participation_rate) > 0.001:
            continue
        if seed is not None and config.get("seed", 42) != seed:
            continue
        r = await _build_compare_result(db, e.id)
        if r and (include_mock or _is_backend_result_source(r.data_source)):
            results.append(r)
        if len(results) >= limit:
            break
    return results


@router.get("/results/{exp_id}/summary", response_model=CompareResult)
async def result_summary(exp_id: str, db: AsyncSession = Depends(get_db)):
    r = await _build_compare_result(db, exp_id)
    if not r:
        raise HTTPException(404, "Result not found")
    return r


@router.post("/results/compare", response_model=list[CompareResult])
async def compare_results(req: CompareRequest, db: AsyncSession = Depends(get_db)):
    results = []
    for exp_id in req.experiment_ids:
        r = await _build_compare_result(db, exp_id)
        if r:
            results.append(r)
    return results


@router.get("/results/recent", response_model=list[CompareResult])
async def recent_results(
    limit: int = Query(10, ge=1, le=50),
    include_mock: bool = Query(False),
    db: AsyncSession = Depends(get_db),
):
    exps = await services.list_experiments(db)
    completed = [e for e in exps if e.status.value in ("completed", "running")]
    results = []
    for e in completed:
        r = await _build_compare_result(db, e.id)
        if r and (include_mock or _is_backend_result_source(r.data_source)):
            results.append(r)
        if len(results) >= limit:
            break
    return results


@router.get("/results/export")
async def export_results(
    format: str = Query("json", pattern="^(json|csv|markdown)$"),
    experiment_ids: str | None = Query(None, description="comma-separated experiment IDs, omit for all completed"),
    db: AsyncSession = Depends(get_db),
):
    if experiment_ids:
        ids = [i.strip() for i in experiment_ids.split(",") if i.strip()]
    else:
        exps = await services.list_experiments(db)
        ids = [e.id for e in exps if e.status.value in ("completed",)]

    rows = []
    for exp_id in ids:
        r = await _build_compare_result(db, exp_id)
        if r:
            rows.append(r.model_dump())

    if format == "json":
        return Response(
            content=json.dumps(rows, ensure_ascii=False, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=results.json"},
        )
    elif format == "csv":
        output = io.StringIO()
        if rows:
            writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return Response(
            content=output.getvalue(),
            media_type="text/csv; charset=utf-8-sig",
            headers={"Content-Disposition": "attachment; filename=results.csv"},
        )
    elif format == "markdown":
        lines = []
        if rows:
            headers = list(rows[0].keys())
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        return Response(
            content="\n".join(lines),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": "attachment; filename=results.md"},
        )
