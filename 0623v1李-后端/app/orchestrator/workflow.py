"""Four-agent experiment workflow and its deterministic policy boundary."""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.operations_agent import OperationsAgent
from app.agents.result_analysis_agent import ResultAnalysisAgent
from app.agents.security_monitor_agent import SecurityMonitorAgent
from app.agents.inspection_protocol import build_security_evidence_explanations
from app.core.database import async_session
from app.models.models import (
    AgentTask,
    ExperimentStatus,
    OperationsReportRecord,
    ResultAnalysisRecord,
    SecurityObservation,
)
from app.services import services


AGENT_ORDER = [
    ("实验场景智能体", "scenario_configuration"),
    ("运行保障智能体", "operations"),
    ("安全监控智能体", "security"),
    ("结果分析智能体", "result_analysis"),
]


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _safe_json(value: Any, depth: int = 0) -> Any:
    """Keep persisted agent output bounded and free of credential-shaped fields."""
    if depth > 6:
        return "[truncated]"
    if isinstance(value, dict):
        output = {}
        for key, item in list(value.items())[:100]:
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("api_key", "authorization", "password", "secret", "token")):
                output[key_text] = "[redacted]"
            else:
                output[key_text] = _safe_json(item, depth + 1)
        return output
    if isinstance(value, (list, tuple)):
        return [_safe_json(item, depth + 1) for item in list(value)[:160]]
    if isinstance(value, str):
        return value[:6000]
    return value


def _task_llm_metadata(metadata: Any) -> tuple[str | None, str | None]:
    """Flatten single-agent or combined security-model metadata for the UI."""
    if not isinstance(metadata, dict):
        return None, None
    if "provider" in metadata or "model" in metadata:
        return metadata.get("provider"), metadata.get("model")
    nested = [item for item in metadata.values() if isinstance(item, dict)]
    providers = {item.get("provider") for item in nested if item.get("provider")}
    models = {item.get("model") for item in nested if item.get("model")}
    provider = next(iter(providers)) if len(providers) == 1 else ("mixed" if providers else None)
    model = next(iter(models)) if len(models) == 1 else ("mixed" if models else None)
    return provider, model


async def _create_task(db: AsyncSession, experiment_id: str, agent_name: str, stage: str, input_payload: dict) -> AgentTask:
    task = AgentTask(
        id=f"agt_{uuid.uuid4().hex[:16]}",
        experiment_id=experiment_id,
        agent_name=agent_name,
        stage=stage,
        status="running",
        input_hash=_hash_payload(input_payload),
        output_json={},
        tool_calls_json=[],
        started_at=datetime.datetime.utcnow(),
    )
    db.add(task)
    await db.commit()
    return task


async def _finish_task(db: AsyncSession, task: AgentTask, output: dict, status: str = "completed", error: str | None = None):
    safe_output = _safe_json(output)
    task.status = status
    task.output_json = safe_output
    task.tool_calls_json = safe_output.get("tool_calls", []) if isinstance(safe_output, dict) else []
    metadata = safe_output.get("llm_metadata", {}) if isinstance(safe_output, dict) else {}
    task.provider, task.model = _task_llm_metadata(metadata)
    snapshot_hash = safe_output.get("inspection_snapshot_hash") if isinstance(safe_output, dict) else None
    if snapshot_hash:
        # The task hash should identify the actual JSON inspected by the LLM,
        # rather than only the small orchestration payload used to queue it.
        task.input_hash = str(snapshot_hash)
    task.error = error
    task.finished_at = datetime.datetime.utcnow()
    await db.commit()


async def _write_agent_event(db: AsyncSession, experiment_id: str, output: dict):
    await services.add_event(
        db,
        experiment_id,
        "agent_update",
        {
            "agent_name": output.get("agent_name"),
            "stage": output.get("stage"),
            "status": output.get("status"),
            "round": output.get("round"),
            "tool_count": len(output.get("tool_calls") or []),
            "finding_count": len(output.get("findings") or output.get("agent_security_findings") or output.get("fl_security_findings") or []),
            "message": (output.get("summary") or {}).get("message", "智能体任务已完成"),
        },
    )


async def _write_operations_report(db: AsyncSession, experiment_id: str, output: dict):
    db.add(
        OperationsReportRecord(
            id=f"ops_{uuid.uuid4().hex[:16]}",
            experiment_id=experiment_id,
            stage=output.get("stage", "unknown"),
            status=output.get("status", "warning"),
            findings_json=_safe_json(output.get("findings", [])),
            manifest_json=_safe_json(output.get("manifest", {})),
        )
    )
    await db.commit()


async def _write_security_observation(
    db: AsyncSession,
    experiment_id: str,
    output: dict,
    category: str,
    round_num: int | None = None,
):
    db.add(
        SecurityObservation(
            experiment_id=experiment_id,
            round=round_num,
            category=category,
            severity=output.get("status", "normal"),
            signal_json=_safe_json(output),
            evidence_json=_safe_json(output.get("evidence_refs", [])),
            recommended_action=output.get("recommended_action", "continue"),
        )
    )
    await db.commit()


async def run_preflight(db: AsyncSession, experiment_id: str) -> dict:
    exp = await services.get_experiment(db, experiment_id)
    if not exp:
        raise ValueError("Experiment not found")
    if exp.status not in {ExperimentStatus.draft, ExperimentStatus.stopped, ExperimentStatus.ready_to_start, ExperimentStatus.blocked}:
        raise ValueError("Experiment is not eligible for preflight")

    await services.update_experiment_status(db, experiment_id, ExperimentStatus.preflight_pending)
    await services.add_event(db, experiment_id, "workflow_stage", {"stage": "preflight", "status": "started"})

    config_payload = {"config": exp.config_json or {}, "task_type": exp.task_type, "dataset": exp.dataset}
    operations_task = await _create_task(db, experiment_id, "运行保障智能体", "preflight", config_payload)
    try:
        operations_output = await OperationsAgent().preflight(db, experiment_id)
        await _finish_task(db, operations_task, operations_output)
    except Exception as exc:
        operations_output = {
            "agent_name": "运行保障智能体",
            "stage": "preflight",
            "status": "blocked",
            "findings": [{"code": "agent_execution_failed", "severity": "blocking", "message": "运行保障任务执行失败。"}],
            "manifest": {},
            "tool_calls": [],
        }
        await _finish_task(db, operations_task, operations_output, status="failed", error=type(exc).__name__)
    await _write_operations_report(db, experiment_id, operations_output)
    await _write_agent_event(db, experiment_id, operations_output)

    security_task = await _create_task(db, experiment_id, "安全监控智能体", "preflight_baseline", config_payload)
    try:
        security_output = await SecurityMonitorAgent().inspect(
            db,
            experiment_id,
            stage="preflight_baseline",
        )
        await _finish_task(db, security_task, security_output)
    except Exception as exc:
        security_output = {
            "agent_name": "安全监控智能体",
            "stage": "preflight_baseline",
            "status": "high_risk",
            "agent_security_findings": [{"code": "agent_execution_failed", "severity": "high_risk", "message": "安全监控任务执行失败。"}],
            "fl_security_findings": [],
            "recommended_action": "pause_review",
            "evidence_refs": [],
            "tool_calls": [],
        }
        await _finish_task(db, security_task, security_output, status="failed", error=type(exc).__name__)
    await _write_security_observation(db, experiment_id, security_output, "agent_security_baseline")
    await _write_agent_event(db, experiment_id, security_output)

    blocked = operations_output.get("status") == "blocked" or security_output.get("status") == "critical"
    next_status = ExperimentStatus.blocked if blocked else ExperimentStatus.ready_to_start
    await services.update_experiment_status(db, experiment_id, next_status)
    await services.add_event(
        db,
        experiment_id,
        "workflow_stage",
        {
            "stage": "preflight",
            "status": "blocked" if blocked else "ready_to_start",
            "message": "预检存在阻断项" if blocked else "预检完成，等待用户确认启动",
        },
    )
    return await get_workflow_snapshot(db, experiment_id)


def _runtime_control_action(operations_output: dict, security_output: dict) -> str:
    """Convert agent observations into a deterministic runner-control decision."""
    statuses = {
        str(operations_output.get("status", "passed")),
        str(security_output.get("status", "normal")),
    }
    actions = {
        str(operations_output.get("recommended_action", "continue")),
        str(security_output.get("recommended_action", "continue")),
    }
    if "blocked" in statuses or "critical" in statuses or "stop_and_review" in actions:
        return "stop_and_review"
    if "high_risk" in statuses or "warning" in statuses or "pause_review" in actions:
        return "pause_review"
    return "continue"


async def run_runtime_checks(db: AsyncSession, experiment_id: str, round_num: int) -> dict:
    """Run both live guard agents after a persisted training round.

    This is intentionally synchronous with the runner's round loop: the next
    round is not entered until the operations and security observations are
    persisted. The LLM can explain findings, but the deterministic control
    action below is the only thing allowed to request a runner stop.
    """
    exp = await services.get_experiment(db, experiment_id)
    if not exp:
        raise ValueError("Experiment not found")

    await services.add_event(
        db,
        experiment_id,
        "runtime_agent_check",
        {"round": round_num, "status": "started", "message": "运行中保障与安全检查开始。"},
    )
    config_payload = {
        "config": exp.config_json or {},
        "current_round": round_num,
        "trigger": "periodic_runtime_check",
    }

    operations_task = await _create_task(db, experiment_id, "运行保障智能体", "runtime", config_payload)
    try:
        operations_output = await OperationsAgent().runtime_check(db, experiment_id, round_num)
        await _finish_task(db, operations_task, operations_output)
    except Exception as exc:
        operations_output = {
            "agent_name": "运行保障智能体",
            "stage": "runtime",
            "round": round_num,
            "status": "blocked",
            "recommended_action": "stop_and_review",
            "summary": {"message": "运行中保障检查执行失败，已请求停止并复核。"},
            "findings": [
                {
                    "code": "runtime_agent_execution_failed",
                    "severity": "blocking",
                    "message": "运行保障智能体在运行期间执行失败。",
                }
            ],
            "manifest": {},
            "tool_calls": [],
        }
        await _finish_task(db, operations_task, operations_output, status="failed", error=type(exc).__name__)
    await _write_operations_report(db, experiment_id, operations_output)
    await _write_agent_event(db, experiment_id, operations_output)

    security_task = await _create_task(db, experiment_id, "安全监控智能体", "runtime", config_payload)
    try:
        security_output = await SecurityMonitorAgent().inspect(
            db,
            experiment_id,
            stage="runtime",
            round_num=round_num,
        )
        await _finish_task(db, security_task, security_output)
    except Exception as exc:
        security_output = {
            "agent_name": "安全监控智能体",
            "stage": "runtime",
            "round": round_num,
            "status": "critical",
            "agent_security_findings": [
                {
                    "code": "runtime_agent_execution_failed",
                    "severity": "critical",
                    "message": "安全监控智能体在运行期间执行失败。",
                }
            ],
            "fl_security_findings": [],
            "recommended_action": "stop_and_review",
            "evidence_refs": [],
            "tool_calls": [],
        }
        await _finish_task(db, security_task, security_output, status="failed", error=type(exc).__name__)
    await _write_security_observation(db, experiment_id, security_output, "runtime_security", round_num)
    await _write_agent_event(db, experiment_id, security_output)

    control_action = _runtime_control_action(operations_output, security_output)
    result = {
        "round": round_num,
        "control_action": control_action,
        "operations_status": operations_output.get("status"),
        "security_status": security_output.get("status"),
    }
    await services.add_event(
        db,
        experiment_id,
        "runtime_agent_check",
        {
            **result,
            "status": "stop_requested" if control_action == "stop_and_review" else "completed",
            "message": (
                "运行中安全/保障检查请求停止并复核。"
                if control_action == "stop_and_review"
                else "运行中保障与安全检查完成。"
            ),
        },
    )
    return result


async def run_runtime_agent_checkpoint(db: AsyncSession, experiment_id: str, round_num: int) -> bool:
    """Run a periodic checkpoint and return whether the runner must stop."""
    exp = await services.get_experiment(db, experiment_id)
    if not exp:
        return True
    config = exp.config_json or {}
    try:
        interval = max(1, int(config.get("runtime_agent_check_interval_rounds", 3) or 3))
        total_rounds = int(config.get("rounds", exp.total_rounds or 0) or 0)
    except (TypeError, ValueError):
        interval, total_rounds = 3, 0
    if round_num % interval != 0 and (not total_rounds or round_num < total_rounds):
        return False
    result = await run_runtime_checks(db, experiment_id, round_num)
    if result.get("control_action") != "stop_and_review":
        return False
    await services.update_experiment_status(db, experiment_id, ExperimentStatus.stopped)
    await services.add_event(
        db,
        experiment_id,
        "stopped",
        {
            "round": round_num,
            "message": "运行中保障或安全检查发现阻断性问题，实验已停止并等待复核。",
            "source": "deterministic_runtime_guard",
        },
    )
    return True


async def execute_runner(runner) -> None:
    """Run a selected runner, then launch final agent tasks in a fresh session."""
    await runner.run()
    async with async_session() as db:
        exp = await services.get_experiment(db, runner.experiment_id)
        if exp and exp.status == ExperimentStatus.completed:
            await finalize_experiment(db, runner.experiment_id)


async def finalize_experiment(db: AsyncSession, experiment_id: str) -> dict:
    exp = await services.get_experiment(db, experiment_id)
    if not exp or exp.status != ExperimentStatus.completed:
        return await get_workflow_snapshot(db, experiment_id)
    await services.update_experiment_status(db, experiment_id, ExperimentStatus.result_analyzing)
    await services.add_event(db, experiment_id, "workflow_stage", {"stage": "post_run", "status": "started"})
    config_payload = {"config": exp.config_json or {}, "current_round": exp.current_round}

    operations_task = await _create_task(db, experiment_id, "运行保障智能体", "finalize", config_payload)
    operations_output = await OperationsAgent().finalize(db, experiment_id)
    await _finish_task(db, operations_task, operations_output)
    await _write_operations_report(db, experiment_id, operations_output)
    await _write_agent_event(db, experiment_id, operations_output)

    security_task = await _create_task(db, experiment_id, "安全监控智能体", "finalize", config_payload)
    security_output = await SecurityMonitorAgent().inspect(db, experiment_id, stage="finalize")
    await _finish_task(db, security_task, security_output)
    await _write_security_observation(db, experiment_id, security_output, "security_summary")
    await _write_agent_event(db, experiment_id, security_output)

    result_task = await _create_task(db, experiment_id, "结果分析智能体", "result_analysis", config_payload)
    result_output = await ResultAnalysisAgent().analyze(db, experiment_id)
    await _finish_task(db, result_task, result_output)
    result_record = ResultAnalysisRecord(
        experiment_id=experiment_id,
        status="completed",
        claim_level=result_output.get("claim_level", "descriptive_only"),
        analysis_json=_safe_json(result_output.get("analysis", {})),
        evidence_refs_json=_safe_json(result_output.get("evidence_refs", [])),
        created_at=datetime.datetime.utcnow(),
        updated_at=datetime.datetime.utcnow(),
    )
    db.add(result_record)
    await db.commit()
    await _write_agent_event(db, experiment_id, result_output)

    await services.update_experiment_status(db, experiment_id, ExperimentStatus.analyzed)
    await services.add_event(
        db,
        experiment_id,
        "result_analysis_completed",
        {
            "claim_level": result_output.get("claim_level", "descriptive_only"),
            "evidence_count": len(result_output.get("evidence_refs") or []),
            "message": "结果分析智能体已生成证据化分析；未执行历史实验比较。",
        },
    )
    return await get_workflow_snapshot(db, experiment_id)


def _serialize_task(task: AgentTask) -> dict:
    return {
        "id": task.id,
        "agent_name": task.agent_name,
        "stage": task.stage,
        "status": task.status,
        "input_hash": task.input_hash,
        "output": task.output_json or {},
        "tool_calls": task.tool_calls_json or [],
        "provider": task.provider,
        "model": task.model,
        "error": task.error,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
    }


async def _latest_operations(db: AsyncSession, experiment_id: str) -> dict | None:
    result = await db.execute(
        select(OperationsReportRecord)
        .where(OperationsReportRecord.experiment_id == experiment_id)
        .order_by(OperationsReportRecord.created_at.desc())
        .limit(1)
    )
    row = result.scalar_one_or_none()
    if not row:
        return None
    return {
        "stage": row.stage,
        "status": row.status,
        "findings": row.findings_json or [],
        "manifest": row.manifest_json or {},
        "created_at": row.created_at,
    }


async def _security_summary(db: AsyncSession, experiment_id: str) -> dict | None:
    result = await db.execute(
        select(SecurityObservation)
        .where(SecurityObservation.experiment_id == experiment_id)
        .order_by(SecurityObservation.created_at.desc())
        .limit(300)
    )
    rows = list(result.scalars())
    if not rows:
        return None
    reviewed_rows = [
        row for row in rows
        if row.severity in {"normal", "warning", "high_risk", "critical"}
    ]
    # The report shown in the current workflow should describe the latest
    # completed review. Older critical observations remain in the audit
    # trail, but must not permanently mask a later successful re-check.
    latest_row = reviewed_rows[0]
    latest_signal = latest_row.signal_json or {}
    agent_findings = latest_signal.get("agent_security_findings") or []
    fl_findings = latest_signal.get("fl_security_findings") or latest_signal.get("findings") or []
    evidence = latest_row.evidence_json or []
    evidence_gap_details = latest_signal.get("evidence_gap_details") or []
    # Older observations were persisted before the UI-facing explanation
    # field was introduced.  Rebuild the explanation from the retained
    # inspection JSON at read time so historical experiments remain useful
    # without mutating their audit records.
    if not evidence_gap_details:
        snapshot = latest_signal.get("inspection_snapshot")
        review = latest_signal.get("llm_assessment")
        if not isinstance(review, dict):
            review = {
                "missing_evidence": latest_signal.get("missing_evidence") or [],
                "findings": [*agent_findings, *fl_findings],
            }
        if isinstance(snapshot, dict):
            evidence_gap_details = build_security_evidence_explanations(review, snapshot)
    severity = latest_row.severity
    evidence_insufficient = bool(latest_signal.get("missing_evidence")) or any(
        finding.get("status") == "not_observed"
        or "证据不足" in str(finding.get("message", ""))
        or "evidence" in str(finding.get("message", "")).lower()
        for finding in [*agent_findings, *fl_findings]
        if isinstance(finding, dict)
    )
    if severity == "normal" and evidence_insufficient:
        # Keep persisted legacy observations internally auditable, while the
        # current workflow view no longer presents missing security evidence
        # as a safe-to-continue conclusion.
        severity = "warning"
    historical_max = max(
        (row.severity for row in reviewed_rows),
        key=lambda value: {"normal": 0, "warning": 1, "high_risk": 2, "critical": 3}.get(value, 0),
        default="normal",
    )
    return {
        "status": severity,
        "agent_security_findings": agent_findings[:120],
        "fl_security_findings": fl_findings[:120],
        "missing_evidence": latest_signal.get("missing_evidence") or [],
        "evidence_gap_details": evidence_gap_details[:120],
        "evidence_refs": evidence[:120],
        "recommended_action": {
            "normal": "continue",
            "warning": "pause_review",
            "high_risk": "pause_review",
            "critical": "stop_and_review",
        }.get(severity, "continue"),
        "observation_count": len(reviewed_rows),
        "historical_max_status": historical_max,
        "unreviewed_snapshot_count": len(rows) - len(reviewed_rows),
        "latest_round": latest_row.round,
    }


async def get_workflow_snapshot(db: AsyncSession, experiment_id: str) -> dict:
    exp = await services.get_experiment(db, experiment_id)
    if not exp:
        raise ValueError("Experiment not found")
    result = await db.execute(
        select(AgentTask).where(AgentTask.experiment_id == experiment_id).order_by(AgentTask.started_at)
    )
    # Keep every runtime checkpoint so the frontend can show that monitoring
    # really happened during training. The UI still chooses the latest task
    # for each agent card, while the snapshot preserves the full audit trail.
    tasks = [_serialize_task(task) for task in result.scalars()]
    analysis_result = await db.execute(
        select(ResultAnalysisRecord).where(ResultAnalysisRecord.experiment_id == experiment_id)
    )
    analysis_row = analysis_result.scalar_one_or_none()
    analysis = None
    if analysis_row:
        analysis = {
            "experiment_id": experiment_id,
            "status": analysis_row.status,
            "claim_level": analysis_row.claim_level,
            "analysis": analysis_row.analysis_json or {},
            "evidence_refs": analysis_row.evidence_refs_json or [],
            "created_at": analysis_row.created_at,
            "updated_at": analysis_row.updated_at,
        }
    runtime_rounds = {
        task.get("output", {}).get("round")
        for task in tasks
        if task.get("stage") == "runtime"
        and task.get("output", {}).get("round") is not None
    }
    return {
        "experiment_id": experiment_id,
        "experiment_status": exp.status.value if hasattr(exp.status, "value") else str(exp.status),
        "agents": tasks,
        "runtime_agent_check_interval_rounds": (exp.config_json or {}).get("runtime_agent_check_interval_rounds", 3),
        # One checkpoint creates one operations task and one security task.
        # Count checkpoint rounds, rather than the two tasks belonging to a
        # single round, so the frontend reports the number of actual checks.
        "runtime_check_count": len(runtime_rounds),
        "operations_report": await _latest_operations(db, experiment_id),
        "security_report": await _security_summary(db, experiment_id),
        "result_analysis": analysis,
    }
