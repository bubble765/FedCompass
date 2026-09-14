"""API surface for the four-agent experiment workflow."""

import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import ExperimentStatus, ResultAnalysisRecord
from app.schemas.schemas import ResultAnalysisOut, WorkflowOut
from app.services import services
from app.orchestrator import workflow
from app.agents.result_analysis_agent import ResultAnalysisAgent


router = APIRouter(prefix="/api", tags=["workflow"])


@router.post("/experiments/{exp_id}/preflight", response_model=WorkflowOut)
async def run_preflight(exp_id: str, db: AsyncSession = Depends(get_db)):
    if not await services.get_experiment(db, exp_id):
        raise HTTPException(404, "Experiment not found")
    try:
        return await workflow.run_preflight(db, exp_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/experiments/{exp_id}/workflow", response_model=WorkflowOut)
async def get_workflow(exp_id: str, db: AsyncSession = Depends(get_db)):
    if not await services.get_experiment(db, exp_id):
        raise HTTPException(404, "Experiment not found")
    return await workflow.get_workflow_snapshot(db, exp_id)


@router.get("/experiments/{exp_id}/operations-report")
async def get_operations_report(exp_id: str, db: AsyncSession = Depends(get_db)):
    snapshot = await get_workflow(exp_id, db)
    return snapshot["operations_report"] or {
        "status": "pending",
        "findings": [],
        "manifest": {},
    }


@router.get("/experiments/{exp_id}/security-report")
async def get_security_report(exp_id: str, db: AsyncSession = Depends(get_db)):
    snapshot = await get_workflow(exp_id, db)
    return snapshot["security_report"] or {
        "status": "normal",
        "agent_security_findings": [],
        "fl_security_findings": [],
        "evidence_refs": [],
        "recommended_action": "continue",
    }


@router.get("/experiments/{exp_id}/result-analysis", response_model=ResultAnalysisOut)
async def get_result_analysis(exp_id: str, db: AsyncSession = Depends(get_db)):
    if not await services.get_experiment(db, exp_id):
        raise HTTPException(404, "Experiment not found")
    snapshot = await workflow.get_workflow_snapshot(db, exp_id)
    analysis = snapshot.get("result_analysis")
    if not analysis:
        raise HTTPException(404, "Result analysis not ready")
    return analysis


@router.post("/experiments/{exp_id}/result-analysis/regenerate", response_model=ResultAnalysisOut)
async def regenerate_result_analysis(exp_id: str, db: AsyncSession = Depends(get_db)):
    exp = await services.get_experiment(db, exp_id)
    if not exp:
        raise HTTPException(404, "Experiment not found")
    if exp.status not in {ExperimentStatus.completed, ExperimentStatus.analyzed, ExperimentStatus.result_analyzing}:
        raise HTTPException(400, "Result analysis requires a completed experiment")
    task = await workflow._create_task(
        db,
        exp_id,
        "结果分析智能体",
        "result_analysis",
        {"config": exp.config_json or {}, "regenerate": True},
    )
    try:
        output = await ResultAnalysisAgent().analyze(db, exp_id)
        await workflow._finish_task(db, task, output)
    except Exception as exc:
        await workflow._finish_task(db, task, {}, status="failed", error=type(exc).__name__)
        raise HTTPException(500, "Result analysis regeneration failed") from exc

    row = await db.get(ResultAnalysisRecord, exp_id)
    if row:
        row.status = "completed"
        row.claim_level = output.get("claim_level", "descriptive_only")
        row.analysis_json = workflow._safe_json(output.get("analysis", {}))
        row.evidence_refs_json = workflow._safe_json(output.get("evidence_refs", []))
        row.updated_at = datetime.datetime.utcnow()
    else:
        row = ResultAnalysisRecord(
            experiment_id=exp_id,
            status="completed",
            claim_level=output.get("claim_level", "descriptive_only"),
            analysis_json=workflow._safe_json(output.get("analysis", {})),
            evidence_refs_json=workflow._safe_json(output.get("evidence_refs", [])),
        )
        db.add(row)
    await db.commit()
    await services.add_event(
        db,
        exp_id,
        "result_analysis_regenerated",
        {"claim_level": output.get("claim_level", "descriptive_only"), "message": "结果分析已重新生成。"},
    )
    return {
        "experiment_id": exp_id,
        "status": row.status,
        "claim_level": row.claim_level,
        "analysis": row.analysis_json or {},
        "evidence_refs": row.evidence_refs_json or [],
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
