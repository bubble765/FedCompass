"""安全监控智能体: inspect a structured LLM/Agent/FL evidence snapshot."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.inspection_protocol import (
    SECURITY_OUTPUT_SCHEMA,
    SECURITY_SKILL,
    SECURITY_SYSTEM_PROMPT,
    build_inspection_snapshot,
    build_security_evidence_explanations,
    compact_metrics,
    model_unavailable_review,
    normalize_security_review,
    security_skill_for_stage,
    validate_llm_review,
)
from app.agents.llm_runtime import LLMRuntime
from app.orchestrator.tools import ToolRegistry


AGENT_NAME = "安全监控智能体"
SECURITY_TOOLS = {
    "experiment.read_config",
    "experiment.manifest",
    "experiment.events.read",
    "experiment.metrics.read",
    "agent_trace.read",
    "security.observations.read",
}

# The security monitor reviews the trace of the other workflow agents too.
# A single ``allowed_tools`` list for the security monitor would incorrectly
# make the operations agent's legitimate dataset/environment calls look like
# unauthorized calls. Keep the boundary keyed by agent name instead.
AGENT_TOOL_BOUNDARIES = {
    "实验场景智能体": sorted({"catalog.read", "config.resolve"}),
    "运行保障智能体": sorted({
        "experiment.read_config",
        "dataset.inspect",
        "environment.inspect",
        "experiment.manifest",
        "experiment.metrics.read",
        "experiment.events.read",
        # FL 轮次快照移交运行保障检查，读取安全观测记录是其合法能力。
        "security.observations.read",
    }),
    "安全监控智能体": sorted(SECURITY_TOOLS),
    "结果分析智能体": sorted({
        "experiment.read_config",
        "experiment.metrics.read",
        "experiment.events.read",
        "experiment.manifest",
        "operations.report.read",
        "security.observations.read",
    }),
}


def _compact_agent_trace(trace: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    """Keep security evidence useful without recursively replaying reports."""
    compact: list[dict[str, Any]] = []
    for item in (trace or [])[-limit:]:
        output = item.get("output") or {}
        findings = output.get("findings") or output.get("agent_security_findings") or output.get("fl_security_findings") or []
        metadata = output.get("llm_metadata") or {}
        compact.append(
            {
                "agent_name": item.get("agent_name"),
                "stage": item.get("stage"),
                "status": item.get("status"),
                "tool_calls": [call.get("tool") for call in (item.get("tool_calls") or []) if isinstance(call, dict)],
                "provider": item.get("provider"),
                "model": item.get("model"),
                "error": item.get("error"),
                "output": {
                    "recommended_action": output.get("recommended_action"),
                    "summary": str((output.get("summary") or {}).get("message", ""))[:800],
                    "finding_count": len(findings),
                    "missing_evidence": list(output.get("missing_evidence") or [])[:12],
                    "inspection_skill": output.get("inspection_skill"),
                    "inspection_snapshot_hash": output.get("inspection_snapshot_hash"),
                    "llm_metadata": {
                        key: metadata.get(key)
                        for key in ("provider", "model", "used_llm", "error", "validation_error")
                        if key in metadata
                    },
                },
            }
        )
    return compact


def _compact_events(events: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    """Retain untrusted event text and control fields, dropping nested reports."""
    compact: list[dict[str, Any]] = []
    for event in (events or [])[-limit:]:
        payload = event.get("payload") or {}
        compact.append(
            {
                "id": event.get("id"),
                "event_type": event.get("event_type"),
                "created_at": event.get("created_at"),
                "payload": {
                    key: (str(payload.get(key))[:800] if isinstance(payload.get(key), str) else payload.get(key))
                    for key in (
                        "stage",
                        "status",
                        "round",
                        "message",
                        "source",
                        "agent_name",
                        "control_action",
                        "tool_count",
                        "finding_count",
                    )
                    if key in payload
                },
            }
        )
    return compact


class SecurityMonitorAgent:
    name = AGENT_NAME

    async def _collect_snapshot(
        self,
        db: AsyncSession,
        experiment_id: str,
        stage: str,
        round_num: int | None,
    ) -> tuple[ToolRegistry, dict[str, Any]]:
        """Collect evidence; no local severity or attack classification occurs here."""
        tools = ToolRegistry(db, SECURITY_TOOLS)
        config_snapshot = await tools.call("experiment.read_config", experiment_id)
        manifest = await tools.call("experiment.manifest", experiment_id)
        trace = await tools.call("agent_trace.read", experiment_id)
        events = await tools.call("experiment.events.read", experiment_id)
        has_runtime_evidence = stage != "preflight_baseline"
        metrics = await tools.call("experiment.metrics.read", experiment_id) if has_runtime_evidence else {}
        observations = await tools.call("security.observations.read", experiment_id) if has_runtime_evidence else []
        config = config_snapshot.get("config") or {}
        raw_round_snapshots = [
            {
                "id": observation.get("id"),
                "round": observation.get("round"),
                "snapshot": observation.get("signal") or {},
            }
            for observation in observations
            if observation.get("category") == "fl_round_snapshot"
        ][-8:]
        previous_reviews = []
        for observation in observations:
            if (
                observation.get("category") == "fl_round_snapshot"
                or observation.get("severity") not in {"normal", "warning", "high_risk", "critical"}
            ):
                continue
            signal = observation.get("signal") or {}
            # Do not recursively feed an earlier full inspection snapshot
            # back into the next prompt.  The current raw FL snapshots remain
            # available above; history only needs its review decision and
            # audit pointers.
            previous_reviews.append(
                {
                    "id": observation.get("id"),
                    "round": observation.get("round"),
                    "category": observation.get("category"),
                    "status": observation.get("severity"),
                    "recommended_action": observation.get("recommended_action"),
                    "summary": signal.get("summary"),
                    "findings": signal.get("findings") or [],
                    "missing_evidence": signal.get("missing_evidence") or [],
                    "inspection_skill": signal.get("inspection_skill"),
                    "inspection_snapshot_hash": signal.get("inspection_snapshot_hash"),
                }
            )
        previous_reviews = previous_reviews[-3:]
        evidence = {
            "experiment": {
                "experiment_id": config_snapshot.get("experiment_id"),
                "name": config_snapshot.get("name"),
                "task_type": config_snapshot.get("task_type"),
                "dataset_name": config_snapshot.get("dataset"),
                "status": config_snapshot.get("status"),
                "config": config,
            },
            "manifest": manifest,
            # The model must see the effective capability boundary for each
            # agent represented in agent_trace. A single security-agent list
            # would produce false unauthorized-call observations for the
            # operations and result agents.
            "allowed_tool_boundaries": AGENT_TOOL_BOUNDARIES,
            "agent_trace": _compact_agent_trace(trace),
            "events": _compact_events(events),
            "monitor_context": {
                "stage": stage,
                "round": round_num,
                "has_runtime_evidence": has_runtime_evidence,
                "raw_round_snapshot_count": len(raw_round_snapshots),
                "previous_security_review_count": len(previous_reviews),
            },
        }
        if has_runtime_evidence:
            evidence.update(
                {
                    "metrics": compact_metrics(metrics, max_points=12),
                    "fl_round_snapshots": raw_round_snapshots,
                    "previous_security_reviews": previous_reviews,
                }
            )
        skill = security_skill_for_stage(stage, config)
        snapshot = build_inspection_snapshot(
            agent_name=self.name,
            stage=stage,
            experiment_id=experiment_id,
            skill=skill,
            evidence=evidence,
        )
        return tools, snapshot

    async def _llm_review(self, snapshot: dict[str, Any], stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
        llm = LLMRuntime(self.name, profile="background_agent")
        config = ((snapshot.get("evidence") or {}).get("experiment") or {}).get("config") or {}
        skill = security_skill_for_stage(stage, config)
        raw_review = await llm.complete_json(
            system_prompt=SECURITY_SYSTEM_PROMPT,
            input_payload={"INSPECTION_JSON": snapshot},
            output_schema=SECURITY_OUTPUT_SCHEMA,
            # Qwen3 会先输出 <think> 思考块再输出正文，预算不足时
            # content 只有思考文本，导致 JSON 解析失败 fail-closed。
            # 请求预算用足 agent_llm_max_tokens（默认 1600），由
            # LLMRuntime 内部按配置截断。
            max_tokens=1600,
        )
        review = validate_llm_review(
            raw_review,
            skill=skill,
            status_values={"normal", "warning", "high_risk", "critical"},
            severity_values={"normal", "warning", "high_risk", "critical"},
            scope_values={"agent", "fl", "shared"},
        )
        if review is None:
            if llm.last_metadata.get("used_llm"):
                llm.last_metadata["validation_error"] = "invalid_inspection_json"
            review = model_unavailable_review("security", stage)
        else:
            review = normalize_security_review(review)
        review["evidence_gap_details"] = build_security_evidence_explanations(review, snapshot)
        metadata = {
            **llm.last_metadata,
            "inspection_skill": skill["name"],
            "inspection_skill_version": skill["version"],
            "inspection_snapshot_hash": snapshot.get("snapshot_hash"),
        }
        return review, metadata

    @staticmethod
    def _split_findings(review: dict[str, Any]) -> tuple[list[dict], list[dict]]:
        agent_findings: list[dict] = []
        fl_findings: list[dict] = []
        for finding in review.get("findings", []):
            scope = finding.get("scope", "shared")
            if scope in {"agent", "shared"}:
                agent_findings.append(finding)
            if scope in {"fl", "shared"}:
                fl_findings.append(finding)
        return agent_findings, fl_findings

    @staticmethod
    def _evidence_refs(review: dict[str, Any], snapshot: dict[str, Any], round_num: int | None) -> list[dict[str, Any]]:
        refs = [
            {
                "type": "inspection_json",
                "path": path,
                **({"round": round_num} if round_num is not None else {}),
            }
            for finding in review.get("findings", [])
            for path in finding.get("evidence_paths", [])
        ]
        refs.append({"type": "inspection_snapshot", "hash": snapshot.get("snapshot_hash")})
        return refs[:160]

    async def inspect_workflow(
        self,
        db: AsyncSession,
        experiment_id: str,
        stage: str = "preflight_baseline",
    ) -> dict[str, Any]:
        """Compatibility entry point; the unified skill checks all scopes."""
        return await self.inspect(db, experiment_id, stage=stage)

    async def inspect_fl_security(
        self,
        db: AsyncSession,
        experiment_id: str,
        stage: str = "finalize",
    ) -> dict[str, Any]:
        """Compatibility entry point; FL checks are part of the unified review."""
        return await self.inspect(db, experiment_id, stage=stage)

    async def inspect(
        self,
        db: AsyncSession,
        experiment_id: str,
        stage: str = "finalize",
        round_num: int | None = None,
    ) -> dict[str, Any]:
        tools, snapshot = await self._collect_snapshot(db, experiment_id, stage, round_num)
        review, metadata = await self._llm_review(snapshot, stage)
        agent_findings, fl_findings = self._split_findings(review)
        return {
            "agent_name": self.name,
            "stage": stage,
            "status": review["overall_status"],
            "recommended_action": review["recommended_action"],
            "summary": {"title": "安全检查已完成", "message": review["summary"]},
            "agent_security_findings": agent_findings,
            "fl_security_findings": fl_findings,
            "findings": review["findings"],
            "missing_evidence": review["missing_evidence"],
            "evidence_gap_details": review.get("evidence_gap_details", []),
            "evidence_refs": self._evidence_refs(review, snapshot, round_num),
            "confidence": review["confidence"],
            "tool_calls": tools.calls,
            "inspection_skill": snapshot["inspection_skill"],
            "inspection_snapshot_hash": snapshot["snapshot_hash"],
            # This is the JSON document actually submitted to Qwen, retained
            # in the task output for audit and frontend inspection.
            "inspection_snapshot": snapshot,
            "llm_assessment": review,
            "llm_metadata": metadata,
            **({"round": round_num} if round_num is not None else {}),
        }
