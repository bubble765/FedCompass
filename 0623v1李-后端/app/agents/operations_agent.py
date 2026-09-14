"""运行保障智能体: collect evidence and ask the LLM to inspect it."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.inspection_protocol import (
    OPERATIONS_OUTPUT_SCHEMA,
    OPERATIONS_SKILL,
    OPERATIONS_SYSTEM_PROMPT,
    build_inspection_snapshot,
    compact_metrics,
    model_unavailable_review,
    normalize_operations_review,
    operations_skill_for_stage,
    validate_llm_review,
)
from app.agents.llm_runtime import LLMRuntime
from app.orchestrator.tools import ToolRegistry


AGENT_NAME = "运行保障智能体"
OPERATIONS_TOOLS = {
    "experiment.read_config",
    "dataset.inspect",
    "environment.inspect",
    "experiment.manifest",
    "experiment.metrics.read",
    "experiment.events.read",
    # FL 轮次快照证据（指标/客户端参与/梯度漂移/防御观测）由安全监控移交本智能体检查。
    "security.observations.read",
}


class OperationsAgent:
    name = AGENT_NAME

    async def _llm_review(self, snapshot: dict[str, Any], stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the operations skill against one JSON snapshot.

        The status and findings in the returned review come from the model.
        Local code only validates the untrusted response and fails closed when
        the configured inspection model is unavailable or malformed.
        """
        llm = LLMRuntime(self.name, profile="background_agent")
        skill = operations_skill_for_stage(stage)
        raw_review = await llm.complete_json(
            system_prompt=OPERATIONS_SYSTEM_PROMPT,
            input_payload={"INSPECTION_JSON": snapshot},
            output_schema=OPERATIONS_OUTPUT_SCHEMA,
            # Qwen3 会先输出 <think> 思考块再输出正文，预算不足时
            # content 只有思考文本，导致 JSON 解析失败 fail-closed。
            # 请求预算用足 agent_llm_max_tokens（默认 1600），由
            # LLMRuntime 内部按配置截断。
            max_tokens=1600,
        )
        review = validate_llm_review(
            raw_review,
            skill=skill,
            status_values={"passed", "warning", "blocked"},
            severity_values={"normal", "warning", "blocking"},
            scope_values={"operations"},
        )
        if review is None:
            if llm.last_metadata.get("used_llm"):
                llm.last_metadata["validation_error"] = "invalid_inspection_json"
            review = model_unavailable_review("operations", stage)
        else:
            # FL 观测移交后保持 fail-closed 语义：证据不足不得判 passed + continue。
            review = normalize_operations_review(review)
        metadata = {
            **llm.last_metadata,
            "inspection_skill": skill["name"],
            "inspection_skill_version": skill["version"],
            "inspection_snapshot_hash": snapshot.get("snapshot_hash"),
        }
        return review, metadata

    @staticmethod
    def _resolve_plugins(config: dict[str, Any]) -> dict[str, Any]:
        """Collect plugin availability as evidence; do not turn it into a finding."""
        result: dict[str, Any] = {}
        try:
            from app.plugins.registry import get_algorithm, get_attack, get_defense

            resolvers = {
                "algorithm": get_algorithm,
                "defense": get_defense,
                "attack": get_attack,
            }
            for kind, resolver in resolvers.items():
                name = config.get(kind, "none")
                try:
                    resolver(name)
                    result[kind] = {"name": name, "availability": "available"}
                except Exception as exc:
                    result[kind] = {"name": name, "availability": "unavailable", "error_type": type(exc).__name__}
        except Exception as exc:
            result["registry"] = {"availability": "unavailable", "error_type": type(exc).__name__}
        return result

    @staticmethod
    def _manifest_with_runtime_context(
        manifest: dict[str, Any],
        *,
        plugin_resolution: dict[str, Any] | None = None,
        smoke_test: str | None = None,
        estimated_client_steps: int | None = None,
    ) -> dict[str, Any]:
        output = dict(manifest)
        if plugin_resolution is not None:
            output["plugin_resolution"] = plugin_resolution
        if smoke_test is not None:
            output["smoke_test"] = smoke_test
        if estimated_client_steps is not None:
            output["estimated_client_steps"] = estimated_client_steps
        output["manifest_hash"] = hashlib.sha256(
            json.dumps(output, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return output

    @staticmethod
    def _base_evidence(config_snapshot: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return {
            "experiment": {
                "experiment_id": config_snapshot.get("experiment_id"),
                "name": config_snapshot.get("name"),
                "task_type": config_snapshot.get("task_type"),
                "dataset_name": config_snapshot.get("dataset"),
                "status": config_snapshot.get("status"),
                "config": config,
            }
        }

    @staticmethod
    def _fl_round_snapshots(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract the most recent FL round snapshots as evidence.

        The snapshots are stored as security observations by the runner's
        per-round persist step; the operations agent inspects them for the
        FL runtime checks (metric / participation / drift / defense).
        """
        return [
            {
                "id": observation.get("id"),
                "round": observation.get("round"),
                "snapshot": observation.get("signal") or {},
            }
            for observation in observations or []
            if observation.get("category") == "fl_round_snapshot"
        ][-8:]

    @staticmethod
    def _output(
        *,
        stage: str,
        review: dict[str, Any],
        metadata: dict[str, Any],
        snapshot: dict[str, Any],
        manifest: dict[str, Any],
        tool_calls: list[dict[str, Any]],
        round_num: int | None = None,
    ) -> dict[str, Any]:
        output: dict[str, Any] = {
            "agent_name": AGENT_NAME,
            "stage": stage,
            "status": review["overall_status"],
            "recommended_action": review["recommended_action"],
            "summary": {
                "title": "运行保障检查已完成" if review["overall_status"] != "blocked" else "运行保障检查被阻断",
                "message": review["summary"],
            },
            "findings": review["findings"],
            "missing_evidence": review["missing_evidence"],
            "manifest": manifest,
            "tool_calls": tool_calls,
            "inspection_skill": OPERATIONS_SKILL["name"],
            "inspection_snapshot_hash": snapshot["snapshot_hash"],
            # Persist the exact bounded JSON input in the AgentTask output so
            # the frontend and audit trail can show what the model inspected.
            "inspection_snapshot": snapshot,
            "llm_assessment": review,
            "llm_metadata": metadata,
        }
        if round_num is not None:
            output["round"] = round_num
        return output

    async def preflight(self, db: AsyncSession, experiment_id: str) -> dict[str, Any]:
        tools = ToolRegistry(db, OPERATIONS_TOOLS)
        config_snapshot = await tools.call("experiment.read_config", experiment_id)
        dataset = await tools.call("dataset.inspect", experiment_id)
        environment = await tools.call("environment.inspect", experiment_id)
        manifest_base = await tools.call("experiment.manifest", experiment_id)
        config = config_snapshot.get("config") or {}

        rounds = int(config.get("rounds", 0) or 0)
        clients = int(config.get("num_clients", 0) or 0)
        rate = float(config.get("participation_rate", -1) or -1)
        plugin_resolution = self._resolve_plugins(config)
        smoke_test = "passed" if all(
            item.get("availability") == "available"
            for key, item in plugin_resolution.items()
            if key != "registry" and isinstance(item, dict)
        ) and plugin_resolution.get("registry", {}).get("availability", "available") != "unavailable" else "failed"
        estimated_steps = max(rounds * max(1, int(clients * max(rate, 0))), 1)
        manifest = self._manifest_with_runtime_context(
            manifest_base,
            plugin_resolution=plugin_resolution,
            smoke_test=smoke_test,
            estimated_client_steps=estimated_steps,
        )
        evidence = self._base_evidence(config_snapshot, config)
        evidence.update(
            {
                "dataset": dataset,
                "environment": environment,
                "plugin_resolution": plugin_resolution,
                "manifest": manifest,
                "progress": {
                    "stage": "preflight",
                    "expected_rounds": rounds,
                    "current_round": 0,
                },
            }
        )
        snapshot = build_inspection_snapshot(
            agent_name=self.name,
            stage="preflight",
            experiment_id=experiment_id,
            skill=operations_skill_for_stage("preflight"),
            evidence=evidence,
        )
        review, metadata = await self._llm_review(snapshot, "preflight")
        return self._output(
            stage="preflight",
            review=review,
            metadata=metadata,
            snapshot=snapshot,
            manifest=manifest,
            tool_calls=tools.calls,
        )

    async def finalize(self, db: AsyncSession, experiment_id: str) -> dict[str, Any]:
        tools = ToolRegistry(db, OPERATIONS_TOOLS)
        config_snapshot = await tools.call("experiment.read_config", experiment_id)
        dataset = await tools.call("dataset.inspect", experiment_id)
        environment = await tools.call("environment.inspect", experiment_id)
        metrics = await tools.call("experiment.metrics.read", experiment_id)
        events = await tools.call("experiment.events.read", experiment_id)
        observations = await tools.call("security.observations.read", experiment_id)
        config = config_snapshot.get("config") or {}
        plugin_resolution = self._resolve_plugins(config)
        manifest_base = await tools.call("experiment.manifest", experiment_id)
        manifest = self._manifest_with_runtime_context(
            manifest_base,
            plugin_resolution=plugin_resolution,
        )
        expected_rounds = int(config.get("rounds", 0) or 0)
        round_complete_count = sum(1 for event in events if event.get("event_type") == "round_complete")
        evidence = self._base_evidence(config_snapshot, config)
        evidence.update(
            {
                "dataset": dataset,
                "environment": environment,
                "plugin_resolution": plugin_resolution,
                "progress": {
                    "stage": "finalize",
                    "expected_rounds": expected_rounds,
                    "completed_rounds": round_complete_count,
                    "current_round": config_snapshot.get("current_round"),
                },
                "metrics": compact_metrics(metrics),
                "fl_round_snapshots": self._fl_round_snapshots(observations),
                "events": events[-100:],
                "manifest": manifest,
            }
        )
        snapshot = build_inspection_snapshot(
            agent_name=self.name,
            stage="finalize",
            experiment_id=experiment_id,
            skill=operations_skill_for_stage("finalize", config),
            evidence=evidence,
        )
        review, metadata = await self._llm_review(snapshot, "finalize")
        return self._output(
            stage="finalize",
            review=review,
            metadata=metadata,
            snapshot=snapshot,
            manifest=manifest,
            tool_calls=tools.calls,
        )

    async def runtime_check(self, db: AsyncSession, experiment_id: str, round_num: int) -> dict[str, Any]:
        """Inspect the live round state from one JSON evidence snapshot."""
        tools = ToolRegistry(db, OPERATIONS_TOOLS)
        config_snapshot = await tools.call("experiment.read_config", experiment_id)
        metrics = await tools.call("experiment.metrics.read", experiment_id)
        events = await tools.call("experiment.events.read", experiment_id)
        environment = await tools.call("environment.inspect", experiment_id)
        manifest = await tools.call("experiment.manifest", experiment_id)
        observations = await tools.call("security.observations.read", experiment_id)
        config = config_snapshot.get("config") or {}
        recent_events = [
            event
            for event in events
            if event.get("event_type") in {"round_complete", "client_participation", "security_snapshot", "security_observation"}
        ][-36:]
        recent_round_events = [
            event
            for event in recent_events
            if int((event.get("payload") or {}).get("round", 0) or 0) >= max(round_num - 2, 1)
        ]
        evidence = self._base_evidence(config_snapshot, config)
        evidence.update(
            {
                "environment": environment,
                "manifest": manifest,
                "progress": {
                    "stage": "runtime",
                    "current_round": round_num,
                    "recent_round_event_count": len(recent_round_events),
                    "current_round_event_count": sum(
                        1
                        for event in recent_round_events
                        if int((event.get("payload") or {}).get("round", 0) or 0) == round_num
                    ),
                },
                "metrics": compact_metrics(metrics, max_points=12),
                "fl_round_snapshots": self._fl_round_snapshots(observations),
                "events": recent_round_events[-24:],
            }
        )
        snapshot = build_inspection_snapshot(
            agent_name=self.name,
            stage="runtime",
            experiment_id=experiment_id,
            skill=operations_skill_for_stage("runtime", config),
            evidence=evidence,
        )
        review, metadata = await self._llm_review(snapshot, "runtime")
        return self._output(
            stage="runtime",
            review=review,
            metadata=metadata,
            snapshot=snapshot,
            manifest=manifest,
            tool_calls=tools.calls,
            round_num=round_num,
        )
