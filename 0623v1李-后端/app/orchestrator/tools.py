"""Allow-listed, read-only tools exposed to backend agents."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    AgentTask,
    Experiment,
    OperationsReportRecord,
    SecurityObservation,
)
from app.services import services


TOOL_DESCRIPTIONS = {
    "catalog.read": "读取模块、算法和配置约束目录",
    "config.resolve": "将场景草案归一化为受约束配置",
    "experiment.read_config": "读取已归一化实验配置（脱敏）",
    "dataset.inspect": "检查数据源和本地数据可用性",
    "environment.inspect": "读取解释器、CPU、内存和磁盘摘要",
    "experiment.manifest": "生成配置、环境和代码指纹",
    "experiment.metrics.read": "读取聚合后的指标序列",
    "experiment.events.read": "读取实验事件摘要",
    "agent_trace.read": "读取智能体任务状态和工具调用名称",
    "operations.report.read": "读取最近运行保障报告",
    "security.observations.read": "读取被动安全观察",
}


class ToolRegistry:
    """Per-agent registry. There is deliberately no shell or arbitrary HTTP tool."""

    def __init__(self, db: AsyncSession, allowed_tools: set[str]):
        self.db = db
        self.allowed_tools = set(allowed_tools)
        self.calls: list[dict[str, Any]] = []

    async def call(self, name: str, experiment_id: str) -> Any:
        if name not in self.allowed_tools:
            raise PermissionError(f"tool not allowed for this agent: {name}")
        method = {
            "experiment.read_config": self.read_config,
            "dataset.inspect": self.inspect_dataset,
            "environment.inspect": self.inspect_environment,
            "experiment.manifest": self.build_manifest,
            "experiment.metrics.read": self.read_metrics,
            "experiment.events.read": self.read_events,
            "agent_trace.read": self.read_agent_trace,
            "operations.report.read": self.read_operations_report,
            "security.observations.read": self.read_security_observations,
        }.get(name)
        if method is None:
            raise KeyError(f"unknown tool: {name}")
        try:
            result = await method(experiment_id)
            self.calls.append({"tool": name, "status": "ok"})
            return result
        except Exception as exc:
            self.calls.append({"tool": name, "status": "error", "error": type(exc).__name__})
            raise

    async def read_config(self, experiment_id: str) -> dict:
        exp = await services.get_experiment(self.db, experiment_id)
        if not exp:
            return {}
        config = dict(exp.config_json or {})
        for key in list(config):
            if any(marker in key.lower() for marker in ("key", "token", "secret", "password")):
                config[key] = "[redacted]"
        return {
            "experiment_id": exp.id,
            "name": exp.name,
            "task_type": exp.task_type,
            "dataset": exp.dataset,
            "algorithm": exp.algorithm,
            "status": exp.status.value if hasattr(exp.status, "value") else str(exp.status),
            "config": config,
        }

    async def inspect_dataset(self, experiment_id: str) -> dict:
        exp = await services.get_experiment(self.db, experiment_id)
        if not exp:
            return {"available": False, "reason": "experiment_not_found"}
        meta = services.get_dataset_runtime_meta(exp.dataset or "")
        return {"dataset": exp.dataset, **meta}

    async def inspect_environment(self, experiment_id: str) -> dict:
        del experiment_id
        disk = shutil.disk_usage(os.getcwd())
        result = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "cpu_count": os.cpu_count() or 1,
            "disk_free_gb": round(disk.free / (1024**3), 2),
        }
        try:
            import torch

            result["torch"] = torch.__version__
            result["cuda_available"] = bool(torch.cuda.is_available())
            result["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        except Exception:
            result["torch"] = "unavailable"
            result["cuda_available"] = False
            result["cuda_device_count"] = 0
        return result

    async def build_manifest(self, experiment_id: str) -> dict:
        config = await self.read_config(experiment_id)
        environment = await self.inspect_environment(experiment_id)
        fingerprint_input = {"config": config, "environment": environment}
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_input, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {
            "config_hash": hashlib.sha256(
                json.dumps(config.get("config", {}), ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "environment_hash": hashlib.sha256(
                json.dumps(environment, ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "reproduction_fingerprint": fingerprint,
            "seed": (config.get("config") or {}).get("seed"),
            "data_source": (config.get("config") or {}).get("data_source"),
            "training_mode": (config.get("config") or {}).get("training_mode"),
            "python": environment.get("python"),
            "torch": environment.get("torch"),
        }

    async def read_metrics(self, experiment_id: str) -> dict:
        return await services.get_metrics(self.db, experiment_id)

    async def read_events(self, experiment_id: str) -> list[dict]:
        logs = list(reversed(await services.get_logs(self.db, experiment_id, 500)))
        return [
            {
                "id": event.id,
                "event_type": event.event_type,
                "payload": event.payload_json or {},
                "created_at": event.created_at.isoformat() if event.created_at else None,
            }
            for event in logs
        ]

    async def read_agent_trace(self, experiment_id: str) -> list[dict]:
        rows = await self.db.execute(
            select(AgentTask).where(AgentTask.experiment_id == experiment_id).order_by(AgentTask.started_at)
        )
        output_keys = {
            "stage",
            "status",
            "recommended_action",
            "summary",
            "findings",
            "agent_security_findings",
            "fl_security_findings",
            "missing_evidence",
            "inspection_skill",
            "inspection_snapshot_hash",
            "llm_metadata",
        }
        return [
            {
                "agent_name": row.agent_name,
                "stage": row.stage,
                "status": row.status,
                "tool_calls": row.tool_calls_json or [],
                "output": {
                    key: (row.output_json or {}).get(key)
                    for key in sorted(output_keys)
                    if key in (row.output_json or {})
                },
                "provider": row.provider,
                "model": row.model,
                "error": row.error,
            }
            for row in rows.scalars()
        ]

    async def read_operations_report(self, experiment_id: str) -> dict:
        result = await self.db.execute(
            select(OperationsReportRecord)
            .where(OperationsReportRecord.experiment_id == experiment_id)
            .order_by(OperationsReportRecord.created_at.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        if not row:
            return {}
        return {
            "stage": row.stage,
            "status": row.status,
            "findings": row.findings_json or [],
            "manifest": row.manifest_json or {},
            "created_at": row.created_at.isoformat() if row.created_at else None,
        }

    async def read_security_observations(self, experiment_id: str) -> list[dict]:
        result = await self.db.execute(
            select(SecurityObservation)
            .where(SecurityObservation.experiment_id == experiment_id)
            .order_by(SecurityObservation.created_at.desc())
            .limit(200)
        )
        return [
            {
                "id": row.id,
                "round": row.round,
                "category": row.category,
                "severity": row.severity,
                "signal": row.signal_json or {},
                "evidence": row.evidence_json or [],
                "recommended_action": row.recommended_action,
            }
            for row in result.scalars()
        ]
