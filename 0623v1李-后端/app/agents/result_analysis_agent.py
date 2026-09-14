"""结果分析智能体: explain the current run and bind every claim to evidence."""

from __future__ import annotations

import math
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.llm_runtime import LLMRuntime
from app.orchestrator.tools import ToolRegistry


AGENT_NAME = "结果分析智能体"
RESULT_TOOLS = {
    "experiment.read_config",
    "experiment.metrics.read",
    "experiment.events.read",
    "experiment.manifest",
    "operations.report.read",
    "security.observations.read",
}


def _values(metrics: dict, name: str) -> list[tuple[int, float]]:
    values = []
    for point in metrics.get(name, []) or []:
        try:
            value = float(point.get("value"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append((int(point.get("round", 0) or 0), value))
    return values


def _round_value(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


class ResultAnalysisAgent:
    name = AGENT_NAME

    async def analyze(self, db: AsyncSession, experiment_id: str) -> dict[str, Any]:
        tools = ToolRegistry(db, RESULT_TOOLS)
        config_snapshot = await tools.call("experiment.read_config", experiment_id)
        metrics = await tools.call("experiment.metrics.read", experiment_id)
        events = await tools.call("experiment.events.read", experiment_id)
        manifest = await tools.call("experiment.manifest", experiment_id)
        operations = await tools.call("operations.report.read", experiment_id)
        security_observations = await tools.call("security.observations.read", experiment_id)
        config = config_snapshot.get("config") or {}
        algorithm = config.get("algorithm", "unknown")
        dataset = config.get("dataset", "unknown")
        data_source = config.get("data_source", "unknown")
        training_mode = config.get("training_mode", "unknown")

        accuracy = _values(metrics, "test_accuracy")
        loss = _values(metrics, "train_loss")
        drift = _values(metrics, "gradient_drift")
        detection = _values(metrics, "detection_accuracy")
        round_complete = [event for event in events if event.get("event_type") == "round_complete"]
        completed_rounds = len(round_complete)

        statistics = {
            "completed_rounds": completed_rounds,
            "metric_rounds": len(accuracy or loss),
            "final_test_accuracy": _round_value(accuracy[-1][1] if accuracy else None),
            "best_test_accuracy": _round_value(max((value for _, value in accuracy), default=None)),
            "final_train_loss": _round_value(loss[-1][1] if loss else None),
            "best_train_loss": _round_value(min((value for _, value in loss), default=None)),
            "final_gradient_drift": _round_value(drift[-1][1] if drift else None),
            "best_detection_accuracy": _round_value(max((value for _, value in detection), default=None)),
        }
        if len(accuracy) >= 2:
            statistics["accuracy_change_first_to_last"] = _round_value(accuracy[-1][1] - accuracy[0][1])
        if len(loss) >= 2:
            statistics["loss_change_first_to_last"] = _round_value(loss[-1][1] - loss[0][1])

        observed_outcome = [
            f"本次实验在 {dataset} 上使用 {algorithm} 完成了 {completed_rounds} 个已记录轮次。",
        ]
        if statistics["final_test_accuracy"] is not None:
            observed_outcome.append(
                f"最终 test_accuracy 为 {statistics['final_test_accuracy']:.4f}，本次运行最高为 {statistics['best_test_accuracy']:.4f}。"
            )
        if statistics["final_train_loss"] is not None:
            observed_outcome.append(f"最终 train_loss 为 {statistics['final_train_loss']:.4f}。")
        if statistics["final_gradient_drift"] is not None:
            observed_outcome.append(f"最后记录的 gradient_drift 为 {statistics['final_gradient_drift']:.4f}。")
        observed_outcome.append(f"数据来源为 {data_source}，训练模式为 {training_mode}。")

        supported_claims = [
            f"在当前 {dataset}、数据划分、模型、客户端规模、参与率、seed 和训练轮数下，可以描述 {algorithm} 的本次运行表现。",
        ]
        unsupported_claims = [
            (
                f"单独一次 {algorithm} 运行不能证明它优于 FedAvg 或任何其他算法。"
                if algorithm != "fedavg"
                else "单独一次 FedAvg 运行不能证明它普遍优于其他算法。"
            ),
            "当前分析不执行历史实验比较，也不能据此推出跨数据集、跨 seed 的普遍结论。",
        ]
        limitations = [
            "当前结果没有在同一实验计划中提供保持其他变量一致的 FedAvg 对照。",
            "当前只有一个 seed，不能据此估计均值、方差或统计显著性。",
        ]
        if data_source != "real":
            limitations.append("当前不是原始真实数据运行，结果只适合作为系统链路或合成场景证据。")
        if operations.get("status") == "warning":
            limitations.append("运行保障智能体发现完整性提醒，结论应结合其证据复核。")

        anomalies = []
        for observation in security_observations:
            if observation.get("severity") in {"warning", "high_risk", "critical"}:
                anomalies.extend(observation.get("signal", {}).get("findings", []))
        anomalies = anomalies[:80]

        next_experiments = []
        if algorithm != "fedavg":
            next_experiments.append(
                "补充 FedAvg 对照：固定 dataset、split、network、num_clients、participation_rate、rounds、local_epochs、batch_size、learning_rate 和 seed，只替换 algorithm。"
            )
        next_experiments.extend(
            [
                "使用至少 3 个不同 seed 重复当前配置，报告均值、标准差和每个 seed 的完整曲线。",
                "若要支持鲁棒性结论，补充明确的攻击/防御实验，并同时报告 clean accuracy、检测指标和误报/漏报证据。",
            ]
        )
        if data_source != "real":
            next_experiments.append("在真实本地数据可用后复现实验，并单独标记真实数据结果。")

        evidence_refs: list[dict] = []
        for metric_name in ("test_accuracy", "train_loss", "gradient_drift", "detection_accuracy"):
            points = _values(metrics, metric_name)
            if points:
                evidence_refs.append({"type": "metric", "name": metric_name, "round": points[-1][0]})
        evidence_refs.extend(
            {"type": "event", "event_id": event.get("id"), "event_type": event.get("event_type")}
            for event in round_complete[-3:]
        )
        evidence_refs.append({"type": "manifest", "fingerprint": manifest.get("reproduction_fingerprint")})
        evidence_refs = [ref for ref in evidence_refs if ref.get("event_id") or ref.get("fingerprint") or ref.get("name")]

        deterministic_analysis = {
            "observed_outcome": observed_outcome,
            "supported_claims": supported_claims,
            "unsupported_claims": unsupported_claims,
            "limitations": limitations,
            "anomalies": anomalies,
            "claim_level": "descriptive_only",
            "statistics": statistics,
            "next_experiments": next_experiments,
            "evidence_refs": evidence_refs,
            "data_source": data_source,
            "training_mode": training_mode,
        }
        llm = LLMRuntime(self.name, profile="background_agent")
        llm_interpretation = await llm.complete_json(
            system_prompt=(
                "你是结果分析智能体。只能解释给定的确定性统计事实，不能补造基线、历史实验或统计显著性。"
                "不要把一次实验写成算法优越性结论。严格输出 JSON。"
            ),
            input_payload=deterministic_analysis,
            output_schema={"interpretation": "string", "caveats": ["string"]},
            # 本地 Qwen3 会先输出 <think> 思考块，预算不足时正文为空。
            # 用足 agent_llm_max_tokens（默认 1600）配置预算。
            max_tokens=1600,
        )
        deterministic_analysis["llm_interpretation"] = llm_interpretation
        deterministic_analysis["llm_metadata"] = llm.last_metadata
        return {
            "agent_name": self.name,
            "stage": "result_analysis",
            "status": "completed",
            "claim_level": "descriptive_only",
            "analysis": deterministic_analysis,
            "evidence_refs": evidence_refs,
            "tool_calls": tools.calls,
            "llm_metadata": llm.last_metadata,
        }
