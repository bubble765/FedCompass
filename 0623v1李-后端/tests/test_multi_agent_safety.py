import asyncio
import math
import unittest

from app.agents.llm_runtime import LLMRuntime
from app.agents.inspection_protocol import (
    OPERATIONS_SKILL,
    SECURITY_SKILL,
    build_inspection_snapshot,
    build_security_evidence_explanations,
    model_unavailable_review,
    normalize_operations_review,
    normalize_security_review,
    operations_skill_for_stage,
    security_skill_for_stage,
    validate_llm_review,
)
from app.agents.security_monitor_agent import AGENT_TOOL_BOUNDARIES, SECURITY_TOOLS
from app.agents.security_signals import build_round_snapshot
from app.orchestrator.workflow import _runtime_control_action
from app.orchestrator.tools import ToolRegistry


class MultiAgentSafetyTest(unittest.TestCase):
    def test_round_snapshot_contains_evidence_but_no_rule_based_verdict(self):
        snapshot = build_round_snapshot(
            {"num_clients": 2, "participation_rate": 1.0, "defense": "vert"},
            4,
            {"test_accuracy": math.nan, "gradient_drift": 20.0},
            [("round_complete", {"round": 4, "message": "untrusted text"})],
        )

        self.assertEqual(snapshot["schema_version"], "fl_round_snapshot.v1")
        self.assertEqual(snapshot["round"], 4)
        self.assertIn("observed_metrics", snapshot)
        self.assertNotIn("status", snapshot)
        self.assertNotIn("recommended_action", snapshot)
        self.assertNotIn("findings", snapshot)

    def test_llm_review_status_is_taken_from_validated_model_output(self):
        snapshot = build_inspection_snapshot(
            agent_name="运行保障智能体",
            stage="runtime",
            experiment_id="exp_test",
            skill=OPERATIONS_SKILL,
            evidence={"experiment": {"config": {"rounds": 3}}},
        )
        review = validate_llm_review(
            {
                "overall_status": "warning",
                "recommended_action": "pause_review",
                "summary": "模型发现需要人工复核的运行迹象。",
                "confidence": 0.8,
                "findings": [
                    {
                        "check_id": "execution_config",
                        "scope": "operations",
                        "status": "warning",
                        "severity": "warning",
                        "message": "需要复核配置与当前运行状态。",
                        "evidence_paths": ["evidence.experiment.config"],
                        "confidence": 0.8,
                    }
                ],
                "missing_evidence": [],
            },
            skill=OPERATIONS_SKILL,
            status_values={"passed", "warning", "blocked"},
            severity_values={"normal", "warning", "blocking"},
        )

        self.assertIsNotNone(review)
        self.assertEqual(review["overall_status"], "warning")
        self.assertEqual(review["recommended_action"], "pause_review")
        self.assertEqual(review["findings"][0]["evidence_paths"], ["evidence.experiment.config"])
        self.assertEqual(snapshot["inspection_skill"], "operations_inspection")

    def test_model_unavailable_fails_closed_instead_of_claiming_passed(self):
        review = model_unavailable_review("security", "runtime")
        self.assertEqual(review["overall_status"], "critical")
        self.assertEqual(review["recommended_action"], "stop_and_review")

    def test_security_missing_evidence_cannot_remain_normal_continue(self):
        review = validate_llm_review(
            {
                "overall_status": "normal",
                "recommended_action": "continue",
                "summary": "All checks passed, but evidence is missing.",
                "confidence": 0.6,
                "checked_checks": ["agent_tool_authorization"],
                "findings": [
                    {
                        "check_id": "agent_tool_authorization",
                        "scope": "agent",
                        "status": "not_observed",
                        "severity": "warning",
                        "message": "Evidence is insufficient.",
                        "evidence_paths": ["evidence.agent_trace"],
                        "confidence": 0.2,
                    }
                ],
                "missing_evidence": ["evidence.agent_trace"],
            },
            skill=SECURITY_SKILL,
            status_values={"normal", "warning", "high_risk", "critical"},
            severity_values={"normal", "warning", "high_risk", "critical"},
            scope_values={"agent", "fl", "shared"},
        )

        self.assertIsNotNone(review)
        normalized = normalize_security_review(review)
        self.assertEqual(normalized["overall_status"], "warning")
        self.assertEqual(normalized["recommended_action"], "pause_review")
        self.assertIn("证据不足", normalized["summary"])
        self.assertIn("证据不足", normalized["findings"][0]["message"])

    def test_security_evidence_gap_explains_missing_input_and_consequence(self):
        snapshot = build_inspection_snapshot(
            agent_name="安全监控智能体",
            stage="runtime",
            experiment_id="exp_test",
            skill=SECURITY_SKILL,
            evidence={
                "agent_trace": [],
                "allowed_tool_boundaries": {"安全监控智能体": ["agent_trace.read"]},
            },
        )
        review = {
            "overall_status": "warning",
            "recommended_action": "pause_review",
            "missing_evidence": ["evidence.agent_trace"],
            "findings": [],
        }

        details = build_security_evidence_explanations(review, snapshot)

        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["evidence_name"], "智能体调用轨迹")
        self.assertEqual(details[0]["observed_state"], "已定义但内容为空")
        self.assertIn("工具越权", details[0]["possible_consequence"])
        self.assertIn("不代表已经确认发生攻击", details[0]["interpretation"])

    def test_model_unavailable_explains_inspection_result_gap(self):
        snapshot = build_inspection_snapshot(
            agent_name="安全监控智能体",
            stage="runtime",
            experiment_id="exp_test",
            skill=SECURITY_SKILL,
            evidence={},
        )
        details = build_security_evidence_explanations(
            model_unavailable_review("security", "runtime"),
            snapshot,
        )

        self.assertEqual(len(details), 1)
        self.assertEqual(details[0]["evidence_name"], "本轮安全监控模型检查结果")
        self.assertIn("Qwen JSON", details[0]["what_is_missing"])
        self.assertIn("停止并等待复核", details[0]["interpretation"])
        self.assertIn("不代表已经确认发生攻击", details[0]["interpretation"])

    def test_operations_checklist_matches_the_inspection_stage(self):
        preflight_ids = {item["check_id"] for item in operations_skill_for_stage("preflight")["checks"]}
        runtime_ids = {item["check_id"] for item in operations_skill_for_stage("runtime")["checks"]}

        self.assertIn("plugin_resolution", preflight_ids)
        self.assertNotIn("metric_completeness", preflight_ids)
        self.assertNotIn("round_completeness", preflight_ids)
        self.assertNotIn("metric_integrity", preflight_ids)
        self.assertNotIn("client_participation", preflight_ids)
        self.assertNotIn("update_and_gradient_drift", preflight_ids)
        self.assertNotIn("configuration_integrity", preflight_ids)
        self.assertIn("metric_completeness", runtime_ids)
        self.assertIn("resource_continuity", runtime_ids)
        self.assertNotIn("plugin_resolution", runtime_ids)
        # FL 运行时观测检查项已由安全监控移交运行保障。
        self.assertIn("metric_integrity", runtime_ids)
        self.assertIn("client_participation", runtime_ids)
        self.assertIn("update_and_gradient_drift", runtime_ids)
        self.assertIn("configuration_integrity", runtime_ids)

    def test_security_checklist_keeps_agent_scope_and_drops_fl_observations(self):
        # 安全监控只保留智能体自身安全检查，FL 观测全部移交运行保障。
        preflight_ids = {item["check_id"] for item in security_skill_for_stage("preflight_baseline")["checks"]}
        runtime_ids = {
            item["check_id"]
            for item in security_skill_for_stage(
                "runtime",
                {"attack": "model_poisoning", "defense": "none"},
            )["checks"]
        }

        self.assertIn("agent_tool_authorization", preflight_ids)
        self.assertIn("agent_task_integrity", preflight_ids)
        self.assertIn("credential_and_secret_exposure", preflight_ids)
        self.assertIn("indirect_prompt_injection", preflight_ids)
        self.assertNotIn("configuration_integrity", preflight_ids)
        self.assertNotIn("metric_integrity", preflight_ids)
        self.assertNotIn("client_participation", preflight_ids)
        self.assertNotIn("update_and_gradient_drift", preflight_ids)
        self.assertNotIn("defense_observability", runtime_ids)

    def test_operations_runtime_adds_defense_observability_only_with_attack_or_defense(self):
        runtime_without = {
            item["check_id"]
            for item in operations_skill_for_stage("runtime", {"attack": "none", "defense": "none"})["checks"]
        }
        runtime_with_attack = {
            item["check_id"]
            for item in operations_skill_for_stage(
                "runtime",
                {"attack": "model_poisoning", "defense": "none"},
            )["checks"]
        }
        preflight_with_attack = {
            item["check_id"]
            for item in operations_skill_for_stage(
                "preflight",
                {"attack": "model_poisoning", "defense": "none"},
            )["checks"]
        }

        self.assertNotIn("defense_observability", runtime_without)
        self.assertIn("defense_observability", runtime_with_attack)
        # 预检阶段没有 FL 证据，不得追加防御可观测性检查。
        self.assertNotIn("defense_observability", preflight_with_attack)

    def test_operations_missing_evidence_cannot_remain_passed_continue(self):
        review = validate_llm_review(
            {
                "overall_status": "passed",
                "recommended_action": "continue",
                "summary": "All runtime checks passed, but evidence is missing.",
                "confidence": 0.6,
                "checked_checks": ["metric_integrity"],
                "findings": [
                    {
                        "check_id": "metric_integrity",
                        "scope": "operations",
                        "status": "not_observed",
                        "severity": "warning",
                        "message": "Evidence is insufficient.",
                        "evidence_paths": ["evidence.metrics"],
                        "confidence": 0.2,
                    }
                ],
                "missing_evidence": ["evidence.metrics"],
            },
            skill=operations_skill_for_stage("runtime", {"defense": "vert", "attack": "none"}),
            status_values={"passed", "warning", "blocked"},
            severity_values={"normal", "warning", "blocking"},
            scope_values={"operations"},
        )

        self.assertIsNotNone(review)
        normalized = normalize_operations_review(review)
        self.assertEqual(normalized["overall_status"], "warning")
        self.assertEqual(normalized["recommended_action"], "pause_review")
        self.assertIn("证据不足", normalized["summary"])

    def test_invalid_english_natural_language_gets_chinese_fallback(self):
        review = validate_llm_review(
            {
                "overall_status": "passed",
                "recommended_action": "continue",
                "summary": "All required checks passed.",
                "confidence": 0.9,
                "checked_checks": ["execution_config"],
                "findings": [
                    {
                        "check_id": "execution_config",
                        "scope": "operations",
                        "status": "warning",
                        "severity": "warning",
                        "message": "Review this configuration.",
                        "evidence_paths": ["evidence.experiment.config"],
                        "confidence": 0.5,
                    }
                ],
                "missing_evidence": [],
            },
            skill=OPERATIONS_SKILL,
            status_values={"passed", "warning", "blocked"},
            severity_values={"normal", "warning", "blocking"},
        )

        self.assertIsNotNone(review)
        self.assertIn("检查", review["summary"])
        self.assertIn("运行保障", review["findings"][0]["message"])

    def test_tool_registry_rejects_unlisted_execution_capability(self):
        async def run():
            registry = ToolRegistry(None, {"experiment.read_config"})
            with self.assertRaises(PermissionError):
                await registry.call("shell.execute", "exp_test")

        asyncio.run(run())

    def test_llm_runtime_falls_back_without_configured_endpoint(self):
        async def run():
            runtime = LLMRuntime("结果分析智能体")
            output = await runtime.complete_json(
                system_prompt="只输出 JSON",
                input_payload={"api_key": "should-not-be-stored", "value": 1},
                output_schema={"interpretation": "string"},
            )
            self.assertEqual(output, {})
            self.assertFalse(runtime.last_metadata["used_llm"])
            self.assertNotIn("should-not-be-stored", str(runtime.last_metadata))

        asyncio.run(run())

    def test_background_agents_use_local_qwen_profile(self):
        runtime = LLMRuntime("运行保障智能体", profile="background_agent")
        provider, endpoint, model, headers = runtime._connection()
        self.assertEqual(provider, "local")
        self.assertIn("8001", endpoint)
        self.assertEqual(model, "Qwen/Qwen3-0.6B")
        self.assertNotIn("Authorization", headers)

    def test_security_allowlist_is_agent_specific(self):
        self.assertIn("agent_trace.read", SECURITY_TOOLS)
        self.assertNotIn("config.resolve", SECURITY_TOOLS)
        self.assertNotIn("shell.execute", SECURITY_TOOLS)

    def test_security_snapshot_has_boundaries_for_each_workflow_agent(self):
        self.assertIn("dataset.inspect", AGENT_TOOL_BOUNDARIES["运行保障智能体"])
        self.assertIn("security.observations.read", AGENT_TOOL_BOUNDARIES["结果分析智能体"])
        # 运行保障智能体检查 FL 轮次快照，读取安全观测记录必须在其边界内。
        self.assertIn("security.observations.read", AGENT_TOOL_BOUNDARIES["运行保障智能体"])
        self.assertNotIn("shell.execute", str(AGENT_TOOL_BOUNDARIES))

    def test_runtime_guard_stops_only_on_critical_observation(self):
        self.assertEqual(
            _runtime_control_action(
                {"status": "passed", "recommended_action": "continue"},
                {"status": "normal", "recommended_action": "continue"},
            ),
            "continue",
        )
        self.assertEqual(
            _runtime_control_action(
                {"status": "warning", "recommended_action": "pause_review"},
                {"status": "normal", "recommended_action": "continue"},
            ),
            "pause_review",
        )
        self.assertEqual(
            _runtime_control_action(
                {"status": "passed", "recommended_action": "continue"},
                {"status": "critical", "recommended_action": "stop_and_review"},
            ),
            "stop_and_review",
        )


if __name__ == "__main__":
    unittest.main()
