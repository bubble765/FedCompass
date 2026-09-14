import asyncio
import unittest
from unittest.mock import patch

from app.agent.config_assistant import ConfigAssistantError, parse_config_request
from app.schemas.schemas import ConfigAssistantLLMConfig, ConfigAssistantRequest


class ConfigAssistantTest(unittest.TestCase):
    def run_request(self, message, current_config=None):
        request = ConfigAssistantRequest(
            message=message,
            current_config=current_config
            or {
                "module_id": "vision_benchmark",
                "task_type": "vision_classification",
                "dataset": "cifar10",
                "split": "iid",
                "network": "resnet18",
                "algorithm": "fedavg",
                "optimizer": "sgd",
                "defense": "none",
                "attack": "none",
                "num_clients": 30,
                "participation_rate": 0.1,
                "rounds": 10,
            },
        )
        with patch("app.agent.config_assistant.settings.llm_api_key", ""), patch(
            "app.agent.config_assistant.settings.llm_base_url", ""
        ):
            return asyncio.run(parse_config_request(request))

    def test_local_parser_extracts_common_chinese_experiment_fields(self):
        result = self.run_request(
            "在 CIFAR-10 上使用 FedCVC 做 30 个客户端的非 IID 实验，每轮参与率 20%，训练 10 轮，使用 SAM，不启用攻击和防御。"
        )

        self.assertEqual(result.provider, "local_fallback")
        self.assertEqual(result.config["module_id"], "participation_adaptation")
        self.assertEqual(result.config["dataset"], "cifar10")
        self.assertEqual(result.config["algorithm"], "fedcvc")
        self.assertEqual(result.config["split"], "dirichlet")
        self.assertEqual(result.config["optimizer"], "sgd")
        self.assertEqual(result.config["participation_rate"], 0.2)
        self.assertTrue(result.warnings)
        self.assertTrue(any("优化器" in warning for warning in result.warnings))
        self.assertIn("network", result.missing_fields)
        self.assertIn("local_epochs", result.missing_fields)
        self.assertTrue(any("未在本次输入中明确" in warning for warning in result.warnings))

    def test_local_parser_selects_reid_module_and_domain_split(self):
        result = self.run_request(
            "在 Market-1501 上做 16 客户端的 ReID 域泛化实验，使用 CO-EVO，按域划分，训练 10 轮。"
        )

        self.assertEqual(result.config["module_id"], "reid_generalization")
        self.assertEqual(result.config["task_type"], "reid")
        self.assertEqual(result.config["dataset"], "market1501")
        self.assertEqual(result.config["algorithm"], "co_evo")
        self.assertEqual(result.config["split"], "domain_as_client")
        self.assertEqual(result.config["network"], "tiny_reid")

    def test_high_level_reid_scenario_gets_a_complete_recommendation(self):
        result = self.run_request(
            "我想跑一个 ReID 的实验，用 Market1501 数据集，留一域协议。"
        )

        self.assertEqual(result.mode, "scenario_recommendation")
        self.assertEqual(result.missing_fields, [])
        self.assertEqual(result.warnings, [])
        self.assertIsNone(result.follow_up_question)
        self.assertEqual(result.config["module_id"], "reid_generalization")
        self.assertEqual(result.config["task_type"], "reid")
        self.assertEqual(result.config["dataset"], "market1501")
        self.assertEqual(result.config["split"], "domain_as_client")
        self.assertEqual(result.config["protocol"], "leave_one_domain_out")
        self.assertEqual(result.config["network"], "tiny_reid")
        self.assertEqual(result.config["algorithm"], "co_evo")
        self.assertEqual(result.config["optimizer"], "sgd")
        self.assertEqual(result.config["num_clients"], 16)
        self.assertEqual(result.config["participation_rate"], 1.0)
        self.assertEqual(result.config["rounds"], 10)
        self.assertEqual(result.config["local_epochs"], 2)
        self.assertEqual(result.config["batch_size"], 8)
        self.assertEqual(result.config["learning_rate"], 0.001)
        self.assertEqual(result.config["seed"], 42)
        self.assertIn("algorithm", result.suggested_fields)
        self.assertIn("network", result.suggested_fields)
        self.assertIn("learning_rate", result.suggested_fields)
        self.assertIn("protocol", result.explicit_fields)
        self.assertNotIn("trust_threshold", result.config)
        self.assertNotIn("distill_weight", result.config)
        self.assertNotIn("attack", result.recommended_fields)
        self.assertTrue(set(result.recommended_fields).issuperset({
            "module_id", "task_type", "dataset", "split", "protocol", "network",
            "algorithm", "optimizer", "num_clients", "participation_rate", "rounds",
            "local_epochs", "batch_size", "learning_rate", "seed",
        }))

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ConfigAssistantError):
            asyncio.run(parse_config_request(ConfigAssistantRequest(message="  ")))

    def test_model_english_confirmation_text_is_normalized_to_chinese(self):
        request = ConfigAssistantRequest(
            message="只使用 CIFAR-10",
            current_config={"module_id": "vision_benchmark", "dataset": "cifar10"},
        )
        model_payload = {
            "config": {"dataset": "cifar10"},
            "explicit_fields": ["dataset"],
            "assumptions": [{"field": "network", "reason": "Use the current network."}],
            "warnings": ["Please confirm the missing parameters."],
            "follow_up_question": "Please confirm the configuration.",
        }
        with patch("app.agent.config_assistant._call_llm_sync", return_value=model_payload), patch(
            "app.agent.config_assistant.settings.llm_api_key", "configured"
        ), patch("app.agent.config_assistant.settings.llm_base_url", "https://example.test/v1"):
            result = asyncio.run(parse_config_request(request))

        self.assertTrue(all(any("\u4e00" <= char <= "\u9fff" for char in text) for text in result.warnings))
        self.assertTrue(all(any("\u4e00" <= char <= "\u9fff" for char in note.reason) for note in result.assumptions))
        self.assertTrue(any("\u4e00" <= char <= "\u9fff" for char in (result.follow_up_question or "")))

    def test_request_api_model_settings_are_forwarded_for_one_call_only(self):
        request = ConfigAssistantRequest(
            message="我想跑一个 ReID 实验，用 Market1501 数据集，留一域协议。",
            current_config={"module_id": "vision_benchmark"},
            llm=ConfigAssistantLLMConfig(
                provider="api",
                base_url="https://api.example.test/v1",
                api_key="test-secret",
                model="qwen-plus",
            ),
        )
        model_payload = {
            "config": {
                "module_id": "reid_generalization",
                "task_type": "reid",
                "dataset": "market1501",
                "split": "domain_as_client",
                "protocol": "leave_one_domain_out",
                "network": "tiny_reid",
                "algorithm": "co_evo",
                "optimizer": "sgd",
            },
            "explicit_fields": ["module_id", "task_type", "dataset", "split", "protocol"],
        }
        with patch("app.agent.config_assistant._call_llm_sync", return_value=model_payload) as call_llm:
            result = asyncio.run(parse_config_request(request))

        self.assertEqual(result.provider, "llm_api")
        self.assertEqual(call_llm.call_args.args[1]["provider"], "api")
        self.assertEqual(call_llm.call_args.args[1]["model"], "qwen-plus")
        self.assertEqual(call_llm.call_args.args[1]["api_key"], "test-secret")
        self.assertNotIn("api_key", result.model_dump())
        self.assertNotIn("test-secret", str(result.model_dump()))

    def test_api_model_requires_key_when_explicitly_selected(self):
        request = ConfigAssistantRequest(
            message="使用 CIFAR-10 做视觉分类实验。",
            llm=ConfigAssistantLLMConfig(
                provider="api",
                base_url="https://api.example.test/v1",
                model="qwen-plus",
            ),
        )
        with self.assertRaisesRegex(ConfigAssistantError, "API Key"):
            asyncio.run(parse_config_request(request))

    def test_model_cannot_treat_current_defaults_as_user_input(self):
        message = "在 3D 多模态应用模块中使用 ModelNet40，IID 划分、base-to-new 协议、PointBERT、FedULIP、SGD，10 个客户端，参与率 100%，训练 5 轮，本地 Epoch 1，批大小 16。"
        request = ConfigAssistantRequest(
            message=message,
            current_config={
                "module_id": "vision_benchmark",
                "task_type": "vision_classification",
                "dataset": "cifar10",
                "split": "iid",
                "network": "resnet18",
                "algorithm": "fedavg",
                "optimizer": "sgd",
                "defense": "none",
                "attack": "none",
                "num_clients": 30,
                "participation_rate": 0.1,
                "rounds": 10,
                "local_epochs": 5,
                "batch_size": 64,
                "learning_rate": 0.01,
                "seed": 42,
            },
        )
        model_payload = {
            "config": {
                "module_id": "multimodal_3d",
                "task_type": "ulip3d",
                "dataset": "modelnet40",
                "split": "iid",
                "protocol": "base2new",
                "network": "pointbert",
                "algorithm": "fedulip",
                "optimizer": "sgd",
                "num_clients": 10,
                "participation_rate": 1.0,
                "rounds": 5,
                "local_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "seed": 5,
            },
            "explicit_fields": [
                "module_id", "task_type", "dataset", "split", "protocol", "network",
                "algorithm", "optimizer", "num_clients", "participation_rate", "rounds",
                "local_epochs", "batch_size", "learning_rate", "seed",
            ],
        }
        with patch("app.agent.config_assistant._call_llm_sync", return_value=model_payload), patch(
            "app.agent.config_assistant.settings.llm_api_key", "configured"
        ), patch("app.agent.config_assistant.settings.llm_base_url", "https://example.test/v1"):
            result = asyncio.run(parse_config_request(request))

        self.assertIn("learning_rate", result.missing_fields)
        self.assertIn("seed", result.missing_fields)
        self.assertIn("学习率", "".join(result.warnings))
        self.assertIn("随机种子", "".join(result.warnings))

    def test_model_empty_explicit_fields_uses_local_field_detection(self):
        message = "在 3D 多模态应用模块中使用 ModelNet40，IID 划分、base-to-new 协议、PointBERT、FedULIP、SGD，10 个客户端，参与率 100%，训练 5 轮，本地 Epoch 1，批大小 16，学习率 0.001，随机种子 5。"
        request = ConfigAssistantRequest(
            message=message,
            current_config={
                "module_id": "multimodal_3d",
                "task_type": "ulip3d",
                "dataset": "modelnet40",
                "split": "iid",
                "protocol": "base2new",
                "network": "pointbert",
                "algorithm": "fedulip",
                "optimizer": "sgd",
                "num_clients": 10,
                "participation_rate": 1.0,
                "rounds": 5,
                "local_epochs": 1,
                "batch_size": 16,
                "learning_rate": 0.001,
                "seed": 5,
            },
        )
        model_payload = {
            "config": dict(request.current_config),
            "explicit_fields": [],
        }
        with patch("app.agent.config_assistant._call_llm_sync", return_value=model_payload), patch(
            "app.agent.config_assistant.settings.llm_api_key", "configured"
        ), patch("app.agent.config_assistant.settings.llm_base_url", "https://example.test/v1"):
            result = asyncio.run(parse_config_request(request))

        self.assertEqual(result.missing_fields, [])
        self.assertEqual(result.warnings, [])
        self.assertIn("learning_rate", result.explicit_fields)
        self.assertIn("seed", result.explicit_fields)

    def test_module_name_does_not_count_as_explicit_data_split(self):
        message = "在低参与异构适配模块中使用 CIFAR-10 做视觉分类，低参与协议、Small CNN、FedCVC、SGD，不启用攻击和防御，30 个客户端，参与率 20%，训练 10 轮，本地 Epoch 1，批大小 32，学习率 0.01，随机种子 42。"
        result = self.run_request(message)

        self.assertIn("split", result.missing_fields)
        self.assertNotIn("split", [change.field for change in result.changes])
        self.assertIn("数据划分", "".join(result.warnings))

    def test_every_visible_base_field_is_required_when_omitted(self):
        complete = "在低参与异构适配模块中使用 CIFAR-10 做视觉分类，Dirichlet 划分、低参与协议、Small CNN、FedCVC、SGD，30 个客户端，参与率 20%，训练 10 轮，本地 Epoch 1，批大小 32，学习率 0.01，随机种子 42。"
        field_fragments = {
            "module_id": "在低参与异构适配模块中",
            "task_type": "做视觉分类",
            "dataset": "CIFAR-10",
            "split": "Dirichlet 划分",
            "protocol": "低参与协议",
            "network": "Small CNN",
            "algorithm": "FedCVC",
            "optimizer": "SGD",
            "num_clients": "30 个客户端",
            "participation_rate": "参与率 20%",
            "rounds": "训练 10 轮",
            "local_epochs": "本地 Epoch 1",
            "batch_size": "批大小 32",
            "learning_rate": "学习率 0.01",
            "seed": "随机种子 42",
        }

        for field, fragment in field_fragments.items():
            with self.subTest(field=field):
                result = self.run_request(complete.replace(fragment, ""))
                self.assertIn(field, result.missing_fields)
                self.assertNotIn(field, [change.field for change in result.changes])

    def test_security_fields_are_required_for_defense_module(self):
        complete = "在鲁棒聚合防护模块中做攻防演示，使用 MNIST，Dirichlet 划分、模型投毒协议、Small CNN、FedAvg、SGD，防御 VERT，攻击高斯噪声，30 个客户端，参与率 50%，训练 10 轮，本地 Epoch 1，批大小 32，学习率 0.01，随机种子 42，可信筛选阈值 0.8。"
        self.assertEqual(self.run_request(complete).missing_fields, [])

        for field, fragment in {
            "defense": "防御 VERT",
            "attack": "攻击高斯噪声",
            "trust_threshold": "可信筛选阈值 0.8",
        }.items():
            with self.subTest(field=field):
                result = self.run_request(complete.replace(fragment, ""))
                self.assertIn(field, result.missing_fields)
                self.assertNotIn(field, [change.field for change in result.changes])

    def test_distillation_fields_are_required_and_parsed_independently(self):
        complete = "在蒸馏对齐增强模块中使用 CIFAR-10 做视觉分类，Dirichlet 划分、低参与蒸馏协议、Small CNN、FedCADS、SGD，30 个客户端，参与率 20%，训练 10 轮，本地 Epoch 1，批大小 32，学习率 0.01，随机种子 42，本地蒸馏权重 0.4，全局蒸馏权重 0.6，蒸馏温度 2。"
        self.assertEqual(self.run_request(complete).missing_fields, [])

        for field, fragment in {
            "distill_weight": "本地蒸馏权重 0.4",
            "global_distill_weight": "全局蒸馏权重 0.6",
            "temperature": "蒸馏温度 2",
        }.items():
            with self.subTest(field=field):
                result = self.run_request(complete.replace(fragment, ""))
                self.assertIn(field, result.missing_fields)
                self.assertNotIn(field, [change.field for change in result.changes])

    def test_builtin_complete_examples_need_no_confirmation(self):
        examples = [
            "在低参与异构适配模块中使用 CIFAR-10 做视觉分类，Dirichlet 划分、低参与协议、Small CNN、FedCVC、SGD，不启用攻击和防御，30 个客户端，参与率 20%，训练 10 轮，本地 Epoch 1，批大小 32，学习率 0.01，随机种子 42。",
            "在 ReID 域泛化应用模块中使用 Market-1501，按域划分、留一域协议、Tiny ReID、CO-EVO、SGD，16 个客户端，参与率 100%，训练 10 轮，本地 Epoch 2，批大小 8，学习率 0.001，随机种子 42。",
            "在 3D 多模态应用模块中使用 ModelNet40，IID 划分、base-to-new 协议、PointBERT、FedULIP、SGD，10 个客户端，参与率 100%，训练 5 轮，本地 Epoch 1，批大小 16，学习率 0.001，随机种子 5。",
        ]
        for example in examples:
            result = self.run_request(example)
            self.assertEqual(result.missing_fields, [])
            self.assertEqual(result.warnings, [])
            self.assertIsNone(result.follow_up_question)
            self.assertNotIn("lambda_anchor", result.config)
            self.assertNotIn("lambda_style", result.config)
            self.assertNotIn("adapter_type", result.config)


if __name__ == "__main__":
    unittest.main()
