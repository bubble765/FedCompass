import math
import unittest

import torch

from app.runners.real_federated_reid import (
    build_reid_model,
    evaluate_reid_model,
    local_train_reid,
    prepare_federated_reid_dataset,
    state_dict_to_cpu as reid_state_dict_to_cpu,
)
from app.runners.real_federated_ulip import (
    build_ulip_model,
    evaluate_ulip_model,
    local_train_ulip,
    prepare_federated_ulip_dataset,
    state_dict_to_cpu as ulip_state_dict_to_cpu,
)
from app.runners.real_federated_vision import (
    aggregate_state_dicts,
    build_model,
    evaluate_model,
    local_train,
    prepare_federated_dataset,
    state_dict_to_cpu,
)
from app.seed.data import SEED_DATA


class PyTorchReproductionSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_backend_synthetic_vision_round_runs_with_fedadam(self):
        config = {
            "algorithm": "fedcads",
            "optimizer": "sam",
            "network": "small_cnn",
            "batch_size": 8,
            "local_epochs": 1,
            "learning_rate": 0.01,
            "train_sample_cap_per_client": 12,
            "eval_sample_cap": 24,
            "device": "cpu",
            "seed": 7,
            "temperature": 2.0,
            "distill_weight": 0.2,
        }
        prepared = prepare_federated_dataset(
            "cifar10",
            "dirichlet",
            num_clients=3,
            seed=7,
            data_source="backend_synthetic",
            train_samples=96,
            test_samples=36,
        )
        self.assertEqual(prepared.data_source, "backend_synthetic")
        global_state = state_dict_to_cpu(build_model("small_cnn", prepared.num_classes).state_dict())
        client_states = []
        weights = []
        for client_id in [1, 2]:
            state, meta = local_train(
                global_state,
                prepared.train_dataset,
                prepared.client_indices[client_id - 1],
                config,
                round_num=1,
                client_id=client_id,
                num_classes=prepared.num_classes,
            )
            self.assertGreater(meta["num_samples"], 0)
            self.assertTrue(math.isfinite(meta["train_loss"]))
            client_states.append(state)
            weights.append(meta["num_samples"])

        aggregated = aggregate_state_dicts(
            client_states,
            weights,
            algorithm="fedadam",
            server_state={},
            global_state=global_state,
            round_num=1,
        )
        metrics = evaluate_model(aggregated, prepared.test_dataset, config, prepared.num_classes)
        self.assertGreaterEqual(metrics["test_accuracy"], 0.0)
        self.assertLessEqual(metrics["test_accuracy"], 1.0)
        self.assertTrue(math.isfinite(metrics["eval_loss"]))

    def test_backend_synthetic_reid_round_runs(self):
        config = {
            "network": "tiny_reid",
            "batch_size": 4,
            "local_epochs": 1,
            "learning_rate": 0.001,
            "train_sample_cap_per_client": 8,
            "eval_sample_cap": 12,
            "embedding_dim": 64,
            "device": "cpu",
            "seed": 11,
        }
        prepared = prepare_federated_reid_dataset(
            "market1501",
            "domain_as_client",
            num_clients=3,
            seed=11,
            data_source="backend_synthetic",
        )
        self.assertEqual(prepared.data_source, "backend_synthetic")
        global_state = reid_state_dict_to_cpu(build_reid_model("tiny_reid", prepared.num_classes, 64).state_dict())
        state, meta = local_train_reid(
            global_state,
            prepared.train_dataset,
            prepared.client_indices[0],
            config,
            round_num=1,
            client_id=1,
            num_classes=prepared.num_classes,
        )
        self.assertGreater(meta["num_samples"], 0)
        metrics = evaluate_reid_model(state, prepared.query_dataset, prepared.gallery_dataset, config, prepared.num_classes)
        self.assertGreaterEqual(metrics["rank1"], 0.0)
        self.assertLessEqual(metrics["rank1"], 1.0)
        self.assertGreaterEqual(metrics["map"], 0.0)
        self.assertLessEqual(metrics["map"], 1.0)

    def test_backend_synthetic_ulip_round_runs(self):
        config = {
            "network": "pointbert",
            "batch_size": 8,
            "local_epochs": 1,
            "learning_rate": 0.001,
            "train_sample_cap_per_client": 16,
            "eval_sample_cap": 32,
            "embedding_dim": 64,
            "adapter_weight": 0.1,
            "device": "cpu",
            "seed": 13,
        }
        prepared = prepare_federated_ulip_dataset(
            "modelnet40",
            "iid",
            num_clients=3,
            seed=13,
            data_source="backend_synthetic",
        )
        self.assertEqual(prepared.data_source, "backend_synthetic")
        global_state = ulip_state_dict_to_cpu(build_ulip_model("pointbert", prepared.num_classes, 64).state_dict())
        state, meta = local_train_ulip(
            global_state,
            prepared.train_dataset,
            prepared.client_indices[0],
            config,
            round_num=1,
            client_id=1,
            num_classes=prepared.num_classes,
        )
        self.assertGreater(meta["num_samples"], 0)
        metrics = evaluate_ulip_model(state, prepared.test_dataset, config, prepared.num_classes)
        self.assertGreaterEqual(metrics["test_accuracy"], 0.0)
        self.assertLessEqual(metrics["test_accuracy"], 1.0)
        self.assertTrue(math.isfinite(metrics["modality_alignment"]))

    def test_seed_task_algorithms_run_on_pytorch_backends(self):
        prepared_vision = prepare_federated_dataset(
            "cifar10",
            "dirichlet",
            num_clients=2,
            seed=23,
            data_source="backend_synthetic",
            train_samples=48,
            test_samples=16,
        )
        vision_global = state_dict_to_cpu(build_model("small_cnn", prepared_vision.num_classes).state_dict())
        prepared_reid = prepare_federated_reid_dataset(
            "market1501",
            "domain_as_client",
            num_clients=2,
            seed=29,
            data_source="backend_synthetic",
        )
        reid_global = reid_state_dict_to_cpu(build_reid_model("tiny_reid", prepared_reid.num_classes, 32).state_dict())
        prepared_ulip = prepare_federated_ulip_dataset(
            "modelnet40",
            "iid",
            num_clients=2,
            seed=31,
            data_source="backend_synthetic",
        )
        ulip_global = ulip_state_dict_to_cpu(build_ulip_model("pointbert", prepared_ulip.num_classes, 32).state_dict())

        for algorithm in SEED_DATA["algorithms"]:
            if algorithm["category"] in {"attack", "defense"}:
                continue
            algorithm_id = algorithm["id"]
            task_types = algorithm.get("task_types") or []
            task_type = (
                "vision_classification"
                if "vision_classification" in task_types
                else "reid"
                if "reid" in task_types
                else "ulip3d"
                if "ulip3d" in task_types
                else None
            )
            if task_type is None:
                continue

            with self.subTest(algorithm=algorithm_id, task_type=task_type):
                if task_type == "vision_classification":
                    optimizer = algorithm_id if algorithm["category"] == "optimizer" else "sgd"
                    train_algorithm = "fedavg" if algorithm["category"] == "optimizer" else algorithm_id
                    config = {
                        "algorithm": train_algorithm,
                        "optimizer": optimizer,
                        "network": "small_cnn",
                        "batch_size": 4,
                        "local_epochs": 1,
                        "learning_rate": 0.005,
                        "train_sample_cap_per_client": 4,
                        "eval_sample_cap": 12,
                        "device": "cpu",
                        "seed": 23,
                    }
                    local_state, meta = local_train(
                        vision_global,
                        prepared_vision.train_dataset,
                        prepared_vision.client_indices[0],
                        config,
                        round_num=1,
                        client_id=1,
                        num_classes=prepared_vision.num_classes,
                    )
                    self.assertGreater(meta["num_samples"], 0)
                    aggregated = aggregate_state_dicts(
                        [local_state],
                        [meta["num_samples"]],
                        algorithm=train_algorithm,
                        server_state={"server_lr": 0.05, "beta1": 0.9, "beta2": 0.99},
                        global_state=vision_global,
                        round_num=1,
                    )
                    metrics = evaluate_model(aggregated, prepared_vision.test_dataset, config, prepared_vision.num_classes)
                    self.assertTrue(math.isfinite(metrics["eval_loss"]))
                elif task_type == "reid":
                    config = {
                        "algorithm": algorithm_id,
                        "network": "tiny_reid",
                        "batch_size": 4,
                        "local_epochs": 1,
                        "learning_rate": 0.001,
                        "train_sample_cap_per_client": 4,
                        "eval_sample_cap": 8,
                        "embedding_dim": 32,
                        "device": "cpu",
                        "seed": 29,
                    }
                    local_state, meta = local_train_reid(
                        reid_global,
                        prepared_reid.train_dataset,
                        prepared_reid.client_indices[0],
                        config,
                        round_num=1,
                        client_id=1,
                        num_classes=prepared_reid.num_classes,
                    )
                    self.assertGreater(meta["num_samples"], 0)
                    metrics = evaluate_reid_model(local_state, prepared_reid.query_dataset, prepared_reid.gallery_dataset, config, prepared_reid.num_classes)
                    self.assertTrue(math.isfinite(metrics["eval_loss"]))
                else:
                    config = {
                        "algorithm": algorithm_id,
                        "network": "pointbert",
                        "batch_size": 4,
                        "local_epochs": 1,
                        "learning_rate": 0.001,
                        "train_sample_cap_per_client": 8,
                        "eval_sample_cap": 16,
                        "embedding_dim": 32,
                        "adapter_weight": 0.1,
                        "device": "cpu",
                        "seed": 31,
                    }
                    local_state, meta = local_train_ulip(
                        ulip_global,
                        prepared_ulip.train_dataset,
                        prepared_ulip.client_indices[0],
                        config,
                        round_num=1,
                        client_id=1,
                        num_classes=prepared_ulip.num_classes,
                    )
                    self.assertGreater(meta["num_samples"], 0)
                    metrics = evaluate_ulip_model(local_state, prepared_ulip.test_dataset, config, prepared_ulip.num_classes)
                    self.assertTrue(math.isfinite(metrics["eval_loss"]))


if __name__ == "__main__":
    unittest.main()
