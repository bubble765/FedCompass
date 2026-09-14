import unittest

from app.schemas.schemas import ExperimentConfig
from app.services.services import get_experiment_config_options, resolve_experiment_config


class ExperimentConfigConstraintTest(unittest.TestCase):
    def test_multimodal_3d_options_follow_fedulip_paper_scope(self):
        options = get_experiment_config_options("multimodal_3d")
        self.assertEqual(options["task_types"], ["ulip3d"])
        self.assertEqual(options["splits"], ["iid"])
        self.assertEqual(options["protocols"], ["base2new", "cross_dataset", "domain_abcd"])
        self.assertEqual(
            options["datasets"],
            ["modelnet40", "scanobjectnn", "shapenetcore", "mvtec3d", "mnist3d", "3dimage"],
        )
        self.assertNotIn("cifar10", options["datasets"])

    def test_vision_benchmark_is_visual_only(self):
        options = get_experiment_config_options("vision_benchmark")
        self.assertEqual(options["task_types"], ["vision_classification"])
        self.assertNotIn("text_classification", options["task_types"])
        self.assertNotIn("ag_news", options["datasets"])
        self.assertIn("tiny_imagenet", options["datasets"])

    def test_reid_options_follow_co_evo_datasets(self):
        options = get_experiment_config_options("reid_generalization")
        self.assertEqual(options["task_types"], ["reid"])
        self.assertEqual(options["splits"], ["domain_as_client"])
        self.assertEqual(options["datasets"], ["cuhk02", "cuhk03", "msmt17", "market1501"])
        self.assertNotIn("dukemtmc", options["datasets"])

    def test_resolve_config_coerces_to_selected_module_constraints(self):
        resolved = resolve_experiment_config(
            ExperimentConfig(
                module_id="multimodal_3d",
                task_type="vision_classification",
                dataset="cifar10",
                split="dirichlet",
                network="resnet18",
                algorithm="fedavg",
            )
        )
        self.assertEqual(resolved["task_type"], "ulip3d")
        self.assertEqual(resolved["dataset"], "modelnet40")
        self.assertEqual(resolved["split"], "iid")
        self.assertEqual(resolved["network"], "pointbert")
        self.assertEqual(resolved["algorithm"], "fedavg")
        self.assertIn(resolved["training_mode"], {"real_federated", "pytorch_federated"})


if __name__ == "__main__":
    unittest.main()
