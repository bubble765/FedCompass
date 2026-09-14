import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.plugins.base import ClientState
from app.plugins.registry import (
    ALGORITHM_REGISTRY,
    ATTACK_REGISTRY,
    DEFENSE_REGISTRY,
    get_algorithm,
)
from app.seed.data import SEED_DATA


class AlgorithmSmokeTest(unittest.TestCase):
    def setUp(self):
        self.global_params = np.linspace(-0.1, 0.1, 48, dtype=np.float32)
        self.config = {
            "learning_rate": 0.01,
            "local_epochs": 2,
            "participation_rate": 0.5,
            "distill_weight": 0.4,
            "rho": 0.05,
        }

    def test_seed_algorithms_are_registered(self):
        seed_ids = {item["id"] for item in SEED_DATA["algorithms"]}
        self.assertEqual(sorted(seed_ids - set(ALGORITHM_REGISTRY)), [])

    def test_seed_algorithms_are_marked_smoke_validated(self):
        unvalidated = [
            item["id"]
            for item in SEED_DATA["algorithms"]
            if getattr(item["status"], "value", item["status"]) != "validated"
        ]
        self.assertEqual(unvalidated, [])

    def test_all_registered_algorithms_run_one_round(self):
        for algorithm_id in sorted(ALGORITHM_REGISTRY):
            with self.subTest(algorithm=algorithm_id):
                aggregate_fn, local_train_fn = get_algorithm(algorithm_id)
                client_params = []
                for client_id in range(1, 5):
                    state = ClientState(client_id=client_id, model_params=self.global_params.copy())
                    params, meta = local_train_fn(state, self.global_params.copy(), 1, dict(self.config))
                    self.assertEqual(params.shape, self.global_params.shape)
                    self.assertTrue(np.isfinite(params).all())
                    self.assertTrue(math.isfinite(float(meta.get("train_loss", 1.0))))
                    client_params.append(params)
                aggregated, agg_meta = aggregate_fn(client_params, np.ones(len(client_params), dtype=np.float32))
                self.assertEqual(aggregated.shape, self.global_params.shape)
                self.assertTrue(np.isfinite(aggregated).all())
                self.assertIn("method", agg_meta)

    def test_all_defenses_run_filter_and_aggregate(self):
        client_params = [
            self.global_params + np.full_like(self.global_params, 0.01 * idx)
            for idx in range(4)
        ]
        client_updates = [(idx + 1, params) for idx, params in enumerate(client_params)]
        for defense_id, (aggregate_fn, filter_fn) in sorted(DEFENSE_REGISTRY.items()):
            with self.subTest(defense=defense_id):
                aggregated, meta = aggregate_fn(client_params, np.ones(len(client_params), dtype=np.float32))
                self.assertEqual(aggregated.shape, self.global_params.shape)
                self.assertTrue(np.isfinite(aggregated).all())
                self.assertIn("method", meta)
                if filter_fn is not None:
                    trusted, filtered, filter_meta = filter_fn(client_updates, self.global_params, 1)
                    self.assertGreaterEqual(len(trusted), 1)
                    self.assertEqual(set(trusted).isdisjoint(set(filtered)), True)
                    self.assertIn("method", filter_meta)

    def test_all_attacks_are_bounded_and_finite(self):
        params = self.global_params + 0.01
        state = ClientState(client_id=1, model_params=self.global_params.copy())
        for attack_id, attack_fn in sorted(ATTACK_REGISTRY.items()):
            with self.subTest(attack=attack_id):
                attacked = attack_fn(state, params.copy(), self.global_params.copy(), 1)
                self.assertEqual(attacked.shape, self.global_params.shape)
                self.assertTrue(np.isfinite(attacked).all())


if __name__ == "__main__":
    unittest.main()
