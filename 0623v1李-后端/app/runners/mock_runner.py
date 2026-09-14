"""MockRunner: generates realistic fake metrics for demo experiments."""
import asyncio
import datetime
import math
import random
import hashlib
from app.core.database import async_session
from app.models.models import ExperimentStatus
from app.orchestrator import workflow
from app.services import services


def _seeded(experiment_id, round_num):
    """Return a deterministic random.Random for the given experiment+round."""
    seed_str = f"{experiment_id}:{round_num}"
    seed_int = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    return random.Random(seed_int)


class MockRunner:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self._stop = False

    def stop(self):
        self._stop = True

    async def run(self):
        async with async_session() as db:
            exp = await services.get_experiment(db, self.experiment_id)
            total_rounds = exp.total_rounds if exp else 100
            config = exp.config_json or {}
            num_clients = config.get("num_clients", 100)
            participation_rate = config.get("participation_rate", 0.1)

            await services.update_experiment_status(db, self.experiment_id, ExperimentStatus.running)

            for rnd in range(1, total_rounds + 1):
                if self._stop:
                    await services.update_experiment_status(db, self.experiment_id, ExperimentStatus.stopped)
                    await services.add_event(db, self.experiment_id, "stopped",
                        {"round": rnd, "message": "Experiment stopped by user"})
                    return

                progress = rnd / total_rounds
                train_loss = max(0.05, 2.0 * math.exp(-3 * progress) + 0.1 * random.uniform(-0.05, 0.05))
                test_accuracy = min(0.95, 0.1 + 0.85 * (1 - math.exp(-5 * progress)) + random.uniform(-0.02, 0.02))
                comm_cost = 142.3 + random.uniform(-5, 5)
                client_part = max(0.05, 0.1 + random.uniform(-0.02, 0.02))
                gradient_drift = max(0.001, 0.05 * math.exp(-2 * progress) + random.uniform(-0.005, 0.005))

                rng = _seeded(self.experiment_id, rnd)
                n_participating = max(1, int(num_clients * participation_rate * rng.uniform(0.8, 1.2)))
                all_ids = list(range(1, num_clients + 1))
                participating = sorted(rng.sample(all_ids, min(n_participating, num_clients)))
                remaining = [c for c in all_ids if c not in participating]
                n_malicious = rng.randint(0, max(1, int(num_clients * 0.06))) if rnd > 3 else 0
                malicious = sorted(rng.sample(remaining, min(n_malicious, len(remaining)))) if n_malicious > 0 and remaining else []
                round_events = [
                    (
                        "round_complete",
                        {
                            "round": rnd,
                            "train_loss": round(train_loss, 4),
                            "test_accuracy": round(test_accuracy, 4),
                            "timestamp": datetime.datetime.utcnow().isoformat(),
                        },
                    ),
                    (
                        "client_participation",
                        {
                            "round": rnd,
                            "participating": participating,
                            "silent": [],
                            "malicious": malicious,
                        },
                    ),
                ]
                if rnd % 10 == 0 and rnd > 5:
                    rng2 = _seeded(self.experiment_id, rnd + 10000)
                    n_trusted = rng2.randint(max(1, int(num_clients * 0.8)), num_clients)
                    n_filtered = rng2.randint(1, max(1, int(num_clients * 0.1)))
                    trusted = sorted(rng2.sample(all_ids, min(n_trusted, num_clients)))
                    rest = [c for c in all_ids if c not in trusted]
                    n_filt = min(n_filtered, len(rest))
                    filtered = sorted(rng2.sample(rest, n_filt)) if n_filt > 0 and rest else []
                    round_events.append(
                        (
                            "defense_detection",
                            {
                                "round": rnd,
                                "trusted_clients": trusted,
                                "filtered_clients": filtered,
                                "detection_accuracy": round(0.85 + 0.1 * rng2.random(), 4),
                            },
                        )
                    )

                await services.persist_round_update(
                    db,
                    self.experiment_id,
                    rnd,
                    {
                        "train_loss": round(train_loss, 4),
                        "test_accuracy": round(test_accuracy, 4),
                        "communication_cost_mb": round(comm_cost, 2),
                        "client_participation_rate": round(client_part, 4),
                        "gradient_drift": round(gradient_drift, 4),
                    },
                    round_events,
                )
                if await workflow.run_runtime_agent_checkpoint(db, self.experiment_id, rnd):
                    return
                await asyncio.sleep(0.3)

            await services.update_experiment_status(db, self.experiment_id, ExperimentStatus.completed)
            await services.add_event(db, self.experiment_id, "completed", {
                "final_accuracy": round(test_accuracy, 4),
                "final_loss": round(train_loss, 4),
                "message": "Experiment completed successfully",
            })
