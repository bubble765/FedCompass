"""FastAPI application entry point."""
import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import settings
from app.core.database import init_db, async_session
from app.seed.data import SEED_DATA
from app.models.models import (
    Module, Algorithm, ModuleAlgorithm,
    ReferenceAsset, ModuleReference, AlgorithmReference,
    Dataset, Experiment, ExperimentMetric, ExperimentEvent,
    ExperimentStatus,
)
from app.schemas.schemas import ExperimentConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    await seed_database()
    await stop_orphan_running_experiments()
    yield


async def stop_orphan_running_experiments():
    """Clear DB-level running states left behind by a previous server process."""
    from sqlalchemy import select

    async with async_session() as db:
        result = await db.execute(select(Experiment).where(Experiment.status == ExperimentStatus.running))
        orphan_experiments = list(result.scalars())
        if not orphan_experiments:
            return
        now = datetime.datetime.utcnow()
        for exp in orphan_experiments:
            exp.status = ExperimentStatus.stopped
            exp.finished_at = now
            db.add(
                ExperimentEvent(
                    experiment_id=exp.id,
                    event_type="stopped",
                    payload_json={
                        "round": exp.current_round or 0,
                        "message": "Experiment marked stopped after backend restart.",
                    },
                )
            )
        await db.commit()
        print(f"[startup] Marked {len(orphan_experiments)} orphan running experiment(s) as stopped.")


async def seed_database():
    async with async_session() as db:
        from sqlalchemy import select

        async def upsert_by_id(model_cls, payload):
            existing = await db.get(model_cls, payload["id"])
            if existing:
                for key, value in payload.items():
                    setattr(existing, key, value)
                return existing
            obj = model_cls(**payload)
            db.add(obj)
            return obj

        async def upsert_relation(model_cls, keys, role):
            stmt = select(model_cls)
            for key, value in keys.items():
                stmt = stmt.where(getattr(model_cls, key) == value)
            existing = (await db.execute(stmt)).scalar_one_or_none()
            if existing:
                existing.role = role
                return existing
            obj = model_cls(**keys, role=role)
            db.add(obj)
            return obj

        for m in SEED_DATA["modules"]:
            await upsert_by_id(Module, m)
        for a in SEED_DATA["algorithms"]:
            await upsert_by_id(Algorithm, a)
        for r in SEED_DATA["references"]:
            await upsert_by_id(ReferenceAsset, r)
        for d in SEED_DATA["datasets"]:
            await upsert_by_id(Dataset, d)
        await db.flush()

        for mid, aid, role in SEED_DATA["module_algorithms"]:
            await upsert_relation(ModuleAlgorithm, {"module_id": mid, "algorithm_id": aid}, role)
        for mid, rid, role in SEED_DATA["module_references"]:
            await upsert_relation(ModuleReference, {"module_id": mid, "reference_asset_id": rid}, role)
        for aid, rid, role in SEED_DATA["algorithm_references"]:
            await upsert_relation(AlgorithmReference, {"algorithm_id": aid, "reference_asset_id": rid}, role)

        seed_module_algorithm_keys = {(mid, aid) for mid, aid, _ in SEED_DATA["module_algorithms"]}
        seed_module_reference_keys = {(mid, rid) for mid, rid, _ in SEED_DATA["module_references"]}
        seed_algorithm_reference_keys = {(aid, rid) for aid, rid, _ in SEED_DATA["algorithm_references"]}

        for row in (await db.execute(select(ModuleAlgorithm))).scalars():
            if (row.module_id, row.algorithm_id) not in seed_module_algorithm_keys:
                await db.delete(row)
        for row in (await db.execute(select(ModuleReference))).scalars():
            if (row.module_id, row.reference_asset_id) not in seed_module_reference_keys:
                await db.delete(row)
        for row in (await db.execute(select(AlgorithmReference))).scalars():
            if (row.algorithm_id, row.reference_asset_id) not in seed_algorithm_reference_keys:
                await db.delete(row)
        await db.commit()


async def seed_demo_experiments():
    """Pre-generate 4 demo experiments with 100 rounds of mock data directly in DB."""
    import math, random, hashlib, datetime, uuid

    demo_configs = [
        {"task_type": "vision_classification", "dataset": "cifar10", "split": "dirichlet",
         "algorithm": "fedcvc", "optimizer": "sam", "defense": "vert", "attack": "featurepoison",
         "num_clients": 50, "participation_rate": 0.2, "rounds": 100, "local_epochs": 2, "batch_size": 64,
         "learning_rate": 0.01, "seed": 42, "trust_threshold": 0.8},
        {"task_type": "reid", "dataset": "market1501", "split": "domain_as_client",
         "algorithm": "co_evo", "optimizer": "sgd", "defense": "none", "attack": "none",
         "num_clients": 16, "participation_rate": 1.0, "rounds": 100, "local_epochs": 2, "batch_size": 32,
         "learning_rate": 0.001, "seed": 12, "lambda_anchor": 1.0, "lambda_style": 0.1},
        {"task_type": "vision_classification", "dataset": "mnist", "split": "iid",
         "algorithm": "feddyn", "optimizer": "sgd", "defense": "flbeeline", "attack": "gn",
         "num_clients": 100, "participation_rate": 0.15, "rounds": 100, "local_epochs": 2, "batch_size": 64,
         "learning_rate": 0.01, "seed": 99},
        {"task_type": "ulip3d", "dataset": "modelnet40", "split": "iid",
         "algorithm": "fedulip", "optimizer": "sgd", "defense": "none", "attack": "none",
         "num_clients": 10, "participation_rate": 1.0, "rounds": 100, "local_epochs": 1, "batch_size": 16,
         "learning_rate": 0.001, "seed": 5, "adapter_type": "saca"},
    ]

    from app.services import services

    for cfg in demo_configs:
        config = ExperimentConfig(**cfg)
        async with async_session() as db:
            exp_id = f"exp_{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            exp = Experiment(
                id=exp_id,
                name=f"{config.algorithm} on {config.dataset}",
                task_type=config.task_type,
                dataset=config.dataset,
                algorithm=config.algorithm,
                optimizer=config.optimizer,
                defense=config.defense,
                attack=config.attack,
                status=ExperimentStatus.completed,
                current_round=100,
                total_rounds=100,
                config_json=config.model_dump(),
            )
            db.add(exp)
            await db.flush()

            num_clients = config.num_clients
            total_rounds = 100

            # Pre-generate metrics and events for all 100 rounds
            for rnd in range(1, total_rounds + 1):
                progress = rnd / total_rounds
                rng = random.Random((hash(config.algorithm + str(rnd)) % (2**31)))

                train_loss = max(0.05, 2.0 * math.exp(-3 * progress) + 0.1 * rng.uniform(-0.05, 0.05))
                test_accuracy = min(0.95, 0.1 + 0.85 * (1 - math.exp(-5 * progress)) + rng.uniform(-0.02, 0.02))
                comm_cost = 142.3 + rng.uniform(-5, 5)
                client_part = max(0.05, 0.1 + rng.uniform(-0.02, 0.02))
                grad_drift = max(0.001, 0.05 * math.exp(-2 * progress) + rng.uniform(-0.005, 0.005))

                db.add(ExperimentMetric(experiment_id=exp_id, round=rnd, metric_name="train_loss", metric_value=round(train_loss, 4)))
                db.add(ExperimentMetric(experiment_id=exp_id, round=rnd, metric_name="test_accuracy", metric_value=round(test_accuracy, 4)))
                db.add(ExperimentMetric(experiment_id=exp_id, round=rnd, metric_name="communication_cost_mb", metric_value=round(comm_cost, 2)))
                db.add(ExperimentMetric(experiment_id=exp_id, round=rnd, metric_name="client_participation_rate", metric_value=round(client_part, 4)))
                db.add(ExperimentMetric(experiment_id=exp_id, round=rnd, metric_name="gradient_drift", metric_value=round(grad_drift, 4)))

                # Independent metrics with per-round fluctuations
                map_v = max(0.0, min(0.95, 0.55 + 0.35 * (1 - math.exp(-4 * progress)) + rng.uniform(-0.03, 0.03)))
                rank1_v = max(0.0, min(0.95, 0.62 + 0.30 * (1 - math.exp(-4 * progress)) + rng.uniform(-0.03, 0.03)))
                det_v = max(0.0, min(0.95, 0.50 + 0.38 * (1 - math.exp(-3.5 * progress)) + rng.uniform(-0.04, 0.04)))
                db.add(ExperimentMetric(experiment_id=exp_id, round=rnd, metric_name="map", metric_value=round(map_v, 4)))
                db.add(ExperimentMetric(experiment_id=exp_id, round=rnd, metric_name="rank1", metric_value=round(rank1_v, 4)))
                db.add(ExperimentMetric(experiment_id=exp_id, round=rnd, metric_name="detection_accuracy", metric_value=round(det_v, 4)))

                db.add(ExperimentEvent(experiment_id=exp_id, event_type="round_complete", payload_json={
                    "round": rnd, "train_loss": round(train_loss, 4),
                    "test_accuracy": round(test_accuracy, 4),
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }))

                # Participation every round keeps the monitor heatmap tied to communication rounds.
                seed_str = f"{exp_id}:part:{rnd}"
                prng = random.Random(int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16))
                n_p = max(1, int(num_clients * config.participation_rate * prng.uniform(0.8, 1.2)))
                all_ids = list(range(1, num_clients + 1))
                participating = sorted(prng.sample(all_ids, min(n_p, num_clients)))
                remaining = [c for c in all_ids if c not in participating]
                n_m = prng.randint(0, max(1, int(num_clients * 0.06))) if rnd > 3 else 0
                malicious = sorted(prng.sample(remaining, min(n_m, len(remaining)))) if n_m > 0 and remaining else []
                db.add(ExperimentEvent(experiment_id=exp_id, event_type="client_participation", payload_json={
                    "round": rnd, "participating": participating, "silent": [], "malicious": malicious,
                }))

                # Defense every 10 rounds after round 5
                if rnd % 10 == 0 and rnd > 5:
                    drng = random.Random(int(hashlib.md5(f"{exp_id}:def:{rnd}".encode()).hexdigest()[:8], 16))
                    n_trusted = drng.randint(max(1, int(num_clients * 0.8)), num_clients)
                    n_filtered = drng.randint(1, max(1, int(num_clients * 0.1)))
                    all_ids2 = list(range(1, num_clients + 1))
                    trusted = sorted(drng.sample(all_ids2, min(n_trusted, num_clients)))
                    rest = [c for c in all_ids2 if c not in trusted]
                    n_filt = min(n_filtered, len(rest))
                    filtered = sorted(drng.sample(rest, n_filt)) if n_filt > 0 and rest else []
                    db.add(ExperimentEvent(experiment_id=exp_id, event_type="defense_detection", payload_json={
                        "round": rnd, "trusted_clients": trusted, "filtered_clients": filtered,
                        "detection_accuracy": round(0.85 + 0.1 * drng.random(), 4),
                    }))

            # Completed event
            db.add(ExperimentEvent(experiment_id=exp_id, event_type="completed", payload_json={
                "final_accuracy": round(test_accuracy, 4), "final_loss": round(train_loss, 4),
                "message": "Experiment completed successfully",
            }))

            await db.commit()

    print(f"[seed] Pre-generated {len(demo_configs)} demo experiments with 100 rounds each.")


app = FastAPI(
    title="FedCompass API",
    description="FedCompass Federated Learning Display System Backend",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api import overview, modules, algorithms, references, resources, experiments, results, events, config_assistant, workflow

app.include_router(overview.router)
app.include_router(modules.router)
app.include_router(algorithms.router)
app.include_router(references.router)
app.include_router(experiments.router)
app.include_router(results.router)
app.include_router(resources.router)
app.include_router(events.router)
app.include_router(config_assistant.router)
app.include_router(workflow.router)


@app.get("/")
async def root():
    return {"service": "FedCompass API", "version": "0.1.0"}


@app.get("/healthz")
async def healthz():
    try:
        async with async_session() as db:
            await db.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
