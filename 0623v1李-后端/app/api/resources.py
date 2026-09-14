"""Datasets, Optimizers, Defenses, Attacks API."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import DatasetOut, AlgorithmOut
from app.services.services import get_experiment_config_options, list_datasets, list_algorithms
from app.plugins.Nets import get_networks_for_task

router = APIRouter(prefix="/api", tags=["resources"])


@router.get("/datasets", response_model=list[DatasetOut])
async def datasets(db: AsyncSession = Depends(get_db)):
    return [DatasetOut.model_validate(d) for d in await list_datasets(db)]


@router.get("/optimizers", response_model=list[AlgorithmOut])
async def optimizers(db: AsyncSession = Depends(get_db)):
    all_algs = await list_algorithms(db)
    return [AlgorithmOut.model_validate(a) for a in all_algs
            if a.category in ("optimization", "optimizer")]


@router.get("/defenses", response_model=list[AlgorithmOut])
async def defenses(db: AsyncSession = Depends(get_db)):
    all_algs = await list_algorithms(db)
    return [AlgorithmOut.model_validate(a) for a in all_algs if a.category == "defense"]


@router.get("/attacks", response_model=list[AlgorithmOut])
async def attacks(db: AsyncSession = Depends(get_db)):
    all_algs = await list_algorithms(db)
    return [AlgorithmOut.model_validate(a) for a in all_algs if a.category == "attack"]


@router.get("/experiment-config-options")
async def experiment_config_options(module_id: str | None = Query(None)):
    """Return paper-grounded experiment options for the selected module."""
    return get_experiment_config_options(module_id)


@router.get("/networks")
async def networks(task_type: str | None = Query(None)):
    """Return available neural networks, optionally filtered by task type."""
    if task_type:
        return [{"id": nid, "name": name} for nid, name in get_networks_for_task(task_type)]
    all_nets = []
    seen = set()
    for tt in ["vision_classification", "text_classification", "reid", "ulip3d"]:
        for nid, name in get_networks_for_task(tt):
            if nid not in seen:
                seen.add(nid)
                all_nets.append({"id": nid, "name": name})
    return all_nets
