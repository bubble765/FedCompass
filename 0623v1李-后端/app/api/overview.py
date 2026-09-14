"""Overview API."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import OverviewOut, ModuleBrief, KeyMetrics
from app.services.services import get_overview, list_modules

router = APIRouter(prefix="/api", tags=["overview"])


@router.get("/overview", response_model=OverviewOut)
async def overview(db: AsyncSession = Depends(get_db)):
    data = await get_overview(db)
    data["algorithm_count"] = data.pop("registered_algorithms", 0)
    data["dataset_count"] = data.pop("registered_datasets", 0)
    data["defense_count"] = data.pop("available_defenses", 0)
    data["key_metrics"] = KeyMetrics(
        accuracy=0.847,
        map=0.781,
        rank1=0.823,
        attack_detection_accuracy=0.884,
        communication_cost=128.0,
    )
    return OverviewOut(**data)


@router.get("/overview/capability-map", response_model=list[ModuleBrief])
async def capability_map(db: AsyncSession = Depends(get_db)):
    mods = await list_modules(db)
    result = []
    for m in mods:
        primary_ids = m.primary_algorithms[:3] if m.primary_algorithms else []
        alg_map = {a.id: a.name for a in (m.algorithms or [])}
        tags = [alg_map.get(aid, aid) for aid in primary_ids]
        result.append(ModuleBrief(
            id=m.id,
            name=m.name,
            positioning=m.positioning,
            status=m.status.value if hasattr(m.status, "value") else str(m.status),
            tags=tags,
        ))
    return result
