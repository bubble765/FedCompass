"""Algorithms API."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import AlgorithmOut, ReferenceAssetOut
from app.services.services import list_algorithms, get_algorithm, get_algorithm_references

router = APIRouter(prefix="/api", tags=["algorithms"])


@router.get("/algorithms", response_model=list[AlgorithmOut])
async def algorithms(db: AsyncSession = Depends(get_db)):
    return [AlgorithmOut.model_validate(a) for a in await list_algorithms(db)]


@router.get("/algorithms/{algorithm_id}", response_model=AlgorithmOut)
async def algorithm_detail(algorithm_id: str, db: AsyncSession = Depends(get_db)):
    a = await get_algorithm(db, algorithm_id)
    if not a:
        raise HTTPException(404, "Algorithm not found")
    return AlgorithmOut.model_validate(a)


@router.get("/algorithms/{algorithm_id}/references", response_model=list[ReferenceAssetOut])
async def algorithm_references(algorithm_id: str, db: AsyncSession = Depends(get_db)):
    refs = await get_algorithm_references(db, algorithm_id)
    return [ReferenceAssetOut.model_validate(r) for r in refs]
