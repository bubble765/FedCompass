"""References API."""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import ReferenceAssetOut
from app.services.services import list_references

router = APIRouter(prefix="/api", tags=["references"])


@router.get("/references", response_model=list[ReferenceAssetOut])
async def references(db: AsyncSession = Depends(get_db)):
    return [ReferenceAssetOut.model_validate(r) for r in await list_references(db)]
