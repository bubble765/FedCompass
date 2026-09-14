"""Modules API."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import ModuleOut
from app.services.services import list_modules, get_module, get_module_references

router = APIRouter(prefix="/api", tags=["modules"])


@router.get("/modules", response_model=list[ModuleOut])
async def modules(db: AsyncSession = Depends(get_db)):
    return [ModuleOut.model_validate(m) for m in await list_modules(db)]


@router.get("/modules/{module_id}", response_model=ModuleOut)
async def module_detail(module_id: str, db: AsyncSession = Depends(get_db)):
    m = await get_module(db, module_id)
    if not m:
        raise HTTPException(404, "Module not found")
    return ModuleOut.model_validate(m)


@router.get("/modules/{module_id}/references")
async def module_references(module_id: str, db: AsyncSession = Depends(get_db)):
    """Return module references in frontend-compatible format."""
    m = await get_module(db, module_id)
    if not m:
        raise HTTPException(404, "Module not found")
    refs = await get_module_references(db, module_id)
    papers = [r.title for r in refs if r.asset_type and r.asset_type.value == "paper"]
    code_sources = [r.title for r in refs if r.asset_type and r.asset_type.value in ("external_code", "local_archive")]
    meta = getattr(m, "meta_json", None) or {}
    return {
        "implementation_status": meta.get("implementation_status", "planned"),
        "papers": papers,
        "code_sources": code_sources,
        "frontend_views": meta.get("frontend_views", []),
        "backend_plugins": meta.get("backend_plugins", []),
    }
