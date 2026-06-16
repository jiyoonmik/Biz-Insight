from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.dependencies import require_stock_code
from src.context.queries import get_company_profile, get_knowledge_context, search_company_by_name


router = APIRouter(tags=["companies"])


@router.get("/companies/search")
def company_search(name: str = Query(..., min_length=1), limit: int = Query(default=10, ge=1, le=50)) -> dict:
    return {"query": name, "results": search_company_by_name(name, limit=limit)}


@router.get("/companies/{stock_code}")
def company_profile(stock_code: str) -> dict:
    return get_company_profile(require_stock_code(stock_code))


@router.get("/companies/{stock_code}/aliases")
def company_aliases(stock_code: str) -> dict:
    profile = get_company_profile(require_stock_code(stock_code))
    return {"stock_code": profile["stock_code"], "aliases": profile.get("aliases", [])}


@router.get("/companies/{stock_code}/knowledge-context")
def company_knowledge_context(stock_code: str, year: int | None = None) -> dict:
    return get_knowledge_context(require_stock_code(stock_code), year=year)
