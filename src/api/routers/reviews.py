from __future__ import annotations

from fastapi import APIRouter, Query

from src.api.dependencies import require_stock_code
from src.dynamic.review_store import get_review_summary, search_review_evidence


router = APIRouter(tags=["reviews"])


@router.get("/companies/{stock_code}/review-summary")
def review_summary(stock_code: str, period: str | None = None) -> dict:
    code = require_stock_code(stock_code)
    return get_review_summary(code, period=period)


@router.get("/companies/{stock_code}/review-evidence")
def review_evidence(stock_code: str, query: str = Query(..., min_length=1), top_k: int = Query(default=5, ge=1, le=20)) -> dict:
    code = require_stock_code(stock_code)
    return {"stock_code": code, "query": query, "evidence": search_review_evidence(code, query=query, top_k=top_k)}

