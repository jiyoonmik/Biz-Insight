from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import require_stock_code
from src.context.queries import get_credit_ratings


router = APIRouter(tags=["ratings"])


@router.get("/companies/{stock_code}/credit-ratings")
def credit_ratings(stock_code: str, year: int | None = None) -> dict:
    code = require_stock_code(stock_code)
    return {"stock_code": code, "year": year, "credit_ratings": get_credit_ratings(code, year=year)}
