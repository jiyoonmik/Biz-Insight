from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import require_stock_code
from src.dynamic.macro_store import get_macro_context
from src.dynamic.stock_store import get_stock_series, get_stock_summary
from src.context.queries import get_criteria_evidence, get_risk_context


router = APIRouter(tags=["signals"])


@router.get("/companies/{stock_code}/risk-signals")
def risk_signals(stock_code: str, year: int | None = None) -> dict:
    code = require_stock_code(stock_code)
    result = get_risk_context(code, year=year)
    result["stock_code"] = code
    result["year"] = year
    return result


@router.get("/companies/{stock_code}/criteria-evidence")
def criteria_evidence(stock_code: str, year: int | None = None) -> dict:
    code = require_stock_code(stock_code)
    return {"stock_code": code, "year": year, "criteria_evidence": get_criteria_evidence(code, year=year)}


@router.get("/companies/{stock_code}/stock-series")
def stock_series(stock_code: str, start: str | None = None, end: str | None = None) -> dict:
    code = require_stock_code(stock_code)
    return {"stock_code": code, "series": get_stock_series(code, start=start, end=end)}


@router.get("/companies/{stock_code}/stock-summary")
def stock_summary(stock_code: str, months: int = 12) -> dict:
    code = require_stock_code(stock_code)
    return get_stock_summary(code, months=months)


@router.get("/macro/context")
def macro_context(year: int | None = None) -> dict:
    return get_macro_context(year=year)
