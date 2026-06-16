from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import require_stock_code
from src.context.queries import get_company_observations, get_sector_observations


router = APIRouter(tags=["financials"])


@router.get("/companies/{stock_code}/financial-observations")
def financial_observations(stock_code: str, year: int | None = None, limit: int = 250) -> dict:
    code = require_stock_code(stock_code)
    return {"stock_code": code, "year": year, "observations": get_company_observations(code, year=year, limit=limit)}


@router.get("/companies/{stock_code}/metrics/{metric_code}")
def company_metric(stock_code: str, metric_code: str, year: int | None = None) -> dict:
    code = require_stock_code(stock_code)
    observations = get_company_observations(code, year=year, limit=5000)
    return {
        "stock_code": code,
        "metric_code": metric_code,
        "year": year,
        "observations": [row for row in observations if row.get("metric_code") == metric_code],
    }


@router.get("/sectors/{sector_code}/observations")
def sector_observations(sector_code: str, year: int | None = None, limit: int = 250) -> dict:
    return {"sector_code": sector_code, "year": year, "observations": get_sector_observations(sector_code, year=year, limit=limit)}
