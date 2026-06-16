from __future__ import annotations

from fastapi import HTTPException

from src.canonical.schema import normalize_stock_code
from src.context.queries import get_company_profile


def require_stock_code(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if not code:
        raise HTTPException(status_code=422, detail="stock_code must contain digits")
    if not get_company_profile(code):
        raise HTTPException(status_code=404, detail=f"company not found: {code}")
    return code
