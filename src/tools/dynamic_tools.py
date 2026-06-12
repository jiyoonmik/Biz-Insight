from __future__ import annotations

from langchain_core.tools import tool

from src.dynamic.macro_store import get_macro_context
from src.dynamic.review_store import get_review_summary, search_review_evidence
from src.dynamic.stock_store import get_stock_summary


@tool
def tool_get_stock_summary(stock_code: str, months: int = 12) -> dict:
    """stock_code 기준 최근 월별 주가 요약을 조회합니다."""
    return get_stock_summary(stock_code, months=months)


@tool
def tool_get_review_summary(stock_code: str, period: str | None = None) -> dict:
    """stock_code 기준 직원 리뷰 요약/평점 snapshot을 조회합니다."""
    return get_review_summary(stock_code, period=period)


@tool
def tool_search_review_evidence(stock_code: str, query: str, top_k: int = 5) -> dict:
    """stock_code 기준 직원 리뷰 원문 evidence를 검색합니다. 원문은 KG가 아니라 dynamic store에서 조회합니다."""
    return {"stock_code": stock_code, "query": query, "evidence": search_review_evidence(stock_code, query=query, top_k=top_k)}


@tool
def tool_get_macro_context(year: int | None = None) -> dict:
    """기준연도 또는 최신 거시경제 context를 dynamic store에서 조회합니다."""
    return get_macro_context(year=year)


DYNAMIC_TOOLS = [
    tool_get_stock_summary,
    tool_get_review_summary,
    tool_search_review_evidence,
    tool_get_macro_context,
]

