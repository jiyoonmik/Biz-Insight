from __future__ import annotations

from langchain_core.tools import tool

from src.context.queries import (
    get_credit_ratings,
    get_criteria_evidence,
    get_knowledge_context,
    search_company_by_name,
)


@tool
def tool_get_company_knowledge_context(stock_code: str, year: int | None = None) -> dict:
    """stock_code 기준으로 기업 분석 컨텍스트, 재무 관측값, 신용등급, 평가기준 근거, 위험 신호를 조회합니다."""
    return get_knowledge_context(stock_code, year=year)


@tool
def tool_search_company_by_name(name: str) -> dict:
    """기업명을 alias table에서 검색하여 분석에 사용할 stock_code 후보를 확정합니다."""
    return {"query": name, "results": search_company_by_name(name)}


@tool
def tool_get_criteria_evidence(stock_code: str, year: int | None = None) -> dict:
    """stock_code와 기준연도로 평가기준별 수치 근거를 조회합니다."""
    return {"stock_code": stock_code, "year": year, "criteria_evidence": get_criteria_evidence(stock_code, year=year)}


@tool
def tool_get_credit_rating_history(stock_code: str) -> dict:
    """stock_code 기준 KIS/NICE/통합/모델 신용등급 이력을 조회합니다."""
    return {"stock_code": stock_code, "credit_ratings": get_credit_ratings(stock_code)}


CONTEXT_TOOLS = [
    tool_search_company_by_name,
    tool_get_company_knowledge_context,
    tool_get_criteria_evidence,
    tool_get_credit_rating_history,
]
