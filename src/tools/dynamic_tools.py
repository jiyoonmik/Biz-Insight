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
    """stock_code 기준으로 질의와 유사한 직원 리뷰 원문을 TF-IDF로 검색합니다.
    정확 부분문자열이 아니라 문자 n-gram 유사도라, 띄어쓰기·조사가 달라도
    '커리어 향상 부족'으로 '커리어 향상이 안 된다' 후기를 찾습니다. 각 결과는
    직무·재직상태·평점·장점·단점을 포함합니다. 조직문화, 이탈 신호, 내부 리스크
    근거에 사용하세요."""
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
