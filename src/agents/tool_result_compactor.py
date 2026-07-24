"""Utilities for keeping tool results small enough for agent loops."""
from __future__ import annotations

import ast
import math
from typing import Any

from langchain_core.messages import ToolMessage


KEY_METRICS = {
    "revenue",
    "operating_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "equity",
    "current_ratio",
    "quick_ratio",
    "debt_ratio",
    "borrowing_dependency",
    "borrowings_dependency",
    "dependence_on_net_borrowings",
    "interest_coverage_ratio",
    "operating_cash_flow",
    "cash_flow_operating",
    "free_cash_flow",
    "ebitda_margin",
    "gross_profit_margin",
    "net_profit_margin",
    "earnings_per_share",
    "book_value_per_share",
    "enterprise_value_to_ebitda",
    "price_to_book_ratio",
    "price_earnings_ratio",
    "operating_income_growth_rate",
    "net_income_growth_rate",
    "rating",
    "employee_rating",
    "market_cap",
    "kosdaq_rank",
    "business_thesis",
    "risk_focus",
    "stock_close",
    "foreign_ownership_ratio",
    "worklifebal",
}


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value)
    if text == "nan":
        return None
    return value


def _records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        return [row for row in value["data"] if isinstance(row, dict)]
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def _latest_year(records: list[dict[str, Any]]) -> str | None:
    years = []
    for row in records:
        raw = row.get("period_value") or row.get("year")
        try:
            years.append(int(str(raw)))
        except Exception:
            pass
    return str(max(years)) if years else None


def _metric_row(row: dict[str, Any]) -> dict[str, Any]:
    metric = row.get("metric_code") or row.get("label_en") or row.get("label_ko")
    value = row.get("numeric_value")
    if value is None:
        value = row.get("text_value")
    year = row.get("period_value")
    if value is None and year is not None:
        value = row.get(str(year))
    if value is None:
        for candidate in ["2022", "2021", "2020", "YoY"]:
            if candidate in row:
                value = row.get(candidate)
                year = candidate
                break
    return {
        "metric": metric,
        "name": row.get("metric_name_ko") or row.get("label_ko"),
        "year": year or row.get("year"),
        "value": _clean(value),
        "source": row.get("source_file"),
    }


def _compact_metric_records(records: list[dict[str, Any]], limit: int = 40) -> list[dict[str, Any]]:
    latest = _latest_year(records)
    selected = []
    seen = set()
    for row in records:
        metric = row.get("metric_code") or row.get("label_en")
        year = str(row.get("period_value") or row.get("year") or "")
        if metric not in KEY_METRICS:
            continue
        if latest and year and year != latest:
            continue
        key = (metric, year)
        if key in seen:
            continue
        seen.add(key)
        selected.append(_metric_row(row))
        if len(selected) >= limit:
            break
    if selected:
        return selected
    return [_metric_row(row) for row in records[:limit]]


def _compact_company_search(result: dict[str, Any]) -> dict[str, Any]:
    rows = _records(result.get("results", []))
    query = str(result.get("query", ""))

    def score(row: dict[str, Any]) -> tuple[int, str]:
        alias = str(row.get("alias_name") or "")
        preferred = str(row.get("preferred_name") or "")
        exact = alias == query or preferred == query
        preferred_alias = row.get("alias_type") == "preferred"
        return (0 if exact else 1, 0 if preferred_alias else 1, preferred)

    ranked = sorted(rows, key=score)
    return {
        "query": query,
        "results": [
            {
                "stock_code": row.get("stock_code"),
                "alias_name": row.get("alias_name"),
                "preferred_name": row.get("preferred_name"),
                "sector_name": row.get("sector_name"),
                "main_product": row.get("main_product"),
            }
            for row in ranked[:5]
        ],
    }


def _compact_knowledge_context(result: dict[str, Any]) -> dict[str, Any]:
    observations = _records(result.get("financial_observations", []))
    sector_rows = _records(result.get("sector_comparison", []))
    criteria = result.get("criteria_evidence") or {}
    return {
        "company": result.get("company"),
        "latest_year": _latest_year(observations),
        "financial_observations": _compact_metric_records(observations),
        "credit_ratings": result.get("credit_ratings", [])[:10],
        "sector_comparison": _compact_metric_records(sector_rows, limit=20),
        "criteria_available": sorted(criteria.keys()) if isinstance(criteria, dict) else [],
        "risk_signals": result.get("risk_signals", [])[:10],
    }


def _compact_criteria_evidence(result: dict[str, Any]) -> dict[str, Any]:
    criteria = result.get("criteria_evidence") or {}
    compacted = {}
    if isinstance(criteria, dict):
        for criterion, rows in criteria.items():
            compacted[criterion] = _compact_metric_records(_records(rows), limit=8)
    return {
        "stock_code": result.get("stock_code"),
        "year": result.get("year"),
        "criteria_evidence": compacted,
    }


def _compact_review_rows(result: dict[str, Any]) -> dict[str, Any]:
    rows = _records(result)
    return {
        "total_reviews": result.get("total_reviews") if isinstance(result, dict) else len(rows),
        "columns": result.get("columns") if isinstance(result, dict) else None,
        "samples": [
            {
                "corp": row.get("corp"),
                "stock_code": row.get("stock_code"),
                "year": row.get("year"),
                "positive": row.get("up"),
                "negative": row.get("down"),
                "source": row.get("source_file"),
            }
            for row in rows[:8]
        ],
    }


def _compact_series(result: dict[str, Any]) -> dict[str, Any]:
    rows = _records(result)
    if not rows:
        return result
    return {
        "row_count": len(rows),
        "first": rows[0],
        "latest": rows[-1],
    }


def _scrub(value: Any) -> Any:
    """repr이 다시 파싱 가능하도록 값을 정리한다.

    도구 결과는 `str(dict)`로 ToolMessage에 실리고, 나중에 `ast.literal_eval`로
    되읽어 출처 추적·히스토리 요약·근거 추출에 쓰인다. 그런데 pandas가 돌려주는
    `nan`이나 numpy 스칼라(`np.float64(3.0)`)가 섞이면 그 repr은 literal_eval로
    파싱되지 않는다. 파싱이 실패해도 예외는 삼켜지므로 **출처가 조용히 사라지고**
    히스토리 요약이 "사실 없음"으로 잘못 표시된다.

    실제로 이 버그는 히스토리 압축 효과를 측정하다가 드러났다. 계측을 붙이지
    않았다면 트레이스의 출처 목록이 왜 비는지 알아내기 어려웠을 것이다.
    """
    if isinstance(value, dict):
        return {key: _scrub(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(item) for item in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()  # numpy 스칼라 → 파이썬 기본형
        except Exception:
            return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def compact_tool_result(tool_name: str, result: Any, max_chars: int = 12_000) -> Any:
    if not isinstance(result, dict):
        text = str(result)
        return text[:max_chars] + "...[truncated]" if len(text) > max_chars else result

    if tool_name == "tool_search_company_by_name":
        compacted = _compact_company_search(result)
    elif tool_name == "tool_get_company_knowledge_context":
        compacted = _compact_knowledge_context(result)
    elif tool_name == "tool_get_criteria_evidence":
        compacted = _compact_criteria_evidence(result)
    elif tool_name == "tool_get_employee_reviews":
        compacted = _compact_review_rows(result)
    elif tool_name in {"tool_get_stock_data"}:
        compacted = _compact_series(result)
    elif "data" in result and isinstance(result.get("data"), list):
        compacted = {
            "columns": result.get("columns"),
            "data": _compact_metric_records(_records(result), limit=40),
        }
    else:
        compacted = result

    compacted = _scrub(compacted)
    text = str(compacted)
    if len(text) <= max_chars:
        return compacted
    return text[:max_chars] + "...[truncated]"


# ReAct 루프에서 원문 그대로 재전송할 최근 도구 결과 개수.
HISTORY_KEEP_FULL = 1


def _history_digest(message: Any, ledger: list[dict[str, Any]] | None = None) -> str:
    """오래된 도구 결과를 원장 참조 한 줄로 바꾼다.

    요약은 원장에서 직접 만든다. 도구 결과 문자열을 다시 파싱해 재추출하면 같은
    일을 두 번 하는 데다, 파싱이 실패하는 순간 "사실 없음"이라는 틀린 요약을
    LLM에게 보내게 된다. 이미 수집 시점에 뽑아 둔 것을 쓰는 편이 정확하고 싸다.
    """
    from src.agents.evidence import extract_facts

    name = getattr(message, "name", "tool") or "tool"

    if ledger:
        facts = [fact for fact in ledger if isinstance(fact, dict) and fact.get("tool") == name]
    else:
        content = getattr(message, "content", "")
        try:
            parsed = ast.literal_eval(content) if isinstance(content, str) else content
        except Exception:
            parsed = content
        facts = extract_facts(name, parsed)

    if not facts:
        return f"[히스토리 압축] {name} 호출 완료 — 구조화된 사실 없음. 원문 생략."

    metrics = sorted({str(fact["metric"]) for fact in facts})
    shown = ", ".join(metrics[:8])
    more = f" 외 {len(metrics) - 8}종" if len(metrics) > 8 else ""
    return (
        f"[히스토리 압축] {name} 호출 완료 — 사실 {len(facts)}건이 근거 원장에 적재됨. "
        f"수집 지표: {shown}{more}. "
        "원문은 생략했으며 최종 리포트는 원장에서 인용한다. 같은 도구를 다시 부를 필요 없음."
    )


def compact_message_history(
    messages: list[Any],
    keep_full: int | None = None,
    ledger: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """ReAct 루프가 LLM에 재전송하는 히스토리를 줄인다.

    ReAct는 매 턴 전체 히스토리를 다시 보낸다. 도구 결과 하나가 최대 12,000자이므로
    4턴이면 같은 원문을 네 번 지불하게 된다. 그런데 이 프로젝트는 도구 결과를 이미
    **근거 원장**으로 구조화해 상태에 들고 있다(README §2.5). 원문이 사라져도 근거는
    남아 있고, Synthesis와 Verifier는 원장에서 인용하고 대조한다.

    ReAct 루프에서 오래된 원문이 필요한 경우는 "다음에 무엇을 더 모을까"를 정할 때
    거의 없으므로, 최근 `keep_full`건만 원문으로 두고 나머지는 '무엇을 모았는지'를
    알려주는 참조 한 줄로 바꾼다.

    상태(`researcher_messages`)는 건드리지 않는다. 여기서 만드는 것은 LLM에 보낼
    사본이며, 트레이스와 폴백 요약은 계속 전체 히스토리를 본다.
    """
    # 모듈 상수를 호출 시점에 읽는다. 기본 인자에 바인딩하면 설정을 바꿔도
    # 반영되지 않아 측정과 실제 동작이 어긋난다.
    keep_full = HISTORY_KEEP_FULL if keep_full is None else keep_full

    tool_positions = [
        index for index, message in enumerate(messages)
        if getattr(message, "type", None) == "tool" or isinstance(message, ToolMessage)
    ]
    if len(tool_positions) <= keep_full:
        return messages

    to_compact = set(tool_positions[:-keep_full] if keep_full else tool_positions)
    compacted = []
    for index, message in enumerate(messages):
        if index not in to_compact:
            compacted.append(message)
            continue
        compacted.append(
            ToolMessage(
                content=_history_digest(message, ledger=ledger),
                tool_call_id=getattr(message, "tool_call_id", str(index)),
                name=getattr(message, "name", None),
            )
        )
    return compacted


def summarize_tool_messages(messages: list[Any], max_chars: int = 16_000) -> str:
    blocks = []
    for message in messages:
        if not isinstance(message, ToolMessage) and getattr(message, "type", None) != "tool":
            continue
        name = getattr(message, "name", "tool")
        content = getattr(message, "content", "")
        try:
            parsed = ast.literal_eval(content) if isinstance(content, str) else content
        except Exception:
            parsed = content
        compacted = compact_tool_result(name, parsed, max_chars=4_000)
        blocks.append(f"### {name}\n{compacted}")
    if not blocks:
        return "수집된 tool 결과가 없습니다."
    text = "\n\n".join(blocks)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text
