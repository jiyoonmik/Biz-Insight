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

    text = str(compacted)
    if len(text) <= max_chars:
        return compacted
    return text[:max_chars] + "...[truncated]"


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
