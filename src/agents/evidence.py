"""근거 원장(Evidence Ledger).

파이프라인의 구조적 약점 하나를 메우는 계층이다. 원래 도구 결과는 첫 홉에서
LLM 요약 문장이 되어 버렸고, 그 시점에 `metric_code / period / value / source_file`
이라는 구조가 사라졌다. 그러면 최종 리포트가 인용한 수치가 실제로 도구에서 나온
값인지 확인할 방법이 없다.

그래서 도구를 호출한 자리에서 구조화된 사실만 따로 뽑아 상태에 누적한다.
- Researcher/Analyst는 사실을 '수집'하고
- Synthesis는 이 원장 안에서만 수치를 인용하도록 지시받고
- Verifier는 리포트 수치를 이 원장과 대조한다

fact 하나의 스키마:
    {metric, label, period, value, kind, source, tool}

`kind`를 스키마에 넣은 이유는 canonical(회계연도 기준 사실)과 dynamic(스냅샷 기준
시그널)의 구분이 이 프로젝트에서 가장 자주 깨지는 지점이기 때문이다. 원장 단계에서
기준을 분리해 두면 Synthesis 프롬프트와 Verifier 리포트가 같은 구분을 재사용한다.
"""
from __future__ import annotations

from typing import Any


# dynamic store 계열 도구. canonical 사실과 기준 시점이 다르므로 kind를 분리한다.
DYNAMIC_TOOLS = {
    "tool_get_stock_summary",
    "tool_get_review_summary",
    "tool_search_review_evidence",
    "tool_get_macro_context",
    "tool_get_stock_data",
    "tool_get_employee_reviews",
}

# 플랫 dict(주가 요약 등)에서 사실로 승격할 수치 키.
FLAT_NUMERIC_KEYS = {
    "latest_close",
    "period_return",
    "volatility",
    "average_volume",
    "foreign_ownership_change",
    "rating",
    "employee_rating",
    "paywellfare",
    "worklifebal",
    "culture",
    "opportunity",
    "manager",
    "recommend",
    "ceo",
    "potential",
    "total_reviews",
}

# 사실이 아니라 식별자/메타에 해당하므로 수치 대조 대상에서 제외한다.
SKIP_KEYS = {
    "stock_code",
    "months",
    "available",
    "year",
    "period_value",
    "subject_id",
    "count",
}

MAX_FACTS_PER_TOOL = 60


def _to_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _fact(
    metric: str | None,
    value: Any,
    *,
    tool: str,
    kind: str,
    label: str | None = None,
    period: str | None = None,
    source: str | None = None,
) -> dict[str, Any] | None:
    if not metric:
        return None
    number = _to_number(value)
    if number is None:
        # 텍스트 사실(등급 문자열 등)도 원장에는 남긴다. 수치 대조에서는 제외된다.
        text = None if value is None else str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        return {
            "metric": str(metric),
            "label": label,
            "period": None if period is None else str(period),
            "value": text,
            "numeric": None,
            "kind": kind,
            "source": source,
            "tool": tool,
        }
    return {
        "metric": str(metric),
        "label": label,
        "period": None if period is None else str(period),
        "value": number,
        "numeric": number,
        "kind": kind,
        "source": source,
        "tool": tool,
    }


def _from_metric_row(row: dict[str, Any], *, tool: str, kind: str) -> dict[str, Any] | None:
    """compact_tool_result가 만든 `{metric, name, year, value, source}` 행."""
    return _fact(
        row.get("metric"),
        row.get("value"),
        tool=tool,
        kind=kind,
        label=row.get("name"),
        period=row.get("year"),
        source=row.get("source"),
    )


def _from_observation_row(row: dict[str, Any], *, tool: str, kind: str) -> dict[str, Any] | None:
    """canonical observations 원본 행."""
    value = row.get("numeric_value")
    if value is None:
        value = row.get("text_value")
    return _fact(
        row.get("metric_code"),
        value,
        tool=tool,
        kind=kind,
        label=row.get("metric_name_ko"),
        period=row.get("period_value"),
        source=row.get("source_file"),
    )


def _from_rating_row(row: dict[str, Any], *, tool: str) -> dict[str, Any] | None:
    agency = row.get("agency") or "rating"
    return _fact(
        f"credit_rating_{agency}",
        row.get("rating") or row.get("rating_value"),
        tool=tool,
        kind="rating",
        label=f"{agency} 신용등급",
        period=row.get("year"),
        source=row.get("source_file"),
    )


def _from_review_row(row: dict[str, Any], *, tool: str) -> dict[str, Any] | None:
    """직원 후기 행. 질적 근거로 다룬다.

    개별 후기의 별점(1~5)을 수치 사실로 흘리면 원장이 리포트의 무관한 숫자와
    우연히 매칭되어 근거 대조율을 왜곡한다. 후기는 텍스트 근거이므로 numeric을
    비우고, 직무·재직상태와 장/단점 요지를 라벨에 담아 원장에 남긴다.
    """
    snippet = normalize_text(row.get("summary")) or normalize_text(row.get("cons")) or normalize_text(row.get("pros"))
    if not snippet:
        return None
    who = " ".join(str(row.get(field)) for field in ("status", "position") if row.get(field))
    label = f"{who} 후기".strip() if who else "직원 후기"
    return {
        "metric": "employee_review",
        "label": label,
        "period": (str(row.get("period_value")) if row.get("period_value") else None)
        or (str(row.get("review_date")) if row.get("review_date") else None),
        "value": str(snippet)[:120],
        "numeric": None,
        "kind": "dynamic",
        "source": row.get("source_file"),
        "tool": tool,
    }


def _row_facts(row: dict[str, Any], *, tool: str, kind: str) -> list[dict[str, Any]]:
    """dict 하나를 알려진 행 모양 중 하나로 해석한다."""
    facts: list[dict[str, Any]] = []

    if "agency" in row and ("rating" in row or "rating_value" in row):
        fact = _from_rating_row(row, tool=tool)
        return [fact] if fact else []

    # 직원 후기 행(장/단점 텍스트를 가진다)은 질적 근거로 다룬다.
    if ("pros" in row or "cons" in row) and ("summary" in row or "position" in row or "status" in row):
        fact = _from_review_row(row, tool=tool)
        return [fact] if fact else []

    if "metric" in row and "value" in row:
        fact = _from_metric_row(row, tool=tool, kind=kind)
        return [fact] if fact else []

    if "metric_code" in row:
        fact = _from_observation_row(row, tool=tool, kind=kind)
        return [fact] if fact else []

    # 플랫 요약 dict(주가 요약, 리뷰 요약)는 화이트리스트 키만 승격한다.
    period = row.get("end_date") or row.get("period_value") or row.get("dynamic_data_as_of")
    source = row.get("source_file") or row.get("source")
    for key, value in row.items():
        if key in SKIP_KEYS or key not in FLAT_NUMERIC_KEYS:
            continue
        fact = _fact(key, value, tool=tool, kind=kind, period=period, source=source)
        if fact:
            facts.append(fact)
    return facts


def extract_facts(tool_name: str, result: Any) -> list[dict[str, Any]]:
    """압축된 도구 결과에서 구조화된 사실만 재귀적으로 수집한다.

    도구마다 반환 모양이 달라 파서를 도구별로 두면 도구가 늘 때마다 깨진다.
    대신 '알아볼 수 있는 행 모양'을 정의하고 결과 전체를 순회한다. 모르는 모양은
    조용히 건너뛴다 — 원장에 없는 수치는 Verifier가 미확인으로 보고하므로
    누락이 조용한 통과로 이어지지 않는다.
    """
    kind = "dynamic" if tool_name in DYNAMIC_TOOLS else "canonical"
    facts: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if len(facts) >= MAX_FACTS_PER_TOOL:
            return
        if isinstance(node, dict):
            facts.extend(f for f in _row_facts(node, tool=tool_name, kind=kind) if f)
            for value in node.values():
                if isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(result)
    return _dedupe(facts)[:MAX_FACTS_PER_TOOL]


def _dedupe(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple] = set()
    result = []
    for fact in facts:
        key = (fact["metric"], fact["period"], str(fact["value"]))
        if key in seen:
            continue
        seen.add(key)
        result.append(fact)
    return result


def ledger_digest(facts: list[dict[str, Any]], limit: int = 60) -> str:
    """Synthesis 프롬프트에 넣을 근거표. 토큰을 아끼려 한 줄 표기로 압축한다."""
    if not facts:
        return "(수집된 구조화 근거 없음)"

    ordered = sorted(
        _dedupe([f for f in facts if isinstance(f, dict)]),
        key=lambda f: (0 if f.get("kind") == "canonical" else 1, str(f.get("metric"))),
    )
    lines = []
    for fact in ordered[:limit]:
        label = fact.get("label") or fact.get("metric")
        period = fact.get("period") or "기준일 미상"
        source = fact.get("source") or fact.get("tool")
        basis = "회계연도" if fact.get("kind") == "canonical" else "스냅샷"
        lines.append(f"- [{basis}] {label}({fact.get('metric')}) / {period} / {fact.get('value')} / 출처: {source}")
    if len(ordered) > limit:
        lines.append(f"- ... 외 {len(ordered) - limit}건")
    return "\n".join(lines)
