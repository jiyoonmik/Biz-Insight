"""Verifier: 결정론적 근거 대조 게이트.

Reviewer가 이미 품질 게이트인데 왜 하나 더 두는가. 둘은 판정 방식이 다르다.

- Reviewer는 LLM이다. "근거가 충분한가" 같은 판단은 사람이 읽어야 하는 문제라
  LLM이 맞지만, 대신 판정이 흔들리고 회귀 측정에 쓸 수 없다.
- Verifier는 코드다. "리포트에 적힌 이 수치가 도구가 반환한 값에 실제로 있는가"는
  문자열과 숫자 비교로 끝나는 문제다. 여기에 LLM을 쓰면 검증자 자신이 환각의
  원천이 되고, 같은 리포트를 두 번 채점했을 때 점수가 달라진다.

그래서 판정 가능한 것은 코드로 내리고, LLM 게이트는 판단이 필요한 곳에만 남겼다.

측정하는 값은 정확도(correctness)가 아니라 **근거 대조율(coverage)** 이다.
LLM이 원장의 값 두 개로 증감률을 계산해 쓰면 그 파생 수치는 원장에 없고
미확인으로 잡힌다. 이는 오류가 아니라 '원장에서 직접 확인되지 않음'이라는 뜻이고,
리포트에도 그렇게 적는다. 지표를 실제보다 좋게 보이게 만드는 해석은 넣지 않는다.
"""
from __future__ import annotations

import re
from typing import Any


# 숫자 + 선택적 단위. 콤마 구분 정수와 소수를 함께 잡는다.
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(조원|억원|만원|조|억|만|%|퍼센트|배|원)?"
)

UNIT_SCALE = {
    "조": 1e12,
    "조원": 1e12,
    "억": 1e8,
    "억원": 1e8,
    "만": 1e4,
    "만원": 1e4,
}

# 연도 표기는 수치 인용이 아니라 기준 시점 표기이므로 별도로 다룬다.
YEAR_PATTERN = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?=\s*년|\D|$)")

REL_TOLERANCE = 0.01
ABS_TOLERANCE = 0.01
MAX_UNMATCHED_SAMPLES = 12


def _line_tokens(line: str) -> list[tuple[str, str, int, int]]:
    """한 줄에서 (숫자문자열, 단위, 시작, 끝) 토큰을 뽑는다."""
    tokens = []
    for match in NUMBER_PATTERN.finditer(line):
        tokens.append((match.group(1), match.group(2) or "", match.start(), match.end()))
    return tokens


def _merge_compound(line: str, tokens: list[tuple[str, str, int, int]]) -> list[tuple[str, list[float]]]:
    """한국어 복합 단위 표기를 하나의 수치로 합친다.

    '1조 2,345억원'은 토큰 두 개로 잡히지만 실제로는 1.2345e12라는 값 하나다.
    쪼갠 채로 대조하면 원장에 있는 값인데도 미확인으로 잡혀 대조율이 실제보다
    나쁘게 나온다. 단위 스케일이 내림차순으로 이어지고 사이에 공백뿐이면 합친다.
    """
    merged: list[tuple[str, list[float]]] = []
    index = 0
    while index < len(tokens):
        token, unit, start, end = tokens[index]
        scale = UNIT_SCALE.get(unit)
        if scale:
            total = float(token.replace(",", "")) * scale
            text_end = end
            last_scale = scale
            cursor = index + 1
            while cursor < len(tokens):
                next_token, next_unit, next_start, next_end = tokens[cursor]
                next_scale = UNIT_SCALE.get(next_unit)
                gap = line[text_end:next_start]
                if next_scale is None or next_scale >= last_scale or gap.strip():
                    break
                total += float(next_token.replace(",", "")) * next_scale
                last_scale, text_end, cursor = next_scale, next_end, cursor + 1
            if cursor > index + 1:
                merged.append((line[start:text_end].strip(), [abs(total)]))
                index = cursor
                continue
        merged.append((f"{token}{unit}", _candidate_values(float(token.replace(",", "")), unit)))
        index += 1
    return merged


def _report_numbers(report: str) -> list[tuple[str, list[float]]]:
    """리포트 본문에서 대조 대상 수치와 그 후보 스케일을 뽑는다.

    제외 대상:
    - 마크다운 리스트/제목의 순번 (`1.`, `## 3`)
    - 연도 표기 (기준 시점이지 인용 수치가 아님)
    - 표 구분선 등 숫자가 아닌 토큰
    """
    numbers: list[tuple[str, list[float]]] = []
    for raw_line in report.splitlines():
        line = raw_line.strip()
        if not line or set(line) <= set("|-: "):
            continue
        # 리스트 순번과 제목 번호 제거
        line = re.sub(r"^\s*(?:[-*+]\s+)?\d{1,2}[.)]\s+", "", line)
        line = re.sub(r"^#{1,6}\s*\d{1,2}[.)]?\s*", "", line)

        years = {match.group(1) for match in YEAR_PATTERN.finditer(line)}
        tokens = [
            token for token in _line_tokens(line)
            if not (token[0] in years and not token[1])
        ]
        numbers.extend(_merge_compound(line, tokens))
    return numbers


def _candidate_values(value: float, unit: str) -> list[float]:
    """표기 단위가 원장 단위와 다를 수 있으므로 후보 스케일을 넓게 잡는다.

    canonical 관측값은 원 단위 원본이 많고 리포트는 '억원'으로 줄여 쓴다.
    비율은 12.3(%)과 0.123 두 관행이 섞여 있어 양쪽을 모두 후보로 둔다.
    """
    candidates = {value}
    scale = UNIT_SCALE.get(unit)
    if scale:
        candidates.add(value * scale)
    if unit in {"%", "퍼센트"}:
        candidates.add(value / 100)
    else:
        # 단위 없는 수치도 비율 표기 차이를 흡수한다.
        candidates.add(value / 100)
        candidates.add(value * 100)
    return [abs(candidate) for candidate in candidates]


def _matches(candidates: list[float], ledger_values: list[float]) -> bool:
    for candidate in candidates:
        for known in ledger_values:
            tolerance = max(ABS_TOLERANCE, REL_TOLERANCE * abs(known))
            if abs(candidate - abs(known)) <= tolerance:
                return True
    return False


def verify_report(report: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """리포트 수치를 근거 원장과 대조한다. LLM을 호출하지 않는다."""
    facts = [fact for fact in (evidence or []) if isinstance(fact, dict)]
    ledger_values = [
        abs(float(fact["numeric"]))
        for fact in facts
        if fact.get("numeric") is not None
    ]
    text_values = {
        str(fact.get("value")).strip()
        for fact in facts
        if fact.get("numeric") is None and fact.get("value") is not None
    }

    numbers = _report_numbers(report or "")
    matched = 0
    unmatched: list[str] = []
    for display, candidates in numbers:
        if _matches(candidates, ledger_values):
            matched += 1
        elif len(unmatched) < MAX_UNMATCHED_SAMPLES:
            unmatched.append(display)

    checked = len(numbers)
    canonical_facts = sum(1 for fact in facts if fact.get("kind") == "canonical")
    dynamic_facts = sum(1 for fact in facts if fact.get("kind") == "dynamic")

    return {
        "checked": checked,
        "matched": matched,
        "unmatched": checked - matched,
        "coverage": round(matched / checked, 4) if checked else None,
        "unmatched_samples": unmatched,
        "evidence_count": len(facts),
        "canonical_facts": canonical_facts,
        "dynamic_facts": dynamic_facts,
        "rating_values": sorted(text_values)[:10],
    }


def render_verification_section(result: dict[str, Any]) -> str:
    """검증 결과를 리포트 하단 섹션으로 만든다.

    개발자 로그가 아니라 리포트의 일부로 노출한다. 근거 대조 결과를 숨기면
    리포트를 읽는 사람이 어디까지 믿어야 하는지 알 수 없다.
    """
    checked = result.get("checked", 0)
    if not result.get("evidence_count"):
        return (
            "\n\n---\n\n## 🔎 근거 대조\n\n"
            "구조화된 근거를 수집하지 못해 수치 대조를 수행하지 못했습니다. "
            "본문 수치는 도구 원본과 대조되지 않은 상태입니다."
        )
    if not checked:
        return (
            "\n\n---\n\n## 🔎 근거 대조\n\n"
            f"근거 원장 {result['evidence_count']}건을 수집했으나 본문에서 대조할 수치를 찾지 못했습니다."
        )

    coverage = result.get("coverage") or 0
    lines = [
        "\n\n---\n\n## 🔎 근거 대조",
        "",
        f"- 근거 원장: **{result['evidence_count']}건** "
        f"(회계연도 기준 {result['canonical_facts']} / 스냅샷 기준 {result['dynamic_facts']})",
        f"- 본문 수치 대조: **{result['matched']}/{checked}건 일치** (대조율 {coverage:.0%})",
    ]
    if result.get("unmatched_samples"):
        samples = ", ".join(result["unmatched_samples"])
        lines.append(f"- 원장에서 직접 확인되지 않은 수치: {samples}")
        lines.append(
            "  - 계산된 증감률·비율처럼 원장 값에서 파생된 수치가 여기 포함될 수 있습니다. "
            "오류 판정이 아니라 '원본 대조 불가' 표시입니다."
        )
    return "\n".join(lines)


def verifier_node(state: dict) -> dict:
    """Synthesis 결과를 근거 원장과 대조하는 종단 노드.

    LLM을 호출하지 않으므로 rate limit 지연도 붙이지 않는다.
    """
    report = state.get("final_report", "") or ""
    evidence = state.get("evidence", []) or []
    result = verify_report(report, evidence)

    errors = []
    coverage = result.get("coverage")
    if result["evidence_count"] == 0:
        errors.append("verifier_no_evidence_collected")
    elif coverage is not None and coverage < 0.5:
        # 실패로 처리하지 않고 표면화만 한다. 파생 수치가 많은 리포트도 정상이며,
        # 임계값을 강제하면 Synthesis가 수치를 덜 쓰는 방향으로 퇴화한다.
        errors.append(f"verifier_low_grounding_coverage: {coverage:.2f}")

    coverage_text = "n/a" if coverage is None else f"{coverage:.0%}"
    return {
        "final_report": report + render_verification_section(result),
        "grounding": result,
        "errors": errors,
        "analyses": [{
            "agent": "verifier",
            "content": (
                f"🔎 근거 대조 완료: 원장 {result['evidence_count']}건, "
                f"본문 수치 {result['matched']}/{result['checked']}건 일치 (대조율 {coverage_text})"
            ),
        }],
        "next_agent": "END",
    }
