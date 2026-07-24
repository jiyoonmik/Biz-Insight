"""Reviewer의 기계적 점검 항목.

Reviewer 프롬프트는 원래 다섯 가지를 한꺼번에 LLM에게 물었다. 그런데 그중 상당수는
판단이 아니라 확인이다 — stock_code가 본문에 있는가, 기준연도를 적었는가, 수치
근거를 인용했는가, 리스크를 언급했는가. LLM은 이런 기계적 확인에 오히려 약하다.
빠진 항목을 그냥 지나치고 "전반적으로 충실합니다"라고 통과시키는 실패를 실제로 봤다.

그래서 체크리스트를 성격에 따라 나눴다.

- **확인**(여기) → 코드. 절대 놓치지 않고, 피드백이 구체적이며, 재현 가능하다.
- **판단**(Reviewer LLM) → 논리 전개가 타당한가, 근거 없이 단정한 곳은 없는가.

부수 효과로 비용도 준다. 기계적 점검에서 이미 떨어질 초안을 두고 LLM에게
"이거 괜찮나요"를 묻는 것은 낭비이므로, 실패 시 LLM 호출 없이 곧바로 반려한다.
Reviewer 프롬프트에서 확인 항목이 빠져 프롬프트 자체도 짧아진다.
"""
from __future__ import annotations

import re
from typing import Any

from src.agents.verifier import verify_report


STOCK_CODE = re.compile(r"\b\d{6}\b")
FISCAL_YEAR = re.compile(r"(19|20)\d{2}\s*년|회계연도|기준\s*연도|FY\s*\d{2,4}")
RISK_MENTION = re.compile(r"리스크|위험|취약|우려|불확실")
NUMERIC_EVIDENCE = re.compile(r"\d")

# 원장 대조율이 이 값 미만이면 본문 수치가 근거에서 떠 있다고 본다.
MIN_GROUNDING_COVERAGE = 0.3
# 최소 본문 길이. 이보다 짧으면 분석이라고 보기 어렵다.
MIN_CONTENT_CHARS = 200


def run_deterministic_checks(content: str, evidence: list[dict[str, Any]] | None) -> dict[str, Any]:
    """분석 초안의 기계적 요건을 점검한다. LLM을 호출하지 않는다."""
    text = content or ""
    facts = evidence or []
    failures: list[str] = []

    if len(text.strip()) < MIN_CONTENT_CHARS:
        failures.append(f"분석 본문이 {len(text.strip())}자로 너무 짧습니다. 근거와 해석을 채우세요.")

    if not STOCK_CODE.search(text):
        failures.append("stock_code(6자리 종목코드)가 본문에 명시되지 않았습니다.")

    if not FISCAL_YEAR.search(text):
        failures.append("기준 회계연도가 본문에 명시되지 않았습니다.")

    if not NUMERIC_EVIDENCE.search(text):
        failures.append("수치 근거가 하나도 인용되지 않았습니다.")

    if not RISK_MENTION.search(text):
        failures.append("리스크 관련 언급이 없습니다. 최소한 확인된 위험 요인을 정리하세요.")

    grounding = verify_report(text, facts)
    coverage = grounding.get("coverage")
    if facts and coverage is not None and coverage < MIN_GROUNDING_COVERAGE:
        samples = ", ".join(grounding.get("unmatched_samples", [])[:5])
        failures.append(
            f"본문 수치 중 근거 원장과 일치하는 비율이 {coverage:.0%}로 낮습니다. "
            f"원장에 없는 수치({samples})는 근거를 밝히거나 제거하세요."
        )

    return {
        "passed": not failures,
        "failures": failures,
        "grounding": grounding,
        "feedback": " ".join(failures),
    }
