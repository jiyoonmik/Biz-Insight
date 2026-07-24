import json

from src.agents.review_checks import run_deterministic_checks
from src.config import get_llm


def reviewer_node(state: dict) -> dict:
    """
    Reviewer 에이전트: Analyst의 분석 결과를 검토하고 품질을 확인합니다.
    분석이 부족하면 피드백을 주어 다시 Analyst로 보내고(Self-Correction),
    충분하면 Synthesizer로 넘깁니다.

    두 단계로 나눠 검토합니다.
    1. 기계적 요건 점검 (코드) — 실패하면 LLM을 부르지 않고 곧바로 반려
    2. 판단이 필요한 검토 (LLM) — 1단계를 통과한 초안만
    """
    company = state["company"]
    user_request = state.get("user_request") or state.get("query_type", "full_report")
    requested_domains = state.get("requested_domains", [])
    revision_count = state.get("revision_count", 0)
    max_revisions = state.get("max_revisions", 1)
    analyses = state.get("analyses", [])
    
    # 마지막으로 Analyst가 작성한 내용을 찾음
    analyst_content = ""
    for a in reversed(analyses):
        if a["agent"] == "analyst":
            analyst_content = a["content"]
            break
            
    if not analyst_content:
        feedback = "Analyst 산출물이 비어 있어 최종 리포트 품질 검증을 수행할 수 없습니다."
        if revision_count < max_revisions:
            return {
                "feedback": feedback,
                "revision_count": revision_count + 1,
                "recursion_count": 0,
                "max_recursions": 1,
                "next_agent": "analyst",
                "errors": ["reviewer_empty_analyst_output"],
                "analyses": [{"agent": "reviewer", "content": f"🔄 검토 피드백: {feedback}"}],
            }
        return {
            "next_agent": "synthesis",
            "errors": ["reviewer_empty_analyst_output"],
            "analyses": [{"agent": "reviewer", "content": f"⚠️ 추가 보완 필요: {feedback}"}],
        }
        
    # ── 1단계: 기계적 요건 점검 (LLM 호출 없음) ──
    checks = run_deterministic_checks(analyst_content, state.get("evidence", []))
    if not checks["passed"]:
        feedback = checks["feedback"]
        detail = "\n".join(f"- {item}" for item in checks["failures"])
        if revision_count < max_revisions:
            return {
                "feedback": feedback,
                "revision_count": revision_count + 1,
                "recursion_count": 0,
                "max_recursions": 2,
                "next_agent": "analyst",
                "analyses": [{
                    "agent": "reviewer",
                    "content": f"🔄 기계적 요건 미충족으로 반려 (LLM 검토 생략)\n{detail}",
                }],
            }
        return {
            "next_agent": "synthesis",
            "errors": [f"reviewer_deterministic_checks_failed: {len(checks['failures'])}건"],
            "analyses": [{
                "agent": "reviewer",
                "content": f"⚠️ 추가 보완 필요 (재검토 한도 도달)\n{detail}",
            }],
        }

    # ── 2단계: 판단이 필요한 검토만 LLM에 맡긴다 ──
    llm = get_llm(agent="reviewer")
    prompt = f"""당신은 기업 분석 리포트의 품질을 검증하는 수석 검토자(Reviewer)입니다.
대상 기업: '{company}'
사용자 요청: {user_request}
분석 도메인: {requested_domains}

아래는 애널리스트가 작성한 초안입니다:
---
{analyst_content}
---

형식 요건(stock_code 명시, 기준연도 명시, 수치 인용, 리스크 언급, 근거 원장 대조)은
이미 자동 점검을 통과했습니다. 당신은 **자동 점검이 판정할 수 없는 것만** 보세요.

- 근거와 결론 사이의 논리 전개가 타당한가
- 수치는 있지만 그 수치가 뒷받침하지 못하는 단정을 하지 않았는가
- 회계연도 기준 재무 사실과 스냅샷 기준 동적 시그널(주가/리뷰/거시)을 혼동해 서술하지 않았는가
- 신용등급, 산업평균처럼 출처가 필요한 주장에 출처가 붙어 있는가

검토 결과는 반드시 아래 JSON 형식으로만 응답하세요:
{{
    "status": "PASS" 혹은 "REVISE",
    "feedback": "PASS인 경우 빈 문자열. REVISE인 경우 구체적으로 보완해야 할 점을 1~2문장으로 작성"
}}
"""

    response = llm.invoke(prompt)
    
    # JSON 파싱 (간단히 처리)
    content = response.content.strip()
    # LLM이 markdown block(```json ... ```)으로 감쌌을 경우 제거
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
        
    try:
        result = json.loads(content)
        status = result.get("status", "PASS")
        feedback = result.get("feedback", "")
    except Exception:
        status = "REVISE"
        feedback = "Reviewer 응답 JSON 파싱에 실패했습니다. 분석 본문과 근거를 더 명확히 구조화해야 합니다."
        
    if status == "REVISE" and feedback and revision_count < max_revisions:
        return {
            "feedback": feedback,
            "revision_count": revision_count + 1,
            "recursion_count": 0,
            "max_recursions": 2,
            "next_agent": "analyst", # 다시 Analyst로 돌아감
            "analyses": [{"agent": "reviewer", "content": f"🔄 검토 피드백: {feedback}"}]
        }
    elif status == "REVISE" and feedback:
        return {
            "next_agent": "synthesis",
            "analyses": [{"agent": "reviewer", "content": f"⚠️ 추가 보완 필요: {feedback}\n\n최대 재검토 횟수에 도달하여 현재 분석으로 최종 리포트를 생성합니다."}]
        }
    else:
        return {
            "next_agent": "synthesis", # 문제 없으면 최종 작성으로 넘어감
            "analyses": [{"agent": "reviewer", "content": "✅ 검토 완료: 분석 품질이 양호합니다."}]
        }
