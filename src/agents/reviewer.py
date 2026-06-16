import time
import json
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import get_llm, rate_limited, INTER_AGENT_DELAY_SEC

@rate_limited
def reviewer_node(state: dict) -> dict:
    """
    Reviewer 에이전트: Analyst의 분석 결과를 검토하고 품질을 확인합니다.
    분석이 부족하면 피드백을 주어 다시 Analyst로 보내고(Self-Correction),
    충분하면 Synthesizer로 넘깁니다.
    """
    llm = get_llm()
    
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
        
    prompt = f"""당신은 기업 분석 리포트의 품질을 검증하는 수석 검토자(Reviewer)입니다.
대상 기업: '{company}'
사용자 요청: {user_request}
분석 도메인: {requested_domains}

아래는 애널리스트가 작성한 초안입니다:
---
{analyst_content}
---

이 분석이 투자자에게 제공될 수 있을 만큼 객관적이고 논리적인지 검토하세요.
특히 아래 품질 기준을 반드시 확인하세요.
- 기업명만으로 단정하지 않고 stock_code가 확정되어 있는가
- canonical context/재무제표의 기준 회계연도와 주가/리뷰/거시지표 같은 동적 데이터 기준일을 구분했는가
- 평가기준별 근거(수익성, 유동성, 재무부담, 현금창출력, 계속기업 등)가 수치와 기간을 포함하는가
- 신용등급, 산업평균, 위험 신호가 근거 없이 주장되지 않았는가
- 주가/직원 리뷰 원문을 재무제표 사실처럼 혼동하지 않았는가

검토 결과는 반드시 아래 JSON 형식으로만 응답하세요:
{{
    "status": "PASS" 혹은 "REVISE",
    "feedback": "PASS인 경우 빈 문자열. REVISE인 경우 구체적으로 보완해야 할 점을 1~2문장으로 작성"
}}
"""
    
    response = llm.invoke(prompt)
    time.sleep(INTER_AGENT_DELAY_SEC)
    
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
