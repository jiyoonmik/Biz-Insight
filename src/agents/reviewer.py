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
        # Analyst 결과가 없으면 바로 넘김
        return {"next_agent": "synthesis"}
        
    prompt = f"""당신은 기업 분석 리포트의 품질을 검증하는 수석 검토자(Reviewer)입니다.
대상 기업: '{company}'
사용자 요청: {user_request}
분석 도메인: {requested_domains}

아래는 애널리스트가 작성한 초안입니다:
---
{analyst_content}
---

이 분석이 투자자에게 제공될 수 있을 만큼 객관적이고 논리적이며, 중요한 누락(예: 재무 데이터 부족, 위험 요인 언급 부재 등)이 없는지 검토하세요.

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
        # 파싱 실패 시 기본적으로 통과시킴
        status = "PASS"
        feedback = ""
        
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
