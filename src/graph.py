"""
Biz-Insight LangGraph 메인 그래프 (Multi-Agent Architecture)
- 각 에이전트 노드(Researcher, Analyst, Reviewer, Synthesis)가 자율적으로 동작하며
- 상태(state)의 next_agent 값을 기반으로 동적 라우팅 수행
- Reviewer <-> Analyst 간의 피드백 기반 Self-Correction 루프 포함
"""
from langgraph.graph import StateGraph, START, END

from src.state import BizInsightState

# ── 새로운 에이전트 노드 임포트 ──
from src.agents.supervisor import supervisor_node
from src.agents.researcher import researcher_node
from src.agents.analyst import analyst_node
from src.agents.reviewer import reviewer_node
from src.agents.synthesis import synthesis_node

def dynamic_router(state: dict) -> str:
    """
    현재 상태의 next_agent 값에 따라 다음 실행할 노드를 결정합니다.
    이 라우팅을 통해 ReAct 루프 및 에이전트 간 순환(Cyclic) 호출이 가능해집니다.
    """
    next_agent = state.get("next_agent")
    
    if next_agent == "END" or next_agent is None:
        return END
        
    return next_agent

# ── 그래프 구성 ──
def build_graph():
    """LangGraph 그래프를 구성하고 컴파일합니다."""
    workflow = StateGraph(BizInsightState)

    # 1. 노드 추가
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("synthesis", synthesis_node)

    # 2. 엣지 정의
    # 무조건 supervisor부터 시작
    workflow.add_edge(START, "supervisor")

    # Supervisor 이후 라우팅
    workflow.add_conditional_edges("supervisor", dynamic_router)
    
    # 각 에이전트 실행 후 라우팅 (next_agent 값에 따라 분기)
    # Researcher는 도구 사용 중이면 자신을 반복 호출, 끝나면 Analyst로
    workflow.add_conditional_edges("researcher", dynamic_router)
    
    # Analyst는 도구 사용 중이면 자신을 반복 호출, 끝나면 Reviewer로
    workflow.add_conditional_edges("analyst", dynamic_router)
    
    # Reviewer는 품질 평가 후 통과면 Synthesis로, 반려면 다시 Analyst로
    workflow.add_conditional_edges("reviewer", dynamic_router)
    
    # Synthesis는 최종 결과물 작성 후 종료
    workflow.add_conditional_edges("synthesis", dynamic_router)

    return workflow.compile()


# ── 컴파일된 그래프 (싱글턴) ──
biz_insight_graph = build_graph()


def generate_ai_report(
    company_name: str,
    query_type: str = "full_report",
    user_request: str | None = None,
) -> str:
    """
    외부에서 호출하기 위한 진입점 함수.

    Args:
        company_name: 분석할 기업명
        query_type: 분석 유형 ("full_report", "financial", "credit", "overview", "investment", "review", "risk")
        user_request: 사용자의 자연어 분석 요청

    Returns:
        최종 종합 리포트 (마크다운 문자열)
    """
    request_text = user_request or query_type
    initial_state = {
        "company": company_name,
        "query_type": query_type,
        "user_request": request_text,
        "active_agents": [],
        "current_agent": None,
        "next_agent": None,
        "recursion_count": 0,
        "max_recursions": 5, # 에이전트 내부 루프 최대 횟수 제어용
        "requested_domains": [],
        "allowed_research_tools": [],
        "needs_analysis": True,
        "revision_count": 0,
        "max_revisions": 1,
        "feedback": None,
        "final_report": "",
    }

    try:
        # 설정된 recursion limit 내에서 그래프 실행
        # 그래프 전체 레벨의 깊이 제한을 설정 (ReAct 루프 등 무한 반복 방지)
        result = biz_insight_graph.invoke(initial_state, {"recursion_limit": 30})
        return result.get("final_report", "리포트 생성 실패 (데이터 없음)")
    except Exception as e:
        return f"## ❌ {company_name} 분석 중 예외 발생\n\n상세 오류: {str(e)}"
