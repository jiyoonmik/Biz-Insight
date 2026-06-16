"""
Biz-Insight LangGraph 메인 그래프 (Multi-Agent Architecture)
- 각 에이전트 노드(Researcher, Analyst, Reviewer, Synthesis)가 자율적으로 동작하며
- 상태(state)의 next_agent 값을 기반으로 동적 라우팅 수행
- Reviewer <-> Analyst 간의 피드백 기반 Self-Correction 루프 포함
"""
from langgraph.graph import StateGraph, START, END
import ast
from typing import Any

from src.state import BizInsightState

# ── 새로운 에이전트 노드 임포트 ──
from src.agents.supervisor import supervisor_node
from src.agents.researcher import researcher_node
from src.agents.analyst import analyst_node
from src.agents.reviewer import reviewer_node
from src.agents.synthesis import synthesis_node
from src.config import langsmith_project_name, langsmith_tracing_enabled

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


def _build_invoke_config(company_name: str, query_type: str, user_request: str) -> dict:
    """Build LangGraph runtime config, including LangSmith trace metadata."""
    return {
        "recursion_limit": 30,
        "run_name": "biz-insight-report",
        "tags": [
            "biz-insight",
            "streamlit",
            f"query_type:{query_type}",
        ],
        "metadata": {
            "company": company_name,
            "query_type": query_type,
            "user_request": user_request,
        },
    }


def _invoke_with_optional_tracing(initial_state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if not langsmith_tracing_enabled():
        return biz_insight_graph.invoke(initial_state, config)

    from langchain_core.tracers.context import tracing_v2_enabled

    with tracing_v2_enabled(project_name=langsmith_project_name(), tags=config.get("tags")):
        return biz_insight_graph.invoke(initial_state, config)


def _parse_tool_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return ast.literal_eval(content)
    except Exception:
        return content


def _collect_sources(value: Any) -> set[str]:
    sources: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"source", "source_file"} and item:
                if isinstance(item, str):
                    sources.update(part.strip() for part in item.split(";") if part.strip())
                else:
                    sources.add(str(item))
            else:
                sources.update(_collect_sources(item))
    elif isinstance(value, list):
        for item in value:
            sources.update(_collect_sources(item))
    return sources


def _tool_messages(messages: list[Any]) -> list[dict[str, Any]]:
    calls = []
    for message in messages or []:
        name = getattr(message, "name", None)
        if not name or getattr(message, "type", None) != "tool":
            continue
        parsed = _parse_tool_content(getattr(message, "content", ""))
        calls.append(
            {
                "tool": name,
                "sources": sorted(_collect_sources(parsed)),
                "preview": str(parsed)[:1200],
            }
        )
    return calls


def build_execution_trace(result: dict[str, Any]) -> dict[str, Any]:
    analyses = result.get("analyses", []) or []
    agents = []
    for item in analyses:
        if isinstance(item, dict) and item.get("agent") not in agents:
            agents.append(item.get("agent"))

    tool_calls = _tool_messages(result.get("researcher_messages", []))
    tool_calls.extend(_tool_messages(result.get("analyst_messages", [])))

    sources = set()
    for call in tool_calls:
        sources.update(call.get("sources", []))

    return {
        "agents": agents,
        "requested_domains": result.get("requested_domains", []),
        "allowed_research_tools": result.get("allowed_research_tools", []),
        "tool_calls": tool_calls,
        "data_sources": sorted(sources),
        "errors": result.get("errors", []),
    }


def generate_ai_report_result(
    company_name: str,
    query_type: str = "full_report",
    user_request: str | None = None,
) -> dict[str, Any]:
    request_text = user_request or query_type
    initial_state = {
        "company": company_name,
        "query_type": query_type,
        "user_request": request_text,
        "active_agents": [],
        "current_agent": None,
        "next_agent": None,
        "recursion_count": 0,
        "max_recursions": 5,
        "requested_domains": [],
        "allowed_research_tools": [],
        "needs_analysis": True,
        "revision_count": 0,
        "max_revisions": 1,
        "feedback": None,
        "final_report": "",
    }

    try:
        config = _build_invoke_config(company_name, query_type, request_text)
        result = _invoke_with_optional_tracing(initial_state, config)
        report = result.get("final_report", "리포트 생성 실패 (데이터 없음)")
        return {
            "report": report,
            "trace": build_execution_trace(result),
            "state": result,
        }
    except Exception as e:
        error_report = f"## ❌ {company_name} 분석 중 예외 발생\n\n상세 오류: {str(e)}"
        return {
            "report": error_report,
            "trace": {
                "agents": [],
                "requested_domains": [],
                "allowed_research_tools": [],
                "tool_calls": [],
                "data_sources": [],
                "errors": [str(e)],
            },
            "state": {},
        }


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
    return generate_ai_report_result(company_name, query_type, user_request)["report"]
