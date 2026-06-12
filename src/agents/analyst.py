import time
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.config import get_llm, rate_limited, INTER_AGENT_DELAY_SEC
from src.agents.tool_result_compactor import compact_tool_result, summarize_tool_messages
from src.tools.langchain_tools import ANALYST_TOOLS


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


@rate_limited
def analyst_node(state: dict) -> dict:
    """
    Analyst 에이전트: 수집된 데이터를 바탕으로 분석을 수행하고 예측 모델을 활용합니다.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(ANALYST_TOOLS)
    
    company = state["company"]
    user_request = state.get("user_request") or state.get("query_type", "full_report")
    requested_domains = state.get("requested_domains", [])
    analyses = state.get("analyses", [])
    messages = state.get("analyst_messages", [])
    recursion_count = state.get("recursion_count", 0)
    max_recursions = state.get("max_recursions", 3)
    feedback = state.get("feedback")
    
    # 시스템 프롬프트 설정
    if not any(isinstance(m, SystemMessage) for m in messages):
        research_summary = "\n\n".join(
            _content_to_text(a.get("content", ""))
            for a in analyses
            if a.get("agent") in {"supervisor", "researcher"}
        )
        sys_msg = SystemMessage(
            content=f"당신은 '{company}' 기업에 대한 데이터를 분석하는 전문 애널리스트입니다. "
                    f"사용자의 자연어 요청은 '{user_request}'입니다. "
                    f"분석 도메인은 {requested_domains}입니다. "
                    "필요하다면 ML 모델 예측 도구를 사용하여 분석의 깊이를 더하세요. "
                    "최종적으로 투자자 및 경영진에게 유용한 통찰력 있는 분석 리포트를 작성하세요."
        )
        context_msg = HumanMessage(content=f"Supervisor/Researcher 산출물:\n{research_summary}")
        messages = [sys_msg, context_msg] + messages
        
    # Reviewer의 피드백이 있다면 반영
    if feedback:
        feedback_msg = HumanMessage(content=f"Reviewer의 피드백이 있습니다: {feedback}\n이를 반영하여 분석을 수정/보완해주세요.")
        messages.append(feedback_msg)
        # 피드백을 반영했으므로 상태에서 삭제 (다음 턴을 위해)
        feedback_update = {"feedback": None}
    else:
        feedback_update = {}

    # ── 호출 횟수 제어 ──
    if recursion_count >= max_recursions:
        summary = summarize_tool_messages(messages)
        return {
            "analyses": [{
                "agent": "analyst",
                "content": "⚠️ 최대 분석 횟수에 도달하여 tool 사용을 종료합니다.\n\n"
                           "아래 도구 근거를 바탕으로 최종 종합 단계에서 보수적으로 요약해야 합니다.\n\n"
                           f"{summary}",
            }],
            "errors": [f"analyst_react_limit_reached: {recursion_count}/{max_recursions}"],
            "next_agent": "reviewer",
            **feedback_update
        }
        
    # 모델 호출
    response = llm_with_tools.invoke(messages)
    
    # 도구 호출 확인
    if response.tool_calls:
        new_messages = messages + [response]
        for tool_call in response.tool_calls:
            tool_func = next((t for t in ANALYST_TOOLS if t.name == tool_call["name"]), None)
            if tool_func:
                try:
                    tool_result = tool_func.invoke(tool_call["args"])
                except Exception as e:
                    tool_result = f"Error: {str(e)}"
                compact_result = compact_tool_result(tool_call["name"], tool_result)
                
                tool_msg = ToolMessage(
                    content=str(compact_result),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                new_messages.append(tool_msg)
            else:
                new_messages.append(ToolMessage(
                    content=f"Error: Tool {tool_call['name']} not found.",
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                ))
        
        return {
            "analyst_messages": new_messages[len(messages):],
            "recursion_count": recursion_count + 1,
            "next_agent": "analyst",
            **feedback_update
        }
    else:
        # 분석 완료
        content = _content_to_text(response.content)
        finish_reason = response.response_metadata.get("finish_reason") if response.response_metadata else None
        malformed = finish_reason == "MALFORMED_FUNCTION_CALL"
        if malformed or not content.strip():
            tool_summary = summarize_tool_messages(messages)
            content = (
                "⚠️ Analyst LLM이 유효한 분석 본문을 반환하지 못했습니다.\n\n"
                f"- finish_reason: {finish_reason}\n"
                "- 처리: 수집된 tool 근거 요약을 fallback 분석으로 전달합니다.\n\n"
                f"{tool_summary}"
            )
            errors = [f"analyst_invalid_response: finish_reason={finish_reason}"]
        else:
            errors = []
        time.sleep(INTER_AGENT_DELAY_SEC)
        return {
            "analyst_messages": [response],
            "analyses": [{"agent": "analyst", "content": content}],
            "errors": errors,
            "recursion_count": 0, # 다음 에이전트를 위해 초기화
            "next_agent": "reviewer", # 다음 단계는 품질 검토 (Reviewer)
            **feedback_update
        }
