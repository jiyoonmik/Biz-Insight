from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from src.config import get_llm
from src.agents.evidence import extract_facts
from src.agents.tool_result_compactor import (
    compact_message_history,
    compact_tool_result,
    summarize_tool_messages,
)
from src.tools.langchain_tools import RESEARCHER_TOOLS


def _message_content_to_text(content) -> str:
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


def researcher_node(state: dict) -> dict:
    """
    Researcher 에이전트: 도구를 사용해 기업 데이터를 수집합니다.
    """
    llm = get_llm(agent="researcher")
    
    company = state["company"]
    query_type = state["query_type"]
    user_request = state.get("user_request") or query_type
    requested_domains = state.get("requested_domains", [])
    allowed_tool_names = state.get("allowed_research_tools", [])
    available_tools = [
        tool for tool in RESEARCHER_TOOLS
        if not allowed_tool_names or tool.name in allowed_tool_names
    ]
    llm_with_tools = llm.bind_tools(available_tools)

    messages = state.get("researcher_messages", [])
    recursion_count = state.get("recursion_count", 0)
    max_recursions = state.get("max_recursions", 5)
    
    # 시스템 프롬프트 설정 (최초 실행 시)
    if not any(isinstance(m, SystemMessage) for m in messages):
        sys_msg = SystemMessage(
            content=f"당신은 '{company}' 기업에 대한 데이터를 수집하는 전문 리서처입니다. "
                    f"사용자의 자연어 요청은 '{user_request}'입니다. "
                    f"추론된 분석 도메인은 {requested_domains}입니다. "
                    f"반드시 허용된 도구({allowed_tool_names}) 안에서만 필요한 데이터를 수집하세요. "
                    "기업명만으로 분석하지 말고 먼저 alias 검색 도구로 stock_code를 확정한 뒤, "
                    "canonical context와 dynamic signal 도구를 우선 사용하세요. "
                    "CSV 조회 도구는 context/API 계층에서 찾지 못한 경우의 fallback으로만 사용하세요. "
                    "요청과 관련 없는 도구는 호출하지 마세요. "
                    "모든 필요한 데이터를 충분히 수집했다고 판단되면, 최종 수집 결과를 요약해서 답변하세요."
        )
        request_msg = HumanMessage(content=f"분석 대상 기업: {company}\n분석 요청: {user_request}")
        messages = [sys_msg, request_msg] + messages

    # Supervisor가 근거 공백을 보고 재수집을 지시했다면 목표를 좁혀 이어간다.
    directive = state.get("research_directive")
    directive_update: dict = {}
    if directive:
        messages = messages + [HumanMessage(content=directive)]
        directive_update = {"research_directive": None}
    
    # ── 호출 횟수(Recursion) 제어 ──
    if recursion_count >= max_recursions:
        summary = summarize_tool_messages(messages)
        return {
            "analyses": [{
                "agent": "researcher",
                "content": "⚠️ 최대 탐색 횟수에 도달하여 tool 사용을 종료합니다.\n\n"
                           "아래는 지금까지 성공적으로 수집한 데이터의 압축 요약입니다.\n\n"
                           f"{summary}",
            }],
            "errors": [f"researcher_react_limit_reached: {recursion_count}/{max_recursions}"],
            "research_done": True,
            "next_agent": "supervisor",
            **directive_update,
        }
        
    # 모델 호출. 히스토리는 원장 참조로 압축한 사본을 보낸다(§2.5).
    response = llm_with_tools.invoke(
        compact_message_history(messages, ledger=state.get("evidence", []))
    )
    
    # 도구 호출이 있는 경우
    if response.tool_calls:
        new_messages = messages + [response]
        collected_facts = []
        for tool_call in response.tool_calls:
            # 도구 이름에 맞는 함수 찾기
            tool_func = next((t for t in available_tools if t.name == tool_call["name"]), None)
            if tool_func:
                try:
                    tool_result = tool_func.invoke(tool_call["args"])
                except Exception as e:
                    tool_result = f"Error: {str(e)}"
                compact_result = compact_tool_result(tool_call["name"], tool_result)
                # 요약 문장이 되기 전에 구조화된 사실을 원장으로 빼돌린다.
                collected_facts.extend(extract_facts(tool_call["name"], compact_result))

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
        
        # 도구 실행 후 다시 모델을 호출하기 위해 recursion 증가 후 현재 노드 반환
        return {
            "researcher_messages": new_messages[len(messages):],
            "evidence": collected_facts,
            "recursion_count": recursion_count + 1,
            # 도구 루프는 supervisor를 거치지 않는다. 허브는 단계 사이를 조율하지
            # 도구 턴 하나하나를 중개하지 않는다.
            "next_agent": "researcher",
            **directive_update,
        }
    else:
        # 도구 호출이 끝나고 최종 요약을 내놓은 경우
        content = _message_content_to_text(response.content)
        finish_reason = response.response_metadata.get("finish_reason") if response.response_metadata else None
        if not content.strip():
            content = "⚠️ Researcher LLM이 빈 응답을 반환했습니다.\n\n" + summarize_tool_messages(messages)
            errors = [f"researcher_empty_response: finish_reason={finish_reason}"]
        else:
            errors = []
        return {
            "researcher_messages": [response],
            "analyses": [{"agent": "researcher", "content": content}],
            "errors": errors,
            "research_done": True,
            "next_agent": "supervisor",
            **directive_update,
        }
