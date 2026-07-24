"""
Supervisor Agent.

사용자의 자연어 요청을 분석해 필요한 분석 도메인, Researcher 도구 후보,
후속 에이전트 흐름을 결정합니다. 비용 예측성을 위해 기본 라우팅은 규칙 기반으로
처리하고, 모호한 요청은 전체 리포트로 보수적으로 확장합니다.
"""
import re


DOMAIN_TOOL_MAP = {
    "overview": ["tool_search_company_by_name", "tool_get_company_knowledge_context", "tool_get_company_info"],
    "financial": [
        "tool_search_company_by_name",
        "tool_get_company_knowledge_context",
        "tool_get_criteria_evidence",
        "tool_get_company_info",
        "tool_get_financial_data",
        "tool_get_industry_average",
    ],
    "credit": [
        "tool_search_company_by_name",
        "tool_get_company_knowledge_context",
        "tool_get_credit_rating_history",
        "tool_get_criteria_evidence",
        "tool_get_company_info",
        "tool_get_financial_data",
        "tool_get_credit_data",
        "tool_get_credit_rank",
        "tool_get_industry_average",
    ],
    "investment": [
        "tool_search_company_by_name",
        "tool_get_company_knowledge_context",
        "tool_get_stock_summary",
        "tool_get_company_info",
        "tool_get_financial_data",
        "tool_get_investment_data",
        "tool_get_stock_data",
    ],
    "review": [
        "tool_search_company_by_name",
        "tool_get_company_knowledge_context",
        "tool_get_review_summary",
        "tool_search_review_evidence",
        "tool_get_company_info",
        "tool_get_employee_reviews",
    ],
    "risk": [
        "tool_search_company_by_name",
        "tool_get_company_knowledge_context",
        "tool_get_criteria_evidence",
        "tool_get_credit_rating_history",
        "tool_get_stock_summary",
        "tool_get_review_summary",
        "tool_search_review_evidence",
        "tool_get_macro_context",
        "tool_get_company_info",
        "tool_get_financial_data",
        "tool_get_credit_data",
        "tool_get_credit_rank",
        "tool_get_stock_data",
        "tool_get_employee_reviews",
    ],
}

QUERY_TYPE_DOMAINS = {
    "overview": ["overview"],
    "financial": ["financial"],
    "credit": ["credit"],
    "investment": ["investment"],
    "review": ["review"],
    "risk": ["risk"],
    "full_report": ["overview", "financial", "credit", "investment", "review", "risk"],
}

# 도메인을 확정할 수 있는 신호. 이 단어가 잡히면 요청이 그 도메인을 명시했다고 본다.
DOMAIN_KEYWORDS = {
    "overview": [
        "개요", "기업 정보", "대표", "제품", "업종", "요약", "사업 구조",
        "overview", "profile",
    ],
    "financial": [
        "재무", "매출", "영업이익", "순이익", "수익성", "성장성", "안정성",
        "부채", "자산", "현금흐름", "실적", "financial", "finance", "profit",
        "profitability", "revenue",
    ],
    "credit": [
        "신용", "등급", "상환", "채무", "부도", "부실", "차입", "credit",
        "rating", "default",
    ],
    "investment": [
        "주가", "밸류에이션", "가치평가", "per", "pbr", "roe",
        "모멘텀", "매수", "매도", "stock", "valuation",
    ],
    "review": [
        "직원", "리뷰", "평판", "문화", "복지", "워라밸", "조직", "퇴사",
        "review", "employee", "culture",
    ],
    "risk": [
        "리스크", "위험", "취약", "악재", "변동성", "불확실", "위기",
        "risk", "downside", "volatility",
    ],
}

# 도메인을 확정하기엔 신호가 약한 단어. 라우팅 결정에서 의도적으로 제외한다.
#
# "이 회사 어때?"의 '회사'나 "투자해도 될까?"의 '투자'는 도메인 지정이 아니라
# 한국어 요청에서 거의 조사처럼 쓰이는 일반 명사다. 이 단어들을 확정 신호로 두면
# 종합 판단을 원한 요청이 단일 도메인으로 좁혀지고, 사용자는 자기가 요청한 적 없는
# 축소를 당한다. 좁혀서 근거를 빠뜨리는 쪽이 조금 더 수집하는 쪽보다 나쁘다는
# 원칙(README §2.1)을 여기서도 그대로 적용해, 약한 신호뿐이면 전체로 확장한다.
#
# 목록을 지우지 않고 남겨 두는 이유는 '왜 이 단어로는 라우팅하지 않는가'가
# 코드에 남아 있어야 다음 사람이 같은 실수를 되돌리지 않기 때문이다.
WEAK_DOMAIN_KEYWORDS = {
    "overview": ["회사", "기본", "summary"],
    "investment": ["투자", "investment"],
}

# ASCII 키워드는 단어 경계로 매칭한다. 부분문자열로 두면 'per'가 'performance'에,
# 'roe'가 'europe'에 걸린다. 한글은 조사가 붙어 오므로 부분문자열 매칭을 유지한다.
ASCII_KEYWORD = re.compile(r"^[a-z0-9 ]+$")


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _keyword_hit(keyword: str, lowered_text: str) -> bool:
    if ASCII_KEYWORD.match(keyword):
        return re.search(rf"\b{re.escape(keyword)}\b", lowered_text) is not None
    return keyword in lowered_text


def _match_domains(text: str, table: dict[str, list[str]]) -> list[str]:
    lowered = text.lower()
    return [
        domain
        for domain, keywords in table.items()
        if any(_keyword_hit(keyword.lower(), lowered) for keyword in keywords)
    ]


def _domains_from_text(text: str) -> tuple[list[str], list[str]]:
    """(확정 신호 도메인, 약한 신호 도메인)을 함께 돌려준다."""
    return _match_domains(text, DOMAIN_KEYWORDS), _match_domains(text, WEAK_DOMAIN_KEYWORDS)


def _build_plan(query_type: str, user_request: str) -> tuple[list[str], list[str], bool]:
    query_domains = QUERY_TYPE_DOMAINS.get(query_type, [])
    request_domains, weak_domains = _domains_from_text(user_request)

    if query_type == "full_report":
        # 확정 신호가 있을 때만 좁힌다. 약한 신호뿐이면 전체로 확장한다.
        domains = request_domains or QUERY_TYPE_DOMAINS["full_report"]
    elif query_domains:
        # 명시적 query_type은 유지하고, 확정 신호로 잡힌 도메인만 덧붙인다.
        domains = _dedupe(query_domains + request_domains)
    elif request_domains:
        domains = request_domains
    else:
        domains = QUERY_TYPE_DOMAINS["full_report"]

    tools = []
    for domain in domains:
        tools.extend(DOMAIN_TOOL_MAP.get(domain, []))

    allowed_tools = _dedupe(tools)
    needs_analysis = domains != ["overview"]
    return domains, allowed_tools, needs_analysis


def supervisor_node(state: dict) -> dict:
    """
    자연어 요청 기반 Manager/Supervisor 노드.

    Returns:
        requested_domains: 필요한 분석 도메인
        allowed_research_tools: Researcher가 사용할 도구 allow-list
        active_agents: 실행 예정 에이전트
        next_agent: 다음 실행 노드
    """
    query_type = state.get("query_type", "full_report")
    company = state.get("company", "Unknown")
    user_request = state.get("user_request") or query_type

    domains, allowed_tools, needs_analysis = _build_plan(query_type, user_request)
    strong_domains, weak_domains = _domains_from_text(user_request)

    active_agents = ["researcher", "synthesis"]
    if needs_analysis:
        active_agents = ["researcher", "analyst", "reviewer", "synthesis"]
    active_agents.append("verifier")

    # 왜 이 범위가 선택됐는지를 계획서에 남긴다. 라우팅 결과만 남기면
    # 범위가 예상과 다를 때 사용자도 개발자도 원인을 알 수 없다.
    if strong_domains:
        reason = f"요청에서 확정 신호 감지: {', '.join(strong_domains)}"
    elif weak_domains:
        reason = (
            f"약한 신호({', '.join(weak_domains)})만 감지되어 좁히지 않고 전체 도메인으로 확장"
        )
    else:
        reason = "요청에서 도메인 신호를 찾지 못해 전체 도메인으로 확장"

    plan_msg = f"""## 초기 분석 계획
- 대상 기업: {company}
- 사용자 요청: {user_request}
- 추론 도메인: {', '.join(domains)}
- 선택 근거: {reason}
- 수집 도구: {', '.join(allowed_tools)}
- 실행 흐름: {' -> '.join(active_agents)}
"""

    return {
        "active_agents": active_agents,
        "requested_domains": domains,
        "allowed_research_tools": allowed_tools,
        "needs_analysis": needs_analysis,
        "next_agent": "researcher",
        "recursion_count": 0,
        "max_recursions": 4,
        "revision_count": 0,
        "max_revisions": 1,
        "analyses": [{"agent": "supervisor", "content": plan_msg}],
        "errors": [],
    }
