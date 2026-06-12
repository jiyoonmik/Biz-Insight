"""
Supervisor Agent.

사용자의 자연어 요청을 분석해 필요한 분석 도메인, Researcher 도구 후보,
후속 에이전트 흐름을 결정합니다. 비용 예측성을 위해 기본 라우팅은 규칙 기반으로
처리하고, 모호한 요청은 전체 리포트로 보수적으로 확장합니다.
"""
from src.config import rate_limited


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

DOMAIN_KEYWORDS = {
    "overview": [
        "개요", "기본", "회사", "기업 정보", "대표", "제품", "업종", "요약",
        "overview", "profile", "summary",
    ],
    "financial": [
        "재무", "매출", "영업이익", "순이익", "수익성", "성장성", "안정성",
        "부채", "자산", "현금흐름", "financial", "finance", "profit", "revenue",
    ],
    "credit": [
        "신용", "등급", "상환", "채무", "부도", "부실", "차입", "credit",
        "rating", "default",
    ],
    "investment": [
        "투자", "주가", "밸류에이션", "가치평가", "per", "pbr", "roe",
        "모멘텀", "매수", "매도", "investment", "stock", "valuation",
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


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _domains_from_text(text: str) -> list[str]:
    lowered = text.lower()
    domains = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword.lower() in lowered for keyword in keywords):
            domains.append(domain)
    return domains


def _build_plan(query_type: str, user_request: str) -> tuple[list[str], list[str], bool]:
    query_domains = QUERY_TYPE_DOMAINS.get(query_type, [])
    request_domains = _domains_from_text(user_request)

    if query_type == "full_report" and request_domains:
        domains = request_domains
    elif query_domains and query_type != "full_report":
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


@rate_limited
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
    active_agents = ["researcher", "synthesis"]
    if needs_analysis:
        active_agents = ["researcher", "analyst", "reviewer", "synthesis"]

    plan_msg = f"""## 초기 분석 계획
- 대상 기업: {company}
- 사용자 요청: {user_request}
- 추론 도메인: {', '.join(domains)}
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
