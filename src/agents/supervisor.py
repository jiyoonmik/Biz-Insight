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


# ── 예산 (Supervisor가 단독 소유) ──────────────────────────
# 예전에는 각 워커가 다음 워커의 예산을 덮어썼다(researcher가 3, reviewer가 1이나 2).
# 소유자가 없으니 값이 어디서 정해지는지 추적이 안 됐다. 파견하는 쪽이 예산을 준다.
RESEARCH_BUDGET = 4       # Researcher ReAct 턴
RECOLLECT_BUDGET = 2      # 재수집 시 ReAct 턴 (좁은 목표라 짧게)
ANALYSIS_BUDGET = 3       # Analyst ReAct 턴
REVISION_BUDGET = 2       # Reviewer 피드백 반영 재분석 턴
MAX_REVISIONS = 1         # Reviewer -> Analyst 재검토 횟수
MAX_RECOLLECTIONS = 1     # 근거 공백 재수집 횟수


def domains_without_evidence(domains: list[str], evidence: list[dict]) -> list[str]:
    """요청된 도메인 중 근거를 하나도 얻지 못한 것을 찾는다.

    원장의 각 사실은 자기를 만든 도구 이름을 갖고 있으므로, 도메인의 도구 번들과
    교집합이 비면 그 도메인은 '수집 실패'다. 이걸 보지 않으면 Analyst가 데이터
    없는 도메인에 대해 그럴듯한 문장을 쓰게 된다.
    """
    used_tools = {fact.get("tool") for fact in (evidence or []) if isinstance(fact, dict)}
    return [
        domain for domain in domains
        if not (set(DOMAIN_TOOL_MAP.get(domain, [])) & used_tools)
    ]


def _tools_for(domains: list[str]) -> list[str]:
    tools: list[str] = []
    for domain in domains:
        tools.extend(DOMAIN_TOOL_MAP.get(domain, []))
    return _dedupe(tools)


def _initial_plan(state: dict) -> dict:
    """진입 시 1회: 요청을 해석해 범위와 도구를 정한다."""
    query_type = state.get("query_type", "full_report")
    company = state.get("company", "Unknown")
    user_request = state.get("user_request") or query_type

    domains, allowed_tools, needs_analysis = _build_plan(query_type, user_request)
    strong_domains, weak_domains = _domains_from_text(user_request)

    active_agents = ["researcher", "synthesis"] if not needs_analysis else [
        "researcher", "analyst", "reviewer", "synthesis"
    ]
    active_agents.append("verifier")

    # 왜 이 범위가 선택됐는지를 계획서에 남긴다. 결과만 남기면 범위가 예상과
    # 다를 때 사용자도 개발자도 원인을 알 수 없다.
    if strong_domains:
        reason = f"요청에서 확정 신호 감지: {', '.join(strong_domains)}"
    elif weak_domains:
        reason = f"약한 신호({', '.join(weak_domains)})만 감지되어 좁히지 않고 전체 도메인으로 확장"
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
        "planned": True,
        "active_agents": active_agents,
        "requested_domains": domains,
        "allowed_research_tools": allowed_tools,
        "needs_analysis": needs_analysis,
        "next_agent": "researcher",
        "recursion_count": 0,
        "max_recursions": RESEARCH_BUDGET,
        "revision_count": 0,
        "max_revisions": MAX_REVISIONS,
        "recollect_count": 0,
        "gap_handled": False,
        "research_done": False,
        "analysis_done": False,
        "review_verdict": None,
        "synthesis_done": False,
        "verification_done": False,
        "research_directive": None,
        "analyses": [{"agent": "supervisor", "content": plan_msg}],
        "errors": [],
    }


def _dispatch(agent: str, **updates) -> dict:
    return {"next_agent": agent, **updates}


def decide_next(state: dict) -> dict:
    """워커가 한 단계를 마칠 때마다 호출되어 다음 행선지와 예산을 정한다.

    LLM을 쓰지 않는다. 상태만 보고 결정하므로 같은 상태에서 항상 같은 결정이 나오고,
    분기 하나하나를 모델 없이 테스트할 수 있다(README 4절).
    """
    domains = state.get("requested_domains", []) or []
    evidence = state.get("evidence", []) or []
    needs_analysis = state.get("needs_analysis", True)

    # 1) 수집 단계 ─ 근거 공백을 보고 재수집할지, 도메인을 접을지 결정한다.
    if not state.get("research_done"):
        return _dispatch("researcher")

    # 근거 공백은 수집 직후 한 번만 판정한다. `gap_handled`로 닫지 않으면,
    # 접을 도메인이 남지 않는 경우(전부 공백) 같은 분기가 매 홉 반복되어
    # 그래프가 recursion limit까지 돌게 된다.
    missing = [] if state.get("gap_handled") else domains_without_evidence(domains, evidence)
    if missing:
        recollect_count = state.get("recollect_count", 0)
        if recollect_count < MAX_RECOLLECTIONS:
            # 좁힌 allow-list로 한 번 더. 계획을 고정해 두면 수집 실패가
            # 그대로 하류로 흘러 근거 없는 분석이 된다.
            focus = ", ".join(missing)
            return _dispatch(
                "researcher",
                allowed_research_tools=_tools_for(missing),
                research_directive=(
                    f"다음 도메인에서 근거를 전혀 얻지 못했습니다: {focus}. "
                    "해당 도메인 도구만 사용해 다시 수집하세요. "
                    "데이터가 없으면 없다고 명시하고 종료하세요."
                ),
                research_done=False,
                recollect_count=recollect_count + 1,
                recursion_count=0,
                max_recursions=RECOLLECT_BUDGET,
                analyses=[{
                    "agent": "supervisor",
                    "content": f"🔁 계획 수정: 근거를 얻지 못한 도메인({focus}) 재수집 지시",
                }],
            )

        # 재수집도 실패 ─ 계획에서 접고 사유를 남긴다. 조용히 넘기면
        # Analyst가 데이터 없는 도메인을 상상해서 쓴다.
        remaining = [domain for domain in domains if domain not in missing]
        dropped = [{"domain": domain, "reason": "재수집 후에도 근거 0건"} for domain in missing]
        if not remaining:
            # 전부 비었으면 접을 것이 없다. 있는 그대로 진행하고 리포트에 남긴다.
            return _dispatch(
                "analyst" if needs_analysis else "synthesis",
                recursion_count=0,
                max_recursions=ANALYSIS_BUDGET,
                dropped_domains=dropped,
                gap_handled=True,
                errors=[f"supervisor_no_evidence_for_any_domain: {', '.join(missing)}"],
                analyses=[{
                    "agent": "supervisor",
                    "content": f"⚠️ 모든 요청 도메인({', '.join(missing)})에서 근거를 얻지 못했습니다. "
                               "수집된 범위 안에서만 서술하도록 진행합니다.",
                }],
            )
        return _dispatch(
            "analyst" if needs_analysis else "synthesis",
            requested_domains=remaining,
            dropped_domains=dropped,
            gap_handled=True,
            recursion_count=0,
            max_recursions=ANALYSIS_BUDGET,
            errors=[f"supervisor_dropped_domains: {', '.join(missing)}"],
            analyses=[{
                "agent": "supervisor",
                "content": f"✂️ 계획 축소: 근거를 얻지 못한 도메인({', '.join(missing)})을 분석 범위에서 제외. "
                           f"남은 범위: {', '.join(remaining)}",
            }],
        )

    # 공백 없이 통과 ─ 판정이 끝났음을 표시하고 분석 단계로 넘긴다.
    if not state.get("gap_handled"):
        return _dispatch(
            "analyst" if needs_analysis else "synthesis",
            gap_handled=True,
            recursion_count=0,
            max_recursions=ANALYSIS_BUDGET,
        )

    # 2) 분석 단계
    if needs_analysis:
        if not state.get("analysis_done"):
            return _dispatch("analyst", recursion_count=0, max_recursions=ANALYSIS_BUDGET)

        verdict = state.get("review_verdict")
        if verdict is None:
            return _dispatch("reviewer")

        if verdict == "REVISE":
            revision_count = state.get("revision_count", 0)
            if revision_count < state.get("max_revisions", MAX_REVISIONS):
                return _dispatch(
                    "analyst",
                    analysis_done=False,
                    review_verdict=None,
                    revision_count=revision_count + 1,
                    recursion_count=0,
                    max_recursions=REVISION_BUDGET,
                )
            # 재검토 예산 소진 ─ 실패로 처리하지 않고 현재 분석으로 진행한다.
            # 빈 화면보다 미완성임을 아는 리포트가 낫다.
            #
            # 판정을 EXHAUSTED로 바꿔 두는 것이 중요하다. REVISE로 남겨 두면
            # 이 분기가 매 홉 다시 걸려 synthesis_done을 무시한 채 종합을 무한히
            # 재파견한다(실제로 그렇게 돌다 recursion limit에 걸렸다).
            return _dispatch(
                "synthesis",
                review_verdict="EXHAUSTED",
                analyses=[{
                    "agent": "supervisor",
                    "content": f"⚠️ 재검토 예산({state.get('max_revisions', MAX_REVISIONS)}회) 소진. "
                               "현재 분석으로 최종 리포트를 생성합니다. "
                               f"미해결 피드백: {state.get('feedback') or '없음'}",
                }],
            )

    # 3) 종합과 검증
    if not state.get("synthesis_done"):
        return _dispatch("synthesis")
    if not state.get("verification_done"):
        return _dispatch("verifier")
    return _dispatch("END")


def supervisor_node(state: dict) -> dict:
    """재진입형 Supervisor.

    모든 워커가 한 단계를 마치면 여기로 돌아온다. 진입 시 1회는 계획을 세우고,
    이후에는 관측된 결과를 보고 다음 행선지와 예산을 정한다. 예전에는 진입 시
    한 번만 실행되는 정적 플래너였고, 각 워커가 다음 행선지와 예산을 스스로
    정했다. 그래서 수집이 실패해도 계획에 반영할 주체가 없었다.
    """
    if not state.get("planned"):
        return _initial_plan(state)
    return decide_next(state)
