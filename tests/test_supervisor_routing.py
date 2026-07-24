"""Supervisor 라우팅 테스트.

라우터를 LLM 없이 규칙으로 만든 선택의 직접적인 이득이 이 파일이다.
LLM 호출 없이 밀리초 단위로 라우팅 정책 전체를 회귀 검증할 수 있다.
"""
import unittest

from src.agents.supervisor import DOMAIN_TOOL_MAP, _build_plan, supervisor_node


class SupervisorRoutingTests(unittest.TestCase):
    def test_explicit_query_type_selects_its_domain(self) -> None:
        domains, tools, needs_analysis = _build_plan("credit", "신용등급을 알려줘")
        self.assertEqual(domains, ["credit"])
        self.assertTrue(needs_analysis)
        self.assertEqual(set(tools), set(DOMAIN_TOOL_MAP["credit"]))

    def test_natural_language_narrows_full_report(self) -> None:
        """명시적 신호가 있으면 사용자의 의도가 기본값(전체)을 이긴다."""
        domains, _, _ = _build_plan("full_report", "리스크 위주로 봐줘")
        self.assertEqual(domains, ["risk"])

    def test_weak_signal_alone_expands_instead_of_narrowing(self) -> None:
        """'회사'/'투자' 같은 일반 명사는 도메인을 확정하지 못한다.

        좁혀서 근거를 빠뜨리는 쪽이 더 수집하는 쪽보다 나쁘다는 원칙.
        """
        for request in ["이 회사 어때?", "투자해도 될까?"]:
            with self.subTest(request=request):
                domains, _, _ = _build_plan("full_report", request)
                self.assertEqual(len(domains), 6, f"{request} -> {domains}")

    def test_weak_signal_does_not_pollute_explicit_query_type(self) -> None:
        domains, _, _ = _build_plan("risk", "이 회사가 무너질 가능성이 있는지 알려줘")
        self.assertEqual(domains, ["risk"])

    def test_ascii_keyword_uses_word_boundary(self) -> None:
        """'per'가 'performance'에 부분문자열로 걸리면 안 된다."""
        domains, _, _ = _build_plan("full_report", "analyze financial performance")
        self.assertEqual(domains, ["financial"])

    def test_unknown_signal_falls_back_to_all_domains(self) -> None:
        domains, _, _ = _build_plan("full_report", "여기 들어가도 괜찮을지 판단해줘")
        self.assertEqual(len(domains), 6)

    def test_multi_domain_request(self) -> None:
        domains, _, _ = _build_plan("financial", "재무와 함께 리스크도 봐줘")
        self.assertEqual(set(domains), {"financial", "risk"})

    def test_overview_skips_analysis_stage(self) -> None:
        """개요 요약에 품질 게이트 2회를 태우지 않는다."""
        domains, _, needs_analysis = _build_plan("overview", "사업 구조를 알려줘")
        self.assertEqual(domains, ["overview"])
        self.assertFalse(needs_analysis)

    def test_tool_allow_list_is_scoped_to_selected_domains(self) -> None:
        """allow-list가 실제로 도구를 잘라내는지 확인한다."""
        _, overview_tools, _ = _build_plan("overview", "개요만")
        _, full_tools, _ = _build_plan("full_report", "전부 분석해줘")
        self.assertLess(len(overview_tools), len(full_tools))
        self.assertNotIn("tool_get_stock_summary", overview_tools)
        self.assertIn("tool_search_company_by_name", overview_tools)

    def test_node_emits_plan_with_reason_and_verifier_in_path(self) -> None:
        state = supervisor_node({
            "company": "테스트기업",
            "query_type": "full_report",
            "user_request": "이 회사 어때?",
        })
        self.assertEqual(state["next_agent"], "researcher")
        self.assertIn("verifier", state["active_agents"])
        plan = state["analyses"][0]["content"]
        self.assertIn("약한 신호", plan)


if __name__ == "__main__":
    unittest.main()
