"""Reviewer 기계적 사전 점검 테스트.

핵심 주장 두 가지를 고정한다.
1. 코드가 확인할 수 있는 요건은 LLM에 묻지 않고 코드가 판정한다.
2. 그 판정에서 떨어진 초안은 LLM 호출 없이 곧바로 반려된다.
"""
import unittest
from unittest import mock

from src.agents.review_checks import run_deterministic_checks
from src.agents.reviewer import reviewer_node
from tests.fakes import ANALYST_DRAFT, ANALYST_DRAFT_LEDGER


GOOD_DRAFT = ANALYST_DRAFT
LEDGER = ANALYST_DRAFT_LEDGER


class DeterministicCheckTests(unittest.TestCase):
    def test_complete_draft_passes(self) -> None:
        result = run_deterministic_checks(GOOD_DRAFT, LEDGER)
        self.assertTrue(result["passed"], result["failures"])

    def test_missing_stock_code_is_caught(self) -> None:
        draft = GOOD_DRAFT.replace("(095570)", "")
        result = run_deterministic_checks(draft, LEDGER)
        self.assertFalse(result["passed"])
        self.assertTrue(any("stock_code" in f for f in result["failures"]))

    def test_missing_fiscal_year_is_caught(self) -> None:
        draft = GOOD_DRAFT.replace("2022 회계연도 기준", "최근")
        result = run_deterministic_checks(draft, LEDGER)
        self.assertFalse(result["passed"])
        self.assertTrue(any("회계연도" in f for f in result["failures"]))

    def test_missing_risk_mention_is_caught(self) -> None:
        # 공유 픽스처에 문자열 치환을 거는 대신 독립 초안을 쓴다.
        # 픽스처 문구가 바뀌면 치환이 조용히 무효가 되어 테스트가 통과해 버린다.
        draft = (
            "AJ네트웍스(095570)의 2022 회계연도 기준 분석입니다. "
            "매출액은 1조 2,345억원이며 부채비율은 312.5%로 나타났습니다. "
            "유동비율은 95%로 집계되었고 영업현금흐름은 개선 추세를 보였습니다. "
            "전반적으로 안정적인 흐름을 유지하고 있으며 추가로 지적할 항목은 없습니다. "
            "향후에도 현재 수준의 재무 구조가 유지될 것으로 보입니다."
        )
        result = run_deterministic_checks(draft, LEDGER)
        self.assertFalse(result["passed"])
        self.assertTrue(any("리스크" in f for f in result["failures"]))

    def test_too_short_draft_is_caught(self) -> None:
        result = run_deterministic_checks("095570 2022년 리스크 있음", LEDGER)
        self.assertFalse(result["passed"])
        self.assertTrue(any("짧" in f for f in result["failures"]))

    def test_numbers_detached_from_ledger_are_caught(self) -> None:
        """원장과 전혀 맞지 않는 수치만 늘어놓은 초안은 반려된다."""
        draft = (
            "AJ네트웍스(095570)의 2022 회계연도 분석입니다. "
            "매출액은 7,777억원, 부채비율은 88.8%, 유동비율은 44.4%로 추정됩니다. "
            "영업이익률은 33.3%이며 자기자본비율은 22.2%로 파악됩니다. "
            "주요 리스크는 추정치 기반 판단에 따른 불확실성입니다."
        )
        result = run_deterministic_checks(draft, LEDGER)
        self.assertFalse(result["passed"])
        self.assertTrue(any("원장" in f for f in result["failures"]))

    def test_empty_ledger_does_not_trigger_grounding_failure(self) -> None:
        """원장이 비었으면 대조할 수 없을 뿐, 초안 탓으로 돌리지 않는다."""
        result = run_deterministic_checks(GOOD_DRAFT, [])
        self.assertTrue(result["passed"], result["failures"])


class ReviewerGateTests(unittest.TestCase):
    def _state(self, draft: str) -> dict:
        return {
            "company": "AJ네트웍스",
            "query_type": "financial",
            "user_request": "재무를 분석해줘",
            "requested_domains": ["financial"],
            "revision_count": 0,
            "max_revisions": 1,
            "evidence": LEDGER,
            "analyses": [{"agent": "analyst", "content": draft}],
        }

    def test_failing_draft_is_rejected_without_calling_the_llm(self) -> None:
        with mock.patch("src.agents.reviewer.get_llm") as get_llm:
            state = reviewer_node(self._state("너무 짧은 초안"))
            get_llm.assert_not_called()
        # Reviewer는 판정만 보고한다. 재분석 여부와 예산은 Supervisor가 정한다.
        self.assertEqual(state["next_agent"], "supervisor")
        self.assertEqual(state["review_verdict"], "REVISE")
        self.assertIn("LLM 검토 생략", state["analyses"][0]["content"])
        self.assertTrue(any("deterministic_checks_failed" in e for e in state["errors"]))

    def test_passing_draft_reaches_the_llm_stage(self) -> None:
        class Stub:
            def invoke(self, prompt):
                Stub.prompt = prompt
                return type("R", (), {"content": '{"status": "PASS", "feedback": ""}'})()

        with mock.patch("src.agents.reviewer.get_llm", return_value=Stub()) as get_llm:
            state = reviewer_node(self._state(GOOD_DRAFT))
            get_llm.assert_called_once()
        self.assertEqual(state["next_agent"], "supervisor")
        self.assertEqual(state["review_verdict"], "PASS")
        # LLM 프롬프트는 판단 항목만 남고 형식 요건은 빠져 있어야 한다
        self.assertIn("자동 점검이 판정할 수 없는 것만", Stub.prompt)

    def test_reviewer_does_not_own_the_revision_budget(self) -> None:
        """예산 소진 판단은 Supervisor 몫이다. Reviewer는 몇 번째든 같은 판정을 낸다."""
        state = self._state("너무 짧은 초안")
        state["revision_count"] = 99
        with mock.patch("src.agents.reviewer.get_llm") as get_llm:
            result = reviewer_node(state)
            get_llm.assert_not_called()
        self.assertEqual(result["review_verdict"], "REVISE")
        self.assertEqual(result["next_agent"], "supervisor")
        self.assertNotIn("revision_count", result)


if __name__ == "__main__":
    unittest.main()
