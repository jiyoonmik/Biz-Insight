"""LangGraph 제어 흐름 테스트 (LLM 대역 사용).

검증 대상은 리포트 문장 품질이 아니라 그래프가 설계대로 도는가이다.
- overview 요청이 Analyst/Reviewer를 건너뛰는가
- Reviewer의 REVISE가 실제로 Analyst로 되돌리는가
- ReAct 도구 루프에서 근거 원장이 쌓이는가
- LLM이 깨진 응답을 줬을 때 조용히 통과하지 않고 errors에 남는가

이 네 가지는 모두 LLM 없이 확인할 수 있어야 하며, 그렇지 않으면 그래프를 고칠
때마다 API 키와 수십 초를 지불하게 된다.
"""
import json
import unittest
from unittest import mock

from src.graph import generate_ai_report_result
from tests.fakes import ANALYST_DRAFT, FakeLLM, FakeResponse, tool_call


REPORT_TEXT = "# 리포트\n2022년 매출액은 1조 2,345억원입니다."



def _patch_llms(**per_agent):
    """에이전트별 LLM 대역을 주입한다. 각 노드는 자기 모듈에서 get_llm을 부른다."""
    patchers = [
        mock.patch(f"src.agents.{agent}.get_llm", return_value=llm)
        for agent, llm in per_agent.items()
    ]
    return patchers


class GraphFlowTests(unittest.TestCase):
    def setUp(self) -> None:
        # rate limit 지연은 실행 안정성을 위한 것이라 테스트에서는 제거한다.
        sleep_patcher = mock.patch("time.sleep")
        sleep_patcher.start()
        self.addCleanup(sleep_patcher.stop)

    def _run(self, company, query_type, request, **llms):
        patchers = _patch_llms(**llms)
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return generate_ai_report_result(company, query_type, request)

    def test_overview_skips_analyst_and_reviewer(self) -> None:
        result = self._run(
            "AJ네트웍스", "overview", "사업 구조와 주요 제품을 요약해줘",
            researcher=FakeLLM([FakeResponse(content="기업 개요 수집 완료")]),
            synthesis=FakeLLM([FakeResponse(content=REPORT_TEXT)]),
        )
        agents = result["trace"]["agents"]
        self.assertIn("researcher", agents)
        self.assertIn("synthesis", agents)
        self.assertIn("verifier", agents)
        self.assertNotIn("analyst", agents)
        self.assertNotIn("reviewer", agents)

    def test_reviewer_revise_sends_work_back_to_analyst(self) -> None:
        revise = json.dumps({"status": "REVISE", "feedback": "부채비율 근거 수치를 추가하세요."})
        result = self._run(
            "AJ네트웍스", "financial", "수익성과 재무 안정성을 분석해줘",
            researcher=FakeLLM([FakeResponse(content="재무 데이터 수집 완료")]),
            analyst=FakeLLM([
                FakeResponse(content=ANALYST_DRAFT),
                FakeResponse(content=ANALYST_DRAFT + " 피드백을 반영해 보완했습니다."),
            ]),
            reviewer=FakeLLM([
                FakeResponse(content=revise),
                FakeResponse(content=json.dumps({"status": "PASS", "feedback": ""})),
            ]),
            synthesis=FakeLLM([FakeResponse(content=REPORT_TEXT)]),
        )
        state = result["state"]
        self.assertEqual(state["revision_count"], 1)
        contents = [a["content"] for a in state["analyses"] if a["agent"] == "analyst"]
        self.assertEqual(len(contents), 2, "Analyst가 두 번 실행되어야 한다")
        self.assertTrue(any("검토 피드백" in a["content"] for a in state["analyses"]))

    def test_revision_budget_stops_the_loop(self) -> None:
        """max_revisions=1이므로 두 번째 REVISE는 루프를 더 돌리지 않는다."""
        revise = json.dumps({"status": "REVISE", "feedback": "근거가 여전히 부족합니다."})
        result = self._run(
            "AJ네트웍스", "financial", "수익성을 분석해줘",
            researcher=FakeLLM([FakeResponse(content="수집 완료")]),
            analyst=FakeLLM([FakeResponse(content=ANALYST_DRAFT),
                             FakeResponse(content=ANALYST_DRAFT + " 보완")]),
            reviewer=FakeLLM([FakeResponse(content=revise), FakeResponse(content=revise)]),
            synthesis=FakeLLM([FakeResponse(content=REPORT_TEXT)]),
        )
        state = result["state"]
        self.assertEqual(state["revision_count"], 1)
        # 예산 소진 판단과 그 기록은 Reviewer가 아니라 Supervisor가 남긴다.
        self.assertTrue(
            any(a["agent"] == "supervisor" and "재검토 예산" in a["content"]
                for a in state["analyses"]),
            [a["content"][:40] for a in state["analyses"]],
        )
        self.assertEqual(state["review_verdict"], "EXHAUSTED")
        self.assertIn("verifier", result["trace"]["agents"])

    def test_react_tool_loop_accumulates_evidence_ledger(self) -> None:
        result = self._run(
            "AJ네트웍스", "overview", "기업 개요를 알려줘",
            researcher=FakeLLM([
                FakeResponse(tool_calls=[
                    tool_call("tool_get_company_knowledge_context", {"stock_code": "095570", "year": 2022})
                ]),
                FakeResponse(content="수집 완료"),
            ]),
            synthesis=FakeLLM([FakeResponse(content=REPORT_TEXT)]),
        )
        evidence = result["state"]["evidence"]
        self.assertTrue(evidence, "도구 호출 결과가 원장에 쌓여야 한다")
        self.assertTrue(all("metric" in fact and "kind" in fact for fact in evidence))
        self.assertEqual(result["trace"]["evidence_count"], len(evidence))
        self.assertIn("근거 대조", result["report"])

    def test_malformed_llm_response_is_recorded_not_swallowed(self) -> None:
        result = self._run(
            "AJ네트웍스", "financial", "재무를 분석해줘",
            researcher=FakeLLM([FakeResponse(content="수집 완료")]),
            analyst=FakeLLM([FakeResponse(content="", finish_reason="MALFORMED_FUNCTION_CALL")]),
            reviewer=FakeLLM([FakeResponse(content=json.dumps({"status": "PASS", "feedback": ""}))]),
            synthesis=FakeLLM([FakeResponse(content=REPORT_TEXT)]),
        )
        errors = result["trace"]["errors"]
        self.assertTrue(any("analyst_invalid_response" in err for err in errors))
        # 실패해도 파이프라인은 끝까지 진행되어 리포트를 낸다
        self.assertIn("verifier", result["trace"]["agents"])

    def test_supervisor_replans_when_research_yields_no_evidence(self) -> None:
        """그래프 수준에서 재계획이 실제로 도는지 확인한다.

        Researcher가 도구를 한 번도 못 부르면 원장이 비고, 예전 구조에서는 그대로
        Analyst로 흘러 근거 없는 분석이 나왔다. 이제는 Supervisor가 재수집을
        지시하고, 그래도 없으면 도메인을 접고 사유를 남긴다.
        """
        result = self._run(
            "AJ네트웍스", "financial", "수익성을 분석해줘",
            researcher=FakeLLM([
                FakeResponse(content="수집 실패"),
                FakeResponse(content="재수집도 실패"),
            ]),
            analyst=FakeLLM([FakeResponse(content=ANALYST_DRAFT)]),
            reviewer=FakeLLM([FakeResponse(content=json.dumps({"status": "PASS", "feedback": ""}))]),
            synthesis=FakeLLM([FakeResponse(content=REPORT_TEXT)]),
        )
        state = result["state"]
        self.assertEqual(state["recollect_count"], 1, "재수집을 1회 지시해야 한다")
        self.assertTrue(
            any("재수집" in a["content"] for a in state["analyses"] if a["agent"] == "supervisor")
        )
        self.assertTrue(
            any("no_evidence" in err or "dropped_domains" in err
                for err in result["trace"]["errors"]),
            result["trace"]["errors"],
        )
        # 재계획을 해도 파이프라인은 끝까지 진행해 리포트를 낸다
        self.assertIn("verifier", result["trace"]["agents"])

    def test_reviewer_unparsable_json_fails_closed(self) -> None:
        """검토 결과를 읽지 못했으면 통과시키지 않고 한 번 더 돌린다."""
        result = self._run(
            "AJ네트웍스", "financial", "재무를 분석해줘",
            researcher=FakeLLM([FakeResponse(content="수집 완료")]),
            analyst=FakeLLM([FakeResponse(content=ANALYST_DRAFT),
                             FakeResponse(content=ANALYST_DRAFT + " 보완")]),
            reviewer=FakeLLM([
                FakeResponse(content="JSON이 아닌 응답"),
                FakeResponse(content=json.dumps({"status": "PASS", "feedback": ""})),
            ]),
            synthesis=FakeLLM([FakeResponse(content=REPORT_TEXT)]),
        )
        self.assertEqual(result["state"]["revision_count"], 1)


if __name__ == "__main__":
    unittest.main()
