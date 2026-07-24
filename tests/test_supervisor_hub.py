"""재진입 Supervisor(hub) 테스트.

Supervisor를 진입 1회 정적 플래너에서 허브로 바꾼 이유는 이름을 맞추기 위해서가
아니라 **관측된 결과로 계획을 고치기 위해서**다. 그 능력이 실제로 도는지 고정한다.

- 워커가 단계를 마치면 supervisor가 다음 행선지와 예산을 정하는가
- 근거를 못 얻은 도메인을 재수집하고, 그래도 없으면 계획에서 접는가
- 재검토/재수집 예산이 supervisor 단독 소유인가

decide_next는 LLM을 쓰지 않으므로 분기 하나하나를 모델 없이 검증할 수 있다.
"""
import unittest

from src.agents.supervisor import (
    ANALYSIS_BUDGET,
    MAX_RECOLLECTIONS,
    RECOLLECT_BUDGET,
    RESEARCH_BUDGET,
    decide_next,
    domains_without_evidence,
    supervisor_node,
)


def _fact(tool: str) -> dict:
    return {"metric": "revenue", "period": "2022", "value": 1.0,
            "numeric": 1.0, "kind": "canonical", "source": "fs.csv", "tool": tool}


def _state(**overrides) -> dict:
    base = {
        "company": "테스트기업",
        "query_type": "full_report",
        "user_request": "전반적으로 분석해줘",
        "planned": True,
        "needs_analysis": True,
        "requested_domains": ["financial", "review"],
        "evidence": [_fact("tool_get_financial_data"), _fact("tool_get_review_summary")],
        "research_done": True,
        "analysis_done": False,
        "review_verdict": None,
        "synthesis_done": False,
        "verification_done": False,
        "recollect_count": 0,
        "gap_handled": True,
        "revision_count": 0,
        "max_revisions": 1,
    }
    base.update(overrides)
    return base


class EvidenceGapTests(unittest.TestCase):
    def test_domain_with_no_matching_tool_is_reported(self) -> None:
        missing = domains_without_evidence(
            ["financial", "review"], [_fact("tool_get_financial_data")]
        )
        self.assertEqual(missing, ["review"])

    def test_all_domains_served_reports_nothing(self) -> None:
        missing = domains_without_evidence(
            ["financial"], [_fact("tool_get_financial_data")]
        )
        self.assertEqual(missing, [])


class DispatchTests(unittest.TestCase):
    def test_first_entry_builds_plan_and_dispatches_researcher(self) -> None:
        state = supervisor_node({
            "company": "테스트기업", "query_type": "credit",
            "user_request": "신용등급을 알려줘",
        })
        self.assertTrue(state["planned"])
        self.assertEqual(state["next_agent"], "researcher")
        self.assertEqual(state["max_recursions"], RESEARCH_BUDGET)
        self.assertEqual(state["requested_domains"], ["credit"])

    def test_research_not_done_dispatches_researcher(self) -> None:
        self.assertEqual(decide_next(_state(research_done=False))["next_agent"], "researcher")

    def test_served_domains_go_to_analyst_with_budget(self) -> None:
        decision = decide_next(_state())
        self.assertEqual(decision["next_agent"], "analyst")
        self.assertEqual(decision["max_recursions"], ANALYSIS_BUDGET)

    def test_analysis_done_goes_to_reviewer(self) -> None:
        self.assertEqual(decide_next(_state(analysis_done=True))["next_agent"], "reviewer")

    def test_pass_goes_to_synthesis_then_verifier_then_end(self) -> None:
        after_review = decide_next(_state(analysis_done=True, review_verdict="PASS"))
        self.assertEqual(after_review["next_agent"], "synthesis")

        after_synth = decide_next(
            _state(analysis_done=True, review_verdict="PASS", synthesis_done=True)
        )
        self.assertEqual(after_synth["next_agent"], "verifier")

        after_verify = decide_next(_state(
            analysis_done=True, review_verdict="PASS",
            synthesis_done=True, verification_done=True,
        ))
        self.assertEqual(after_verify["next_agent"], "END")

    def test_overview_skips_analysis_stage(self) -> None:
        decision = decide_next(_state(
            needs_analysis=False, requested_domains=["overview"],
            evidence=[_fact("tool_get_company_knowledge_context")],
        ))
        self.assertEqual(decision["next_agent"], "synthesis")


class ReplanningTests(unittest.TestCase):
    """관측된 수집 실패를 계획에 반영하는가 — 허브로 바꾼 실질적 이유."""

    def test_domain_without_evidence_triggers_narrowed_recollection(self) -> None:
        decision = decide_next(_state(gap_handled=False, evidence=[_fact("tool_get_financial_data")]))

        self.assertEqual(decision["next_agent"], "researcher")
        self.assertEqual(decision["recollect_count"], 1)
        self.assertEqual(decision["max_recursions"], RECOLLECT_BUDGET)
        self.assertFalse(decision["research_done"])
        # allow-list가 실패한 도메인으로 좁혀져야 한다
        self.assertIn("tool_get_review_summary", decision["allowed_research_tools"])
        self.assertNotIn("tool_get_financial_data", decision["allowed_research_tools"])
        # 재수집 목표를 Researcher에게 전달한다
        self.assertIn("review", decision["research_directive"])

    def test_domain_is_dropped_after_recollection_fails(self) -> None:
        decision = decide_next(_state(
            gap_handled=False,
            evidence=[_fact("tool_get_financial_data")],
            recollect_count=MAX_RECOLLECTIONS,
        ))
        self.assertEqual(decision["next_agent"], "analyst")
        self.assertEqual(decision["requested_domains"], ["financial"])
        self.assertEqual(decision["dropped_domains"][0]["domain"], "review")
        self.assertTrue(any("dropped_domains" in err for err in decision["errors"]))

    def test_no_evidence_at_all_proceeds_but_records_the_gap(self) -> None:
        """전부 비어도 멈추지 않는다. 빈 화면보다 한계를 밝힌 리포트가 낫다."""
        decision = decide_next(_state(gap_handled=False, evidence=[], recollect_count=MAX_RECOLLECTIONS))
        self.assertEqual(decision["next_agent"], "analyst")
        self.assertTrue(any("no_evidence_for_any_domain" in err for err in decision["errors"]))

    def test_revise_returns_to_analyst_and_resets_stage(self) -> None:
        decision = decide_next(_state(analysis_done=True, review_verdict="REVISE"))
        self.assertEqual(decision["next_agent"], "analyst")
        self.assertFalse(decision["analysis_done"])
        self.assertIsNone(decision["review_verdict"])
        self.assertEqual(decision["revision_count"], 1)

    def test_revision_budget_exhausted_proceeds_to_synthesis(self) -> None:
        decision = decide_next(_state(
            analysis_done=True, review_verdict="REVISE", revision_count=1,
        ))
        self.assertEqual(decision["next_agent"], "synthesis")
        self.assertIn("재검토 예산", decision["analyses"][0]["content"])


if __name__ == "__main__":
    unittest.main()
