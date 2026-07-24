"""근거 원장과 결정론적 근거 대조 테스트."""
import unittest

from src.agents.evidence import extract_facts, ledger_digest
from src.agents.verifier import render_verification_section, verifier_node, verify_report


KNOWLEDGE_CONTEXT = {
    "company": {"stock_code": "095570", "preferred_name": "AJ네트웍스"},
    "financial_observations": [
        {"metric": "revenue", "name": "매출액", "year": "2022", "value": 1234500000000, "source": "fs.csv"},
        {"metric": "debt_ratio", "name": "부채비율", "year": "2022", "value": 312.5, "source": "fs.csv"},
    ],
    "credit_ratings": [
        {"agency": "KIS", "rating": "BBB", "year": "2022", "source_file": "credit_rank.csv"},
    ],
}


class EvidenceLedgerTests(unittest.TestCase):
    def test_extracts_metric_rows_with_period_and_source(self) -> None:
        facts = extract_facts("tool_get_company_knowledge_context", KNOWLEDGE_CONTEXT)
        by_metric = {fact["metric"]: fact for fact in facts}
        self.assertEqual(by_metric["revenue"]["numeric"], 1234500000000.0)
        self.assertEqual(by_metric["revenue"]["period"], "2022")
        self.assertEqual(by_metric["revenue"]["source"], "fs.csv")
        self.assertEqual(by_metric["debt_ratio"]["kind"], "canonical")

    def test_rating_rows_keep_text_value(self) -> None:
        facts = extract_facts("tool_get_credit_rating_history", KNOWLEDGE_CONTEXT)
        rating = next(f for f in facts if f["metric"] == "credit_rating_KIS")
        self.assertEqual(rating["value"], "BBB")
        self.assertIsNone(rating["numeric"])

    def test_dynamic_tool_results_are_tagged_as_snapshot(self) -> None:
        """canonical(회계연도)과 dynamic(스냅샷)의 기준 구분을 원장이 보존해야 한다."""
        facts = extract_facts(
            "tool_get_stock_summary",
            {"stock_code": "095570", "available": True, "end_date": "2026-06-13",
             "latest_close": 5230.0, "volatility": 0.031},
        )
        self.assertTrue(facts)
        self.assertTrue(all(fact["kind"] == "dynamic" for fact in facts))
        self.assertEqual({f["metric"] for f in facts}, {"latest_close", "volatility"})
        self.assertTrue(all(fact["period"] == "2026-06-13" for fact in facts))

    def test_identifier_keys_are_not_treated_as_facts(self) -> None:
        facts = extract_facts("tool_get_stock_summary", {"stock_code": "095570", "months": 12})
        self.assertEqual(facts, [])

    def test_unknown_shapes_are_skipped_quietly(self) -> None:
        self.assertEqual(extract_facts("tool_unknown", {"foo": "bar"}), [])
        self.assertEqual(extract_facts("tool_unknown", "plain string"), [])

    def test_digest_marks_period_basis(self) -> None:
        facts = extract_facts("tool_get_company_knowledge_context", KNOWLEDGE_CONTEXT)
        digest = ledger_digest(facts)
        self.assertIn("[회계연도]", digest)
        self.assertIn("fs.csv", digest)


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.facts = extract_facts("tool_get_company_knowledge_context", KNOWLEDGE_CONTEXT)

    def test_matches_number_written_with_korean_compound_units(self) -> None:
        """'1조 2,345억원'은 토큰 두 개지만 값 하나(1.2345e12)다."""
        result = verify_report("매출액은 1조 2,345억원입니다.", self.facts)
        self.assertEqual(result["checked"], 1)
        self.assertEqual(result["matched"], 1)

    def test_matches_percentage_regardless_of_ratio_convention(self) -> None:
        result = verify_report("부채비율은 312.5%입니다.", self.facts)
        self.assertEqual(result["matched"], 1)

    def test_flags_fabricated_number(self) -> None:
        result = verify_report("목표주가는 99,999원입니다.", self.facts)
        self.assertEqual(result["matched"], 0)
        self.assertIn("99,999원", result["unmatched_samples"])

    def test_year_reference_is_not_counted_as_a_cited_number(self) -> None:
        result = verify_report("2022년 실적 기준입니다.", self.facts)
        self.assertEqual(result["checked"], 0)

    def test_markdown_list_ordinals_are_ignored(self) -> None:
        result = verify_report("1. 첫째 항목\n2. 둘째 항목", self.facts)
        self.assertEqual(result["checked"], 0)

    def test_coverage_is_reported_as_ratio(self) -> None:
        report = "매출액 1조 2,345억원, 부채비율 312.5%, 목표주가 99,999원"
        result = verify_report(report, self.facts)
        self.assertEqual((result["checked"], result["matched"]), (3, 2))
        self.assertAlmostEqual(result["coverage"], 2 / 3, places=3)

    def test_empty_ledger_is_surfaced_not_silently_passed(self) -> None:
        result = verify_report("매출액은 100억원입니다.", [])
        self.assertEqual(result["evidence_count"], 0)
        self.assertIn("대조되지 않은", render_verification_section(result))

    def test_node_appends_section_and_terminates(self) -> None:
        state = verifier_node({
            "final_report": "매출액은 1조 2,345억원입니다.",
            "evidence": self.facts,
        })
        self.assertEqual(state["next_agent"], "END")
        self.assertIn("근거 대조", state["final_report"])
        self.assertEqual(state["grounding"]["matched"], 1)
        self.assertEqual(state["analyses"][0]["agent"], "verifier")

    def test_node_records_low_coverage_without_failing_the_run(self) -> None:
        state = verifier_node({
            "final_report": "목표주가 99,999원, 예상 매출 88,888원",
            "evidence": self.facts,
        })
        self.assertEqual(state["next_agent"], "END")
        self.assertTrue(any("low_grounding_coverage" in err for err in state["errors"]))


if __name__ == "__main__":
    unittest.main()
