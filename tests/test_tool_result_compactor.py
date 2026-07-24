"""컨텍스트 예산 관리(tool_result_compactor) 테스트.

도구 하나가 관측값 수백 행을 돌려주기 때문에, 압축을 LLM의 요약 능력에 맡기지
않고 결정론적 코드로 처리한다. 그 압축 규칙이 의도대로 도는지 확인한다.
특히 '필터가 데이터를 전부 날려 에이전트가 데이터 없음으로 오판하는' 실패는
잘림보다 나쁘므로 폴백 동작을 명시적으로 검증한다.
"""
import ast
import math
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.agents.tool_result_compactor import (
    compact_message_history,
    compact_tool_result,
    summarize_tool_messages,
)


class CompactorTests(unittest.TestCase):
    def test_keeps_only_whitelisted_metrics_of_latest_year(self) -> None:
        result = compact_tool_result(
            "tool_get_financial_data",
            {
                "columns": ["metric_code", "period_value", "numeric_value"],
                "data": [
                    {"metric_code": "revenue", "period_value": "2022", "numeric_value": 100, "source_file": "fs.csv"},
                    {"metric_code": "revenue", "period_value": "2021", "numeric_value": 90, "source_file": "fs.csv"},
                    {"metric_code": "some_noise_metric", "period_value": "2022", "numeric_value": 1, "source_file": "fs.csv"},
                ],
            },
        )
        rows = result["data"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["metric"], "revenue")
        self.assertEqual(rows[0]["year"], "2022")
        self.assertEqual(rows[0]["source"], "fs.csv")

    def test_falls_back_to_raw_rows_when_whitelist_matches_nothing(self) -> None:
        """압축 결과가 비면 '데이터 없음'으로 오판된다. 그보다는 자르는 편이 낫다."""
        result = compact_tool_result(
            "tool_get_financial_data",
            {
                "columns": ["metric_code"],
                "data": [
                    {"metric_code": "unknown_a", "period_value": "2022", "numeric_value": 1},
                    {"metric_code": "unknown_b", "period_value": "2022", "numeric_value": 2},
                ],
            },
        )
        self.assertEqual(len(result["data"]), 2)

    def test_enforces_hard_character_cap(self) -> None:
        oversized = {"data": [{"metric_code": "revenue", "period_value": "2022",
                               "numeric_value": "x" * 500} for _ in range(200)]}
        compacted = compact_tool_result("tool_unknown_shape", oversized, max_chars=500)
        self.assertIsInstance(compacted, str)
        self.assertTrue(compacted.endswith("...[truncated]"))
        self.assertLessEqual(len(compacted), 500 + len("...[truncated]"))

    def test_company_search_puts_exact_preferred_match_first(self) -> None:
        result = compact_tool_result(
            "tool_search_company_by_name",
            {
                "query": "카카오",
                "results": [
                    {"stock_code": "1", "alias_name": "카카오게임즈", "preferred_name": "카카오게임즈", "alias_type": "alias"},
                    {"stock_code": "2", "alias_name": "카카오", "preferred_name": "카카오", "alias_type": "preferred"},
                ],
            },
        )
        self.assertEqual(result["results"][0]["stock_code"], "2")
        self.assertLessEqual(len(result["results"]), 5)

    def test_result_repr_stays_parseable(self) -> None:
        """도구 결과는 문자열로 실려 나중에 되읽힌다. NaN이 섞이면 파싱이 깨진다.

        파싱 실패는 예외로 드러나지 않고 출처 목록이 조용히 비는 형태로 나타나므로
        압축 단계에서 값을 정리해 둔다.
        """
        result = compact_tool_result(
            "tool_get_stock_summary",
            {"stock_code": "095570", "latest_close": float("nan"), "period_return": float("inf")},
        )
        parsed = ast.literal_eval(str(result))
        self.assertIsInstance(parsed, dict)
        self.assertIsNone(parsed["latest_close"])
        self.assertIsNone(parsed["period_return"])

    def test_nan_inside_records_is_parseable_too(self) -> None:
        result = compact_tool_result(
            "tool_get_financial_data",
            {"columns": ["metric_code"],
             "data": [{"metric_code": "revenue", "period_value": "2022",
                       "numeric_value": 100, "source_file": float("nan")}]},
        )
        parsed = ast.literal_eval(str(result))
        self.assertIsNone(parsed["data"][0]["source"])


class MessageHistoryTests(unittest.TestCase):
    """ReAct 루프가 매 턴 재전송하는 히스토리 압축."""

    def _history(self, tool_results: list[str]) -> list:
        messages = [SystemMessage(content="sys"), HumanMessage(content="req")]
        for index, content in enumerate(tool_results):
            messages.append(AIMessage(content="", tool_calls=[
                {"name": f"tool_{index}", "args": {}, "id": f"c{index}", "type": "tool_call"}
            ]))
            messages.append(ToolMessage(content=content, tool_call_id=f"c{index}", name=f"tool_{index}"))
        return messages

    def test_keeps_recent_result_and_compacts_older_ones(self) -> None:
        messages = self._history(["A" * 5000, "B" * 5000, "C" * 5000])
        compacted = compact_message_history(messages, keep_full=1)

        tool_msgs = [m for m in compacted if isinstance(m, ToolMessage)]
        self.assertEqual(len(tool_msgs), 3, "메시지 개수는 유지되어야 한다")
        self.assertIn("[히스토리 압축]", tool_msgs[0].content)
        self.assertIn("[히스토리 압축]", tool_msgs[1].content)
        self.assertEqual(tool_msgs[2].content, "C" * 5000, "가장 최근 결과는 원문 유지")

    def test_preserves_tool_call_ids(self) -> None:
        """tool_call과 ToolMessage의 짝이 깨지면 모델 호출 자체가 실패한다."""
        messages = self._history(["A" * 100, "B" * 100])
        compacted = compact_message_history(messages, keep_full=1)
        ids = [m.tool_call_id for m in compacted if isinstance(m, ToolMessage)]
        self.assertEqual(ids, ["c0", "c1"])

    def test_does_not_mutate_original_history(self) -> None:
        """상태에 저장된 전체 히스토리는 트레이스와 폴백 요약이 계속 본다."""
        messages = self._history(["A" * 100, "B" * 100])
        compact_message_history(messages, keep_full=1)
        self.assertEqual(messages[3].content, "A" * 100)

    def test_digest_uses_evidence_ledger_when_available(self) -> None:
        messages = self._history(["원문", "원문2"])
        ledger = [
            {"metric": "revenue", "tool": "tool_0", "period": "2022", "value": 1, "numeric": 1.0},
            {"metric": "debt_ratio", "tool": "tool_0", "period": "2022", "value": 2, "numeric": 2.0},
        ]
        compacted = compact_message_history(messages, keep_full=1, ledger=ledger)
        digest = [m for m in compacted if isinstance(m, ToolMessage)][0].content
        self.assertIn("사실 2건", digest)
        self.assertIn("revenue", digest)

    def test_short_history_is_left_alone(self) -> None:
        messages = self._history(["A" * 100])
        self.assertIs(compact_message_history(messages, keep_full=1), messages)


class SummarizeTests(unittest.TestCase):
    def test_summarize_reports_absence_explicitly(self) -> None:
        self.assertEqual(summarize_tool_messages([]), "수집된 tool 결과가 없습니다.")

    def test_summarize_groups_tool_messages_by_name(self) -> None:
        messages = [
            ToolMessage(content=str({"data": [{"metric_code": "revenue", "period_value": "2022",
                                               "numeric_value": 100}]}),
                        tool_call_id="1", name="tool_get_financial_data"),
        ]
        summary = summarize_tool_messages(messages)
        self.assertIn("### tool_get_financial_data", summary)
        self.assertIn("revenue", summary)


if __name__ == "__main__":
    unittest.main()
