"""계측과 rate limit 경계 테스트.

핵심 회귀 방지 지점: 지연은 LLM 호출 직후에만 발생해야 한다. 노드 경계에 걸면
LLM을 부르지 않는 노드(Supervisor, Verifier)까지 대기하고, 부르는 노드는
지연을 이중으로 문다.
"""
import unittest
from unittest import mock

from src.agents import telemetry
from src.agents.telemetry import TrackedLLM


class FakeUsageResponse:
    def __init__(self, input_tokens=10, output_tokens=5):
        self.usage_metadata = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }


class StubLLM:
    def __init__(self, response=None):
        self.calls = 0
        self._response = response or FakeUsageResponse()

    def invoke(self, *args, **kwargs):
        self.calls += 1
        return self._response

    def bind_tools(self, tools, **kwargs):
        return self


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        telemetry.reset()

    def test_counts_calls_and_tokens(self) -> None:
        llm = TrackedLLM(StubLLM(), delay_sec=0, agent="researcher")
        llm.invoke("prompt")
        llm.invoke("prompt")

        usage = telemetry.snapshot()
        self.assertEqual(usage["llm_calls"], 2)
        self.assertEqual(usage["input_tokens"], 20)
        self.assertEqual(usage["output_tokens"], 10)
        self.assertEqual(usage["total_tokens"], 30)
        self.assertEqual(usage["calls_by_agent"], {"researcher": 2})

    def test_delay_is_applied_per_llm_call_only(self) -> None:
        with mock.patch("time.sleep") as sleep:
            llm = TrackedLLM(StubLLM(), delay_sec=1.5, agent="analyst")
            llm.invoke("prompt")
            self.assertEqual(sleep.call_count, 1)
            sleep.assert_called_with(1.5)
        self.assertEqual(telemetry.snapshot()["throttle_seconds"], 1.5)

    def test_no_call_means_no_delay(self) -> None:
        """LLM을 부르지 않으면 대기도 계측도 없다."""
        with mock.patch("time.sleep") as sleep:
            TrackedLLM(StubLLM(), delay_sec=1.5)
            self.assertEqual(sleep.call_count, 0)
        self.assertEqual(telemetry.snapshot()["llm_calls"], 0)

    def test_bind_tools_keeps_tracking(self) -> None:
        stub = StubLLM()
        llm = TrackedLLM(stub, delay_sec=0, agent="researcher").bind_tools([])
        llm.invoke("prompt")
        self.assertEqual(telemetry.snapshot()["llm_calls"], 1)

    def test_missing_usage_metadata_does_not_break_counting(self) -> None:
        class NoUsage:
            pass

        llm = TrackedLLM(StubLLM(response=NoUsage()), delay_sec=0)
        llm.invoke("prompt")
        usage = telemetry.snapshot()
        self.assertEqual(usage["llm_calls"], 1)
        self.assertEqual(usage["total_tokens"], 0)

    def test_reset_clears_previous_run(self) -> None:
        TrackedLLM(StubLLM(), delay_sec=0).invoke("prompt")
        telemetry.reset()
        self.assertEqual(telemetry.snapshot()["llm_calls"], 0)


class NodeBoundaryTests(unittest.TestCase):
    def test_supervisor_makes_no_llm_call_and_no_delay(self) -> None:
        """Supervisor는 규칙 기반이므로 호출도 대기도 0이어야 한다."""
        from src.agents.supervisor import supervisor_node

        telemetry.reset()
        with mock.patch("time.sleep") as sleep:
            supervisor_node({
                "company": "테스트기업",
                "query_type": "financial",
                "user_request": "재무를 분석해줘",
            })
            self.assertEqual(sleep.call_count, 0)
        self.assertEqual(telemetry.snapshot()["llm_calls"], 0)

    def test_verifier_makes_no_llm_call(self) -> None:
        from src.agents.verifier import verifier_node

        telemetry.reset()
        with mock.patch("time.sleep") as sleep:
            verifier_node({"final_report": "매출 100억원", "evidence": []})
            self.assertEqual(sleep.call_count, 0)
        self.assertEqual(telemetry.snapshot()["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
