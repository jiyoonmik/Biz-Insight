"""테스트용 LLM 대역.

멀티에이전트 그래프에서 정작 검증하고 싶은 것은 LLM의 문장력이 아니라 **제어
흐름**이다. Reviewer가 REVISE를 내면 정말 Analyst로 돌아가는지, 재검토 한도에
걸리면 멈추는지, 도구 실패가 errors 채널에 남는지, ReAct 루프가 카운터를 제대로
증가시키는지. 이건 모델 없이 확인할 수 있어야 하고, 그렇지 않으면 그래프 수정
때마다 API 키와 수십 초를 써야 한다.

그래서 응답을 미리 적어 둔 대역을 주입한다. 각 노드가 `get_llm()`을 자기 모듈
네임스페이스에서 부르므로 모듈 단위로 patch하면 된다.
"""
from __future__ import annotations

from typing import Any


class FakeResponse:
    """langchain AIMessage 중 노드가 실제로 읽는 속성만 흉내낸다."""

    def __init__(
        self,
        content: Any = "",
        tool_calls: list[dict] | None = None,
        finish_reason: str = "STOP",
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.response_metadata = {"finish_reason": finish_reason}
        self.type = "ai"
        self.name = None


class FakeLLM:
    """미리 정한 순서대로 응답을 돌려주는 LLM 대역."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = list(responses)
        self.invocations: list[Any] = []
        self.bound_tools: list[Any] = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        if not self._responses:
            return FakeResponse(content="(스크립트 소진)")
        return self._responses.pop(0)


def tool_call(name: str, args: dict, call_id: str = "call-1") -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


# Reviewer 1단계(기계적 요건 점검)를 통과하는 분석 초안 픽스처.
# stock_code / 기준 회계연도 / 수치 인용 / 리스크 언급이 모두 있어야 LLM 검토까지 간다.
# 길이도 실제 Analyst 산출물에 가깝게 둔다 — 짧은 더미 문자열을 쓰면 테스트가
# 검증하려는 경로 대신 최소 길이 요건에 걸려 넘어진다.
ANALYST_DRAFT = (
    "AJ네트웍스(095570)의 2022 회계연도 기준 분석입니다. "
    "매출액은 1조 2,345억원이며 부채비율은 312.5%로 산업 평균 대비 높은 수준입니다. "
    "유동비율은 95%로 단기 유동성에 부담이 있고, 영업현금흐름은 개선 추세를 보였습니다. "
    "수익성 측면에서는 이자보상배율이 낮아 금융비용 부담이 영업이익을 잠식할 여지가 있으며, "
    "차입 구조가 단기에 몰려 있어 차환 일정 관리가 중요한 상황으로 판단됩니다. "
    "주요 리스크는 재무 레버리지 부담과 금리 상승에 따른 이자비용 증가이며, "
    "신용등급 이력은 추가 확인이 필요합니다."
)

# 위 초안이 인용한 수치에 대응하는 근거 원장.
ANALYST_DRAFT_LEDGER = [
    {"metric": "revenue", "period": "2022", "value": 1234500000000.0,
     "numeric": 1234500000000.0, "kind": "canonical", "source": "fs.csv"},
    {"metric": "debt_ratio", "period": "2022", "value": 312.5,
     "numeric": 312.5, "kind": "canonical", "source": "fs.csv"},
    {"metric": "current_ratio", "period": "2022", "value": 95.0,
     "numeric": 95.0, "kind": "canonical", "source": "fs.csv"},
]
