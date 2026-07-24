"""실행 계측과 rate limit 경계.

두 가지를 함께 다루는 이유는 둘 다 **LLM 호출의 성질**이지 노드의 성질이 아니기
때문이다. 원래 이 프로젝트는 `@rate_limited`를 노드에 붙여 두었는데, 그 결과
LLM을 한 번도 부르지 않는 Supervisor가 매 실행 1.5초씩 자고, LLM을 한 번 부르는
노드는 노드 지연(1.5초)과 handoff 지연(2초)을 이중으로 물었다. 지연을 호출 지점
하나로 모으면 낭비가 사라지고, 같은 자리에서 호출 수와 토큰을 셀 수 있다.

계측이 필요한 이유는 더 단순하다. 토큰과 지연을 재지 않으면 "최적화했더니
빨라졌다"를 검증할 수 없고, 이 프로젝트는 이미 "재현 가능한 것만 잰다"는 원칙으로
평가 계층을 만들었다(README §4). 비용도 같은 규칙을 따라야 한다.

한 프로세스에서 리포트를 하나씩 생성하는 현재 사용 형태를 전제로 스레드 로컬
누적기를 쓴다. 동시 실행이 필요해지면 run_id 키를 가진 저장소로 바꿔야 한다.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunUsage:
    """리포트 1건을 만드는 동안의 누적 사용량."""

    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    llm_seconds: float = 0.0
    throttle_seconds: float = 0.0
    calls_by_agent: dict[str, int] = field(default_factory=dict)


_local = threading.local()


def current() -> RunUsage:
    usage = getattr(_local, "usage", None)
    if usage is None:
        usage = RunUsage()
        _local.usage = usage
    return usage


def reset() -> None:
    _local.usage = RunUsage()


def snapshot() -> dict[str, Any]:
    usage = current()
    data = asdict(usage)
    data["total_tokens"] = usage.input_tokens + usage.output_tokens
    data["llm_seconds"] = round(usage.llm_seconds, 2)
    data["throttle_seconds"] = round(usage.throttle_seconds, 2)
    return data


def _extract_usage(response: Any) -> tuple[int, int]:
    """langchain 응답에서 토큰 사용량을 꺼낸다. 없으면 0으로 둔다."""
    meta = getattr(response, "usage_metadata", None)
    if not isinstance(meta, dict):
        meta = (getattr(response, "response_metadata", None) or {}).get("usage_metadata")
    if not isinstance(meta, dict):
        return 0, 0
    return int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0)


class TrackedLLM:
    """LLM 호출을 세고, 토큰을 기록하고, 호출 간 간격을 두는 래퍼.

    LangChain Runnable 전체를 흉내내지 않고 이 파이프라인이 실제로 쓰는
    `bind_tools`와 `invoke`만 감싼다. 모르는 속성은 원본으로 넘긴다.
    """

    def __init__(self, llm: Any, delay_sec: float = 0.0, agent: str | None = None) -> None:
        self._llm = llm
        self._delay = max(0.0, delay_sec)
        self._agent = agent

    def bind_tools(self, tools, **kwargs):
        return TrackedLLM(self._llm.bind_tools(tools, **kwargs), self._delay, self._agent)

    def invoke(self, *args, **kwargs):
        started = time.perf_counter()
        response = self._llm.invoke(*args, **kwargs)
        elapsed = time.perf_counter() - started

        usage = current()
        usage.llm_calls += 1
        usage.llm_seconds += elapsed
        input_tokens, output_tokens = _extract_usage(response)
        usage.input_tokens += input_tokens
        usage.output_tokens += output_tokens
        if self._agent:
            usage.calls_by_agent[self._agent] = usage.calls_by_agent.get(self._agent, 0) + 1

        # rate limit 간격은 호출 직후에만 둔다. 호출하지 않은 노드는 대기하지 않는다.
        if self._delay:
            time.sleep(self._delay)
            usage.throttle_seconds += self._delay
        return response

    def __getattr__(self, name):
        return getattr(self._llm, name)
