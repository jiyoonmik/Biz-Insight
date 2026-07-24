import operator
from typing import TypedDict, Annotated, List, Optional
from langchain_core.messages import BaseMessage


class BizInsightState(TypedDict):
    """LangGraph 그래프 전체에서 공유되는 상태."""

    # ── 입력 ──
    company: str  # 검색 기업명
    query_type: str  # 사용자 질문/요청 유형
    user_request: str  # 사용자의 자연어 분석 요청

    # ── 에이전트 대화 및 ReAct 루프 메시지 ──
    messages: Annotated[List[BaseMessage], operator.add]
    researcher_messages: Annotated[List[BaseMessage], operator.add]
    analyst_messages: Annotated[List[BaseMessage], operator.add]

    # ── 라우팅 및 제어 상태 ──
    active_agents: List[str]  # Supervisor가 계획한 에이전트 실행 목록
    current_agent: Optional[str]  # 현재 실행 중인 에이전트
    next_agent: Optional[str]  # 다음에 실행할 노드/에이전트 이름
    recursion_count: int  # 재귀(반복) 횟수 제어용 카운터
    max_recursions: int  # 현재 에이전트의 최대 tool-calling 반복 횟수
    requested_domains: List[str]  # Supervisor가 자연어 요청에서 추론한 분석 도메인
    allowed_research_tools: List[str]  # Researcher가 사용할 수 있는 도구명
    needs_analysis: bool  # Analyst/Reviewer 단계를 수행할지 여부
    revision_count: int  # Reviewer 피드백으로 재분석한 횟수
    max_revisions: int  # Reviewer 피드백 루프 최대 횟수
    
    # ── 중간 데이터 및 피드백 (Self-Correction 용도) ──
    feedback: Optional[str]  # Reviewer가 남긴 피드백 메시지

    # ── 에이전트별 분석 결과 (순차 축적) ──
    analyses: Annotated[List[dict], operator.add]
    # 각 dict: {"agent": str, "content": str}

    # ── 근거 원장 (도구 결과에서 뽑은 구조화된 사실) ──
    # 도구 호출 시점에 누적한다. 요약 문장이 아니라 값 자체를 상태로 흘려야
    # Synthesis가 인용할 근거를 갖고 Verifier가 대조할 대상을 갖는다.
    evidence: Annotated[List[dict], operator.add]
    # 각 dict: {"metric", "label", "period", "value", "numeric", "kind", "source", "tool"}

    # ── 최종 출력 ──
    final_report: str
    grounding: dict  # Verifier의 근거 대조 결과

    # ── 에러 추적 ──
    errors: Annotated[List[str], operator.add]
