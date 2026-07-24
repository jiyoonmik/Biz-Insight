"""
Synthesis Agent (종합 리포트)
- 개별 에이전트 분석 결과를 종합하여 SWOT 포함 최종 리포트 생성
- LLM 1회 호출
"""
from datetime import datetime
from src.agents.evidence import ledger_digest
from src.config import get_llm


def _flatten_to_strings(items) -> list[str]:
    """LangGraph additive state에 중첩 리스트가 섞여도 안전하게 문자열 목록으로 변환합니다."""
    if items is None:
        return []
    if isinstance(items, (str, int, float, bool)):
        return [str(items)]
    if isinstance(items, dict):
        return [str(items)]

    result = []
    try:
        iterator = iter(items)
    except TypeError:
        return [str(items)]

    for item in iterator:
        result.extend(_flatten_to_strings(item))
    return result


def _generate_synthesis_report(corp_name: str, analyses: list[dict], evidence: list[dict]) -> str:
    """모든 에이전트 결과를 종합하여 최종 리포트를 생성합니다.

    분석 텍스트만 넘기면 Synthesis는 상류에서 이미 문장이 된 수치를 다시 옮겨 적을
    뿐이라, 값이 어디서 왔는지 확인할 방법이 없다. 그래서 도구 결과에서 뽑아 둔
    근거 원장을 함께 넘기고 인용 범위를 원장으로 제한한다.
    """
    llm = get_llm(agent="synthesis")

    # 각 에이전트 결과를 정리
    analyses_text = ""
    for a in analyses:
        agent_name = a.get("agent", "unknown")
        content = "\n".join(_flatten_to_strings(a.get("content", "분석 결과 없음")))
        analyses_text += f"\n### [{agent_name}]\n{content}\n"

    evidence_text = ledger_digest(evidence)

    prompt = f"""당신은 '{corp_name}' 기업을 깊이 있게 분석하는 종합 기업 분석 전문가입니다.
아래에 제공된 개별 AI 에이전트 분석 결과와 근거 원장을 종합하여,
투자자와 경영진이 읽기 좋은 전문적이고 포괄적인 마크다운 형식의 최종 리포트를 작성해 주세요.

<개별 분석 결과>
{analyses_text}
</개별 분석 결과>

<근거 원장 - 도구가 실제로 반환한 값>
{evidence_text}
</근거 원장>

<작성 지침>
1. 마크다운 형식으로 소제목, 글머리 기호 등을 적절히 활용
2. 각 분석 결과의 핵심을 요약하여 종합
3. 본문 앞부분에 기준 회계연도, 동적 데이터 기준일, stock_code를 명시
4. 평가기준별 근거(수익성, 유동성, 재무부담, 현금창출력, 계속기업/내부통제)를 수치와 기간 중심으로 정리
5. 주요 위험 신호와 신용등급 이력을 별도 섹션으로 정리
6. 주가, 직원 리뷰, 거시지표는 최신 동적 시그널로만 사용하고 재무제표 근거와 혼동하지 않음
7. **SWOT 분석** (강점, 약점, 기회, 위협)을 반드시 포함
8. 종합 투자 의견 및 리스크 요약으로 마무리
9. **수치는 근거 원장에 있는 값만 인용한다.** 원장에 없는 수치를 새로 만들어 쓰지 않는다.
   원장 값에서 증감률처럼 파생 수치를 계산했다면 계산 근거가 된 원본 값을 함께 적는다.
10. 원장에 근거가 없어 판단할 수 없는 항목은 추정하지 말고 '데이터 미확보'로 명시한다
11. 언어는 한국어로 작성
</작성 지침>"""

    response = llm.invoke(prompt)
    return response.content.strip()


def synthesis_node(state: dict) -> dict:
    """종합 리포트 에이전트 노드. 작성 후 결정론적 Verifier로 넘긴다."""
    company = state["company"]
    analyses = state.get("analyses", [])
    evidence = state.get("evidence", []) or []

    if not analyses:
        return {
            "final_report": f"## ❌ {company} 분석 실패\n\n분석할 데이터가 없습니다.",
            "next_agent": "END"
        }

    try:
        report = _generate_synthesis_report(company, analyses, evidence)

        # 메타 정보 추가
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # 중복 제거 및 리서처/애널리스트 중심 정리
        agents_used = []
        for a in analyses:
            if isinstance(a, dict):
                agents_used.extend(_flatten_to_strings(a.get("agent", "unknown")))
            else:
                agents_used.extend(_flatten_to_strings(a))
        agents_used = list(dict.fromkeys(agents_used))
        
        errors = _flatten_to_strings(state.get("errors", []))

        header = f"""# ✨ {company} AI 종합 분석 리포트

> 📅 생성일시: {now}
> 🤖 참여 에이전트: {', '.join(agents_used)}
> ⚡ Powered by Gemini 2.5 Flash + LangGraph Multi-Agent

---

"""
        footer = ""
        if errors:
            footer = f"\n\n---\n\n> ⚠️ **분석 중 발생한 이슈**: {'; '.join(errors)}"

        final_report = header + report + footer

        return {
            "final_report": final_report,
            # 실행 경로 트레이스에 synthesis가 빠지지 않도록 기록을 남긴다.
            "analyses": [{
                "agent": "synthesis",
                "content": f"📝 최종 리포트 작성 완료 (근거 원장 {len(evidence)}건 참조)",
            }],
            "next_agent": "verifier"
        }

    except Exception as e:
        # 생성 자체가 실패하면 대조할 본문이 없으므로 Verifier를 건너뛴다.
        return {
            "final_report": f"## ❌ {company} 종합 리포트 생성 실패\n\n오류: {str(e)}",
            "errors": [f"synthesis_failed: {e}"],
            "next_agent": "END"
        }
