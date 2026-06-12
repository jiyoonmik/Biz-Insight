"""
Synthesis Agent (종합 리포트)
- 개별 에이전트 분석 결과를 종합하여 SWOT 포함 최종 리포트 생성
- LLM 1회 호출
"""
import time
from datetime import datetime
from src.config import get_llm, rate_limited, INTER_AGENT_DELAY_SEC


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


@rate_limited
def _generate_synthesis_report(corp_name: str, analyses: list[dict]) -> str:
    """모든 에이전트 결과를 종합하여 최종 리포트를 생성합니다."""
    llm = get_llm()

    # 각 에이전트 결과를 정리
    analyses_text = ""
    for a in analyses:
        agent_name = a.get("agent", "unknown")
        content = "\n".join(_flatten_to_strings(a.get("content", "분석 결과 없음")))
        analyses_text += f"\n### [{agent_name}]\n{content}\n"

    prompt = f"""당신은 '{corp_name}' 기업을 깊이 있게 분석하는 종합 기업 분석 전문가입니다.
아래에 제공된 개별 AI 에이전트 분석 결과를 종합하여,
투자자와 경영진이 읽기 좋은 전문적이고 포괄적인 마크다운 형식의 최종 리포트를 작성해 주세요.

<개별 분석 결과>
{analyses_text}
</개별 분석 결과>

<작성 지침>
1. 마크다운 형식으로 소제목, 글머리 기호 등을 적절히 활용
2. 각 분석 결과의 핵심을 요약하여 종합
3. 본문 앞부분에 기준 회계연도, 동적 데이터 기준일, stock_code를 명시
4. 평가기준별 근거(수익성, 유동성, 재무부담, 현금창출력, 계속기업/내부통제)를 수치와 기간 중심으로 정리
5. 주요 위험 신호와 신용등급 이력을 별도 섹션으로 정리
6. 주가, 직원 리뷰, 거시지표는 최신 동적 시그널로만 사용하고 재무제표 근거와 혼동하지 않음
7. **SWOT 분석** (강점, 약점, 기회, 위협)을 반드시 포함
8. 종합 투자 의견 및 리스크 요약으로 마무리
9. 구체적인 수치가 있다면 적극적으로 인용
10. 언어는 한국어로 작성
</작성 지침>"""

    response = llm.invoke(prompt)
    return response.content.strip()


def synthesis_node(state: dict) -> dict:
    """종합 리포트 에이전트 노드."""
    company = state["company"]
    analyses = state.get("analyses", [])

    if not analyses:
        return {
            "final_report": f"## ❌ {company} 분석 실패\n\n분석할 데이터가 없습니다.",
            "next_agent": "END"
        }

    try:
        report = _generate_synthesis_report(company, analyses)

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
            "next_agent": "END"
        }

    except Exception as e:
        return {
            "final_report": f"## ❌ {company} 종합 리포트 생성 실패\n\n오류: {str(e)}",
            "next_agent": "END"
        }
