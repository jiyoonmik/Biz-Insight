"""
Financial Analysis Agent (재무분석)
- CSV에서 재무 데이터 조회 + 산업 평균 비교 (LLM 호출 없음)
- LLM으로 재무 분석 리포트 생성 (1회 호출)
"""
import time
from src.config import get_llm, rate_limited, INTER_AGENT_DELAY_SEC
from src.tools.csv_tools import get_financial_data, get_industry_average, get_company_info


@rate_limited
def _generate_financial_report(corp_name: str, financial_data: dict, industry_data: dict) -> str:
    """LLM으로 재무 분석 리포트를 생성합니다."""
    llm = get_llm()

    # 데이터를 축약하여 토큰 절약
    data_summary = str(financial_data.get("data", []))[:3000]
    industry_summary = str(industry_data.get("data", []))[:1500]

    prompt = f"""당신은 기업 재무 분석 전문가입니다.
'{corp_name}' 기업의 재무 데이터와 동일 산업군 평균 데이터를 분석하여,
투자자가 읽기 좋은 재무 분석 리포트를 한국어 마크다운 형식으로 작성해주세요.

<재무 데이터>
{data_summary}
</재무 데이터>

<산업 평균>
{industry_summary}
</산업 평균>

<작성 지침>
1. 수익성(매출, 영업이익률 등), 성장성, 안정성 관점에서 분석
2. 동일 산업군 대비 강점/약점 비교
3. 구체적 수치를 인용
4. 간결하게 핵심만 (500자 이내)
</작성 지침>"""

    response = llm.invoke(prompt)
    return response.content.strip()


def financial_analysis_node(state: dict) -> dict:
    """재무분석 에이전트 노드."""
    company = state["company"]

    try:
        # Step 1: 데이터 조회 (LLM 없음)
        financial_data = get_financial_data(company)
        company_info = get_company_info(company)
        sector = company_info.get("업종", "") if isinstance(company_info, dict) else ""
        industry_data = get_industry_average(sector)

        if "error" in financial_data:
            content = f"## 💰 재무분석: {company}\n\n⚠️ {financial_data['error']}\n사용 가능한 컬럼: {financial_data.get('available_columns', [])}"
            time.sleep(INTER_AGENT_DELAY_SEC)
            return {
                "analyses": [{"agent": "financial_analysis", "content": content}],
                "errors": [financial_data["error"]],
            }

        # Step 2: LLM 리포트 생성 (1회 호출)
        report = _generate_financial_report(company, financial_data, industry_data)
        content = f"## 💰 재무분석: {company}\n\n{report}"

        time.sleep(INTER_AGENT_DELAY_SEC)
        return {"analyses": [{"agent": "financial_analysis", "content": content}]}

    except Exception as e:
        error_msg = f"재무분석 실패: {str(e)}"
        time.sleep(INTER_AGENT_DELAY_SEC)
        return {
            "analyses": [{"agent": "financial_analysis", "content": error_msg}],
            "errors": [error_msg],
        }
