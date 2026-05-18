"""
Investment Metrics Agent (상세지표)
- 투자 데이터 + 주가 데이터 조회 (LLM 호출 없음)
- LLM으로 투자 매력도 분석 리포트 생성 (1회 호출)
"""
import time
from src.config import get_llm, rate_limited, INTER_AGENT_DELAY_SEC
from src.tools.csv_tools import get_investment_data, get_stock_data


@rate_limited
def _generate_investment_report(corp_name: str, investment_data: dict, stock_data: dict) -> str:
    """LLM으로 투자 지표 분석 리포트를 생성합니다."""
    llm = get_llm()

    inv_summary = str(investment_data.get("data", []))[:3000]
    stock_summary = str(stock_data.get("data", []))[:1500]

    prompt = f"""당신은 투자 분석 전문가입니다.
'{corp_name}' 기업의 투자 데이터와 주가 추이를 분석하여,
투자 매력도 리포트를 한국어 마크다운으로 작성해주세요.

<투자 지표 데이터>
{inv_summary}
</투자 지표 데이터>

<주가 데이터>
{stock_summary}
</주가 데이터>

<작성 지침>
1. PER, PBR, ROE 등 주요 밸류에이션 지표 분석
2. 주가 추이 및 모멘텀 분석
3. 투자 매력도 총평
4. 간결하게 핵심만 (500자 이내)
</작성 지침>"""

    response = llm.invoke(prompt)
    return response.content.strip()


def investment_metrics_node(state: dict) -> dict:
    """상세지표 에이전트 노드."""
    company = state["company"]

    try:
        # Step 1: 데이터 조회 (LLM 없음)
        investment_data = get_investment_data(company)
        stock_data = get_stock_data(company)

        if "error" in investment_data and "error" in stock_data:
            content = f"## 📊 상세지표: {company}\n\n⚠️ {investment_data['error']}"
            time.sleep(INTER_AGENT_DELAY_SEC)
            return {
                "analyses": [{"agent": "investment_metrics", "content": content}],
                "errors": [investment_data["error"]],
            }

        # Step 2: LLM 리포트 생성 (1회 호출)
        report = _generate_investment_report(company, investment_data, stock_data)
        content = f"## 📊 상세지표: {company}\n\n{report}"

        time.sleep(INTER_AGENT_DELAY_SEC)
        return {"analyses": [{"agent": "investment_metrics", "content": content}]}

    except Exception as e:
        error_msg = f"상세지표 분석 실패: {str(e)}"
        time.sleep(INTER_AGENT_DELAY_SEC)
        return {
            "analyses": [{"agent": "investment_metrics", "content": error_msg}],
            "errors": [error_msg],
        }
