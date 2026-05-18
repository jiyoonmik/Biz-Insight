"""
Company Profile Agent (기업개요)
- CSV에서 기업 기본 정보 조회 (LLM 호출 없음)
- LLM으로 기업 한줄 평가 생성 (1회 호출)
"""
import time
from src.config import get_llm, rate_limited, INTER_AGENT_DELAY_SEC
from src.tools.csv_tools import get_company_info


@rate_limited
def _generate_company_summary(corp_name: str, company_info: dict) -> str:
    """LLM으로 기업 한줄 평가를 생성합니다."""
    llm = get_llm()
    prompt = f"""당신은 기업 분석 전문가입니다.
아래 '{corp_name}' 기업의 기본 정보를 바탕으로, 이 기업에 대한 간결하고 전문적인 한줄 평가(50자 내외)를 한국어로 작성해주세요.

기업 정보:
{company_info}

한줄 평가:"""

    response = llm.invoke(prompt)
    return response.content.strip()


def company_profile_node(state: dict) -> dict:
    """기업개요 에이전트 노드."""
    company = state["company"]

    try:
        # Step 1: CSV 조회 (LLM 호출 없음)
        info = get_company_info(company)
        if "error" in info:
            return {
                "analyses": [{"agent": "company_profile", "content": info["error"]}],
                "errors": [info["error"]],
            }

        # Step 2: LLM으로 기업 평가 생성 (1회 호출)
        summary = _generate_company_summary(company, info)

        content = f"""## 📋 기업개요: {company}

| 항목 | 내용 |
|---|---|
| 종목코드 | {info.get('종목코드', 'N/A')} |
| 업종 | {info.get('업종', 'N/A')} |
| 대표자 | {info.get('대표자명', 'N/A')} |
| 주요제품 | {info.get('주요제품', 'N/A')} |

**AI 한줄 평가:** {summary}
"""
        time.sleep(INTER_AGENT_DELAY_SEC)
        return {"analyses": [{"agent": "company_profile", "content": content}]}

    except Exception as e:
        error_msg = f"기업개요 분석 실패: {str(e)}"
        time.sleep(INTER_AGENT_DELAY_SEC)
        return {
            "analyses": [{"agent": "company_profile", "content": error_msg}],
            "errors": [error_msg],
        }
