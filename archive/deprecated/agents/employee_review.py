"""
Employee Review Agent (직원리뷰/기업문화 분석)
- 블라인드/잡플래닛 리뷰 데이터 조회 (LLM 호출 없음)
- LLM으로 기업문화 분석 리포트 생성 (1회 호출)
"""
import time
from src.config import get_llm, rate_limited, INTER_AGENT_DELAY_SEC
from src.tools.csv_tools import get_employee_reviews


@rate_limited
def _generate_review_report(corp_name: str, review_data: dict) -> str:
    """LLM으로 직원 리뷰 기반 기업문화 분석 리포트를 생성합니다."""
    llm = get_llm()

    review_summary = str(review_data.get("data", []))[:4000]
    total = review_data.get("total_reviews", "N/A")

    prompt = f"""당신은 기업문화 분석 전문가입니다.
'{corp_name}' 기업의 직원 리뷰 데이터(총 {total}건)를 분석하여,
기업문화 및 근무환경 리포트를 한국어 마크다운으로 작성해주세요.

<직원 리뷰 데이터>
{review_summary}
</직원 리뷰 데이터>

<작성 지침>
1. 기업문화 장점/단점 키워드 도출
2. 급여/복지, 워라밸, 성장가능성, 경영진 평가 등 카테고리별 분석
3. 전반적인 직원 만족도 평가
4. 간결하게 핵심만 (500자 이내)
</작성 지침>"""

    response = llm.invoke(prompt)
    return response.content.strip()


def employee_review_node(state: dict) -> dict:
    """직원리뷰 에이전트 노드."""
    company = state["company"]

    try:
        # Step 1: 데이터 조회 (LLM 없음)
        review_data = get_employee_reviews(company)

        if "error" in review_data:
            content = f"## 👥 직원리뷰: {company}\n\n⚠️ {review_data['error']}"
            time.sleep(INTER_AGENT_DELAY_SEC)
            return {
                "analyses": [{"agent": "employee_review", "content": content}],
                "errors": [review_data["error"]],
            }

        # Step 2: LLM 리포트 생성 (1회 호출)
        report = _generate_review_report(company, review_data)
        content = f"## 👥 직원리뷰 분석: {company}\n\n{report}"

        time.sleep(INTER_AGENT_DELAY_SEC)
        return {"analyses": [{"agent": "employee_review", "content": content}]}

    except Exception as e:
        error_msg = f"직원리뷰 분석 실패: {str(e)}"
        time.sleep(INTER_AGENT_DELAY_SEC)
        return {
            "analyses": [{"agent": "employee_review", "content": error_msg}],
            "errors": [error_msg],
        }
