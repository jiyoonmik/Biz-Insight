"""
Credit Analysis Agent (신용분석)
- 신용 데이터 조회 + ML 모델 예측 (LLM 호출 없음)
- LLM으로 신용 분석 해석 리포트 생성 (1회 호출)
"""
import time
from src.config import get_llm, rate_limited, INTER_AGENT_DELAY_SEC
from src.tools.csv_tools import get_credit_data, get_credit_rank
from src.tools.ml_tools import get_feature_importance


@rate_limited
def _generate_credit_report(
    corp_name: str,
    credit_data: dict,
    actual_rating: str | None,
    feature_importance: list | None,
) -> str:
    """LLM으로 신용 분석 리포트를 생성합니다."""
    llm = get_llm()

    data_summary = str(credit_data.get("data", []))[:3000]
    fi_summary = str(feature_importance[:10]) if feature_importance else "피쳐 중요도 데이터 없음"

    rating_info = f"실제 신용등급: {actual_rating}" if actual_rating else "실제 신용등급: 미등록 (예측 필요)"

    prompt = f"""당신은 신용 분석 전문가입니다.
'{corp_name}' 기업의 신용 데이터를 분석하여 신용 위험성 리포트를 한국어 마크다운으로 작성해주세요.

{rating_info}

<신용 데이터>
{data_summary}
</신용 데이터>

<모델 피쳐 중요도 Top-10>
{fi_summary}
</모델 피쳐 중요도 Top-10>

<작성 지침>
1. 신용등급 현황 또는 예측 결과 설명
2. 부채 상환 능력, 재무 건전성 평가
3. 신용등급에 가장 큰 영향을 미친 요인 분석
4. 간결하게 핵심만 (500자 이내)
</작성 지침>"""

    response = llm.invoke(prompt)
    return response.content.strip()


def credit_analysis_node(state: dict) -> dict:
    """신용분석 에이전트 노드."""
    company = state["company"]

    try:
        # Step 1: 데이터 조회 (LLM 없음)
        credit_data = get_credit_data(company)
        actual_rating = get_credit_rank(company)
        feature_importance = get_feature_importance(top_n=10)

        if "error" in credit_data:
            content = f"## 🏦 신용분석: {company}\n\n⚠️ {credit_data['error']}"
            time.sleep(INTER_AGENT_DELAY_SEC)
            return {
                "analyses": [{"agent": "credit_analysis", "content": content}],
                "errors": [credit_data["error"]],
            }

        # Step 2: LLM 리포트 생성 (1회 호출)
        report = _generate_credit_report(company, credit_data, actual_rating, feature_importance)
        content = f"## 🏦 신용분석: {company}\n\n{report}"

        time.sleep(INTER_AGENT_DELAY_SEC)
        return {"analyses": [{"agent": "credit_analysis", "content": content}]}

    except Exception as e:
        error_msg = f"신용분석 실패: {str(e)}"
        time.sleep(INTER_AGENT_DELAY_SEC)
        return {
            "analyses": [{"agent": "credit_analysis", "content": error_msg}],
            "errors": [error_msg],
        }
