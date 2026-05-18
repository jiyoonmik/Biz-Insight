"""
Biz-Insight 공통 설정 모듈
- LLM: Gemini 2.5 Flash (langchain-google-genai)
- Rate limit 대응: 순차 실행 + 호출 간 지연
"""
import os
import time
import functools
from dotenv import load_dotenv

load_dotenv()

# ── LLM 설정 ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2

# ── Rate Limit 설정 ───────────────────────────────────────
# 에이전트 노드 실행 사이 대기 시간 (초)
INTER_AGENT_DELAY_SEC = 2.0
# 단일 LLM 호출 후 대기 시간 (초)
POST_LLM_CALL_DELAY_SEC = 1.5

# ── 데이터 경로 ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── CSV 파일 매핑 ─────────────────────────────────────────
CSV_FILES = {
    "company_info": "company_info.csv",
    "fs": "fs.csv",
    "bs": "bs.csv",
    "incs": "incs.csv",
    "cf": "cf.csv",
    "industry_average": "industry_average_year.csv",
    "credit_data_web": "credit_data_web.csv",
    "credit_model_a": "credit_model_a.csv",
    "credit_rank": "credit_rank.csv",
    "final_features": "final_features.csv",
    "investment_data_web": "investment_data_web.csv",
    "stock_data": "stock_data_per_month.csv",
    "web_visualization": "web_visualization.csv",
    "employee_reviews": "employee_reviews.csv",
    "b_company_review": "b_company_review.csv",
    "j_company_review": "j_company_review.csv",
    "sector": "sector.csv",
    "outstanding_shares": "outstanding_shares.csv",
    "main_fs": "main_fs.csv",
}


def get_llm():
    """Gemini 2.5 Flash LLM 인스턴스를 반환합니다."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )


def rate_limited(func):
    """LLM 호출 함수에 자동으로 지연을 추가하는 데코레이터."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        time.sleep(POST_LLM_CALL_DELAY_SEC)
        return result

    return wrapper
