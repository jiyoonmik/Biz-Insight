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


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return None


def _copy_env(source: str, target: str) -> bool:
    value = os.getenv(source)
    if value and not os.getenv(target):
        os.environ[target] = value
        return True
    return False


def _clear_langsmith_env_cache() -> None:
    try:
        from langsmith.utils import get_env_var

        get_env_var.cache_clear()
    except Exception:
        pass


def configure_langsmith_environment() -> None:
    """Normalize LangSmith/LangChain env aliases before tracing starts."""
    changed = False
    changed |= _copy_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    changed |= _copy_env("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")
    changed |= _copy_env("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")
    changed |= _copy_env("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT")
    changed |= _copy_env("LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT")
    changed |= _copy_env("LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT")

    tracing_value = _first_env(
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
    )
    if tracing_value is not None:
        normalized = "true" if _truthy(tracing_value) else "false"
        for name in (
            "LANGSMITH_TRACING",
            "LANGSMITH_TRACING_V2",
            "LANGCHAIN_TRACING_V2",
            "LANGCHAIN_TRACING",
        ):
            if os.getenv(name) != normalized:
                os.environ[name] = normalized
                changed = True

    if changed:
        _clear_langsmith_env_cache()


def langsmith_tracing_enabled() -> bool:
    configure_langsmith_environment()
    tracing_value = _first_env(
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_TRACING",
    )
    api_key = _first_env("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")
    return _truthy(tracing_value) and bool(api_key)


def langsmith_project_name() -> str | None:
    configure_langsmith_environment()
    return _first_env("LANGSMITH_PROJECT", "LANGCHAIN_PROJECT")


configure_langsmith_environment()

# ── LLM 설정 ──────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLM_MODEL = "gemini-2.5-flash"
LLM_TEMPERATURE = 0.2

# ── Rate Limit 설정 ───────────────────────────────────────
# 무료 티어 quota 대응. 지연을 비용으로 지불해 안정성을 사는 구조라
# 유료 쿼터나 테스트에서는 0으로 낮출 수 있도록 환경변수로 뺐다.
#
# 간격은 LLM 호출 직후에만 둔다. 예전에는 노드 경계에 걸어 두어서 LLM을 부르지
# 않는 노드도 대기했고, 호출하는 노드는 노드 지연과 handoff 지연을 이중으로 물었다.
POST_LLM_CALL_DELAY_SEC = float(os.getenv("BIZINSIGHT_POST_LLM_DELAY_SEC", "1.5"))

# ── 데이터 경로 ───────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.abspath(os.getenv("BIZINSIGHT_DATA_DIR", os.path.join(BASE_DIR, "data")))

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


def get_llm(agent: str | None = None):
    """Gemini LLM 인스턴스를 반환합니다.

    계측과 rate limit 간격을 붙인 래퍼로 감싸서 돌려줍니다. 지연을 노드가 아니라
    호출 지점에 두는 이유는 `src/agents/telemetry.py` 참고.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.agents.telemetry import TrackedLLM

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=GOOGLE_API_KEY,
    )
    return TrackedLLM(llm, delay_sec=POST_LLM_CALL_DELAY_SEC, agent=agent)
