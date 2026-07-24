from langchain_core.tools import tool
from typing import Optional

from src.tools.csv_tools import (
    get_company_info, get_financial_data, get_industry_average,
    get_credit_data, get_credit_rank, get_investment_data,
    get_stock_data, get_employee_reviews
)
from src.tools.ml_tools import predict_credit_rating, get_feature_importance
from src.tools.dynamic_tools import DYNAMIC_TOOLS
from src.tools.context_tools import CONTEXT_TOOLS


@tool
def tool_get_company_info(corp_name: str) -> dict:
    """주어진 기업명(corp_name)의 기본 정보(종목코드, 업종, 대표자, 주요제품 등)를 검색합니다. 
    가장 먼저 호출하여 기업의 존재 여부와 개요를 파악할 때 사용하세요."""
    return get_company_info(corp_name)


@tool
def tool_get_financial_data(corp_name: str) -> dict:
    """기업의 재무제표(fs.csv) 주요 데이터를 반환합니다. 수익성, 성장성 분석 시 필수입니다."""
    return get_financial_data(corp_name)


@tool
def tool_get_industry_average(sector: str) -> dict:
    """특정 산업군(sector)의 평균 재무/신용 지표를 반환합니다. 기업과 산업 평균을 비교할 때 유용합니다."""
    return get_industry_average(sector)


@tool
def tool_get_credit_data(corp_name: str) -> dict:
    """기업의 신용 관련 재무 데이터(부채비율, 유동비율 등)를 반환합니다. 신용 위험성 분석 시 필요합니다."""
    return get_credit_data(corp_name)


@tool
def tool_get_credit_rank(corp_name: str) -> Optional[str]:
    """기업의 실제 신용등급(예: AAA, A+, BBB 등)을 조회합니다. 예측 모델이 아닌 실제 데이터입니다."""
    return get_credit_rank(corp_name)


@tool
def tool_get_investment_data(corp_name: str) -> dict:
    """기업의 투자 지표(PER, PBR, ROE 등)를 반환합니다. 가치 평가(Valuation)에 사용됩니다."""
    return get_investment_data(corp_name)


@tool
def tool_get_stock_data(corp_name: str) -> dict:
    """기업의 최근 월별 주가 추이 데이터를 반환합니다. 모멘텀 분석 시 필요합니다."""
    return get_stock_data(corp_name)


@tool
def tool_get_employee_reviews(corp_name: str) -> dict:
    """기업의 전/현직 직원 리뷰 데이터를 반환합니다. 조직 문화, 장단점 분석 등 정성적 평가에 사용됩니다."""
    return get_employee_reviews(corp_name)


@tool
def tool_predict_credit_rating(stock_code: str, year: Optional[int] = None) -> dict:
    """stock_code로 ML 모델 기반 신용등급 예측값을 조회합니다.
    피처는 도구가 학습 테이블에서 직접 조립하므로 재무 수치를 직접 넘길 필요가 없습니다.
    실제 평가기관 등급이 아니라 모델 예측값이며, 실제 등급은 신용등급 이력 도구를 쓰세요."""
    return predict_credit_rating(stock_code, year=year)


@tool
def tool_get_feature_importance(top_n: int = 10) -> Optional[list]:
    """신용등급 예측 ML 모델에서 가장 중요하게 작용한 피쳐(변수) Top-N을 반환합니다."""
    return get_feature_importance(top_n)

# 에이전트별 제공 도구 목록. canonical context/API 계층 기반 도구를 우선 사용하고,
# 기존 CSV 조회 도구는 legacy fallback으로 남긴다.
RESEARCHER_TOOLS = [
    *CONTEXT_TOOLS,
    *DYNAMIC_TOOLS,
    tool_get_company_info,
    tool_get_financial_data,
    tool_get_industry_average,
    tool_get_credit_data,
    tool_get_credit_rank,
    tool_get_investment_data,
    tool_get_stock_data,
    tool_get_employee_reviews
]

ANALYST_TOOLS = [
    CONTEXT_TOOLS[2],
    CONTEXT_TOOLS[3],
    tool_predict_credit_rating,
    tool_get_feature_importance
]
