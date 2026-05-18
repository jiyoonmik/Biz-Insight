"""
CSV 데이터 조회 도구
- LLM 호출 없이 Pandas로 직접 데이터를 조회/계산하여 LLM 호출 최소화
"""
import os
import pandas as pd
from typing import Optional
from src.config import DATA_DIR, CSV_FILES


def _load_csv(key: str) -> pd.DataFrame:
    """CSV_FILES 키로 DataFrame을 로드합니다."""
    filename = CSV_FILES.get(key)
    if not filename:
        raise ValueError(f"Unknown CSV key: {key}")
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _find_by_corp_name(df: pd.DataFrame, corp_name: str) -> pd.DataFrame:
    """
    DataFrame에서 기업명으로 행을 검색합니다.
    corp/회사명 컬럼을 우선 검색하고, 없으면 모든 문자열 컬럼을 탐색합니다.
    """
    # 우선순위 컬럼 목록
    priority_cols = ["corp", "회사명", "company", "기업명", "corp_name"]
    for col in priority_cols:
        if col in df.columns:
            match = df[df[col].str.contains(corp_name, na=False)]
            if not match.empty:
                return match

    # fallback: 모든 object 컬럼 탐색
    for col in df.columns:
        if df[col].dtype == object:
            try:
                match = df[df[col].str.contains(corp_name, na=False)]
                if not match.empty:
                    return match
            except Exception:
                continue

    return pd.DataFrame()


# ── 기업 기본 정보 ────────────────────────────────────────
def get_company_info(corp_name: str) -> dict:
    """기업 기본 정보(종목코드, 업종, 대표자, 주요제품 등)를 반환합니다."""
    df = _load_csv("company_info")
    match = df[df["회사명"].str.contains(corp_name, na=False)]
    if match.empty:
        return {"error": f"'{corp_name}' 기업 정보를 찾을 수 없습니다."}

    exact = match[match["회사명"] == corp_name]
    row = exact.iloc[0] if not exact.empty else match.iloc[0]
    return row.to_dict()


# ── 재무제표 데이터 ───────────────────────────────────────
def get_financial_data(corp_name: str) -> dict:
    """기업의 재무제표 주요 데이터를 반환합니다."""
    df = _load_csv("fs")
    match = _find_by_corp_name(df, corp_name)

    if match.empty:
        return {"error": f"'{corp_name}' 재무데이터를 찾을 수 없습니다.", "available_columns": list(df.columns[:10])}

    return {
        "data": match.head(20).to_dict(orient="records"),
        "columns": list(match.columns),
        "row_count": len(match),
    }


def get_industry_average(sector: str) -> dict:
    """동일 산업군 평균 지표를 반환합니다."""
    df = _load_csv("industry_average")
    match = df[df["sector"].str.contains(sector, na=False)] if "sector" in df.columns else pd.DataFrame()
    if match.empty:
        return {"error": f"'{sector}' 산업 평균 데이터를 찾을 수 없습니다."}
    return {"data": match.to_dict(orient="records")}


# ── 신용 데이터 ───────────────────────────────────────────
def get_credit_data(corp_name: str) -> dict:
    """기업의 신용 관련 데이터를 반환합니다."""
    df = _load_csv("credit_data_web")
    match = _find_by_corp_name(df, corp_name)

    if match.empty:
        return {"error": f"'{corp_name}' 신용데이터를 찾을 수 없습니다."}

    return {
        "data": match.head(20).to_dict(orient="records"),
        "columns": list(match.columns),
        "row_count": len(match),
    }


def get_credit_rank(corp_name: str) -> Optional[str]:
    """기업의 실제 신용등급을 조회합니다. 없으면 None 반환."""
    df = _load_csv("credit_rank")
    match = _find_by_corp_name(df, corp_name)
    if match.empty:
        return None

    row = match.iloc[0]
    for rank_col in ["rank", "rating", "등급", "credit_rank"]:
        if rank_col in row.index and pd.notna(row[rank_col]):
            return str(row[rank_col])
    return None


# ── 투자 데이터 ───────────────────────────────────────────
def get_investment_data(corp_name: str) -> dict:
    """기업의 투자/주식 관련 데이터를 반환합니다."""
    df = _load_csv("investment_data_web")
    match = _find_by_corp_name(df, corp_name)

    if match.empty:
        return {"error": f"'{corp_name}' 투자데이터를 찾을 수 없습니다."}

    return {
        "data": match.head(20).to_dict(orient="records"),
        "columns": list(match.columns),
    }


def get_stock_data(corp_name: str) -> dict:
    """기업의 월별 주가 데이터를 반환합니다."""
    df = _load_csv("stock_data")
    match = _find_by_corp_name(df, corp_name)

    if match.empty:
        return {"error": f"'{corp_name}' 주가데이터를 찾을 수 없습니다."}

    return {
        "data": match.tail(24).to_dict(orient="records"),  # 최근 24개월
        "columns": list(match.columns),
    }


# ── 직원 리뷰 데이터 ─────────────────────────────────────
def get_employee_reviews(corp_name: str) -> dict:
    """기업의 직원 리뷰 데이터를 반환합니다."""
    df = _load_csv("employee_reviews")
    match = _find_by_corp_name(df, corp_name)

    if match.empty:
        return {"error": f"'{corp_name}' 직원리뷰를 찾을 수 없습니다."}

    return {
        "data": match.head(30).to_dict(orient="records"),
        "columns": list(match.columns),
        "total_reviews": len(match),
    }


# ── 종합 시각화 데이터 ────────────────────────────────────
def get_web_visualization_data(corp_name: str) -> dict:
    """웹 시각화용 종합 데이터를 반환합니다."""
    df = _load_csv("web_visualization")
    match = _find_by_corp_name(df, corp_name)

    if match.empty:
        return {"error": f"'{corp_name}' 시각화 데이터를 찾을 수 없습니다."}

    return {
        "data": match.head(20).to_dict(orient="records"),
        "columns": list(match.columns),
    }
