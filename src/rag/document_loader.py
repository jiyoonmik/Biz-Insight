"""Build retrieval documents from the legacy CSV datasets.

The current data is tabular and mostly company-centric.  For RAG, each document
is a compact textual projection of rows grouped by company and source dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATASETS = [
    "company_info.csv",
    "kospi_company_info.csv",
    "fs.csv",
    "main_fs.csv",
    "bs.csv",
    "incs.csv",
    "cf.csv",
    "credit_data_web.csv",
    "credit_data_model.csv",
    "credit_model_a.csv",
    "credit_model_b.csv",
    "credit_rank.csv",
    "k_credit_rank.csv",
    "n_credit_rank.csv",
    "investment_data_web.csv",
    "investment_data_model.csv",
    "stock_data_per_month.csv",
    "employee_reviews.csv",
    "final_reviews.csv",
    "company_rating.csv",
    "final_rating.csv",
    "rating_filled.csv",
    "web_visualization.csv",
    "industry_average_year.csv",
    "sector_revenue_top_features.csv",
    "economic_indicators.csv",
    "kr_policyratio_month.csv",
    "us_policyratio_month.csv",
    "kr_standard_yield.csv",
    "uskor_exchange_year.csv",
    "ppi_year.csv",
    "crb_index.csv",
    "outstanding_shares.csv",
    "sector.csv",
]

DATASET_DESCRIPTIONS = {
    "company_info.csv": "기업개요 회사 기본정보 대표자 업종 주요제품 상장일 지역",
    "kospi_company_info.csv": "KOSPI 기업개요 회사 기본정보 신용등급 예측등급",
    "fs.csv": "재무제표 전체 재무상태표 손익계산서 현금흐름표 매출 자산 부채 이익",
    "main_fs.csv": "주요 재무제표 재무지표 매출 자산 부채 이익 YoY",
    "bs.csv": "재무상태표 자산 부채 자본 재무 안정성",
    "incs.csv": "손익계산서 매출 영업이익 순이익 수익성 성장성",
    "cf.csv": "현금흐름표 영업현금흐름 투자현금흐름 재무현금흐름",
    "credit_data_web.csv": "신용분석 신용데이터 재무건전성 차입금 부채 상환능력",
    "credit_data_model.csv": "신용모델 신용등급 예측 피쳐 재무건전성 부채 상환능력",
    "credit_model_a.csv": "신용모델 A 신용등급 예측 학습데이터",
    "credit_model_b.csv": "신용모델 B 신용등급 예측 학습데이터",
    "credit_rank.csv": "신용등급 KIS NICE 실제등급 채권등급",
    "k_credit_rank.csv": "한국신용평가 신용등급 채권등급",
    "n_credit_rank.csv": "NICE 신용등급 채권등급",
    "investment_data_web.csv": "투자지표 PER PBR ROE 밸류에이션 투자매력",
    "investment_data_model.csv": "투자모델 투자지표 수익성 안정성 성장성 밸류에이션",
    "stock_data_per_month.csv": "월별 주가 시가 고가 저가 종가 거래량 외국인보유",
    "employee_reviews.csv": "직원리뷰 기업문화 장점 단점 워라밸 복지",
    "final_reviews.csv": "직원리뷰 요약 장점 단점 기업문화",
    "b_company_review.csv": "블라인드 직원리뷰 평점 장점 단점 기업문화",
    "j_company_review.csv": "잡플래닛 직원리뷰 평점 장점 단점 기업문화",
    "company_rating.csv": "직원평점 잡플래닛 블라인드 복지 워라밸 문화 경영진 추천",
    "final_rating.csv": "직원평점 복지 워라밸 문화 성장 경영진 추천",
    "rating_filled.csv": "직원평점 결측보정 복지 워라밸 문화 성장 경영진 추천",
    "web_visualization.csv": "웹시각화 재무 신용 투자 지표",
    "industry_average_year.csv": "산업평균 재무지표 업종평균 비교",
    "sector_revenue_top_features.csv": "산업 매출그룹 상관관계 주요피쳐",
    "economic_indicators.csv": "경제지표 최저임금 환율 생산자물가 금리 국고채",
    "kr_policyratio_month.csv": "한국 기준금리 정책금리 월별",
    "us_policyratio_month.csv": "미국 기준금리 정책금리 월별",
    "kr_standard_yield.csv": "한국 국고채 금리 수익률",
    "uskor_exchange_year.csv": "원달러 환율 평균",
    "ppi_year.csv": "생산자물가지수 PPI",
    "crb_index.csv": "CRB 원자재 지수",
    "outstanding_shares.csv": "발행주식수 상장주식수",
    "sector.csv": "업종 시장구분 시가총액 종가 등락률",
}

ENTITY_COLUMNS = [
    "corp",
    "회사명",
    "corp_name",
    "company_name",
    "기업명",
    "종목명",
]

STOCK_CODE_COLUMNS = ["stock_code", "종목코드"]
SECTOR_COLUMNS = ["sector", "업종", "업종명"]


@dataclass(frozen=True)
class RagDocument:
    """A text document plus metadata for retrieval."""

    doc_id: str
    text: str
    source_file: str
    entity: str | None = None
    stock_code: str | None = None
    sector: str | None = None
    row_count: int = 0
    chunk_index: int = 0

    def to_record(self) -> dict:
        return asdict(self)


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """Read a CSV while tolerating legacy Korean encodings."""
    errors: list[str] = []
    for encoding in ("utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
        except Exception:
            raise
    raise UnicodeDecodeError(
        "csv",
        b"",
        0,
        1,
        f"failed to decode {path}: {'; '.join(errors)}",
    )


def build_documents(
    data_dir: Path,
    dataset_files: Iterable[str] | None = None,
    max_rows_per_doc: int = 80,
) -> list[RagDocument]:
    """Build RAG documents from CSV files in ``data_dir``."""
    files = list(dataset_files or DEFAULT_DATASETS)
    documents: list[RagDocument] = []

    for filename in files:
        path = data_dir / filename
        if not path.exists():
            continue

        df = read_csv_with_fallback(path)
        if df.empty:
            continue

        df = _normalize_dataframe(df)
        entity_col = _first_existing_column(df, ENTITY_COLUMNS)
        if entity_col:
            documents.extend(_documents_by_entity(filename, df, entity_col, max_rows_per_doc))
        else:
            documents.extend(_documents_by_chunks(filename, df, max_rows_per_doc))

    return documents


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[col for col in df.columns if str(col).startswith("Unnamed:")], errors="ignore")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace({"nan": None, "None": None})
    return df


def _documents_by_entity(
    filename: str,
    df: pd.DataFrame,
    entity_col: str,
    max_rows_per_doc: int,
) -> list[RagDocument]:
    documents: list[RagDocument] = []
    grouped = df.groupby(entity_col, dropna=True, sort=True)

    for entity, group in grouped:
        entity_text = _clean_value(entity)
        if not entity_text or _looks_like_bad_entity(entity_text):
            continue

        chunks = _chunk_dataframe(group, max_rows_per_doc)
        for chunk_index, chunk in enumerate(chunks):
            doc_id = _doc_id(filename, entity_text, chunk_index)
            stock_code = _first_non_empty(chunk, STOCK_CODE_COLUMNS)
            sector = _first_non_empty(chunk, SECTOR_COLUMNS)
            text = _format_group_text(filename, entity_text, chunk, stock_code, sector)
            documents.append(
                RagDocument(
                    doc_id=doc_id,
                    text=text,
                    source_file=filename,
                    entity=entity_text,
                    stock_code=stock_code,
                    sector=sector,
                    row_count=len(chunk),
                    chunk_index=chunk_index,
                )
            )

    return documents


def _documents_by_chunks(filename: str, df: pd.DataFrame, max_rows_per_doc: int) -> list[RagDocument]:
    documents: list[RagDocument] = []
    for chunk_index, chunk in enumerate(_chunk_dataframe(df, max_rows_per_doc)):
        doc_id = _doc_id(filename, "global", chunk_index)
        text = _format_group_text(filename, None, chunk, None, None)
        documents.append(
            RagDocument(
                doc_id=doc_id,
                text=text,
                source_file=filename,
                row_count=len(chunk),
                chunk_index=chunk_index,
            )
        )
    return documents


def _format_group_text(
    filename: str,
    entity: str | None,
    df: pd.DataFrame,
    stock_code: str | None,
    sector: str | None,
) -> str:
    header_parts = [f"source={filename}"]
    description = DATASET_DESCRIPTIONS.get(filename)
    if description:
        header_parts.append(f"dataset_description={description}")
    if entity:
        header_parts.append(f"company={entity}")
    if stock_code:
        header_parts.append(f"stock_code={stock_code}")
    if sector:
        header_parts.append(f"sector={sector}")

    lines = [" | ".join(header_parts)]
    lines.extend(_row_to_text(row) for row in df.to_dict(orient="records"))
    return "\n".join(line for line in lines if line)


def _row_to_text(row: dict) -> str:
    parts = []
    for key, value in row.items():
        clean = _clean_value(value)
        if clean:
            parts.append(f"{key}: {clean}")
    return "; ".join(parts)


def _chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> list[pd.DataFrame]:
    return [df.iloc[start : start + chunk_size] for start in range(0, len(df), chunk_size)]


def _first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _first_non_empty(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    col = _first_existing_column(df, candidates)
    if not col:
        return None
    for value in df[col].tolist():
        clean = _clean_value(value)
        if clean:
            return clean
    return None


def _clean_value(value) -> str | None:
    if value is None:
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return text


def _looks_like_bad_entity(entity: str) -> bool:
    """Filter rows where malformed CSV parsing put review text into the entity column."""
    return "\n" in entity or "\r" in entity or len(entity) > 60


def _doc_id(filename: str, entity: str, chunk_index: int) -> str:
    safe_entity = "".join(ch if ch.isalnum() else "_" for ch in entity)[:80]
    return f"{Path(filename).stem}:{safe_entity}:{chunk_index}"
