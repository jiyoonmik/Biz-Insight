"""직원 후기 dynamic store.

이 프로젝트의 원래 차별점은 정규화된 재무 도메인을 **비정형 자연어 후기**로
보강하는 것이다. 그런데 후기 파이프라인이 그 정체성을 배신하고 있었다.

1. 가장 리치한 소스(b_company_review, j_company_review — 직무·재직상태·평점·
   한줄평·장단점을 담는다)가 컬럼명 불일치(`blind_up`/`jobp_up` vs `up`)로
   조용히 누락되고 있었다.
2. 검색은 부분문자열 카운트라, 질의가 후기 원문과 글자 그대로 겹쳐야만 걸렸다.
   "커리어 향상 부족"으로 물으면 후기의 "커리어 향상이 전혀 안 된다"조차 못 찾았다
   (띄어쓰기·조사가 달라 substring이 0). 자연어를 자연어로 검색하지 못한 것이다.

그래서 세 소스를 공통 리치 스키마로 통합하고, 검색을 문자 n-gram TF-IDF 유사도로
승격했다. 문자 2~5그램을 쓰므로 조사·어미·띄어쓰기·부분어에 강인하다 — 정확한
substring이 없어도 "커리어 향상"이 "커리어 향상이"와 높게 매칭된다. 다만 이것은
어휘 유사도이지 임베딩 수준의 교차어휘 의미 매칭이 아니다("성장"과 "커리어 향상"을
같은 뜻으로 이해하지는 못한다). 임베딩 모델 다운로드 없이 얻는 현실적 향상이다.

후기 검색은 "이 회사 후기 중 질의와 가까운 것"을 찾는 문제이므로, 전체를 미리
인덱싱하는 대신 회사 단위로 즉석 인덱싱한다. 회사당 후기는 보통 수십~수백 건이라
fit이 싸다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.canonical.build_reviews import build_reviews
from src.canonical.schema import CANONICAL_DIR, DATA_DIR, normalize_stock_code, normalize_text


DYNAMIC_DIR = DATA_DIR / "dynamic"
REVIEWS_PARQUET = DYNAMIC_DIR / "reviews.parquet"
REVIEW_SUMMARIES_PARQUET = DYNAMIC_DIR / "review_summaries.parquet"
REVIEWS_CSV = DYNAMIC_DIR / "reviews.csv"
REVIEW_SUMMARIES_CSV = DYNAMIC_DIR / "review_summaries.csv"

# 후기 소스를 공통 스키마로 맞추기 위한 매핑. 원본마다 장/단점 컬럼명이 다르고
# (up/down, blind_up/down, jobp_up/down), b_/j_에만 직무·재직상태·평점·한줄평이 있다.
#
# employee_reviews.csv는 제외한다. final_reviews.csv가 그 정제본이며(표본 대조에서
# 100% 중복), 둘을 함께 넣으면 같은 후기를 두 번 세게 된다. 리치 필드를 담은
# b_/j_company_review가 이 프로젝트가 원래 살리려던 자연어 후기의 본체다.
REVIEW_SOURCES = [
    {"file": "final_reviews.csv", "pros": "up", "cons": "down"},
    {"file": "b_company_review.csv", "pros": "blind_up", "cons": "blind_down"},
    {"file": "j_company_review.csv", "pros": "jobp_up", "cons": "jobp_down"},
]

# 개별 후기 하나가 갖는 통합 스키마.
REVIEW_COLUMNS = [
    "corp", "stock_code", "period_value", "review_date",
    "position", "status", "rating", "summary", "pros", "cons", "source_file",
]

# b_/j_에만 있는 리치 필드. 없는 소스에서는 None으로 채운다.
RICH_COLUMNS = ["position", "status", "rating", "summary"]

_YEAR = re.compile(r"(19|20)\d{2}")


def _year_series(df: pd.DataFrame) -> pd.Series:
    """year 또는 date 컬럼에서 4자리 연도를 뽑는다."""
    for column in ("year", "date"):
        if column in df.columns:
            return df[column].astype(str).str.extract(r"((?:19|20)\d{2})")[0]
    return pd.Series([None] * len(df), index=df.index)


def _normalize_source(df: pd.DataFrame, pros: str, cons: str, source_file: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["corp"] = df["corp"] if "corp" in df.columns else None
    out["stock_code"] = df["stock_code"].map(normalize_stock_code)
    out["period_value"] = _year_series(df)
    out["review_date"] = df["date"] if "date" in df.columns else None
    for column in RICH_COLUMNS:
        out[column] = df[column] if column in df.columns else None
    out["pros"] = df[pros] if pros in df.columns else None
    out["cons"] = df[cons] if cons in df.columns else None
    out["source_file"] = source_file
    return out[REVIEW_COLUMNS]


def build_review_store() -> tuple[Path, Path]:
    DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for source in REVIEW_SOURCES:
        path = DATA_DIR / source["file"]
        if not path.exists():
            continue
        # on_bad_lines="skip": 자유텍스트에 escape 안 된 콤마/개행이 있는 원본이 있다.
        # j_company_review.csv가 특히 심해 유효 stock_code 행이 사실상 남지 않는다.
        # 깨진 소스는 best-effort로 통과시키되(살릴 수 있는 만큼만 반영), 파이프라인을
        # 멈추지는 않는다. 리치 후기의 본체는 b_company_review에서 온다.
        df = pd.read_csv(path, engine="python", on_bad_lines="skip", dtype=str)
        if "stock_code" not in df.columns:
            continue
        frames.append(_normalize_source(df, source["pros"], source["cons"], source["file"]))

    if frames:
        reviews = pd.concat(frames, ignore_index=True)
    else:
        reviews = pd.DataFrame(columns=REVIEW_COLUMNS)

    reviews = reviews.dropna(subset=["stock_code"])
    # 텍스트가 하나도 없는 행은 검색·근거로 쓸모가 없으므로 버린다.
    text_cols = reviews[["summary", "pros", "cons"]].fillna("")
    has_text = text_cols.apply(lambda col: col.str.strip()).ne("").any(axis=1)
    reviews = reviews[has_text]
    # 소스가 겹치거나 원본에 중복이 있어도 같은 후기를 두 번 세지 않는다.
    reviews = reviews.drop_duplicates(subset=["stock_code", "pros", "cons", "summary"]).reset_index(drop=True)
    reviews.to_parquet(REVIEWS_PARQUET, index=False)

    summary_path = CANONICAL_DIR / "employee_review_summaries.csv"
    if not summary_path.exists():
        build_reviews()
    summaries = pd.read_csv(summary_path, dtype={"stock_code": str}) if summary_path.exists() else pd.DataFrame()
    summaries.to_parquet(REVIEW_SUMMARIES_PARQUET, index=False)

    by_source = reviews["source_file"].value_counts().to_dict() if not reviews.empty else {}
    print(f"reviews: rows={len(reviews)} by_source={by_source} path={REVIEWS_PARQUET}")
    print(f"review_summaries: rows={len(summaries)} path={REVIEW_SUMMARIES_PARQUET}")
    return REVIEWS_PARQUET, REVIEW_SUMMARIES_PARQUET


def _reviews() -> pd.DataFrame:
    if REVIEWS_PARQUET.exists():
        return pd.read_parquet(REVIEWS_PARQUET)
    if REVIEWS_CSV.exists():
        return pd.read_csv(REVIEWS_CSV, dtype={"stock_code": str})
    build_review_store()
    return pd.read_parquet(REVIEWS_PARQUET)


def _summaries() -> pd.DataFrame:
    if REVIEW_SUMMARIES_PARQUET.exists():
        return pd.read_parquet(REVIEW_SUMMARIES_PARQUET)
    if REVIEW_SUMMARIES_CSV.exists():
        return pd.read_csv(REVIEW_SUMMARIES_CSV, dtype={"stock_code": str})
    build_review_store()
    return pd.read_parquet(REVIEW_SUMMARIES_PARQUET)


def get_review_summary(stock_code: str, period: str | None = None) -> dict[str, Any]:
    code = normalize_stock_code(stock_code)
    if not code:
        return {"available": False}
    df = _summaries()
    subset = df[df["stock_code"].astype(str).str.zfill(6) == code]
    if period:
        subset = subset[subset["period_value"].astype(str) == str(period)]
    if subset.empty:
        return {"stock_code": code, "period": period, "available": False}
    row = subset.sort_values("period_value").tail(1).iloc[0].where(pd.notna(subset.tail(1).iloc[0]), None).to_dict()
    row["available"] = True
    return row


def _tfidf_rank(texts: list[str], query: str, top_k: int) -> list[tuple[int, float]] | None:
    """char n-gram TF-IDF + cosine으로 상위 top_k (index, score)를 돌려준다.

    어휘 유사도이지 임베딩 의미 매칭이 아니다. 문자 2~5그램이라 조사·띄어쓰기·
    부분어에는 강인하지만 서로 다른 단어의 동의성은 잡지 못한다.

    scikit-learn을 못 쓰면 None을 돌려주고 호출부가 부분문자열로 폴백한다.
    vectorizer 설정은 RAG 벡터스토어(src/rag/vectorstore.py)와 동일하게 맞춰
    한/영 혼합 후기에 같은 특성을 쓴다.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True, norm="l2",
    )
    try:
        matrix = vectorizer.fit_transform(texts)
    except ValueError:
        return None  # 어휘가 비었을 때
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    order = scores.argsort()[::-1][:top_k]
    return [(int(index), float(scores[index])) for index in order if scores[index] > 0]


def _substring_rank(texts: list[str], query: str, top_k: int) -> list[tuple[int, float]]:
    lowered = query.lower()
    scored = [(index, float(text.lower().count(lowered))) for index, text in enumerate(texts)]
    scored.sort(key=lambda item: item[1], reverse=True)
    hits = [item for item in scored if item[1] > 0][:top_k]
    return hits or [(index, 0.0) for index in range(min(top_k, len(texts)))]


def rank_review_rows(subset: pd.DataFrame, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """회사 후기 subset을 질의와의 TF-IDF 유사도로 정렬해 상위 top_k를 돌려준다.

    데이터 로딩과 분리한 순수 함수라 parquet 없이 테스트할 수 있다.
    """
    if subset.empty:
        return []
    subset = subset.copy()
    subset["text"] = (
        subset[["summary", "pros", "cons"]].fillna("").agg(" ".join, axis=1).str.strip()
    )
    subset = subset[subset["text"] != ""].reset_index(drop=True)
    if subset.empty:
        return []

    texts = subset["text"].tolist()
    ranked = _tfidf_rank(texts, query, top_k)
    method = "tfidf"
    if ranked is None:
        ranked = _substring_rank(texts, query, top_k)
        method = "substring"

    results = []
    for index, score in ranked:
        row = subset.iloc[index]
        record = {column: row.get(column) for column in REVIEW_COLUMNS}
        record["relevance"] = round(score, 4)
        record["match_method"] = method
        results.append({key: (None if pd.isna(value) else value) for key, value in record.items()})
    return results


def search_review_evidence(stock_code: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    code = normalize_stock_code(stock_code)
    needle = normalize_text(query)
    if not code or not needle:
        return []
    df = _reviews()
    subset = df[df["stock_code"].astype(str).str.zfill(6) == code]
    return rank_review_rows(subset, needle, top_k=top_k)


if __name__ == "__main__":
    build_review_store()
