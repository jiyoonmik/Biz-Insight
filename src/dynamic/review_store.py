from __future__ import annotations

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


def build_review_store() -> tuple[Path, Path]:
    DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
    raw_frames = []
    for source in ["employee_reviews.csv", "b_company_review.csv", "j_company_review.csv", "final_reviews.csv"]:
        path = DATA_DIR / source
        if not path.exists():
            continue
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
        if "stock_code" not in df.columns:
            continue
        df["stock_code"] = df["stock_code"].map(normalize_stock_code)
        df["source_file"] = source
        keep = [column for column in ["corp", "stock_code", "year", "up", "down", "source_file"] if column in df.columns]
        raw_frames.append(df[keep])
    reviews = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame(columns=["corp", "stock_code", "year", "up", "down", "source_file"])
    reviews = reviews.dropna(subset=["stock_code"]).drop_duplicates()
    reviews.to_parquet(REVIEWS_PARQUET, index=False)

    summary_path = CANONICAL_DIR / "employee_review_summaries.csv"
    if not summary_path.exists():
        build_reviews()
    summaries = pd.read_csv(summary_path, dtype={"stock_code": str}) if summary_path.exists() else pd.DataFrame()
    summaries.to_parquet(REVIEW_SUMMARIES_PARQUET, index=False)
    print(f"reviews: rows={len(reviews)} path={REVIEWS_PARQUET}")
    print(f"review_summaries: rows={len(summaries)} path={REVIEW_SUMMARIES_PARQUET}")
    return REVIEWS_PARQUET, REVIEW_SUMMARIES_PARQUET


def _reviews() -> pd.DataFrame:
    if REVIEWS_PARQUET.exists():
        return pd.read_parquet(REVIEWS_PARQUET)
    if REVIEWS_CSV.exists():
        return pd.read_csv(REVIEWS_CSV, dtype={"stock_code": str})
    if not REVIEWS_PARQUET.exists():
        build_review_store()
    return pd.read_parquet(REVIEWS_PARQUET)


def _summaries() -> pd.DataFrame:
    if REVIEW_SUMMARIES_PARQUET.exists():
        return pd.read_parquet(REVIEW_SUMMARIES_PARQUET)
    if REVIEW_SUMMARIES_CSV.exists():
        return pd.read_csv(REVIEW_SUMMARIES_CSV, dtype={"stock_code": str})
    if not REVIEW_SUMMARIES_PARQUET.exists():
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


def search_review_evidence(stock_code: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    code = normalize_stock_code(stock_code)
    needle = normalize_text(query)
    if not code or not needle:
        return []
    df = _reviews()
    subset = df[df["stock_code"].astype(str).str.zfill(6) == code].copy()
    if subset.empty:
        return []
    lowered = needle.lower()
    subset["text"] = subset[["up", "down"]].fillna("").agg(" ".join, axis=1)
    subset["score"] = subset["text"].str.lower().map(lambda text: text.count(lowered))
    matched = subset[subset["score"] > 0].sort_values(["score"], ascending=False).head(top_k)
    if matched.empty:
        matched = subset.head(top_k)
    return matched[["corp", "stock_code", "year", "up", "down", "source_file", "score"]].where(pd.notna(matched), None).to_dict("records")


if __name__ == "__main__":
    build_review_store()
