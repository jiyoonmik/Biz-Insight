from __future__ import annotations

from typing import Any

import pandas as pd

from src.canonical.schema import DATA_DIR, make_hash, normalize_stock_code, normalize_text, read_canonical, report, safe_float, write_csv


def _join_samples(values: pd.Series, limit: int = 3) -> str | None:
    samples = []
    for value in values:
        text = normalize_text(value)
        if text and text not in samples:
            samples.append(text)
        if len(samples) >= limit:
            break
    if not samples:
        return None
    return " / ".join(samples)


def build_reviews() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    ratings: dict[str, dict[str, Any]] = {}

    for source_file in ["final_rating.csv", "rating_filled.csv", "company_rating.csv"]:
        path = DATA_DIR / source_file
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for raw in df.to_dict("records"):
            code = normalize_stock_code(raw.get("stock_code"))
            if not code or code in ratings:
                continue
            ratings[code] = {
                "rating": safe_float(raw.get("rating") or raw.get("jobp_rating") or raw.get("blind_rating")),
                "review_count": int(safe_float(raw.get("count") or raw.get("jobp_cnt") or raw.get("blind_cnt") or 0) or 0),
                "source_file": source_file,
            }

    for source_file in ["final_reviews.csv", "employee_reviews.csv"]:
        path = DATA_DIR / source_file
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["stock_code"] = df["stock_code"].map(normalize_stock_code)
        df["year"] = df["year"].astype(str)
        grouped = df.dropna(subset=["stock_code", "year"]).groupby(["stock_code", "year"], dropna=True)
        for (stock_code, year), group in grouped:
            rating = ratings.get(stock_code, {})
            rows.append(
                {
                    "summary_id": f"{stock_code}_{year}_{make_hash(source_file, stock_code, year, length=8)}",
                    "stock_code": stock_code,
                    "period_type": "year",
                    "period_value": year,
                    "positive_summary": _join_samples(group.get("up", pd.Series(dtype=str))),
                    "negative_summary": _join_samples(group.get("down", pd.Series(dtype=str))),
                    "rating": rating.get("rating"),
                    "review_count": int(group.shape[0]),
                    "source_file": f"{source_file};{rating.get('source_file', '')}".strip(";"),
                }
            )
        break

    summaries = pd.DataFrame(rows)
    if summaries.empty:
        summaries = pd.DataFrame(columns=["summary_id", "stock_code", "period_type", "period_value", "positive_summary", "negative_summary", "rating", "review_count", "source_file"])

    companies = read_canonical("companies.csv")
    company_codes = set(companies.get("stock_code", []))
    without_company = (~summaries["stock_code"].isin(company_codes)).sum() if not summaries.empty else 0
    summaries = summaries[summaries["stock_code"].isin(company_codes)].drop_duplicates("summary_id").sort_values(["stock_code", "period_value"])
    write_csv(summaries, "employee_review_summaries.csv")

    return {
        "reviews": report(
            "reviews",
            rows=len(summaries),
            summaries_without_company=int(without_company),
        )
    }


if __name__ == "__main__":
    build_reviews()

