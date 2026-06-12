from __future__ import annotations

from typing import Any

import pandas as pd

from src.canonical.schema import DATA_DIR, make_hash, normalize_stock_code, normalize_text, normalize_year, read_canonical, report, write_csv


ALLOWED_RATINGS = {"AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC", "CC", "C", "D", "JB"}
ALLOWED_AGENCIES = {"KIS", "NICE", "IntegratedRating", "Model"}


def _add(rows: list[dict[str, Any]], *, stock_code: str | None, year: Any, agency: str, bond_type: Any, rating_value: Any = None, predicted_rating_value: Any = None, source_file: str) -> None:
    code = normalize_stock_code(stock_code)
    year_value = normalize_year(year)
    rating = normalize_text(rating_value)
    predicted = normalize_text(predicted_rating_value)
    if not code or not year_value or agency not in ALLOWED_AGENCIES:
        return
    if not rating and not predicted:
        return
    rating_id = f"{code}_{year_value}_{agency}_{make_hash(bond_type, rating, predicted, source_file, length=8)}"
    rows.append(
        {
            "rating_id": rating_id,
            "stock_code": code,
            "year": int(year_value),
            "agency": agency,
            "bond_type": normalize_text(bond_type),
            "rating_value": rating,
            "predicted_rating_value": predicted,
            "is_predicted": bool(predicted and not rating),
            "source_file": source_file,
        }
    )


def build_credit_ratings() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    path = DATA_DIR / "credit_rank.csv"
    if path.exists():
        df = pd.read_csv(path)
        for raw in df.to_dict("records"):
            _add(rows, stock_code=raw.get("stock_code"), year=raw.get("year"), agency="KIS", bond_type=raw.get("kis_bond_type"), rating_value=raw.get("kis_rank"), source_file="credit_rank.csv")
            _add(rows, stock_code=raw.get("stock_code"), year=raw.get("year"), agency="NICE", bond_type=raw.get("nice_bond_type"), rating_value=raw.get("nice_rank"), source_file="credit_rank.csv")
            _add(rows, stock_code=raw.get("stock_code"), year=raw.get("year"), agency="IntegratedRating", bond_type=None, rating_value=raw.get("rank"), source_file="credit_rank.csv")

    for source_file, agency in [("k_credit_rank.csv", "KIS"), ("n_credit_rank.csv", "NICE")]:
        path = DATA_DIR / source_file
        if path.exists():
            df = pd.read_csv(path)
            for raw in df.to_dict("records"):
                _add(rows, stock_code=raw.get("stock_code"), year=raw.get("year"), agency=agency, bond_type=raw.get("bond_type"), rating_value=raw.get("rank"), source_file=source_file)

    path = DATA_DIR / "kospi_company_info.csv"
    if path.exists():
        df = pd.read_csv(path)
        for raw in df.to_dict("records"):
            _add(rows, stock_code=raw.get("stock_code"), year=2022, agency="IntegratedRating", bond_type=None, rating_value=raw.get("rank"), source_file="kospi_company_info.csv")
            _add(rows, stock_code=raw.get("stock_code"), year=2022, agency="Model", bond_type=None, predicted_rating_value=raw.get("predicted_rank"), source_file="kospi_company_info.csv")

    ratings = pd.DataFrame(rows)
    if ratings.empty:
        ratings = pd.DataFrame(columns=["rating_id", "stock_code", "year", "agency", "bond_type", "rating_value", "predicted_rating_value", "is_predicted", "source_file"])

    companies = read_canonical("companies.csv")
    company_codes = set(companies.get("stock_code", []))
    ratings_without_company = (~ratings["stock_code"].isin(company_codes)).sum() if not ratings.empty else 0
    ratings = ratings[ratings["stock_code"].isin(company_codes)].copy()

    invalid_rating_mask = ~(ratings["rating_value"].fillna("").isin(ALLOWED_RATINGS | {""})) | ~(ratings["predicted_rating_value"].fillna("").isin(ALLOWED_RATINGS | {""}))
    ratings = ratings[~invalid_rating_mask].drop_duplicates("rating_id").sort_values(["stock_code", "year", "agency"])
    write_csv(ratings, "credit_ratings.csv")

    return {
        "credit_ratings": report(
            "credit_ratings",
            rows=len(ratings),
            invalid_rating_values=int(invalid_rating_mask.sum()),
            ratings_without_company=int(ratings_without_company),
        )
    }


if __name__ == "__main__":
    build_credit_ratings()

