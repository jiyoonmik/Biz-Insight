from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd

from src.canonical.schema import (
    CANONICAL_DIR,
    DATA_DIR,
    normalize_stock_code,
    normalize_text,
    report,
    slugify,
    write_csv,
)


COMPANY_COLUMNS = [
    "stock_code",
    "preferred_name",
    "sector_code",
    "sector_name",
    "main_product",
    "listing_date",
    "settlement_month",
    "representative_name",
    "homepage",
    "region",
    "source_file",
]


def _read(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _set_if_missing(target: dict[str, Any], key: str, value: Any) -> None:
    if normalize_text(target.get(key)) is None and normalize_text(value) is not None:
        target[key] = normalize_text(value)


def _upsert_company(
    companies: OrderedDict[str, dict[str, Any]],
    stock_code: str | None,
    source_file: str,
    **values: Any,
) -> None:
    if stock_code is None:
        return
    row = companies.setdefault(
        stock_code,
        {
            "stock_code": stock_code,
            "preferred_name": None,
            "sector_code": None,
            "sector_name": None,
            "main_product": None,
            "listing_date": None,
            "settlement_month": None,
            "representative_name": None,
            "homepage": None,
            "region": None,
            "source_file": source_file,
        },
    )
    if row["source_file"] != source_file:
        row["source_file"] = f"{row['source_file']};{source_file}"
    for key, value in values.items():
        _set_if_missing(row, key, value)
    if row.get("sector_name"):
        row["sector_code"] = slugify(row["sector_name"])


def build_companies() -> dict[str, Any]:
    companies: OrderedDict[str, dict[str, Any]] = OrderedDict()
    aliases: list[dict[str, str]] = []

    def add_alias(stock_code: str | None, alias: Any, source_file: str, alias_type: str) -> None:
        alias_name = normalize_text(alias)
        if stock_code and alias_name:
            aliases.append(
                {
                    "stock_code": stock_code,
                    "alias_name": alias_name,
                    "alias_type": alias_type,
                    "source_file": source_file,
                }
            )

    kospi = _read("kospi_company_info.csv")
    for row in kospi.to_dict("records"):
        code = normalize_stock_code(row.get("stock_code"))
        _upsert_company(
            companies,
            code,
            "kospi_company_info.csv",
            preferred_name=row.get("corp"),
            sector_name=row.get("sector"),
            main_product=row.get("main_product"),
            listing_date=row.get("listing_date"),
            settlement_month=row.get("settlement_month"),
            representative_name=row.get("representative_name"),
            homepage=row.get("homepage"),
            region=row.get("region"),
        )
        add_alias(code, row.get("corp"), "kospi_company_info.csv", "preferred")

    company_info = _read("company_info.csv")
    for row in company_info.to_dict("records"):
        code = normalize_stock_code(row.get("종목코드"))
        _upsert_company(
            companies,
            code,
            "company_info.csv",
            preferred_name=row.get("회사명"),
            sector_name=row.get("업종"),
            main_product=row.get("주요제품"),
            listing_date=row.get("상장일"),
            settlement_month=row.get("결산월"),
            representative_name=row.get("대표자명"),
            homepage=row.get("홈페이지"),
            region=row.get("지역"),
        )
        add_alias(code, row.get("회사명"), "company_info.csv", "source_name")

    sector = _read("sector.csv")
    for row in sector.to_dict("records"):
        code = normalize_stock_code(row.get("종목코드"))
        _upsert_company(
            companies,
            code,
            "sector.csv",
            preferred_name=row.get("종목명"),
            sector_name=row.get("업종명"),
        )
        if code in companies and normalize_text(row.get("업종명")):
            companies[code]["sector_name"] = normalize_text(row.get("업종명"))
            companies[code]["sector_code"] = slugify(row.get("업종명"))
        add_alias(code, row.get("종목명"), "sector.csv", "source_name")

    sector_revenue = _read("sector_revenue_aggregation.csv")
    for row in sector_revenue.to_dict("records"):
        code = normalize_stock_code(row.get("stock_code"))
        _upsert_company(
            companies,
            code,
            "sector_revenue_aggregation.csv",
            preferred_name=row.get("corp"),
            sector_name=row.get("sector"),
            main_product=row.get("main_product"),
            listing_date=row.get("listing_date"),
            settlement_month=row.get("settlement_month"),
            representative_name=row.get("representative_name"),
            homepage=row.get("homepage"),
            region=row.get("region"),
        )
        add_alias(code, row.get("corp"), "sector_revenue_aggregation.csv", "source_name")

    companies_df = pd.DataFrame(companies.values(), columns=COMPANY_COLUMNS)
    companies_df = companies_df.dropna(subset=["stock_code", "preferred_name"]).sort_values("stock_code")

    aliases_df = pd.DataFrame(aliases)
    if not aliases_df.empty:
        aliases_df = aliases_df.drop_duplicates(["stock_code", "alias_name", "source_file"]).sort_values(
            ["stock_code", "alias_name", "source_file"]
        )

    sector_rows = (
        companies_df[["sector_code", "sector_name", "source_file"]]
        .dropna(subset=["sector_code", "sector_name"])
        .drop_duplicates(["sector_code"])
        .rename(columns={"sector_name": "source_name"})
    )
    sector_rows["sector_name"] = sector_rows["source_name"]
    sectors_df = sector_rows[["sector_code", "sector_name", "source_name", "source_file"]].sort_values("sector_code")

    write_csv(companies_df, "companies.csv")
    write_csv(aliases_df, "company_aliases.csv")
    write_csv(sectors_df, "sectors.csv")

    aliases_without_company = 0
    if not aliases_df.empty:
        aliases_without_company = int((~aliases_df["stock_code"].isin(companies_df["stock_code"])).sum())

    return {
        "companies": report(
            "companies",
            rows=len(companies_df),
            unique_stock_codes=companies_df["stock_code"].nunique(),
            duplicate_stock_codes=int(companies_df["stock_code"].duplicated().sum()),
            missing_preferred_name=int(companies_df["preferred_name"].isna().sum()),
        ),
        "company_aliases": report(
            "company_aliases",
            rows=len(aliases_df),
            unique_aliases=aliases_df[["stock_code", "alias_name"]].drop_duplicates().shape[0]
            if not aliases_df.empty
            else 0,
            aliases_without_company=aliases_without_company,
        ),
        "sectors": report("sectors", rows=len(sectors_df), unique_sector_codes=sectors_df["sector_code"].nunique()),
        "output_dir": str(CANONICAL_DIR),
    }


if __name__ == "__main__":
    build_companies()
