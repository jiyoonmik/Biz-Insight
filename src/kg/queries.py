from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from src.canonical.schema import CANONICAL_DIR, normalize_stock_code, normalize_text


@lru_cache(maxsize=32)
def _table(name: str) -> pd.DataFrame:
    path = CANONICAL_DIR / name
    return pd.read_csv(path, dtype=str) if path.exists() else pd.DataFrame()


def clear_cache() -> None:
    _table.cache_clear()


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    if limit is not None:
        df = df.head(limit)
    return df.where(pd.notna(df), None).to_dict("records")


def get_company_profile(stock_code: str) -> dict[str, Any]:
    code = normalize_stock_code(stock_code)
    companies = _table("companies.csv")
    if not code or companies.empty:
        return {}
    matches = companies[companies["stock_code"] == code]
    if matches.empty:
        return {}
    company = matches.iloc[0].where(pd.notna(matches.iloc[0]), None).to_dict()
    aliases = _table("company_aliases.csv")
    if not aliases.empty:
        alias_rows = aliases[aliases["stock_code"] == code].drop_duplicates("alias_name")
        company["aliases"] = _records(alias_rows)
    return company


def search_company_by_name(name: str, limit: int = 10) -> list[dict[str, Any]]:
    query = normalize_text(name)
    if not query:
        return []
    aliases = _table("company_aliases.csv")
    companies = _table("companies.csv")
    if aliases.empty or companies.empty:
        return []
    matched = aliases[aliases["alias_name"].str.contains(query, case=False, na=False)].copy()
    matched["_exact_rank"] = (matched["alias_name"] != query).astype(int)
    matched["_preferred_rank"] = (matched["alias_type"] != "preferred").astype(int)
    matched = matched.sort_values(["_exact_rank", "_preferred_rank", "alias_name"]).head(limit)
    matched = matched.drop(columns=["_exact_rank", "_preferred_rank"])
    result = matched.merge(companies, on="stock_code", how="left", suffixes=("_alias", ""))
    return _records(result, limit)


def get_company_observations(stock_code: str, year: int | None = None, limit: int = 250) -> list[dict[str, Any]]:
    code = normalize_stock_code(stock_code)
    observations = _table("observations.csv")
    metrics = _table("metrics.csv")
    if not code or observations.empty:
        return []
    subset = observations[(observations["subject_type"] == "company") & (observations["subject_id"] == code)]
    if year is not None:
        subset = subset[subset["period_value"] == str(year)]
    if not metrics.empty:
        subset = subset.merge(metrics[["metric_code", "metric_name_ko", "metric_category"]], on="metric_code", how="left")
    priority = {"credit": 0, "financial": 1, "investment": 2, "employee": 3, "macro": 4, "stock": 5}
    if "metric_category" in subset.columns:
        subset["_priority"] = subset["metric_category"].map(priority).fillna(9)
        subset = subset.sort_values(["_priority", "metric_code", "source_file"])
        subset = subset.drop(columns=["_priority"])
    return _records(subset, limit)


def get_sector_observations(sector_code: str, year: int | None = None, limit: int = 250) -> list[dict[str, Any]]:
    observations = _table("observations.csv")
    if observations.empty:
        return []
    subset = observations[(observations["subject_type"] == "sector") & (observations["subject_id"] == sector_code)]
    if year is not None:
        subset = subset[subset["period_value"].isin([str(year), "2018-2022"])]
    return _records(subset, limit)


def get_credit_ratings(stock_code: str, year: int | None = None) -> list[dict[str, Any]]:
    code = normalize_stock_code(stock_code)
    ratings = _table("credit_ratings.csv")
    if not code or ratings.empty:
        return []
    subset = ratings[ratings["stock_code"] == code]
    if year is not None:
        subset = subset[subset["year"].astype(str) == str(year)]
    return _records(subset.sort_values(["year", "agency"]))


def get_metric_criteria(metric_code: str) -> list[dict[str, Any]]:
    criteria = _table("metric_criteria.csv")
    if criteria.empty:
        return []
    return _records(criteria[criteria["metric_code"] == metric_code])


def get_criteria_evidence(stock_code: str, year: int | None = None) -> dict[str, list[dict[str, Any]]]:
    observations = pd.DataFrame(get_company_observations(stock_code, year=year, limit=1000))
    criteria = _table("metric_criteria.csv")
    if observations.empty or criteria.empty:
        return {}
    joined = observations.merge(criteria, on="metric_code", how="inner")
    result: dict[str, list[dict[str, Any]]] = {}
    for criterion, group in joined.groupby("criterion_code"):
        result[str(criterion)] = _records(group.sort_values("metric_code"), 30)
    return result


def get_risk_context(stock_code: str, year: int | None = None) -> dict[str, Any]:
    observations = pd.DataFrame(get_company_observations(stock_code, year=year, limit=1000))
    signals = _table("risk_signals.csv")
    found: list[dict[str, Any]] = []
    if observations.empty or signals.empty:
        return {"risk_signals": found}

    values = {}
    for row in observations.to_dict("records"):
        value = row.get("numeric_value")
        try:
            values[row["metric_code"]] = float(value) if value is not None else None
        except Exception:
            pass

    rules = [
        ("high_leverage", values.get("debt_ratio") is not None and values["debt_ratio"] > 300),
        ("negative_operating_cash_flow", (values.get("operating_cash_flow") or values.get("cash_flow_operating") or 0) < 0),
        ("weak_liquidity", values.get("current_ratio") is not None and values["current_ratio"] < 100),
        ("low_employee_rating", values.get("rating") is not None and values["rating"] < 2.5),
        ("declining_profitability", values.get("operating_income") is not None and values["operating_income"] < 0),
    ]
    for code, applies in rules:
        if applies:
            match = signals[signals["risk_signal_code"] == code]
            if not match.empty:
                found.extend(_records(match))
    return {"risk_signals": found}


def get_knowledge_context(stock_code: str, year: int | None = None) -> dict[str, Any]:
    company = get_company_profile(stock_code)
    if not company:
        return {}
    observations = get_company_observations(stock_code, year=year)
    credit_ratings = get_credit_ratings(stock_code, year=year)
    sector_code = company.get("sector_code")
    sector_comparison = get_sector_observations(sector_code, year=year) if sector_code else []
    criteria_evidence = get_criteria_evidence(stock_code, year=year)
    risk_context = get_risk_context(stock_code, year=year)
    return {
        "company": {
            "stock_code": company.get("stock_code"),
            "preferred_name": company.get("preferred_name"),
            "sector_code": company.get("sector_code"),
            "sector_name": company.get("sector_name"),
            "aliases": [row["alias_name"] for row in company.get("aliases", [])],
        },
        "year": year,
        "financial_observations": observations,
        "credit_ratings": credit_ratings,
        "sector_comparison": sector_comparison,
        "criteria_evidence": criteria_evidence,
        "risk_signals": risk_context.get("risk_signals", []),
    }
