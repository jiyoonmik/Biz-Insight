from __future__ import annotations

from typing import Any

import pandas as pd

from src.canonical.schema import (
    DATA_DIR,
    IDENTIFIER_COLUMNS,
    make_hash,
    normalize_period,
    normalize_stock_code,
    normalize_text,
    read_canonical,
    report,
    safe_decimal,
    sanitize_code,
    slugify,
    write_csv,
)


ROW_YEAR_SOURCES = [
    "final_features.csv",
    "credit_model_a.csv",
    "investment_data_model.csv",
]
WIDE_SOURCES = ["web_visualization.csv", "main_fs.csv", "bs.csv", "incs.csv", "cf.csv"]


def _metric_columns(df: pd.DataFrame) -> list[str]:
    columns = []
    for column in df.columns:
        code = sanitize_code(column)
        if column not in IDENTIFIER_COLUMNS and code not in IDENTIFIER_COLUMNS and code:
            columns.append(column)
    return columns


def _append_observation(rows: list[dict[str, Any]], *, subject_type: str, subject_id: str, metric_code: str, period_type: str, period_value: str, value: Any, unit: str | None, source_file: str) -> None:
    text_value = normalize_text(value)
    numeric = safe_decimal(value)
    if numeric is None and text_value is None:
        return
    source_hash = make_hash(source_file, metric_code, length=8)
    rows.append(
        {
            "observation_id": f"{subject_type}_{subject_id}_{metric_code}_{period_type}_{period_value}_{source_hash}",
            "subject_type": subject_type,
            "subject_id": subject_id,
            "metric_code": metric_code,
            "period_type": period_type,
            "period_value": period_value,
            "numeric_value": str(numeric) if numeric is not None else None,
            "text_value": None if numeric is not None else text_value,
            "unit": unit,
            "source_file": source_file,
        }
    )


def _row_year_observations(rows: list[dict[str, Any]], source: str) -> None:
    path = DATA_DIR / source
    if not path.exists():
        return
    df = pd.read_csv(path)
    for raw in df.to_dict("records"):
        stock_code = normalize_stock_code(raw.get("stock_code"))
        year = normalize_period(raw.get("year"), "fiscal_year")
        if not stock_code or not year:
            continue
        for column in _metric_columns(df):
            _append_observation(
                rows,
                subject_type="company",
                subject_id=stock_code,
                metric_code=sanitize_code(column),
                period_type="fiscal_year",
                period_value=year,
                value=raw.get(column),
                unit=None,
                source_file=source,
            )


def _wide_observations(rows: list[dict[str, Any]], source: str) -> None:
    path = DATA_DIR / source
    if not path.exists():
        return
    df = pd.read_csv(path)
    year_columns = [column for column in df.columns if str(column).isdigit() and len(str(column)) == 4]
    for raw in df.to_dict("records"):
        stock_code = normalize_stock_code(raw.get("stock_code"))
        metric_code = sanitize_code(raw.get("label_en") or raw.get("concept_id"))
        if not stock_code or not metric_code:
            continue
        for year in year_columns:
            _append_observation(
                rows,
                subject_type="company",
                subject_id=stock_code,
                metric_code=metric_code,
                period_type="fiscal_year",
                period_value=str(year),
                value=raw.get(year),
                unit=None,
                source_file=source,
            )


def _sector_observations(rows: list[dict[str, Any]]) -> None:
    path = DATA_DIR / "industry_average_year.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    for raw in df.to_dict("records"):
        sector_name = normalize_text(raw.get("sector"))
        if not sector_name:
            continue
        sector_code = slugify(sector_name)
        for column in _metric_columns(df):
            _append_observation(
                rows,
                subject_type="sector",
                subject_id=sector_code,
                metric_code=sanitize_code(column),
                period_type="fiscal_year",
                period_value="2018-2022",
                value=raw.get(column),
                unit=None,
                source_file="industry_average_year.csv",
            )


def _macro_observations(rows: list[dict[str, Any]]) -> None:
    path = DATA_DIR / "economic_indicators.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    for raw in df.to_dict("records"):
        year = normalize_period(raw.get("year"), "fiscal_year")
        if not year:
            continue
        for column in _metric_columns(df):
            _append_observation(
                rows,
                subject_type="macro",
                subject_id="KR",
                metric_code=sanitize_code(column),
                period_type="fiscal_year",
                period_value=year,
                value=raw.get(column),
                unit=None,
                source_file="economic_indicators.csv",
            )


def build_observations() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in ROW_YEAR_SOURCES:
        _row_year_observations(rows, source)
    for source in WIDE_SOURCES:
        _wide_observations(rows, source)
    _sector_observations(rows)
    _macro_observations(rows)

    observations = pd.DataFrame(rows)
    if observations.empty:
        observations = pd.DataFrame(
            columns=[
                "observation_id",
                "subject_type",
                "subject_id",
                "metric_code",
                "period_type",
                "period_value",
                "numeric_value",
                "text_value",
                "unit",
                "source_file",
            ]
        )

    companies = read_canonical("companies.csv")
    metrics = read_canonical("metrics.csv")
    company_codes = set(companies.get("stock_code", []))
    metric_codes = set(metrics.get("metric_code", []))

    missing_subject_mask = (observations["subject_type"] == "company") & (~observations["subject_id"].isin(company_codes))
    missing_metric_mask = ~observations["metric_code"].isin(metric_codes)
    observations = observations[~missing_subject_mask & ~missing_metric_mask].copy()
    observations = observations.drop_duplicates("observation_id").sort_values(
        ["subject_type", "subject_id", "metric_code", "period_value", "source_file"]
    )
    write_csv(observations, "observations.csv")

    invalid_period = ~observations["period_value"].astype(str).str.match(r"^\d{4}(-\d{2}(-\d{2})?)?$|^\d{4}-\d{4}$")
    non_numeric = observations["numeric_value"].isna() & observations["text_value"].notna()

    return {
        "observations": report(
            "observations",
            rows=len(observations),
            missing_subject=int(missing_subject_mask.sum()),
            missing_metric=int(missing_metric_mask.sum()),
            invalid_period_value=int(invalid_period.sum()),
            non_numeric_values=int(non_numeric.sum()),
        )
    }


if __name__ == "__main__":
    build_observations()

