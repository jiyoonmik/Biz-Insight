from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.canonical.schema import DATA_DIR, normalize_period


DYNAMIC_DIR = DATA_DIR / "dynamic"
MACRO_PARQUET = DYNAMIC_DIR / "macro_monthly.parquet"
MACRO_CSV = DYNAMIC_DIR / "macro_monthly.csv"


def _bok_series(source_file: str, metric_code: str) -> pd.DataFrame:
    path = DATA_DIR / source_file
    if not path.exists():
        return pd.DataFrame(columns=["metric_code", "period_value", "numeric_value", "unit", "source_file"])
    df = pd.read_csv(path)
    period_type = "month" if df["TIME"].astype(str).str.len().max() == 6 else "fiscal_year"
    df["period_value"] = df["TIME"].map(lambda value: normalize_period(value, period_type))
    return pd.DataFrame(
        {
            "metric_code": metric_code,
            "period_type": "month" if period_type == "month" else "fiscal_year",
            "period_value": df["period_value"],
            "numeric_value": pd.to_numeric(df["DATA_VALUE"], errors="coerce"),
            "unit": df.get("UNIT_NAME"),
            "source_file": source_file,
        }
    )


def build_macro_store() -> Path:
    DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
    frames = [
        _bok_series("kr_policyratio_month.csv", "kr_policy_rate"),
        _bok_series("us_policyratio_month.csv", "us_policy_rate"),
        _bok_series("kr_standard_yield.csv", "kr_standard_yield"),
        _bok_series("uskor_exchange_year.csv", "uskor_exchange_average"),
        _bok_series("ppi_year.csv", "ppi_year"),
    ]
    crb_path = DATA_DIR / "crb_index.csv"
    if crb_path.exists():
        crb = pd.read_csv(crb_path)
        crb["date"] = pd.to_datetime(dict(year=crb["year"], month=crb["month"], day=crb["day"]), errors="coerce")
        crb["period_value"] = crb["date"].dt.strftime("%Y-%m")
        monthly = crb.groupby("period_value", as_index=False)["crb_index"].mean()
        frames.append(
            pd.DataFrame(
                {
                    "metric_code": "crb_index_avg",
                    "period_type": "month",
                    "period_value": monthly["period_value"],
                    "numeric_value": monthly["crb_index"],
                    "unit": "index",
                    "source_file": "crb_index.csv",
                }
            )
        )
    macro = pd.concat(frames, ignore_index=True).dropna(subset=["period_value", "numeric_value"])
    macro = macro.sort_values(["metric_code", "period_value"])
    macro.to_parquet(MACRO_PARQUET, index=False)
    print(f"macro_monthly: rows={len(macro)} path={MACRO_PARQUET}")
    return MACRO_PARQUET


def _load() -> pd.DataFrame:
    if MACRO_PARQUET.exists():
        return pd.read_parquet(MACRO_PARQUET)
    if MACRO_CSV.exists():
        return pd.read_csv(MACRO_CSV)
    if not MACRO_PARQUET.exists():
        build_macro_store()
    return pd.read_parquet(MACRO_PARQUET)


def get_macro_series(metric_code: str, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    df = _load()
    subset = df[df["metric_code"] == metric_code].copy()
    if start:
        subset = subset[subset["period_value"].astype(str) >= start]
    if end:
        subset = subset[subset["period_value"].astype(str) <= end]
    return subset.where(pd.notna(subset), None).to_dict("records")


def get_macro_context(year: int | None = None) -> dict[str, Any]:
    df = _load()
    if year is not None:
        year_text = str(year)
        subset = df[df["period_value"].astype(str).str.startswith(year_text)]
    else:
        subset = df
    latest = subset.sort_values("period_value").groupby("metric_code").tail(1)
    return {
        "year": year,
        "metrics": latest.where(pd.notna(latest), None).to_dict("records"),
        "dynamic_data_as_of": str(latest["period_value"].max()) if not latest.empty else None,
    }


if __name__ == "__main__":
    build_macro_store()
