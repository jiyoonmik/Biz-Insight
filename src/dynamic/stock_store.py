from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.canonical.schema import DATA_DIR, normalize_stock_code


DYNAMIC_DIR = DATA_DIR / "dynamic"
STOCK_PARQUET = DYNAMIC_DIR / "stock_prices.parquet"
STOCK_CSV = DYNAMIC_DIR / "stock_prices.csv"


def build_stock_store() -> Path:
    DYNAMIC_DIR.mkdir(parents=True, exist_ok=True)
    source = DATA_DIR / "stock_data_per_month.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    df = pd.read_csv(source)
    df["stock_code"] = df["stock_code"].map(normalize_stock_code)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["stock_code", "date"]).drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df = df.sort_values(["stock_code", "date"])
    df.to_parquet(STOCK_PARQUET, index=False)
    print(f"stock_prices: rows={len(df)} path={STOCK_PARQUET}")
    return STOCK_PARQUET


def _load() -> pd.DataFrame:
    if STOCK_PARQUET.exists():
        return pd.read_parquet(STOCK_PARQUET)
    if STOCK_CSV.exists():
        df = pd.read_csv(STOCK_CSV, dtype={"stock_code": str})
        df["stock_code"] = df["stock_code"].map(normalize_stock_code)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    if not STOCK_PARQUET.exists():
        build_stock_store()
    return pd.read_parquet(STOCK_PARQUET)


def get_stock_series(stock_code: str, start: str | None = None, end: str | None = None) -> list[dict[str, Any]]:
    code = normalize_stock_code(stock_code)
    if not code:
        return []
    df = _load()
    subset = df[df["stock_code"] == code].copy()
    if start:
        subset = subset[subset["date"] >= pd.to_datetime(start)]
    if end:
        subset = subset[subset["date"] <= pd.to_datetime(end)]
    subset["date"] = subset["date"].dt.strftime("%Y-%m-%d")
    return subset.where(pd.notna(subset), None).to_dict("records")


def get_stock_summary(stock_code: str, months: int = 12) -> dict[str, Any]:
    series = pd.DataFrame(get_stock_series(stock_code))
    if series.empty:
        return {"stock_code": normalize_stock_code(stock_code), "months": months, "available": False}
    series["date"] = pd.to_datetime(series["date"])
    series = series.sort_values("date").tail(months)
    first = float(series.iloc[0]["close"])
    last = float(series.iloc[-1]["close"])
    returns = series["close"].astype(float).pct_change().dropna()
    foreign = series["foreign_ownership_ratio"].astype(float) if "foreign_ownership_ratio" in series else pd.Series(dtype=float)
    return {
        "stock_code": normalize_stock_code(stock_code),
        "months": int(months),
        "available": True,
        "start_date": str(series.iloc[0]["date"].date()),
        "end_date": str(series.iloc[-1]["date"].date()),
        "latest_close": last,
        "period_return": (last / first - 1) if first else None,
        "volatility": float(returns.std()) if not returns.empty else None,
        "average_volume": float(series["amount"].astype(float).mean()) if "amount" in series else None,
        "foreign_ownership_change": float(foreign.iloc[-1] - foreign.iloc[0]) if len(foreign) > 1 else None,
    }


if __name__ == "__main__":
    build_stock_store()
