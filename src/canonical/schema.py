from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("BIZINSIGHT_DATA_DIR", PROJECT_ROOT / "data")).resolve()
CANONICAL_DIR = DATA_DIR / "canonical"

MISSING_STRINGS = {"", "nan", "none", "null", "nat", "<na>"}
YEAR_PATTERN = re.compile(r"^\d{4}$")
MONTH_PATTERN = re.compile(r"^\d{4}-\d{2}$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def ensure_dirs() -> None:
    CANONICAL_DIR.mkdir(parents=True, exist_ok=True)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_STRINGS


def normalize_text(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    return None if text.lower() in MISSING_STRINGS else text


def normalize_stock_code(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    return digits.zfill(6)[-6:]


def normalize_year(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if YEAR_PATTERN.match(text):
        return text
    match = re.search(r"(19|20)\d{2}", text)
    return match.group(0) if match else None


def normalize_period(value: Any, period_type: str) -> str | None:
    if is_missing(value):
        return None
    text = str(value).strip()
    if period_type in {"fiscal_year", "year"}:
        return normalize_year(text)
    if period_type == "month":
        if MONTH_PATTERN.match(text):
            return text
        compact = text.replace("-", "")
        if re.match(r"^\d{6}$", compact):
            return f"{compact[:4]}-{compact[4:6]}"
    if period_type == "date":
        if DATE_PATTERN.match(text):
            return text
        parsed = pd.to_datetime(text, errors="coerce")
        if not pd.isna(parsed):
            return str(parsed.date())
    return text


def safe_decimal(value: Any) -> Decimal | None:
    if is_missing(value):
        return None
    try:
        if isinstance(value, str):
            value = value.replace(",", "").strip()
        dec = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    if not dec.is_finite():
        return None
    return dec


def safe_float(value: Any) -> float | None:
    dec = safe_decimal(value)
    return float(dec) if dec is not None else None


def make_hash(*parts: Any, length: int = 12) -> str:
    joined = "|".join("" if part is None else str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def make_id(*parts: Any) -> str:
    cleaned = [sanitize_code(part) for part in parts if normalize_text(part) is not None]
    return "_".join(cleaned)


def sanitize_code(value: Any) -> str:
    text = normalize_text(value)
    if text is None:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii") or str(value)
    text = text.replace("%", "pct")
    text = re.sub(r"[^0-9A-Za-z가-힣]+", "_", text).strip("_").lower()
    text = re.sub(r"_+", "_", text)
    return text or make_hash(value)


def slugify(value: Any, prefix: str = "sector") -> str:
    text = sanitize_code(value)
    if not text or re.search(r"[가-힣]", text):
        return f"{prefix}_{make_hash(value, length=10)}"
    if text[0].isdigit():
        text = f"{prefix}_{text}"
    return text


def read_csv(filename: str, **kwargs: Any) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def write_csv(df: pd.DataFrame, filename: str) -> Path:
    ensure_dirs()
    path = CANONICAL_DIR / filename
    df.to_csv(path, index=False)
    return path


def read_canonical(filename: str) -> pd.DataFrame:
    path = CANONICAL_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str)


def report(name: str, **items: Any) -> dict[str, Any]:
    print(f"{name}:")
    for key, value in items.items():
        print(f"  {key}: {value}")
    return {"name": name, **items}


@dataclass(frozen=True)
class CanonicalResult:
    name: str
    path: Path
    rows: int


IDENTIFIER_COLUMNS = {
    "id",
    "corp",
    "company",
    "company_name",
    "corp_name",
    "stock_code",
    "종목코드",
    "종목명",
    "회사명",
    "sector",
    "업종",
    "업종명",
    "year",
    "fs_type",
    "concept_id",
    "label_en",
    "label_ko",
    "class1",
    "class2",
    "class3",
    "class4",
    "rank",
    "predicted_rank",
    "kis_rank",
    "nice_rank",
    "kis_bond_type",
    "nice_bond_type",
    "bond_type",
    "YoY",
    "Unnamed: 0",
}


ALLOWED_METRIC_CATEGORIES = {
    "financial",
    "credit",
    "investment",
    "employee",
    "stock",
    "macro",
}
