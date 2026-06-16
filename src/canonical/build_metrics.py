from __future__ import annotations

from collections import OrderedDict
from typing import Any

import pandas as pd

from src.canonical.schema import (
    ALLOWED_METRIC_CATEGORIES,
    DATA_DIR,
    IDENTIFIER_COLUMNS,
    normalize_text,
    report,
    sanitize_code,
    write_csv,
)


METRIC_SOURCES = [
    "final_features.csv",
    "credit_model_a.csv",
    "credit_model_b.csv",
    "investment_data_model.csv",
    "web_visualization.csv",
    "main_fs.csv",
    "economic_indicators.csv",
    "industry_average_year.csv",
    "bs.csv",
    "incs.csv",
    "cf.csv",
]

EMPLOYEE_METRICS = {
    "count",
    "rating",
    "paywellfare",
    "worklifebal",
    "culture",
    "opportunity",
    "manager",
    "recommend",
    "ceo",
    "potential",
    "jobp_rating",
    "blind_rating",
}
STOCK_METRICS = {"stock_price", "market_capitalization", "amount", "foreign_ownership_ratio", "close", "open", "high", "low"}
MACRO_METRICS = {
    "minimum_wage",
    "minimum_wag",
    "uskor_exchange_average",
    "us_kor_exchange_avg",
    "ppi_year",
    "PPI_year",
    "kor_usa_ir_diff",
    "kr_standard_yield",
    "crb_index_avg",
}
INVESTMENT_MARKERS = (
    "price_",
    "return_on_",
    "earnings_per_share",
    "book_value_per_share",
    "enterprise_value",
    "turnover",
)
CREDIT_MARKERS = (
    "debt",
    "borrow",
    "liabilit",
    "ratio",
    "coverage",
    "liquidity",
    "quick",
    "current",
    "cash_flow",
    "working_capital",
)

SEED_METRICS: list[dict[str, Any]] = [
    {
        "metric_code": "total_assets",
        "metric_name_ko": "자산총계",
        "metric_name_en": "Total assets",
        "metric_category": "financial",
        "criteria": [],
        "evidence_direction": "context_dependent",
    },
    {
        "metric_code": "revenue",
        "metric_name_ko": "매출액",
        "metric_name_en": "Revenue",
        "metric_category": "financial",
        "criteria": ["growth", "market_position"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "operating_income",
        "metric_name_ko": "영업이익",
        "metric_name_en": "Operating income",
        "metric_category": "financial",
        "criteria": ["profitability", "going_concern"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "ebitda_margin",
        "metric_name_ko": "EBITDA 마진",
        "metric_name_en": "EBITDA margin",
        "metric_category": "financial",
        "criteria": ["profitability", "cash_generation"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "debt_ratio",
        "metric_name_ko": "부채비율",
        "metric_name_en": "Debt ratio",
        "metric_category": "credit",
        "criteria": ["leverage", "going_concern"],
        "evidence_direction": "lower_is_better",
    },
    {
        "metric_code": "current_ratio",
        "metric_name_ko": "유동비율",
        "metric_name_en": "Current ratio",
        "metric_category": "credit",
        "criteria": ["liquidity", "going_concern"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "quick_ratio",
        "metric_name_ko": "당좌비율",
        "metric_name_en": "Quick ratio",
        "metric_category": "credit",
        "criteria": ["liquidity"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "operating_cash_flow",
        "metric_name_ko": "영업활동현금흐름",
        "metric_name_en": "Operating cash flow",
        "metric_category": "credit",
        "criteria": ["cash_generation", "going_concern"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "interest_coverage_ratio",
        "metric_name_ko": "이자보상배율",
        "metric_name_en": "Interest coverage ratio",
        "metric_category": "credit",
        "criteria": ["leverage", "cash_generation"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "return_on_assets",
        "metric_name_ko": "총자산순이익률",
        "metric_name_en": "Return on assets",
        "metric_category": "investment",
        "criteria": ["profitability"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "price_to_book_ratio",
        "metric_name_ko": "PBR",
        "metric_name_en": "Price to book ratio",
        "metric_category": "investment",
        "criteria": ["profitability", "market_position"],
        "evidence_direction": "context_dependent",
    },
    {
        "metric_code": "rating_worklifebal",
        "metric_name_ko": "워라밸 평점",
        "metric_name_en": "Work-life balance rating",
        "metric_category": "employee",
        "criteria": ["internal_control"],
        "evidence_direction": "higher_is_better",
    },
    {
        "metric_code": "us_kor_exchange_avg",
        "metric_name_ko": "원달러 평균환율",
        "metric_name_en": "Average USD/KRW exchange rate",
        "metric_category": "macro",
        "criteria": ["macro_sensitivity"],
        "evidence_direction": "context_dependent",
    },
]


def classify_metric(metric_code: str) -> str:
    if metric_code in EMPLOYEE_METRICS:
        return "employee"
    if metric_code in STOCK_METRICS:
        return "stock"
    if metric_code in {sanitize_code(v) for v in MACRO_METRICS}:
        return "macro"
    if any(marker in metric_code for marker in INVESTMENT_MARKERS):
        return "investment"
    if any(marker in metric_code for marker in CREDIT_MARKERS):
        return "credit"
    return "financial"


def infer_unit(metric_code: str) -> str | None:
    if metric_code.endswith("_ratio") or "margin" in metric_code or "rate" in metric_code or metric_code in {"rating"}:
        return "%"
    if "price" in metric_code or "wage" in metric_code or "revenue" in metric_code or "assets" in metric_code:
        return "KRW"
    if "count" in metric_code:
        return "count"
    return None


def _seed_metrics() -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    metrics: dict[str, dict[str, Any]] = {}
    criteria: list[dict[str, str]] = []
    for seed in SEED_METRICS:
        code = str(seed["metric_code"])
        direction = str(seed.get("evidence_direction") or "context_dependent")
        metrics[code] = {
            "metric_code": code,
            "metric_name_ko": seed.get("metric_name_ko"),
            "metric_name_en": seed.get("metric_name_en"),
            "metric_category": seed.get("metric_category") or classify_metric(code),
            "unit": infer_unit(code),
            "source_file": "canonical_seed",
        }
        for criterion in seed.get("criteria", []):
            criteria.append(
                {
                    "metric_code": code,
                    "criterion_code": criterion,
                    "evidence_direction": direction,
                    "rationale": "Defined in canonical metric seed",
                }
            )
    return metrics, criteria


def _add_metric(metrics: OrderedDict[str, dict[str, Any]], code: str, source_file: str, name_ko: str | None = None) -> None:
    if not code:
        return
    row = metrics.setdefault(
        code,
        {
            "metric_code": code,
            "metric_name_ko": name_ko,
            "metric_name_en": code,
            "metric_category": classify_metric(code),
            "unit": infer_unit(code),
            "source_file": source_file,
        },
    )
    if normalize_text(row.get("metric_name_ko")) is None and normalize_text(name_ko):
        row["metric_name_ko"] = normalize_text(name_ko)
    if source_file not in str(row["source_file"]).split(";"):
        row["source_file"] = f"{row['source_file']};{source_file}"


def build_metrics() -> dict[str, Any]:
    seed_metrics, seed_criteria = _seed_metrics()
    metrics: OrderedDict[str, dict[str, Any]] = OrderedDict((code, row) for code, row in seed_metrics.items())

    for source in METRIC_SOURCES:
        path = DATA_DIR / source
        if not path.exists():
            continue
        header = pd.read_csv(path, nrows=0)
        df = pd.read_csv(path, nrows=2000)
        if {"label_en", "label_ko"}.issubset(header.columns):
            labels = pd.read_csv(path, usecols=["label_en", "label_ko"]).drop_duplicates()
            for row in labels.to_dict("records"):
                _add_metric(metrics, sanitize_code(row["label_en"]), source, normalize_text(row.get("label_ko")))
        if "concept_id" in header.columns:
            usecols = ["concept_id"] + (["label_ko"] if "label_ko" in header.columns else [])
            concepts = pd.read_csv(path, usecols=usecols).drop_duplicates()
            for row in concepts.to_dict("records"):
                _add_metric(metrics, sanitize_code(row["concept_id"]), source, normalize_text(row.get("label_ko")))
        for column in header.columns:
            code = sanitize_code(column)
            if column in IDENTIFIER_COLUMNS or code in IDENTIFIER_COLUMNS or not code:
                continue
            _add_metric(metrics, code, source)

    metrics_df = pd.DataFrame(metrics.values())
    metrics_df = metrics_df[metrics_df["metric_category"].isin(ALLOWED_METRIC_CATEGORIES)]
    metrics_df = metrics_df.drop_duplicates("metric_code").sort_values("metric_code")

    criteria_df = pd.DataFrame(seed_criteria)
    if criteria_df.empty:
        criteria_df = pd.DataFrame(columns=["metric_code", "criterion_code", "evidence_direction", "rationale"])
    else:
        criteria_df = criteria_df.drop_duplicates(["metric_code", "criterion_code"]).sort_values(
            ["criterion_code", "metric_code"]
        )

    risk_signals = pd.DataFrame(
        [
            ["high_leverage", "leverage", "Debt burden is elevated relative to equity, income, or assets.", "debt_ratio/current_ratio thresholds", "high"],
            ["negative_operating_cash_flow", "cash_generation", "Operating cash flow is negative.", "operating_cash_flow < 0", "high"],
            ["low_employee_rating", "internal_control", "Employee rating is weak and may indicate organization risk.", "rating < 2.5", "medium"],
            ["declining_profitability", "profitability", "Profitability metrics declined year over year.", "operating_income or margins decline", "medium"],
            ["weak_liquidity", "liquidity", "Liquidity ratios or working capital are weak.", "current_ratio/quick_ratio thresholds", "high"],
            ["rating_downgrade", "going_concern", "Credit rating deteriorated compared with a prior period.", "rating history downgrade", "high"],
        ],
        columns=["risk_signal_code", "criterion_code", "description", "source_rule", "severity"],
    )

    write_csv(metrics_df, "metrics.csv")
    write_csv(criteria_df, "metric_criteria.csv")
    write_csv(risk_signals, "risk_signals.csv")

    return {
        "metrics": report(
            "metrics",
            rows=len(metrics_df),
            duplicate_metric_codes=int(metrics_df["metric_code"].duplicated().sum()),
            unknown_metric_categories=int((~metrics_df["metric_category"].isin(ALLOWED_METRIC_CATEGORIES)).sum()),
        ),
        "metric_criteria": report("metric_criteria", rows=len(criteria_df)),
        "risk_signals": report("risk_signals", rows=len(risk_signals)),
    }


if __name__ == "__main__":
    build_metrics()
