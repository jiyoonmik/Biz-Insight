from __future__ import annotations

import argparse
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pandas as pd
import requests


DEMO_COMPANIES = {
    "005930": "삼성전자",
    "005380": "현대자동차",
    "035720": "카카오",
    "093320": "케이아이엔엑스",
    "352480": "씨앤씨인터내셔널",
    "054950": "제이브이엠",
    "213420": "덕산네오룩스",
    "189300": "인텔리안테크",
}

ACCOUNT_MAP = {
    "매출액": "revenue",
    "영업수익": "revenue",
    "영업이익": "operating_income",
    "당기순이익": "net_income",
    "분기순이익": "net_income",
    "자산총계": "total_assets",
    "부채총계": "total_liabilities",
    "자본총계": "equity",
}

REPORT_CODES = {
    "annual": "11011",
    "q1": "11013",
    "half": "11012",
    "q3": "11014",
}


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def load_corp_codes(api_key: str) -> dict[str, str]:
    response = requests.get(
        "https://opendart.fss.or.kr/api/corpCode.xml",
        params={"crtfc_key": api_key},
        timeout=60,
    )
    response.raise_for_status()
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        xml_name = archive.namelist()[0]
        root = ElementTree.fromstring(archive.read(xml_name))

    codes: dict[str, str] = {}
    for item in root.findall("list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if stock_code and corp_code:
            codes[stock_code.zfill(6)] = corp_code
    return codes


def _parse_amount(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).replace(",", "").strip()
    if not text or text == "-":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def fetch_financial_rows(api_key: str, corp_code: str, stock_code: str, year: int, report_code: str) -> list[dict[str, Any]]:
    payload = _get_json(
        "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json",
        {
            "crtfc_key": api_key,
            "corp_code": corp_code,
            "bsns_year": str(year),
            "reprt_code": report_code,
            "fs_div": "CFS",
        },
    )
    if payload.get("status") != "000":
        return []

    rows = []
    for item in payload.get("list", []):
        account_name = item.get("account_nm")
        metric_code = ACCOUNT_MAP.get(account_name)
        if not metric_code:
            continue
        amount = _parse_amount(item.get("thstrm_amount"))
        if amount is None:
            continue
        rows.append(
            {
                "observation_id": f"{stock_code}_{metric_code}_{year}_{report_code}",
                "subject_type": "company",
                "subject_id": stock_code,
                "metric_code": metric_code,
                "period_type": "fiscal_year" if report_code == REPORT_CODES["annual"] else "quarter",
                "period_value": str(year) if report_code == REPORT_CODES["annual"] else f"{year}_{report_code}",
                "numeric_value": amount,
                "text_value": "",
                "unit": "KRW",
                "source_file": f"opendart_fnlttSinglAcntAll_{year}_{report_code}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh demo financial observations from OpenDART.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--report", choices=sorted(REPORT_CODES), default="annual")
    parser.add_argument("--out", type=Path, default=Path("data/demo/dart_observations_refresh.csv"))
    args = parser.parse_args()

    api_key = os.getenv("DART_API_KEY")
    if not api_key:
        raise SystemExit("DART_API_KEY is required. Add it to .env or deployment secrets before running this script.")

    corp_codes = load_corp_codes(api_key)
    report_code = REPORT_CODES[args.report]
    rows = []
    missing = []
    for stock_code, name in DEMO_COMPANIES.items():
        corp_code = corp_codes.get(stock_code)
        if not corp_code:
            missing.append(f"{stock_code} {name}: corp_code not found")
            continue
        company_rows = fetch_financial_rows(api_key, corp_code, stock_code, args.year, report_code)
        if not company_rows:
            missing.append(f"{stock_code} {name}: no financial rows")
        rows.extend(company_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"wrote rows={len(rows)} path={args.out}")
    if missing:
        print("missing:")
        for item in missing:
            print(f"- {item}")


if __name__ == "__main__":
    main()
