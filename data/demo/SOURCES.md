# Biz-Insight Demo Data Sources

This directory is a small portfolio/demo dataset, not a full production dataset.

## Scope

- As-of date: 2026-06-13 KST.
- Demo companies:
  - KOSPI benchmarks: Samsung Electronics (`005930`), Hyundai Motor (`005380`), Kakao (`035720`).
  - KOSDAQ growth companies: KINX (`093320`), C&C International (`352480`), JVM (`054950`), Duksan Neolux (`213420`), Intellian Tech (`189300`).
- Runtime target: Streamlit demo with password gate and LangGraph multi-agent workflow.

## Included Data

- `canonical/*.csv`: normalized company, metric, observation, rating, review-summary, and risk-signal tables.
- `dynamic/stock_prices.csv`: recent daily price snapshots for 2026-05-28 through 2026-06-12.
- root CSV files such as `fs.csv`, `credit_data_web.csv`, and `investment_data_web.csv`: small legacy fallback files so older tools do not fail during the demo.

## Source Notes

- Stock prices and foreign ownership ratios were pulled from Naver Finance chart responses for each ticker on 2026-06-13 KST.
- KOSDAQ market-cap/rank values and demo positioning for the five growth companies were supplied as the demo universe definition on 2026-06-13 KST.
- `dart_observations_refresh.csv` was generated from OpenDART `fnlttSinglAcntAll` for 2025 annual reports and merged into `canonical/observations.csv` where the API returned standard financial-statement rows.
- Financial rows are curated public snapshots. Each row keeps its source marker in `source_file`; mixed periods are intentional:
  - Samsung Electronics includes 2025 annual public snapshot rows and 2026 Q1 guidance/news snapshot rows.
  - Hyundai Motor includes 2026 Q1 KRW snapshot rows and a 2024 annual USD summary row because the easily verified public annual summary was USD-denominated.
  - Kakao includes 2024 annual public snapshot rows because a newer verified annual public source was not available in this pass.
  - The five KOSDAQ growth companies include market position, business thesis, risk focus, stock snapshot, synthetic review summaries, and DART-backed 2025 financial rows where available.
- `DemoModel` credit ratings and employee-review summaries are demo-only placeholders. They are labeled with `demo_model_not_external_credit_rating` and `demo_synthetic_portfolio_snapshot` and should not be presented as external ratings or scraped employee-review data.
- `dynamic/macro_monthly.csv` is a placeholder macro context for the demo workflow. Replace it with Bank of Korea/FRED or another verified source before presenting macro claims as factual.

## Refresh Path

For a stronger live demo, replace the curated financial rows with a DART-backed refresh job:

1. Add `DART_API_KEY` to local `.env` or deployment secrets.
2. Fetch latest annual/quarterly filings for the eight demo tickers.
3. Rewrite `canonical/observations.csv`, `fs.csv`, and source markers.
4. Keep the dataset small enough for Streamlit deployment.

Helper script:

```bash
DART_API_KEY=... UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python scripts/refresh_demo_dart.py \
  --year 2025 \
  --report annual \
  --out data/demo/dart_observations_refresh.csv
```
