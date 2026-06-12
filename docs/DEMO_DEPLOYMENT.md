# Demo Deployment

This project is best deployed as a portfolio demo, not as a full data product.

## Product Message

Large-cap companies already have abundant interpreted information. Biz-Insight focuses on KOSDAQ growth companies where filings, news, results, industry context, and employee-review signals are scattered. The demo keeps three KOSPI names as benchmarks and adds five KOSDAQ companies to show how the same multi-agent workflow explains less-covered growth stories.

## Recommended Shape

- Deploy the Streamlit app with the small `data/demo` dataset.
- Keep the full historical `data/` directory out of the deployment artifact.
- Use a simple password gate for reviewer access.
- Keep the README focused on the multi-agent architecture and tool flow.

## Local Demo Run

```bash
BIZINSIGHT_DATA_DIR=data/demo \
BIZINSIGHT_APP_PASSWORD=your-demo-password \
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync streamlit run app.py
```

The app will read:

- `data/demo/company_info.csv` for the sidebar search
- `data/demo/canonical/*.csv` for KG context tools
- `data/demo/dynamic/*.csv` for stock, review, and macro tools

## Report Trace

After a report is generated, open `리포트 작성 과정 보기` below the report. It shows:

- the agent path selected by LangGraph
- the tool calls made during research and analysis
- source markers found in tool outputs
- runtime errors or fallback conditions, if any

This is meant for user-facing transparency, not a developer-only debug panel.

## Refresh Latest Financial Data

The demo includes a DART refresh scaffold for replacing placeholder financial rows with filing-backed observations:

```bash
DART_API_KEY=... \
UV_PROJECT_ENVIRONMENT=bizinsight uv run --no-sync python scripts/refresh_demo_dart.py \
  --year 2025 \
  --report annual \
  --out data/demo/dart_observations_refresh.csv
```

Review the generated CSV before merging it into `data/demo/canonical/observations.csv`. The script intentionally writes a separate file so curated demo rows are not overwritten silently.

Current demo data includes one generated refresh file:

```text
data/demo/dart_observations_refresh.csv
```

## Streamlit Cloud

Use these secrets:

```toml
GOOGLE_API_KEY = "..."
BIZINSIGHT_DATA_DIR = "data/demo"
APP_PASSWORD = "..."
LANGSMITH_TRACING = "false"
```

Set the app entrypoint to:

```text
app.py
```

The root `requirements.txt` is intentionally smaller than `pyproject.toml` so the demo deployment avoids heavy training, crawling, and TensorFlow dependencies.

## Data Caveat

The demo dataset is a curated snapshot as of 2026-06-13 KST. It is designed to demonstrate:

- canonical data modeling
- KG-style query tools
- dynamic signal tools
- LangGraph supervisor/researcher/analyst/reviewer/synthesis flow
- KOSPI benchmark comparison and KOSDAQ growth-company interpretation

It is not intended to be a complete audited financial database.
