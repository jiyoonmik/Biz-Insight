from __future__ import annotations

from fastapi import FastAPI

from src.api.routers import companies, financials, ratings, reports, reviews, signals


app = FastAPI(
    title="Biz-Insight Context API",
    version="0.1.0",
    description="OpenAPI layer for canonical company context and dynamic company signals.",
)

app.include_router(companies.router)
app.include_router(financials.router)
app.include_router(ratings.router)
app.include_router(signals.router)
app.include_router(reviews.router)
app.include_router(reports.router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
