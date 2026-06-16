from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CompanyAnalysisRequest(BaseModel):
    stock_code: str = Field(..., examples=["095570"])
    year: int | None = Field(default=None, examples=[2022])
    include_dynamic_signals: bool = True
    user_question: str | None = None


class CompanyAnalysisResponse(BaseModel):
    stock_code: str
    year: int | None = None
    knowledge_context: dict[str, Any]
    dynamic_signals: dict[str, Any]
    report: str

