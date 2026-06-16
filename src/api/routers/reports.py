from __future__ import annotations

from fastapi import APIRouter

from src.api.dependencies import require_stock_code
from src.api.models import CompanyAnalysisRequest, CompanyAnalysisResponse
from src.dynamic.macro_store import get_macro_context
from src.dynamic.review_store import get_review_summary
from src.dynamic.stock_store import get_stock_summary
from src.context.queries import get_knowledge_context


router = APIRouter(tags=["reports"])


@router.post("/reports/company-analysis", response_model=CompanyAnalysisResponse)
def company_analysis(request: CompanyAnalysisRequest) -> CompanyAnalysisResponse:
    code = require_stock_code(request.stock_code)
    knowledge_context = get_knowledge_context(code, year=request.year)
    dynamic_signals = {}
    if request.include_dynamic_signals:
        dynamic_signals = {
            "stock_summary": get_stock_summary(code, months=12),
            "review_summary": get_review_summary(code, period=str(request.year) if request.year else None),
            "macro_context": get_macro_context(year=request.year),
        }
    company = knowledge_context.get("company", {})
    report = (
        f"{company.get('preferred_name', code)}({code}) analysis context prepared. "
        f"Canonical context fiscal year: {request.year or 'latest available'}; "
        f"dynamic data as of: {dynamic_signals.get('macro_context', {}).get('dynamic_data_as_of', 'n/a')}."
    )
    return CompanyAnalysisResponse(
        stock_code=code,
        year=request.year,
        knowledge_context=knowledge_context,
        dynamic_signals=dynamic_signals,
        report=report,
    )
