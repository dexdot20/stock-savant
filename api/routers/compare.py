"""Company comparison router."""

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from api.models import CompareRequest, CompareResponse
from commands.compare import compare_companies
from core import get_standard_logger

router = APIRouter(prefix="/compare", tags=["Comparison"])
logger = get_standard_logger(__name__)


@router.post(
    "/",
    response_model=CompareResponse,
    summary="Compare companies",
    description="Run the company comparison workflow for two or more ticker symbols.",
    responses={
        400: {"description": "Invalid request."},
        422: {"description": "Validation error."},
        500: {"description": "Unexpected server error."},
    },
)
async def compare_stocks(request: CompareRequest):
    """Compare multiple companies and return the generated report."""
    result = await run_in_threadpool(
        compare_companies,
        symbols=request.symbols,
        criteria=request.criteria,
        depth_mode=request.depth_mode,
        console=None,
    )

    if result.get("error"):
        return CompareResponse(
            success=False,
            symbols=result.get("symbols") or request.symbols,
            analysis=None,
            message=str(result.get("error")),
        )

    warning_text = (
        str(result.get("raw_result", {}).get("warning"))
        if isinstance(result.get("raw_result"), dict)
        and result.get("raw_result", {}).get("step_limit_reached")
        else ""
    )

    return CompareResponse(
        success=True,
        symbols=result.get("symbols") or request.symbols,
        analysis=str(result.get("analysis") or ""),
        message=(
            f"Comparison completed. Warning: {warning_text}"
            if warning_text
            else "Comparison completed."
        ),
    )
