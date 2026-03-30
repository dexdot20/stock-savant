"""
Stock Analysis Router
"""

import uuid
import time
from typing import Union
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from api.models import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    BatchAnalysisRequest,
    BatchAnalysisResponse,
    BatchAnalysisStatusResponse,
)
from api.state import create_task, get_task, update_task
from commands.analyze import (
    analyze_single_stock,
    analyze_multiple_stocks,
    extract_analysis_highlights,
)
from config import get_config
from core import get_standard_logger

router = APIRouter(prefix="/analyze", tags=["Analysis"])
logger = get_standard_logger(__name__)


def _dummy_progress(_msg: str) -> None:
    """Suppress console output during API calls."""
    return None


async def run_analysis_task(task_id: str, symbol: str, request: AnalysisRequest):
    """Run the analysis in the background and update its status."""
    try:
        update_task(task_id, {"status": AnalysisStatus.IN_PROGRESS.value})
        config = get_config()

        result = await run_in_threadpool(
            analyze_single_stock,
            symbol=symbol,
            config=config,
            progress_callback=_dummy_progress,
            investment_horizon=request.investment_horizon,
            user_context=request.user_context,
            is_batch=True,
            console=None,
        )

        if not result:
            update_task(
                task_id,
                {
                    "status": AnalysisStatus.FAILED.value,
                    "success": False,
                    "message": "The analysis did not return any results.",
                },
            )
            return

        highlights = extract_analysis_highlights(result)

        update_task(
            task_id,
            {
                "status": AnalysisStatus.COMPLETED.value,
                "success": True,
                "news_summary": highlights["news_summary"],
                "final_decision": highlights["final_decision"],
                "full_result": result,
            },
        )

    except Exception as e:
        logger.error("Task %s error: %s", task_id, e)
        update_task(
            task_id,
            {
                "status": AnalysisStatus.FAILED.value,
                "success": False,
                "message": f"Error: {str(e)}",
            },
        )


async def run_batch_analysis_task(task_id: str, request: BatchAnalysisRequest):
    """Run batch stock analysis in the background and update its status."""
    try:
        update_task(task_id, {"status": AnalysisStatus.IN_PROGRESS.value})
        config = get_config()

        results = await run_in_threadpool(
            analyze_multiple_stocks,
            symbols=request.symbols,
            max_workers=request.max_workers,
            config=config,
            console=None,
            quiet=True,
            investment_horizon=request.investment_horizon,
            user_context=request.user_context,
        )

        completed = sum(1 for value in results.values() if value)
        failed = len(results) - completed
        update_task(
            task_id,
            {
                "status": AnalysisStatus.COMPLETED.value,
                "success": True,
                "message": f"Batch analysis completed: {completed} succeeded, {failed} failed.",
                "results": results,
                "completed_count": completed,
                "failed_count": failed,
                "total_count": len(results),
            },
        )
    except Exception as exc:
        logger.error("Batch task %s error: %s", task_id, exc)
        update_task(
            task_id,
            {
                "status": AnalysisStatus.FAILED.value,
                "success": False,
                "message": f"Batch analysis error: {exc}",
            },
        )


@router.post(
    "/stock",
    status_code=202,
    response_model=AnalysisResponse,
    summary="Start stock analysis",
    description="Queue a single-symbol analysis workflow and return a task identifier for polling.",
    responses={
        400: {"description": "Invalid request."},
        422: {"description": "Validation error."},
        500: {"description": "Unexpected server error."},
    },
)
async def analyze_stock(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start stock analysis in the background.
    """
    symbol = request.symbol
    task_id = str(uuid.uuid4())

    # Save the task
    create_task(
        task_id,
        {
        "symbol": symbol,
        "status": AnalysisStatus.PENDING.value,
        "success": False,
        "task_id": task_id,
        "created_ts": time.time(),
        },
    )

    # Add the background task
    background_tasks.add_task(run_analysis_task, task_id, symbol, request)

    return AnalysisResponse(
        symbol=symbol,
        task_id=task_id,
        status=AnalysisStatus.PENDING,
        success=True,
        message="Analysis started.",
    )


@router.post(
    "/batch",
    status_code=202,
    response_model=BatchAnalysisResponse,
    summary="Start batch analysis",
    description="Queue a multi-symbol analysis workflow and return a task identifier for polling.",
    responses={
        400: {"description": "Invalid request."},
        422: {"description": "Validation error."},
        500: {"description": "Unexpected server error."},
    },
)
async def analyze_batch(request: BatchAnalysisRequest, background_tasks: BackgroundTasks):
    """Start batch stock analysis in the background."""
    task_id = str(uuid.uuid4())

    create_task(
        task_id,
        {
            "task_id": task_id,
            "symbol": "BATCH",
            "status": AnalysisStatus.PENDING.value,
            "success": False,
            "created_ts": time.time(),
            "symbols": request.symbols,
            "total_symbols": len(request.symbols),
        },
    )

    background_tasks.add_task(run_batch_analysis_task, task_id, request)

    return BatchAnalysisResponse(
        task_id=task_id,
        status=AnalysisStatus.PENDING,
        success=True,
        message="Batch analysis started.",
        total_symbols=len(request.symbols),
    )


@router.get(
    "/status/{task_id}",
    response_model=Union[AnalysisResponse, BatchAnalysisStatusResponse],
    summary="Get analysis status",
    description="Return the latest status and payload for a queued analysis task.",
    responses={
        404: {"description": "Task not found."},
        500: {"description": "Unexpected server error."},
    },
)
async def get_analysis_status(task_id: str):
    """Query the status of an analysis task."""
    task_data = get_task(task_id)
    if not task_data:
        raise HTTPException(status_code=404, detail="Task not found.")

    if task_data.get("symbol") == "BATCH" or "symbols" in task_data:
        return BatchAnalysisStatusResponse(
            task_id=task_data["task_id"],
            status=task_data["status"],
            success=bool(task_data.get("success", False)),
            message=str(task_data.get("message") or ""),
            total_symbols=int(task_data.get("total_symbols") or len(task_data.get("symbols") or [])),
            symbols=list(task_data.get("symbols") or []),
            completed_count=task_data.get("completed_count"),
            failed_count=task_data.get("failed_count"),
            total_count=task_data.get("total_count"),
            results=task_data.get("results"),
        )

    return AnalysisResponse(**task_data)
