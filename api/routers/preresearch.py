"""
Pre-research Router
"""

import uuid
import re
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool

from api.models import PreResearchRequest, PreResearchResponse, AnalysisStatus
from api.state import create_task, get_task, update_task
from config import get_config
from core import get_standard_logger
from services.ai.providers import AIService
from services.factories import get_rag_service
from core.paths import get_runtime_dir

router = APIRouter(prefix="/preresearch", tags=["Pre-Research"])
logger = get_standard_logger(__name__)


async def run_preresearch_task(
    task_id: str, exchange: str, criteria: Optional[str], depth_mode: str
):
    """Run the pre-research workflow in the background."""
    try:
        update_task(task_id, {"status": AnalysisStatus.IN_PROGRESS.value})

        # Initialize the service
        config = get_config()
        ai_service = AIService(config)

        # Run the operation in a thread pool (AI calls can block)
        result = await run_in_threadpool(
            ai_service.pre_research_exchange,
            exchange=exchange,
            criteria=criteria,
            console=None,  # No console output in API mode
            depth_mode=depth_mode,
        )

        if not result or result.get("error"):
            error_msg = (
                result.get("error", "Unknown error")
                if isinstance(result, dict)
                else "Unknown error"
            )
            update_task(
                task_id,
                {
                    "status": AnalysisStatus.FAILED.value,
                    "message": f"Pre-research failed: {error_msg}",
                },
            )
            return

        analysis_text = result.get("analysis") if isinstance(result, dict) else ""
        if analysis_text in ("</tool_call>", "<tool_call>"):
            analysis_text = ""
        warning_text = (
            str(result.get("warning"))
            if isinstance(result, dict) and result.get("step_limit_reached")
            else ""
        )

        # Save the report to disk (preserve CLI behavior)
        try:
            reports_dir = get_runtime_dir() / "instance" / "reports" / "pre_research"
            reports_dir.mkdir(parents=True, exist_ok=True)

            safe_exchange = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", exchange)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{safe_exchange}_research.md"
            filepath = reports_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Pre-Research Report: {exchange}\n")
                f.write(f"**Date:** {datetime.now(timezone.utc).strftime('%d.%m.%Y %H:%M:%S')} UTC\n")
                if criteria:
                    f.write(f"**Criteria:** {criteria}\n")
                f.write("\n---\n\n")
                f.write(analysis_text)

            try:
                rag = get_rag_service(config)
                rag.index_pre_research(
                    exchange=exchange,
                    markdown_content=analysis_text,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as rag_exc:
                logger.debug("API pre-research RAG indexing skipped: %s", rag_exc)

            logger.info("Report saved to %s", filepath)
        except Exception as e:
            logger.error("Failed to save report: %s", e)

        update_task(
            task_id,
            {
                "status": AnalysisStatus.COMPLETED.value,
                "message": (
                    f"Pre-research completed. Warning: {warning_text}"
                    if warning_text
                    else "Pre-research completed"
                ),
                "result": analysis_text,
            },
        )

    except Exception as e:
        logger.error("Task %s error: %s", task_id, e)
        update_task(
            task_id,
            {"status": AnalysisStatus.FAILED.value, "message": f"Error: {str(e)}"},
        )


@router.post(
    "/",
    status_code=202,
    response_model=PreResearchResponse,
    summary="Start pre-research",
    description="Start pre-research for the specified exchange and criteria.",
)
async def start_preresearch(
    request: PreResearchRequest, background_tasks: BackgroundTasks
):
    """Start the pre-research workflow."""
    task_id = str(uuid.uuid4())

    # Save the task
    create_task(
        task_id,
        {
            "task_id": task_id,
            "exchange": request.exchange,
            "status": AnalysisStatus.PENDING.value,
            "created_ts": time.time(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": "Task queued",
        },
    )

    # Start the background task
    background_tasks.add_task(
        run_preresearch_task,
        task_id=task_id,
        exchange=request.exchange,
        criteria=request.criteria,
        depth_mode=request.depth_mode,
    )

    return PreResearchResponse(
        task_id=task_id,
        status=AnalysisStatus.PENDING,
        message="Pre-research started. Check /preresearch/{task_id} for the result.",
    )


@router.get(
    "/{task_id}",
    response_model=PreResearchResponse,
    summary="Get pre-research status",
    description="Return the status and result of the specified task.",
)
async def get_preresearch_status(task_id: str):
    task = get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return PreResearchResponse(
        task_id=task_id,
        status=task["status"],
        message=task.get("message", ""),
        result=task.get("result"),
    )
