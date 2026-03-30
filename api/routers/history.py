"""Analysis history router."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.concurrency import run_in_threadpool

from api.models import HistoryEntryResponse, HistoryListResponse, clean_symbol
from commands.history import list_history, show_history_entry
from core import get_standard_logger

router = APIRouter(prefix="/history", tags=["History"])
logger = get_standard_logger(__name__)


def _normalize_optional_symbol(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    try:
        return clean_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/",
    response_model=HistoryListResponse,
    summary="List history entries",
    description="Return persisted analysis history entries with optional symbol filtering.",
)
async def list_history_endpoint(
    limit: int = Query(10, ge=1, le=100, description="Maximum number of entries to return"),
    symbol: Optional[str] = Query(None, description="Optional ticker symbol filter"),
):
    normalized_symbol = _normalize_optional_symbol(symbol)
    logger.info(
        "API Request: list history — symbol=%s limit=%s",
        normalized_symbol,
        limit,
    )
    entries = await run_in_threadpool(list_history, limit, normalized_symbol)
    return HistoryListResponse(entries=entries, count=len(entries))


@router.get(
    "/{analysis_id}",
    response_model=HistoryEntryResponse,
    summary="Get history entry",
    description="Return the persisted payload for a single analysis history entry.",
)
async def get_history_entry_endpoint(
    analysis_id: int = Path(..., ge=1, description="History entry identifier")
):
    logger.info("API Request: history entry — %s", analysis_id)
    entry = await run_in_threadpool(show_history_entry, analysis_id)
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found.")
    return entry