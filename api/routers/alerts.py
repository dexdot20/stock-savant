"""Alerts router."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool

from api.models import (
    AlertCenterResponse,
    AlertCreateRequest,
    AlertCreateResponse,
    AlertListResponse,
    clean_symbol,
)
from config import get_config
from core import get_standard_logger
from services.alerts import create_price_alert, evaluate_alert_center, list_alerts

router = APIRouter(prefix="/alerts", tags=["Alerts"])
logger = get_standard_logger(__name__)


def _normalize_optional_symbol(symbol: Optional[str]) -> Optional[str]:
    if symbol is None:
        return None
    try:
        return clean_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get(
    "/center",
    response_model=AlertCenterResponse,
    summary="Evaluate alert center",
    description="Evaluate price alerts, portfolio risk alerts, and KAP alerts in one call.",
)
async def get_alert_center_endpoint():
    logger.info("API Request: alert center")
    return await run_in_threadpool(evaluate_alert_center, get_config())


@router.get(
    "/",
    response_model=AlertListResponse,
    summary="List alerts",
    description="Return saved price alerts with optional symbol filtering.",
)
async def list_alerts_endpoint(
    symbol: Optional[str] = Query(None, description="Optional ticker symbol filter"),
    include_triggered: bool = Query(
        False,
        description="Include alerts that have already been triggered",
    ),
):
    normalized_symbol = _normalize_optional_symbol(symbol)
    logger.info(
        "API Request: list alerts — symbol=%s include_triggered=%s",
        normalized_symbol,
        include_triggered,
    )
    return await run_in_threadpool(
        list_alerts,
        symbol=normalized_symbol,
        include_triggered=include_triggered,
    )


@router.post(
    "/price",
    response_model=AlertCreateResponse,
    summary="Create price alert",
    description="Create a persisted price alert for a single symbol.",
)
async def create_price_alert_endpoint(payload: AlertCreateRequest):
    logger.info("API Request: create price alert — %s", payload.symbol)
    result = await run_in_threadpool(
        create_price_alert,
        payload.symbol,
        payload.target_price,
        payload.direction.value,
        payload.note,
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=str(result["error"]))

    return AlertCreateResponse(
        message="Price alert created.",
        alert=result["alert"],
    )