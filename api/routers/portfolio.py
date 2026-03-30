"""Portfolio management router."""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Path
from fastapi.concurrency import run_in_threadpool

from api.models import (
    PortfolioListResponse,
    PortfolioMutationResponse,
    PortfolioPosition,
    PortfolioPositionPayload,
    PortfolioRiskSnapshotResponse,
    PortfolioSnapshotResponse,
    clean_symbol,
)
from commands.portfolio import (
    add_position_with_feedback,
    load_portfolio,
    portfolio_risk_snapshot,
    portfolio_snapshot,
    remove_position,
)
from config import get_config
from core import get_standard_logger

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])
logger = get_standard_logger(__name__)


def _normalize_symbol(symbol: str) -> str:
    try:
        return clean_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


async def _load_positions() -> List[Dict[str, Any]]:
    raw_entries = await run_in_threadpool(load_portfolio)
    positions: List[Dict[str, Any]] = []

    for item in raw_entries:
        if not isinstance(item, dict):
            continue
        try:
            symbol = clean_symbol(item.get("symbol", ""))
            quantity = float(item.get("quantity", 0.0))
            average_cost = float(item.get("average_cost", 0.0))
        except (TypeError, ValueError):
            continue
        positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "average_cost": average_cost,
            }
        )

    return sorted(positions, key=lambda entry: entry["symbol"])


def _find_position(entries: List[Dict[str, Any]], symbol: str) -> Dict[str, Any] | None:
    return next((entry for entry in entries if entry.get("symbol") == symbol), None)


@router.get(
    "/positions",
    response_model=PortfolioListResponse,
    summary="List portfolio positions",
    description="Return the persisted portfolio positions.",
)
async def list_positions_endpoint():
    positions = await _load_positions()
    return PortfolioListResponse(positions=positions, count=len(positions))


@router.post(
    "/positions",
    response_model=PortfolioMutationResponse,
    summary="Save portfolio position",
    description="Create a new portfolio position or increase an existing one.",
)
async def save_position_endpoint(payload: PortfolioPositionPayload):
    logger.info("API Request: save portfolio position — %s", payload.symbol)
    config = get_config()
    success, message = await run_in_threadpool(
        add_position_with_feedback,
        payload.symbol,
        payload.quantity,
        payload.average_cost,
        config,
    )
    if not success:
        raise HTTPException(status_code=422, detail=message)

    positions = await _load_positions()
    position = _find_position(positions, payload.symbol)
    return PortfolioMutationResponse(success=True, message=message, position=position)


@router.delete(
    "/positions/{symbol}",
    response_model=PortfolioMutationResponse,
    summary="Remove portfolio position",
    description="Remove a position from the persisted portfolio.",
)
async def remove_position_endpoint(
    symbol: str = Path(..., description="Ticker symbol to remove")
):
    normalized_symbol = _normalize_symbol(symbol)
    logger.info("API Request: remove portfolio position — %s", normalized_symbol)

    removed = await run_in_threadpool(remove_position, normalized_symbol)
    if not removed:
        raise HTTPException(status_code=404, detail="Portfolio position not found.")

    return PortfolioMutationResponse(
        success=True,
        message="Position removed.",
        position=None,
    )


@router.get(
    "/snapshot",
    response_model=PortfolioSnapshotResponse,
    summary="Portfolio snapshot",
    description="Return portfolio valuation, profit and loss, and current prices.",
)
async def get_portfolio_snapshot_endpoint():
    logger.info("API Request: portfolio snapshot")
    snapshot = await run_in_threadpool(portfolio_snapshot, get_config())
    return snapshot


@router.get(
    "/risk",
    response_model=PortfolioRiskSnapshotResponse,
    summary="Portfolio risk snapshot",
    description="Return the portfolio risk cockpit data, including sector and correlation warnings.",
)
async def get_portfolio_risk_endpoint():
    logger.info("API Request: portfolio risk snapshot")
    snapshot = await run_in_threadpool(portfolio_risk_snapshot, get_config())
    return snapshot