"""Favorites management router."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Path
from fastapi.concurrency import run_in_threadpool

from api.models import FavoriteListResponse, FavoriteMutationResponse, clean_symbol
from commands.favorites import add_favorite, load_favorites, remove_favorite
from core import get_standard_logger

router = APIRouter(prefix="/favorites", tags=["Favorites"])
logger = get_standard_logger(__name__)


def _normalize_symbol(symbol: str) -> str:
    try:
        return clean_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


async def _get_current_favorites() -> list[str]:
    raw_favorites = await run_in_threadpool(load_favorites)
    favorites: list[str] = []
    seen: set[str] = set()

    for item in raw_favorites:
        try:
            symbol = clean_symbol(item)
        except ValueError:
            continue
        if symbol in seen:
            continue
        seen.add(symbol)
        favorites.append(symbol)

    return sorted(favorites)


@router.get(
    "/",
    response_model=FavoriteListResponse,
    summary="List favorites",
    description="Return the persisted favorites watchlist.",
)
async def list_favorites_endpoint():
    favorites = await _get_current_favorites()
    return FavoriteListResponse(favorites=favorites, count=len(favorites))


@router.post(
    "/{symbol}",
    response_model=FavoriteMutationResponse,
    summary="Add favorite",
    description="Add a symbol to the persisted favorites watchlist.",
)
async def add_favorite_endpoint(
    symbol: str = Path(..., description="Ticker symbol to add")
):
    normalized_symbol = _normalize_symbol(symbol)
    logger.info("API Request: add favorite — %s", normalized_symbol)

    added = await run_in_threadpool(add_favorite, normalized_symbol)
    favorites = await _get_current_favorites()
    return FavoriteMutationResponse(
        success=added,
        symbol=normalized_symbol,
        message=(
            "Favorite added." if added else "Symbol already exists in favorites."
        ),
        favorites=favorites,
    )


@router.delete(
    "/{symbol}",
    response_model=FavoriteMutationResponse,
    summary="Remove favorite",
    description="Remove a symbol from the persisted favorites watchlist.",
)
async def remove_favorite_endpoint(
    symbol: str = Path(..., description="Ticker symbol to remove")
):
    normalized_symbol = _normalize_symbol(symbol)
    logger.info("API Request: remove favorite — %s", normalized_symbol)

    removed = await run_in_threadpool(remove_favorite, normalized_symbol)
    if not removed:
        raise HTTPException(status_code=404, detail="Favorite not found.")

    favorites = await _get_current_favorites()
    return FavoriteMutationResponse(
        success=True,
        symbol=normalized_symbol,
        message="Favorite removed.",
        favorites=favorites,
    )