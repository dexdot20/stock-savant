"""Favorites management commands."""

from __future__ import annotations

import json
from typing import List, Optional

from rich.console import Console
from rich.table import Table

from core import get_standard_logger, console as core_console
from core.paths import ensure_json_file, get_favorites_path
from domain.utils import normalize_symbol

logger = get_standard_logger(__name__)

FAVORITES_FILE = get_favorites_path()


def _ensure_favorites_file() -> None:
    """Ensure favorites file exists."""
    ensure_json_file(FAVORITES_FILE, [])


def load_favorites() -> List[str]:
    """Load favorites from JSON file."""
    _ensure_favorites_file()
    try:
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Error loading favorites: %s", e)
        return []


def save_favorites(favorites: List[str]) -> None:
    """Save favorites to JSON file."""
    _ensure_favorites_file()
    with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
        json.dump(favorites, f, indent=2)


def add_favorite(symbol: str) -> bool:
    """Add a symbol to favorites."""
    symbol = normalize_symbol(symbol)
    favorites = load_favorites()
    if symbol not in favorites:
        favorites.append(symbol)
        save_favorites(favorites)
        return True
    return False


def remove_favorite(symbol: str) -> bool:
    """Remove a symbol from favorites."""
    symbol = normalize_symbol(symbol)
    favorites = load_favorites()
    if symbol in favorites:
        favorites.remove(symbol)
        save_favorites(favorites)
        return True
    return False


def list_favorites(
    console: Optional[Console] = None, show_numbers: bool = False
) -> List[str]:
    """List all favorites."""
    if console is None:
        console = core_console

    favorites = sorted(load_favorites())
    if not favorites:
        console.print("[yellow]Your favorites list is empty.[/yellow]")
        return []

    table = Table(title="Favorite Companies")
    if show_numbers:
        table.add_column("#", style="cyan", justify="center")
    table.add_column("Symbol", style="cyan")

    for i, symbol in enumerate(favorites, 1):
        if show_numbers:
            table.add_row(str(i), symbol)
        else:
            table.add_row(symbol)

    console.print(table)
    return favorites
