"""
Startup module for displaying welcome message and market status information.
"""

from datetime import datetime, time, timedelta
from typing import Dict, Tuple
from zoneinfo import ZoneInfo

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


class MarketInfo:
    """Stores market information including timezone and trading hours."""

    def __init__(
        self,
        name: str,
        code: str,
        timezone: str,
        open_time: time,
        close_time: time,
        trading_days: str = "Monday-Friday",
    ):
        self.name = name
        self.code = code
        self.timezone = timezone
        self.open_time = open_time
        self.close_time = close_time
        self.trading_days = trading_days


# Market definitions
MARKETS = {
    "BIST": MarketInfo(
        name="Borsa Istanbul",
        code="BIST",
        timezone="Europe/Istanbul",
        open_time=time(10, 0),
        close_time=time(18, 0),
    ),
    "NYSE": MarketInfo(
        name="New York Stock Exchange",
        code="NYSE",
        timezone="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
    ),
    "NASDAQ": MarketInfo(
        name="NASDAQ",
        code="NASDAQ",
        timezone="America/New_York",
        open_time=time(9, 30),
        close_time=time(16, 0),
    ),
    "LSE": MarketInfo(
        name="London Stock Exchange",
        code="LSE",
        timezone="Europe/London",
        open_time=time(8, 0),
        close_time=time(16, 30),
    ),
}


def is_weekend(dt: datetime) -> bool:
    """Check if the given datetime falls on a weekend."""
    return dt.weekday() in (5, 6)  # Saturday=5, Sunday=6


def is_market_open(market: MarketInfo) -> Tuple[bool, str]:
    """
    Check if a market is currently open.

    Returns:
        Tuple of (is_open, status_message)
    """
    try:
        tz = ZoneInfo(market.timezone)
        now = datetime.now(tz)

        # Check weekend
        if is_weekend(now):
            return False, "Weekend"

        # Check trading hours
        current_time = now.time()
        if market.open_time <= current_time <= market.close_time:
            return True, "Open"
        else:
            return False, "Closed"
    except Exception:
        return False, "Unknown"


def calculate_next_opening(market: MarketInfo) -> Tuple[datetime, str]:
    """
    Calculate the next opening time for a market.

    Returns:
        Tuple of (next_opening_datetime, formatted_countdown_string)
    """
    try:
        tz = ZoneInfo(market.timezone)
        now = datetime.now(tz)

        # Start with today
        next_open = now.replace(
            hour=market.open_time.hour,
            minute=market.open_time.minute,
            second=0,
            microsecond=0,
        )

        # If already past today's opening, or it's a weekend, move to next day
        if now.time() >= market.open_time or is_weekend(now):
            next_open += timedelta(days=1)

        # Skip weekends
        while is_weekend(next_open):
            next_open += timedelta(days=1)

        # Calculate time difference
        time_diff = next_open - now

        # Format countdown
        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)

        if hours > 24:
            days = hours // 24
            remaining_hours = hours % 24
            countdown = f"{days} d {remaining_hours} h"
        elif hours > 0:
            countdown = f"{hours} h {minutes} m"
        else:
            countdown = f"{minutes} m"

        return next_open, countdown
    except Exception:
        return now, "Not calculated"


def display_startup_info(console: Console) -> None:
    """
    Display startup information including welcome message and market status.

    Args:
        console: Rich console instance for displaying formatted output
    """
    market_summary = get_market_summary()
    open_markets = sum(1 for is_open, _ in market_summary.values() if is_open)
    closed_markets = len(market_summary) - open_markets

    # Welcome message
    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🎯 Welcome to Struct[/bold cyan]\n"
            "[dim]AI-powered stock analysis, comparison, and portfolio tracking from the terminal.[/dim]\n\n"
            "[bold]Quick start[/bold]\n"
            "[dim]1[/dim] Analyze stock  •  [dim]2[/dim] Batch analysis  •  [dim]3[/dim] Pre-research  •  [dim]4[/dim] Compare companies\n"
            "[dim]5[/dim] Financial data  •  [dim]6[/dim] Index / ETF data  •  [dim]7[/dim] History  •  [dim]8[/dim] Favorites  •  [dim]9[/dim] Portfolio\n\n"
            f"[dim]Open markets:[/dim] [green]{open_markets}[/green]  [dim]| Closed:[/dim] [red]{closed_markets}[/red]  [dim]| Tip:[/dim] type a menu number and press Enter.",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )

    # Market status table
    table = Table(
        title="📊 Market Snapshot",
        show_header=True,
        header_style="bold cyan",
        box=box.ROUNDED,
        title_style="bold yellow",
    )
    table.add_column("Exchange", style="cyan", justify="left")
    table.add_column("Status", style="white", justify="center")
    table.add_column("Next Open", style="dim", justify="left")

    for market_code, market in MARKETS.items():
        is_open, status = is_market_open(market)

        # Status indicator
        if is_open:
            status_indicator = f"[green]● {status}[/green]"
            info = "Trading now"
        else:
            status_indicator = f"[red]● {status}[/red]"
            _, countdown = calculate_next_opening(market)
            info = f"Opening in {countdown}"

        table.add_row(f"{market.name} ({market.code})", status_indicator, info)

    console.print(table)
    console.print(
        Panel.fit(
            "[bold]Usage hints[/bold]\n"
            "[dim]• Use q, quit, or exit in guided prompts when cancellation is supported.[/dim]\n"
            "[dim]• Saved output and terminal transcripts are written under instance/.[/dim]",
            border_style="cyan",
            box=box.ROUNDED,
        )
    )


def get_market_summary() -> Dict[str, Tuple[bool, str]]:
    """
    Get a summary of all market statuses.

    Returns:
        Dictionary mapping market codes to (is_open, status) tuples
    """
    summary = {}
    for market_code, market in MARKETS.items():
        is_open, status = is_market_open(market)
        summary[market_code] = (is_open, status)
    return summary
