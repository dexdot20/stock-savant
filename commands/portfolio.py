"""Portfolio management commands (CLI only)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from config import get_config
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from core import get_standard_logger
from core.console import console as core_console
from core.paths import ensure_json_file, get_portfolio_path
from domain.utils import normalize_symbol, safe_int_strict as _safe_int
from services.factories import get_financial_service
from services.finance.risk import PortfolioRiskService

logger = get_standard_logger(__name__)

PORTFOLIO_FILE = get_portfolio_path()


def _ensure_portfolio_file() -> None:
    ensure_json_file(PORTFOLIO_FILE, [])


def load_portfolio() -> List[Dict[str, Any]]:
    _ensure_portfolio_file()
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.error("Portfolio load error: %s", exc)
        return []


def save_portfolio(entries: List[Dict[str, Any]]) -> None:
    _ensure_portfolio_file()
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def _portfolio_protection_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or get_config()
    return (
        cfg.get("portfolio", {})
        .get("protection", {})
    )


def _validate_protection_rules(
    entries: List[Dict[str, Any]],
    symbol: str,
    quantity: float,
    average_cost: float,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    protection = _portfolio_protection_config(config)
    if not bool(protection.get("enabled", True)):
        return None

    max_positions = _safe_int(protection.get("max_positions", 20) or 20, 20)
    max_single_pct = float(protection.get("max_single_position_pct", 40.0) or 40.0)
    max_total_cost = float(protection.get("max_total_cost", 0) or 0)

    symbol = normalize_symbol(symbol)
    existing = next((item for item in entries if item.get("symbol") == symbol), None)
    if existing:
        old_qty = float(existing.get("quantity", 0.0))
        old_cost = float(existing.get("average_cost", 0.0))
        new_qty = old_qty + quantity
        new_avg = ((old_qty * old_cost) + (quantity * average_cost)) / new_qty
    else:
        if len(entries) >= max_positions:
            return (
                f"Guardrail: You can open at most {max_positions} positions. "
                "Close one position before adding a new stock."
            )
        new_qty = quantity
        new_avg = average_cost

    if new_qty <= 0 or new_avg <= 0:
        return "Quantity and cost must be greater than zero."

    simulated = [dict(item) for item in entries]
    if existing:
        for item in simulated:
            if item.get("symbol") == symbol:
                item["quantity"] = new_qty
                item["average_cost"] = new_avg
                break
    else:
        simulated.append(
            {"symbol": symbol, "quantity": new_qty, "average_cost": new_avg}
        )

    total_cost = 0.0
    target_cost = 0.0
    for item in simulated:
        cost_value = float(item.get("quantity", 0.0)) * float(item.get("average_cost", 0.0))
        total_cost += cost_value
        if normalize_symbol(item.get("symbol", "")) == symbol:
            target_cost = cost_value

    if max_total_cost > 0 and total_cost > max_total_cost:
        return (
            f"Guardrail: Total cost limit exceeded. "
            f"Limit={max_total_cost:.2f}, new total={total_cost:.2f}."
        )

    if total_cost > 0:
        symbol_weight_pct = (target_cost / total_cost) * 100.0
        if symbol_weight_pct > max_single_pct:
            return (
                f"Guardrail: Single-position weight exceeds %{max_single_pct:.1f}. "
                f"After this trade, {symbol} would weigh %{symbol_weight_pct:.1f}."
            )

    return None


def add_position_with_feedback(
    symbol: str,
    quantity: float,
    average_cost: float,
    config: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    if quantity <= 0 or average_cost <= 0:
        return False, "Quantity and cost must be greater than zero."

    symbol = normalize_symbol(symbol)
    entries = load_portfolio()
    protection_error = _validate_protection_rules(
        entries,
        symbol,
        quantity,
        average_cost,
        config=config,
    )
    if protection_error:
        return False, protection_error

    existing = next((item for item in entries if item.get("symbol") == symbol), None)

    if existing:
        old_qty = float(existing.get("quantity", 0.0))
        old_cost = float(existing.get("average_cost", 0.0))
        new_qty = old_qty + quantity
        if new_qty <= 0:
            return False, "The new total quantity cannot be zero or negative."
        existing["average_cost"] = ((old_qty * old_cost) + (quantity * average_cost)) / new_qty
        existing["quantity"] = new_qty
    else:
        entries.append(
            {
                "symbol": symbol,
                "quantity": float(quantity),
                "average_cost": float(average_cost),
            }
        )

    save_portfolio(entries)
    return True, "Position saved."


def add_position(symbol: str, quantity: float, average_cost: float) -> bool:
    success, _ = add_position_with_feedback(symbol, quantity, average_cost)
    return success


def remove_position(symbol: str) -> bool:
    symbol = symbol.upper().strip()
    entries = load_portfolio()
    filtered = [item for item in entries if item.get("symbol") != symbol]
    if len(filtered) == len(entries):
        return False
    save_portfolio(filtered)
    return True


def list_positions(console: Optional[Console] = None) -> List[Dict[str, Any]]:
    if console is None:
        console = core_console

    entries = sorted(load_portfolio(), key=lambda item: item.get("symbol", ""))
    if not entries:
        console.print("[yellow]Your portfolio list is empty.[/yellow]")
        return []

    table = Table(title="Portfolio Positions")
    table.add_column("Symbol", style="cyan")
    table.add_column("Quantity", style="green", justify="right")
    table.add_column("Average Cost", style="yellow", justify="right")

    for item in entries:
        table.add_row(
            str(item.get("symbol", "")),
            f"{float(item.get('quantity', 0.0)):.4f}",
            f"{float(item.get('average_cost', 0.0)):.4f}",
        )

    console.print(table)
    return entries


def portfolio_snapshot(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entries = load_portfolio()
    if not entries:
        return {
            "positions": [],
            "totals": {
                "cost": 0.0,
                "market_value": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
            },
        }

    service = get_financial_service(config)
    positions: List[Dict[str, Any]] = []
    total_cost = 0.0
    total_market_value = 0.0
    symbols = [str(entry.get("symbol", "")).upper().strip() for entry in entries]
    bulk_prices = {}
    bulk_price_fn = getattr(service, "get_latest_prices_bulk", None)
    if callable(bulk_price_fn):
        try:
            bulk_prices = bulk_price_fn(symbols)
        except Exception as exc:
            logger.debug("Bulk portfolio pricing unavailable: %s", exc)

    for entry in entries:
        symbol = str(entry.get("symbol", "")).upper().strip()
        quantity = float(entry.get("quantity", 0.0))
        average_cost = float(entry.get("average_cost", 0.0))

        current_price = None
        try:
            bulk_data = bulk_prices.get(symbol) or {}
            current_price = bulk_data.get("currentPrice") or bulk_data.get("regularMarketPrice")
            if current_price is None:
                data = service.get_company_data(symbol)
                if data:
                    current_price = data.get("currentPrice") or data.get("regularMarketPrice")
                if current_price is not None:
                    current_price = float(current_price)
        except Exception as exc:
            logger.debug("Current price unavailable for %s: %s", symbol, exc)

        cost_value = quantity * average_cost
        price_available = current_price is not None
        effective_price = current_price if price_available else average_cost
        market_value = quantity * effective_price
        pnl = market_value - cost_value if price_available else 0.0
        pnl_pct = (pnl / cost_value * 100.0) if cost_value > 0 and price_available else 0.0

        total_cost += cost_value
        total_market_value += market_value

        positions.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "average_cost": average_cost,
                "current_price": current_price,
                "cost_value": cost_value,
                "market_value": market_value,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "price_unavailable": not price_available,
            }
        )

    total_pnl = total_market_value - total_cost
    total_pnl_pct = (total_pnl / total_cost * 100.0) if total_cost > 0 else 0.0

    return {
        "positions": positions,
        "totals": {
            "cost": total_cost,
            "market_value": total_market_value,
            "pnl": total_pnl,
            "pnl_pct": total_pnl_pct,
        },
    }


def render_portfolio_snapshot(console: Optional[Console] = None, config: Optional[Dict[str, Any]] = None) -> None:
    if console is None:
        console = core_console

    snapshot = portfolio_snapshot(config)
    positions = snapshot.get("positions", [])
    totals = snapshot.get("totals", {})

    if not positions:
        console.print("[yellow]No positions available to display in the portfolio.[/yellow]")
        return

    table = Table(title="Portfolio Summary")
    table.add_column("Symbol", style="cyan")
    table.add_column("Quantity", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Market Value", justify="right")
    table.add_column("P/L", justify="right")

    for item in positions:
        current = item.get("current_price")
        pnl = float(item.get("pnl", 0.0))
        pnl_style = "green" if pnl >= 0 else "red"
        table.add_row(
            str(item.get("symbol")),
            f"{float(item.get('quantity', 0.0)):.4f}",
            f"{float(item.get('average_cost', 0.0)):.4f}",
            f"{current:.4f}" if current is not None else "N/A",
            f"{float(item.get('cost_value', 0.0)):.2f}",
            f"{float(item.get('market_value', 0.0)):.2f}",
            f"[{pnl_style}]{pnl:.2f} ({float(item.get('pnl_pct', 0.0)):.2f}%)[/{pnl_style}]",
        )

    console.print(table)

    total_pnl = float(totals.get("pnl", 0.0))
    total_style = "green" if total_pnl >= 0 else "red"
    console.print(
        f"[bold]Total Cost:[/bold] {float(totals.get('cost', 0.0)):.2f} | "
        f"[bold]Market Value:[/bold] {float(totals.get('market_value', 0.0)):.2f} | "
        f"[bold]Total P/L:[/bold] [{total_style}]{total_pnl:.2f} ({float(totals.get('pnl_pct', 0.0)):.2f}%)[/{total_style}]"
    )


def portfolio_risk_snapshot(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    entries = load_portfolio()
    risk_service = PortfolioRiskService(config)
    return risk_service.build_snapshot(entries)


def render_portfolio_risk_cockpit(
    console: Optional[Console] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    if console is None:
        console = core_console

    snapshot = portfolio_risk_snapshot(config)
    positions = snapshot.get("positions", [])
    if not positions:
        console.print("[yellow]No positions found for the portfolio risk cockpit.[/yellow]")
        return

    summary = snapshot.get("summary", {})
    console.print(
        Panel.fit(
            "[bold cyan]🛡️ Portfolio Risk Cockpit[/bold cyan]\n"
            f"Risk Score: {float(summary.get('risk_score', 0.0)):.2f} | "
            f"Average Confidence: %{float(summary.get('average_confidence', 0.0)):.2f} | "
            f"Diversification: %{float(summary.get('diversification_score', 0.0)):.2f}",
            border_style="cyan",
        )
    )

    if summary.get("estimated_portfolio_volatility") is not None:
        console.print(
            f"[bold]Estimated annualized volatility:[/bold] %{float(summary.get('estimated_portfolio_volatility', 0.0)):.2f}"
        )

    position_table = Table(title="Risk-Adjusted Positions")
    position_table.add_column("Symbol", style="cyan")
    position_table.add_column("Weight", justify="right")
    position_table.add_column("Sector", style="green")
    position_table.add_column("Confidence", justify="right")
    position_table.add_column("P/L", justify="right")

    for item in positions:
        position_table.add_row(
            str(item.get("symbol") or "-"),
            f"%{float(item.get('weight_pct', 0.0)):.2f}",
            str(item.get("sector") or "Unknown"),
            f"%{float(item.get('confidence_pct', 0.0)):.2f} ({item.get('confidence_level', 'n/a')})",
            f"{float(item.get('pnl', 0.0)):.2f} ({float(item.get('pnl_pct', 0.0)):.2f}%)",
        )

    console.print(position_table)

    sector_table = Table(title="Sector Exposure")
    sector_table.add_column("Sector", style="magenta")
    sector_table.add_column("Weight", justify="right")
    sector_table.add_column("Market Value", justify="right")
    for item in snapshot.get("sector_exposure", []):
        sector_table.add_row(
            str(item.get("sector") or "Unknown"),
            f"%{float(item.get('weight_pct', 0.0)):.2f}",
            f"{float(item.get('market_value', 0.0)):.2f}",
        )
    console.print(sector_table)

    breaches = snapshot.get("breaches", [])
    correlation_alerts = snapshot.get("correlation_alerts", [])
    if breaches or correlation_alerts:
        console.print("\n[bold red]Risk Warnings[/bold red]")
        for item in breaches + correlation_alerts:
            console.print(f"- {item.get('message')}")
