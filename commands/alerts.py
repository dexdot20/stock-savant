from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from core import console as core_console
from services.alerts import create_price_alert, evaluate_alert_center, list_alerts


def render_alert_center(console: Optional[Console] = None) -> None:
    if console is None:
        console = core_console

    result = evaluate_alert_center()
    triggered = result.get("triggered_events") or []
    alerts_payload = result.get("saved_alerts") or {}
    alerts = alerts_payload.get("alerts") or []

    console.print(Panel.fit("[bold cyan]🔔 Alert Center[/bold cyan]", border_style="cyan"))

    if triggered:
        console.print("[bold yellow]New triggered events[/bold yellow]")
        for item in triggered:
            console.print(f"- {item.get('message')}")
        console.print()

    table = Table(title="Saved Alerts")
    table.add_column("Type", style="cyan")
    table.add_column("Symbol", style="green")
    table.add_column("Condition", style="white")
    table.add_column("Status", style="yellow")
    table.add_column("Note", style="magenta")

    if alerts:
        for item in alerts:
            alert_type = str(item.get("type") or "price")
            condition = "-"
            if alert_type == "price":
                condition = f"{item.get('direction')} {float(item.get('target_price', 0.0)):.4f}"
            table.add_row(
                alert_type,
                str(item.get("symbol") or "-"),
                condition,
                str(item.get("status") or ("triggered" if item.get("triggered") else "active")),
                str(item.get("note") or ""),
            )
    else:
        table.add_row("-", "-", "-", "No alerts", "-")

    console.print(table)

    risk_events = result.get("risk_events") or []
    if risk_events:
        console.print("\n[bold red]Portfolio Risk Alerts[/bold red]")
        for item in risk_events:
            console.print(f"- {item.get('message')}")

    kap_events = result.get("kap_events") or []
    if kap_events:
        console.print("\n[bold magenta]KAP Alerts[/bold magenta]")
        for item in kap_events:
            console.print(f"- {item.get('message')}")


def prompt_and_create_price_alert(console: Optional[Console] = None) -> None:
    if console is None:
        console = core_console

    symbol = Prompt.ask("Alert symbol", default="").upper().strip()
    if not symbol:
        console.print("[red]❌ Symbol is required.[/red]")
        return
    direction = Prompt.ask("Direction", choices=["above", "below"], default="above")
    try:
        target_price = float(Prompt.ask("Hedef fiyat", default="1"))
    except ValueError:
        console.print("[red]❌ Target price must be numeric.[/red]")
        return
    note = Prompt.ask("Note", default="")

    result = create_price_alert(symbol, target_price, direction, note)
    if result.get("error"):
        console.print(f"[red]❌ {result.get('error')}[/red]")
        return
    console.print(f"[green]✅ Alert saved: {symbol}[/green]")
