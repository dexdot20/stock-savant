from __future__ import annotations

from typing import Optional

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from core import console as core_console
from services.investor_profile import (
    DEFAULT_PLAYBOOKS,
    get_playbook_choices,
    load_investor_profile,
    save_investor_profile,
)


def show_investor_profile(console: Optional[Console] = None) -> None:
    if console is None:
        console = core_console

    profile = load_investor_profile()
    table = Table(title="Investor Profile")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    for key in (
        "profile_name",
        "risk_tolerance",
        "investment_horizon",
        "market_focus",
        "preferred_sectors",
        "avoided_sectors",
        "max_single_position_pct",
        "alert_sensitivity",
        "active_playbook",
    ):
        value = profile.get(key)
        if isinstance(value, list):
            value = ", ".join(value) or "-"
        table.add_row(key, str(value))

    playbook = str(profile.get("active_playbook") or "balanced")
    table.add_row(
        "playbook_summary",
        str(DEFAULT_PLAYBOOKS.get(playbook, {}).get("summary") or "-"),
    )
    console.print(table)


def edit_investor_profile(console: Optional[Console] = None) -> None:
    if console is None:
        console = core_console

    profile = load_investor_profile()
    profile_name = Prompt.ask(
        "Profile name",
        default=str(profile.get("profile_name") or "Default"),
    ).strip() or "Default"
    risk_tolerance = Prompt.ask(
        "Risk tolerance",
        choices=["low", "medium", "high"],
        default=str(profile.get("risk_tolerance") or "medium"),
    )
    investment_horizon = Prompt.ask(
        "Investment horizon",
        choices=["short-term", "medium-term", "long-term"],
        default=str(profile.get("investment_horizon") or "long-term"),
    )
    market_focus = Prompt.ask(
        "Market focus",
        choices=["BIST", "US", "Crypto", "All"],
        default=str(profile.get("market_focus") or "BIST"),
    )
    preferred_sectors = Prompt.ask(
        "Preferred sectors (comma separated, optional)",
        default=", ".join(profile.get("preferred_sectors") or []),
    )
    avoided_sectors = Prompt.ask(
        "Avoided sectors (comma separated, optional)",
        default=", ".join(profile.get("avoided_sectors") or []),
    )
    max_single_position_pct_raw = Prompt.ask(
        "Preferred max single position %",
        default=str(profile.get("max_single_position_pct") or 25.0),
    )
    alert_sensitivity = Prompt.ask(
        "Alert sensitivity",
        choices=["low", "medium", "high"],
        default=str(profile.get("alert_sensitivity") or "medium"),
    )
    playbook = Prompt.ask(
        "Active playbook",
        choices=get_playbook_choices(),
        default=str(profile.get("active_playbook") or "balanced"),
    )

    try:
        max_single_position_pct = max(1.0, float(max_single_position_pct_raw))
    except ValueError:
        max_single_position_pct = 25.0

    saved = save_investor_profile(
        {
            "profile_name": profile_name,
            "risk_tolerance": risk_tolerance,
            "investment_horizon": investment_horizon,
            "market_focus": market_focus,
            "preferred_sectors": [item.strip() for item in preferred_sectors.split(",") if item.strip()],
            "avoided_sectors": [item.strip() for item in avoided_sectors.split(",") if item.strip()],
            "max_single_position_pct": max_single_position_pct,
            "alert_sensitivity": alert_sensitivity,
            "active_playbook": playbook,
        }
    )
    console.print(f"[green]✅ Investor profile saved: {saved.get('profile_name')}[/green]")