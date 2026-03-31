"""Configuration commands"""

from __future__ import annotations

from typing import Dict, Any, Optional

from rich.console import Console
from rich.table import Table

from config import is_configured_secret
from core import get_standard_logger
from core.console import console as core_console
from services.ai.providers.provider_metadata import get_provider_display_name

logger = get_standard_logger(__name__)


def _display_provider_name(provider_name: str) -> str:
    return get_provider_display_name(provider_name)


def show_config(config: Dict[str, Any], console: Optional[Console] = None) -> None:
    """
    Display current configuration

    Args:
        config: Configuration dictionary
        console: Rich console for output
    """
    if console is None:
        console = core_console

    # AI Configuration
    ai_config = config.get("ai", {})

    console.print("\n[bold cyan]🤖 AI Configuration[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Output Language", ai_config.get("output_language", "English"))

    console.print(table)

    # Model Configuration
    models = ai_config.get("models", {})

    console.print("\n[bold cyan]🎯 AI Models[/bold cyan]\n")

    model_table = Table(show_header=True, header_style="bold magenta")
    model_table.add_column("Purpose", style="cyan")
    model_table.add_column("Provider", style="yellow")
    model_table.add_column("Model", style="green")

    for purpose, model_cfg in models.items():
        if isinstance(model_cfg, dict):
            model_table.add_row(
                purpose.replace("_", " ").title(),
                _display_provider_name(model_cfg.get("provider", "N/A")),
                model_cfg.get("model", "N/A"),
            )

    console.print(model_table)

    # API Keys Status
    console.print("\n[bold cyan]🔑 API Keys Status[/bold cyan]\n")

    key_table = Table(show_header=True, header_style="bold magenta")
    key_table.add_column("Provider", style="cyan")
    key_table.add_column("Status", style="yellow")

    providers = ai_config.get("providers", {})
    for provider_name, provider_cfg in providers.items():
        if isinstance(provider_cfg, dict):
            api_key = provider_cfg.get("api_key", "")
            status = (
                "[green]✅ Set[/green]"
                if is_configured_secret(api_key)
                else "[red]❌ Not Set[/red]"
            )
            key_table.add_row(_display_provider_name(provider_name), status)

    console.print(key_table)

    # Market Data Configuration
    market_config = config.get("market_data", {})

    console.print("\n[bold cyan]📈 Market Data Configuration[/bold cyan]\n")

    market_table = Table(show_header=True, header_style="bold magenta")
    market_table.add_column("Setting", style="cyan")
    market_table.add_column("Value", style="green")

    market_table.add_row("Provider", market_config.get("provider", "yfinance"))
    market_table.add_row("History Period", market_config.get("history_period", "1y"))
    market_table.add_row(
        "History Interval", market_config.get("history_interval", "1d")
    )
    market_table.add_row(
        "Cache TTL (hours)", str(market_config.get("default_cache_ttl_hours", 24))
    )

    console.print(market_table)


def validate_config(config: Dict[str, Any], console: Optional[Console] = None) -> bool:
    """
    Validate configuration and show issues

    Args:
        config: Configuration dictionary
        console: Rich console for output

    Returns:
        True if configuration is valid, False otherwise
    """
    if console is None:
        console = core_console

    issues = []
    warnings = []

    console.print("\n[bold cyan]🔍 Validating Configuration...[/bold cyan]\n")

    # Check AI configuration
    ai_config = config.get("ai", {})
    if not ai_config:
        issues.append("AI configuration is missing")
    else:
        # Check providers
        providers = ai_config.get("providers", {})
        if not providers:
            issues.append("No AI providers configured")
        else:
            # Check API keys
            for provider_name, provider_cfg in providers.items():
                if isinstance(provider_cfg, dict):
                    api_key = provider_cfg.get("api_key", "")
                    if not is_configured_secret(api_key):
                        warnings.append(
                            f"{_display_provider_name(provider_name)} API key is not set"
                        )

        # Check models
        models = ai_config.get("models", {})
        if not models:
            issues.append("No AI models configured")
        else:
            required_models = ["news", "reasoner"]
            for model_type in required_models:
                if model_type not in models:
                    warnings.append(f"'{model_type}' model is not configured")

    # Check market data configuration
    market_config = config.get("market_data", {})
    if not market_config:
        warnings.append("Market data configuration is missing (using defaults)")

    # Display results
    if issues:
        console.print("[bold red]❌ Critical Issues:[/bold red]\n")
        for issue in issues:
            console.print(f"  • [red]{issue}[/red]")
        console.print()

    if warnings:
        console.print("[bold yellow]⚠️  Warnings:[/bold yellow]\n")
        for warning in warnings:
            console.print(f"  • [yellow]{warning}[/yellow]")
        console.print()

    if not issues and not warnings:
        console.print("[green]✅ All checks passed![/green]\n")
        return True
    elif not issues:
        console.print(
            "[yellow]⚠️  Configuration has warnings but is functional[/yellow]\n"
        )
        return True
    else:
        return False
