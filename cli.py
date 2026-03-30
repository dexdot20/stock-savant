#!/usr/bin/env python3

from __future__ import annotations

# Suppress FutureWarning messages from pandas/yfinance before any imports
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional

from rich.console import Console
from rich.status import Status
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from commands.analyze import analyze_multiple_stocks, analyze_single_stock
from commands.config import show_config, validate_config
from commands.history import add_to_history, list_history, show_history_entry
from commands.financials import (
    handle_company_financial_query,
    handle_index_etf_financial_query,
)
from commands.output import (
    display_analysis_result,
    display_batch_results,
    display_history_entry,
)
from commands.profile import edit_investor_profile, show_investor_profile
from commands.compare import handle_compare_companies
from commands.pre_research import handle_pre_research
from commands.favorites import (
    add_favorite,
    list_favorites,
    remove_favorite,
    load_favorites,
)
from commands.portfolio import (
    add_position_with_feedback,
    remove_position,
    list_positions,
    render_portfolio_snapshot,
    render_portfolio_risk_cockpit,
)
from commands.alerts import prompt_and_create_price_alert, render_alert_center
from commands.startup import display_startup_info
from config import get_config
from core import (
    get_standard_logger,
    log_operation_start,
    log_operation_end,
    log_exception,
    BorsaException,
    console,
)
from core.paths import get_runtime_dir
from services.investor_profile import (
    build_investor_context,
    get_analysis_horizon_default,
    load_investor_profile,
)

logger = get_standard_logger(__name__)


class InteractiveCLI:
    """Encapsulates the interactive console workflow."""

    def __init__(self, *, console: Console) -> None:
        self.console = console
        self.config = get_config()
        self._main_menu: Dict[str, tuple[str, Callable[[], None]]] = {
            "1": ("Analyze Stock", self._handle_analyze_stock),
            "2": ("Batch Stock Analysis", self._handle_analyze_batch),
            "3": ("Pre-Research Agent", self._handle_pre_research),
            "4": ("Compare Companies", self._handle_compare_companies),
            "5": ("Query Company Financial Data", self._handle_company_financial_data),
            "6": ("Query Index / ETF Data", self._handle_index_etf_data),
            "7": ("Analysis History", self._show_history_menu),
            "8": ("Favorite Companies", self._handle_favorites),
            "9": ("Portfolio Management", self._handle_portfolio_management),
            "10": ("Settings", self._show_settings_menu),
            "0": ("Exit", lambda: None),
        }
        self._session_start = datetime.now()
        logger.info("InteractiveCLI started - Session ID: %s", id(self))

    _active_status: Optional[Status] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Start the interactive loop."""
        logger.info("═" * 60)
        logger.info("CLI SESSION STARTED")
        logger.info("═" * 60)

        # Display startup information with market status
        display_startup_info(self.console)
        self._preload_services()

        while True:
            try:
                self._show_main_menu()
                choice = Prompt.ask(
                    "Choice",
                    choices=list(self._main_menu.keys()),
                    default="1",
                    show_choices=False,
                )

                logger.info("User choice: %s", choice)

                if choice == "0":
                    session_duration = (
                        datetime.now() - self._session_start
                    ).total_seconds()
                    logger.info(
                        "CLI session ended - Duration: %.2f seconds", session_duration
                    )
                    self.console.print(
                        "[green]👋 Thank you for using Struct. See you soon![/green]"
                    )
                    return

                _, handler = self._main_menu[choice]
                self._execute(handler)
            except EOFError:
                logger.warning("EOF received - ending session")
                self.console.print("[yellow]Session ended. Goodbye![/yellow]")
                return

    # ------------------------------------------------------------------
    # Startup preloading
    # ------------------------------------------------------------------
    def _preload_services(self) -> None:
        """Pre-loads services at program startup to prevent initial usage delays."""
        try:
            with self.console.status(
                "[dim]Services being prepared...[/dim]", spinner="dots"
            ):
                from services.tools import get_tool
                from services.factories import get_rag_service

                # Warm up tool registry — all tools are initialized on first get_tool() call
                get_tool("search_web")
                rag = get_rag_service(self.config)
                if hasattr(rag, "warmup"):
                    rag.warmup()
            self.console.print("")
            logger.info("Service pre-loading completed.")
        except Exception as exc:
            logger.warning("Service pre-loading failed (non-critical): %s", exc)

    # ------------------------------------------------------------------
    # Menu helpers
    # --------
    def _show_main_menu(self) -> None:
        table = Table(title="Main Menu", show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Action", style="green")

        for key, (label, _) in self._main_menu.items():
            if key == "0":
                continue
            table.add_row(key, label)
        table.add_row("0", "Exit")

        self.console.print(table)

    def _execute(self, handler: Callable[[], None]) -> None:
        handler_name = (
            handler.__name__ if hasattr(handler, "__name__") else str(handler)
        )
        start_time = log_operation_start(logger, f"Command: {handler_name}")

        try:
            handler()
            log_operation_end(
                logger, f"Command: {handler_name}", start_time, success=True
            )
        except KeyboardInterrupt:
            logger.warning("Cancelled by user: %s", handler_name)
            self.console.print("[yellow]Operation cancelled.[/yellow]")
        except BorsaException as exc:
            # BorsaException already logged, just show to user
            log_operation_end(
                logger, f"Command: {handler_name}", start_time, success=False
            )
            self.console.print(f"[red]Error: {exc.get_user_message()}[/red]")
        except Exception as exc:  # pragma: no cover - defensive
            log_exception(logger, f"Command failed: {handler_name}", exc)
            log_operation_end(
                logger, f"Command: {handler_name}", start_time, success=False
            )
            self.console.print(f"[red]Error: {exc}[/red]")

    @contextmanager
    def _status_spinner(self, message: str) -> Iterator[Status]:
        """Context manager for status spinner with proper state management."""
        with self.console.status(f"[cyan]{message}[/cyan]", spinner="dots") as status:
            self._active_status = status
            try:
                yield status
            finally:
                self._active_status = None

    def _update_progress(self, message: str) -> None:
        """Update progress message in active status spinner or log if not active."""
        if self._active_status:
            try:
                self._active_status.update(f"[cyan]{message}[/cyan]")
            except Exception:
                # Status might have been stopped, fallback to logging
                logger.debug("Progress (status stopped): %s", message)
        else:
            # No active status, only log (don't print to avoid clutter)
            logger.debug("Progress: %s", message)

    # ------------------------------------------------------------------
    # Analyze actions
    # ------------------------------------------------------------------
    def _handle_company_financial_data(self) -> None:
        """Handle company financial data query."""
        handle_company_financial_query(self.console)
        Prompt.ask("\nPress Enter to continue")

    def _handle_index_etf_data(self) -> None:
        """Handle index/ETF financial data query."""
        handle_index_etf_financial_query(self.console)
        Prompt.ask("\nPress Enter to continue")

    def _handle_pre_research(self) -> None:
        """Handle pre-research agent flow."""
        handle_pre_research(self.console)
        Prompt.ask("\nPress Enter to continue")

    def _handle_compare_companies(self) -> None:
        """Handle company comparison flow."""
        handle_compare_companies(self.console)
        Prompt.ask("\nPress Enter to continue")

    def _handle_favorites(self) -> None:
        """Handle favorites menu."""
        while True:
            self.console.print()
            self.console.print(
                Panel.fit(
                    "[bold cyan]⭐ Favorite Companies[/bold cyan]\n\n"
                    "1. [green]List[/green]\n"
                    "2. [green]Add[/green]\n"
                    "3. [green]Remove[/green]\n"
                    "4. [green]Analyze Favorite (Single)[/green]\n"
                    "5. [green]Analyze All Favorites[/green]\n"
                    "0. [yellow]Go Back[/yellow]",
                    title="Favorites Menu",
                    border_style="cyan",
                )
            )

            choice = Prompt.ask(
                "Choice",
                choices=["1", "2", "3", "4", "5", "0"],
                default="1",
                show_choices=False,
            )

            if choice == "0":
                break
            elif choice == "1":
                list_favorites(self.console)
                Prompt.ask("\nPress Enter to continue")
            elif choice == "2":
                symbol = Prompt.ask("Stock Symbol to Add (e.g., AAPL)").upper()
                if symbol:
                    if add_favorite(symbol):
                        self.console.print(
                            f"[green]✅ {symbol} added to favorites.[/green]"
                        )
                    else:
                        self.console.print(
                            f"[yellow]⚠️ {symbol} is already in favorites.[/yellow]"
                        )
            elif choice == "3":
                favorites = list_favorites(self.console, show_numbers=True)
                if not favorites:
                    continue

                fav_choice = Prompt.ask(
                    "Enter the number or symbol of the stock to remove", default=""
                )
                if not fav_choice:
                    continue

                symbol_to_remove = None
                if fav_choice.isdigit():
                    idx = int(fav_choice) - 1
                    if 0 <= idx < len(favorites):
                        symbol_to_remove = favorites[idx]
                else:
                    symbol_to_remove = fav_choice.upper()

                if symbol_to_remove:
                    if remove_favorite(symbol_to_remove):
                        self.console.print(
                            f"[green]🗑️ {symbol_to_remove} removed from favorites.[/green]"
                        )
                    else:
                        self.console.print(
                            f"[red]❌ {symbol_to_remove} not found in favorites.[/red]"
                        )

            elif choice == "4":
                favorites = list_favorites(self.console, show_numbers=True)
                if not favorites:
                    continue

                fav_choice = Prompt.ask(
                    "Enter the number of the stock to analyze", default=""
                )
                if fav_choice.isdigit():
                    idx = int(fav_choice) - 1
                    if 0 <= idx < len(favorites):
                        symbol = favorites[idx]
                        self._run_single_stock_analysis(symbol)
                    else:
                        self.console.print("[red]Invalid number.[/red]")

            elif choice == "5":
                favorites = load_favorites()
                if not favorites:
                    self.console.print(
                        "[yellow]⚠️ Your favorites list is empty.[/yellow]"
                    )
                    continue

                self.console.print(
                    f"[green]🚀 Starting analysis for {len(favorites)} stocks...[/green]"
                )

                # Analysis options
                investment_horizon = "long-term"
                if Confirm.ask(
                    "Would you like to customize analysis settings?", default=False
                ):
                    investment_horizon = self._prompt_investment_horizon()

                try:
                    results = analyze_multiple_stocks(
                        favorites,
                        max_workers=self.config.get("max_workers", 3),
                        config=self.config,
                        console=self.console,
                        investment_horizon=investment_horizon,
                    )
                    display_batch_results(results, self.console)
                except Exception as e:
                    logger.error("Batch analysis error: %s", e)
                    self.console.print(
                        f"[red]Error occurred during analysis: {e}[/red]"
                    )

    def _handle_portfolio_management(self) -> None:
        """Handle portfolio management menu (CLI only)."""
        while True:
            self.console.print()
            self.console.print(
                Panel.fit(
                    "[bold cyan]📊 Portfolio Management[/bold cyan]\n\n"
                    "1. [green]List Positions[/green]\n"
                    "2. [green]Add Position[/green]\n"
                    "3. [green]Remove Position[/green]\n"
                    "4. [green]Portfolio Summary (P/L)[/green]\n"
                    "5. [green]Risk Cockpit[/green]\n"
                    "6. [green]Alert Center[/green]\n"
                    "7. [green]Create Price Alert[/green]\n"
                    "0. [yellow]Go Back[/yellow]",
                    title="Portfolio Menu",
                    border_style="cyan",
                )
            )

            choice = Prompt.ask(
                "Choice",
                choices=["1", "2", "3", "4", "5", "6", "7", "0"],
                default="1",
                show_choices=False,
            )

            if choice == "0":
                break
            if choice == "1":
                list_positions(self.console)
                Prompt.ask("\nPress Enter to continue")
                continue

            if choice == "2":
                symbol = Prompt.ask("Stock Symbol (e.g., AAPL, THYAO.IS)", default="").upper().strip()
                if not symbol:
                    self.console.print("[red]❌ Symbol is required.[/red]")
                    continue
                try:
                    quantity = float(Prompt.ask("Quantity", default="1"))
                    average_cost = float(Prompt.ask("Average Cost", default="1"))
                except ValueError:
                    self.console.print("[red]❌ Quantity and cost must be numeric.[/red]")
                    continue

                ok, message = add_position_with_feedback(
                    symbol,
                    quantity,
                    average_cost,
                    config=self.config,
                )
                if ok:
                    self.console.print(f"[green]✅ Position saved: {symbol}[/green]")
                else:
                    self.console.print(f"[red]❌ {message}[/red]")
                continue

            if choice == "3":
                symbol = Prompt.ask("Symbol to remove", default="").upper().strip()
                if not symbol:
                    continue
                if remove_position(symbol):
                    self.console.print(f"[green]🗑️ Removed: {symbol}[/green]")
                else:
                    self.console.print(f"[red]❌ Position not found: {symbol}[/red]")
                continue

            if choice == "4":
                render_portfolio_snapshot(self.console, self.config)
                Prompt.ask("\nPress Enter to continue")
                continue

            if choice == "5":
                render_portfolio_risk_cockpit(self.console, self.config)
                Prompt.ask("\nPress Enter to continue")
                continue

            if choice == "6":
                render_alert_center(self.console)
                Prompt.ask("\nPress Enter to continue")
                continue

            if choice == "7":
                prompt_and_create_price_alert(self.console)
                Prompt.ask("\nPress Enter to continue")

    def _handle_edit_config(self) -> None:
        """Open settings.py in default editor."""
        from core.paths import get_app_root
        config_path = get_app_root() / "config" / "settings.py"
        if not config_path.exists():
            self.console.print("[red]❌ config/settings.py not found![/red]")
            return

        self.console.print(
            f"[green]📝 Opening configuration file: {config_path}[/green]"
        )
        self._open_file(config_path)

    def _handle_edit_proxies(self) -> None:
        """Open proxies.txt in default editor."""
        from core.paths import get_app_root
        proxies_path = get_app_root() / "proxies.txt"
        if not proxies_path.exists():
            self.console.print("[yellow]⚠️ proxies.txt not found, creating...[/yellow]")
            proxies_path.touch()

        self.console.print(
            f"[green]📝 Opening proxy list: {proxies_path.absolute()}[/green]"
        )
        self._open_file(proxies_path)

    def _open_file(self, path: Path) -> None:
        try:
            if sys.platform == "win32":
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.call(("open", path))
            else:
                subprocess.call(("xdg-open", path))
        except Exception as e:
            logger.error("File opening error: %s", e)
            self.console.print(f"[red]❌ Could not open file: {e}[/red]")

    def _handle_analyze_stock(self) -> None:
        symbol = self._prompt_text("Enter stock symbol", allow_cancel=True)
        if not symbol:
            logger.debug("Stock analysis cancelled - no symbol provided")
            return

        symbol = symbol.upper()
        self._run_single_stock_analysis(symbol)

    def _run_single_stock_analysis(self, symbol: str) -> None:
        logger.info("Single stock analysis starting: %s", symbol)

        # Ask user for investment horizon
        investment_horizon = self._prompt_investment_horizon(
            default_horizon=get_analysis_horizon_default()
        )

        # Optional: Ask for additional user context
        user_context: Optional[str] = None
        profile = load_investor_profile()
        self.console.print(
            f"[dim]👤 Active investor profile: {profile.get('profile_name')} | playbook={profile.get('active_playbook')}[/dim]"
        )
        if Confirm.ask(
            "[cyan]💡 Would you like to add custom context for a more personalized analysis?[/cyan]\n"
            "   [dim](e.g., budget, experience level, investment goals)[/dim]",
            default=False,
        ):
            user_context = self._prompt_text(
                "Enter your context (e.g., 'Beginner with $10k budget', 'Looking for dividend income')",
                allow_empty=True,
            )
            if user_context:
                logger.info("User context added: %s", user_context[:100])

        user_context = build_investor_context(user_context)

        # News scraping is now mandatory - inform user
        self.console.print(
            "\n[dim]📰 Comprehensive news analysis enabled (Google News with YFinance + Full content scraping)[/dim]"
        )

        logger.info(
            "Analysis parameters - Symbol: %s, Horizon: %s, Google News: True (mandatory)",
            symbol,
            investment_horizon,
        )

        analysis_start = datetime.now()
        with self._status_spinner(
            f"🔍 Performing comprehensive analysis for {symbol} - This may take some time..."
        ) as status:
            result = analyze_single_stock(
                symbol=symbol,
                config=self.config,
                progress_callback=self._update_progress,
                investment_horizon=investment_horizon,
                user_context=user_context,
                console=self.console,
                status_ctx=status,
            )

        analysis_duration = (datetime.now() - analysis_start).total_seconds()

        if not result:
            logger.warning(
                "Analysis failed: %s - Duration: %.2f seconds",
                symbol,
                analysis_duration,
            )
            self.console.print(
                f"[red]❌ {symbol} could not be analyzed. Please check the symbol and try again.[/red]"
            )
            return

        logger.info(
            "Analysis completed: %s - Duration: %.2f seconds - News count: %d",
            symbol,
            analysis_duration,
            result.get("news_count", 0),
        )

        readable_default = (
            self.config.get("ai", {})
            .get("reporting", {})
            .get("readable_mode_default", True)
        )
        readable_mode = Confirm.ask(
            "🗣️ Show easy-to-read report summary?",
            default=bool(readable_default),
        )
        display_analysis_result(
            result,
            console=self.console,
            readable_mode=readable_mode,
        )

        if Confirm.ask(
            "💾 Would you like to save this analysis for future reference?",
            default=False,
        ):
            entry_id = add_to_history(
                symbol=symbol,
                analysis_type="stock",
                data=result,
                config=self.config,
            )
            logger.info("Analysis saved to history: %s - ID: %s", symbol, entry_id)
            self.console.print(
                f"[green]✅ Analysis successfully saved! (ID: {entry_id})[/green]"
            )
            history_dir = get_runtime_dir() / "instance" / "history"
            self.console.print(
                f"[dim]📁 Location: {history_dir} (in JSON format)[/dim]"
            )

    def _handle_analyze_batch(self) -> None:
        manual_input = self._prompt_text(
            "Enter comma-separated stock symbols (e.g., AAPL, MSFT, GOOGL)",
            allow_cancel=True,
        )
        symbols: List[str] = []
        if manual_input:
            symbols.extend(self._parse_symbol_list(manual_input))

        if Confirm.ask(
            "📄 Would you like to load additional symbols from a file?", default=False
        ):
            file_path_text = self._prompt_text("Enter file path", allow_cancel=True)
            if file_path_text:
                file_path = Path(file_path_text).expanduser()
                if not file_path.exists():
                    logger.warning("File not found: %s", file_path)
                    self.console.print(f"[red]❌ File not found: {file_path}[/red]")
                    return
                file_symbols = [
                    line.strip().upper()
                    for line in file_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                logger.info(
                    "Loaded %d symbols from file: %s", len(file_symbols), file_path
                )
                symbols.extend(file_symbols)

        if not symbols:
            logger.debug("Batch analysis cancelled - no symbols provided")
            self.console.print(
                "[yellow]⚠️ No symbols specified. Returning to main menu.[/yellow]"
            )
            return

        symbols = self._deduplicate(symbols)
        logger.info(
            "Batch analysis starting: %d symbols - %s",
            len(symbols),
            ", ".join(symbols[:10]),
        )

        # Batch analysis options
        investment_horizon = None
        user_context = None
        profile = load_investor_profile()
        self.console.print(
            f"[dim]👤 Active investor profile: {profile.get('profile_name')} | playbook={profile.get('active_playbook')}[/dim]"
        )

        if Confirm.ask(
            "🎯 Would you like to specify investment horizon and context for all stocks?",
            default=False,
        ):
            # Horizon
            investment_horizon = self._prompt_investment_horizon(
                default_horizon=get_analysis_horizon_default()
            )

            # Context
            user_context = self._prompt_text(
                "Custom context (leave empty if not wanted)",
                allow_empty=True,
            )

        user_context = build_investor_context(user_context)

        # Batch analysis: Full analysis for each symbol runs in parallel

        # Ask for concurrency
        max_workers_str = Prompt.ask(
            "[cyan]Number of concurrent operations (Thread count)[/cyan]",
            default=str(len(symbols)),
            show_default=True,
        )
        try:
            max_workers = int(max_workers_str)
            if max_workers <= 0:
                max_workers = 1
        except ValueError:
            max_workers = len(symbols)

        # Ask for wait time
        wait_time_str = Prompt.ask(
            "[cyan]Wait time between operations (seconds)[/cyan]",
            default="0",
            show_default=True,
        )
        try:
            wait_time = float(wait_time_str)
            if wait_time < 0:
                wait_time = 0.0
        except ValueError:
            wait_time = 0.0

        # Ask for save destination upfront
        save_to_history_flag = Confirm.ask(
            "💾 Save analysis results to history?", default=True
        )

        logger.debug(
            "Batch analysis - worker count: %d, wait: %.1fs", max_workers, wait_time
        )

        batch_start = datetime.now()
        # quiet=False: Show all analysis steps and logs (like single stock analysis)
        results = analyze_multiple_stocks(
            symbols=symbols,
            max_workers=max_workers,
            config=self.config,
            console=self.console,
            quiet=False,  # Show all steps
            investment_horizon=investment_horizon,
            user_context=user_context,
            wait_time=wait_time,
        )
        batch_duration = (datetime.now() - batch_start).total_seconds()

        display_batch_results(results, console=self.console)

        total = len(results)
        completed = sum(1 for value in results.values() if value)
        failed = total - completed

        logger.info(
            "Batch analysis completed - Success: %d, Failed: %d, Total: %d, Duration: %.2f seconds",
            completed,
            failed,
            total,
            batch_duration,
        )

        self.console.print(
            f"\n[cyan]📊 Batch Analysis Summary: [green]{completed} successful[/green] | [red]{failed} failed[/red] | {total} total stocks[/cyan]"
        )

        # Save results if requested
        if save_to_history_flag and completed > 0:
            saved_count = 0
            for symbol, result in results.items():
                if result:
                    try:
                        add_to_history(
                            symbol=symbol,
                            analysis_type="stock",
                            data=result,
                            config=self.config,
                        )
                        saved_count += 1
                    except Exception as e:
                        logger.error("Save to history error (%s): %s", symbol, e)

            if saved_count > 0:
                self.console.print(
                    f"[green]✅ {saved_count} analyses successfully saved to history![/green]"
                )
                history_dir = get_runtime_dir() / "instance" / "history"
                self.console.print(
                    f"[dim]📁 Location: {history_dir} (in JSON format)[/dim]"
                )

        if completed > 0:
            self.console.print(
                "[dim]💡 Tip: You can use single analysis for a detailed view.[/dim]"
            )

    # ------------------------------------------------------------------
    # History menu
    # ------------------------------------------------------------------
    def _show_history_menu(self) -> None:
        options: Dict[str, tuple[str, Callable[[], None]]] = {
            "1": ("List history", self._handle_history_list),
            "2": ("Show a record", self._handle_history_show),
            "0": ("Go Back", lambda: None),
        }

        while True:
            table = Table(
                title="History", show_header=True, header_style="bold magenta"
            )
            table.add_column("#", style="cyan", justify="center")
            table.add_column("Action", style="green")
            for key, (label, _) in options.items():
                table.add_row(key, label)
            self.console.print(table)

            choice = Prompt.ask(
                "Choice",
                choices=list(options.keys()),
                default="0",
                show_choices=False,
            )

            if choice == "0":
                return

            _, handler = options[choice]
            self._execute(handler)

    def _handle_history_list(self) -> None:
        limit = self._prompt_int("How many records to show?", default=10, minimum=1)
        symbol = self._prompt_text(
            "🔍 Filter by symbol (Press Enter for all)", allow_empty=True
        )
        symbol = symbol or None

        entries = list_history(limit=limit, symbol=symbol, config=self.config)
        if not entries:
            self.console.print(
                "[yellow]📚 Your analysis history is empty. Analyze some stocks first![/yellow]"
            )
            return

        table = Table(
            title="📚 Your Analysis History",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("ID", style="cyan")
        table.add_column("Symbol", style="green")
        table.add_column("Type", style="yellow")
        table.add_column("Timestamp", style="dim")

        for entry in entries:
            table.add_row(
                str(entry["id"]),
                entry["symbol"],
                entry["type"],
                entry["timestamp"],
            )

        self.console.print(table)

    def _handle_history_show(self) -> None:
        analysis_id = self._prompt_int(
            "Enter the ID of the record you want to view", minimum=1, allow_cancel=True
        )
        if analysis_id is None:
            return

        entry = show_history_entry(analysis_id=int(analysis_id), config=self.config)
        if not entry:
            self.console.print(
                f"[yellow]⚠️ Record with ID {analysis_id} not found. Please check the ID.[/yellow]"
            )
            return

        display_history_entry(entry, console=self.console)

    # ------------------------------------------------------------------
    # Config menu
    # ------------------------------------------------------------------
    def _show_config_menu(self) -> None:
        options: Dict[str, tuple[str, Callable[[], None]]] = {
            "1": ("Show configuration", self._handle_config_show),
            "2": ("Validate configuration", self._handle_config_validate),
            "0": ("Go Back", lambda: None),
        }

        while True:
            table = Table(
                title="Configuration", show_header=True, header_style="bold magenta"
            )
            table.add_column("#", style="cyan", justify="center")
            table.add_column("Action", style="green")
            for key, (label, _) in options.items():
                table.add_row(key, label)
            self.console.print(table)

            choice = Prompt.ask(
                "Choice",
                choices=list(options.keys()),
                default="0",
                show_choices=False,
            )

            if choice == "0":
                return

            _, handler = options[choice]
            self._execute(handler)

    def _handle_config_show(self) -> None:
        show_config(config=self.config, console=self.console)

    def _handle_config_validate(self) -> None:
        is_valid = validate_config(config=self.config, console=self.console)
        if is_valid:
            self.console.print(
                "[green]✅ Configuration is valid and ready to use![/green]"
            )
        else:
            self.console.print(
                "[red]⚠️ Configuration issues detected. Please review the warnings above.[/red]"
            )

    # ------------------------------------------------------------------
    # Settings menu
    # ------------------------------------------------------------------
    def _show_settings_menu(self) -> None:
        options: Dict[str, tuple[str, Callable[[], None]]] = {
            "1": ("Configuration", self._show_config_menu),
            "2": ("Edit configuration", self._handle_edit_config),
            "3": ("Edit proxy list", self._handle_edit_proxies),
            "4": ("Investor profile", self._handle_investor_profile_menu),
            "0": ("Go Back", lambda: None),
        }

        while True:
            table = Table(
                title="Settings", show_header=True, header_style="bold magenta"
            )
            table.add_column("#", style="cyan", justify="center")
            table.add_column("Action", style="green")
            for key, (label, _) in options.items():
                table.add_row(key, label)
            self.console.print(table)

            choice = Prompt.ask(
                "Choice",
                choices=list(options.keys()),
                default="0",
                show_choices=False,
            )

            if choice == "0":
                return

            _, handler = options[choice]
            self._execute(handler)

    def _handle_investor_profile_menu(self) -> None:
        while True:
            table = Table(
                title="Investor Profile",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("#", style="cyan", justify="center")
            table.add_column("Action", style="green")
            table.add_row("1", "Show active profile")
            table.add_row("2", "Edit active profile")
            table.add_row("0", "Go Back")
            self.console.print(table)

            choice = Prompt.ask(
                "Choice",
                choices=["1", "2", "0"],
                default="0",
                show_choices=False,
            )

            if choice == "0":
                return
            if choice == "1":
                show_investor_profile(self.console)
                Prompt.ask("\nPress Enter to continue")
            elif choice == "2":
                edit_investor_profile(self.console)
                Prompt.ask("\nPress Enter to continue")

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------
    def _prompt_investment_horizon(self, default_horizon: str = "long-term") -> str:
        """Prompts the user to select an investment horizon."""
        options = {
            "1": ("📅 Short-Term (days/weeks) - For active traders", "short-term"),
            "2": ("📆 Medium-Term (months) - For swing traders", "medium-term"),
            "3": ("📈 Long-Term (years) - For buy-and-hold investors", "long-term"),
        }
        default_choice_map = {
            "short-term": "1",
            "medium-term": "2",
            "long-term": "3",
        }

        table = Table(
            title="🎯 Select Your Investment Horizon",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("#", style="cyan", justify="center")
        table.add_column("Investment Style", style="green")
        for key, (label, _) in options.items():
            table.add_row(key, label)
        self.console.print(table)

        choice = Prompt.ask(
            "[cyan]Select your investment horizon[/cyan]",
            choices=list(options.keys()),
            default=default_choice_map.get(default_horizon, "3"),
            show_choices=False,
        )
        _, horizon = options[choice]
        logger.info("Investment horizon selected: %s", horizon)
        return horizon

    def _prompt_text(
        self, message: str, *, allow_cancel: bool = False, allow_empty: bool = False
    ) -> Optional[str]:
        cancel_hint = " (type 'q' to cancel)" if allow_cancel else ""
        while True:
            value = self.console.input(
                f"[bold cyan]{message}{cancel_hint}[/bold cyan]: "
            ).strip()
            if allow_cancel and value.lower() in {"q", "quit", "exit"}:
                return None
            if value:
                return value
            if allow_empty:
                return ""
            if allow_cancel:
                return None
            self.console.print("[yellow]Value is required. Please try again.[/yellow]")

    def _prompt_int(
        self,
        message: str,
        *,
        default: Optional[int] = None,
        minimum: Optional[int] = None,
        allow_cancel: bool = False,
    ) -> Optional[int]:
        while True:
            cancel_hint = " (type 'q' to cancel)" if allow_cancel else ""
            prompt_message = f"[bold cyan]{message}{cancel_hint}[/bold cyan]"
            if default is not None:
                prompt_message += f" [{default}]"
            prompt_message += ": "

            raw = self.console.input(prompt_message).strip()
            if allow_cancel and raw.lower() in {"q", "quit", "exit"}:
                return None
            if not raw:
                if default is not None:
                    return default
                return None

            try:
                value = int(raw)
            except ValueError:
                self.console.print("[red]Please enter a valid number.[/red]")
                continue

            if minimum is not None and value < minimum:
                self.console.print(f"[yellow]Value must be ≥ {minimum}.[/yellow]")
                continue

            return value

    @staticmethod
    def _parse_symbol_list(raw: str) -> List[str]:
        return [token.strip().upper() for token in raw.split(",") if token.strip()]

    @staticmethod
    def _deduplicate(symbols: List[str]) -> List[str]:
        """Cleans duplicate symbols while preserving order."""
        return list(dict.fromkeys(symbols))


def main() -> int:
    """Application entry point."""
    logger.info("=" * 60)
    logger.info("STARTING STRUCT CLI APPLICATION")
    logger.info("=" * 60)

    try:
        InteractiveCLI(console=console).run()
        logger.info("Application exited normally")
        return 0
    except (KeyboardInterrupt, EOFError):
        logger.warning("Interrupt signal received from user (Ctrl+C)")
        console.print("\n[yellow]Operation stopped by user.[/yellow]")
        return 0
    except BorsaException as exc:
        # BorsaException already logged
        logger.error("Application exited with BorsaException: %s", exc.error_id)
        console.print(f"[red]Unexpected error: {exc.get_user_message()}[/red]")
        return 1
    except Exception as exc:  # pragma: no cover - defensive
        log_exception(logger, "Application crashed with unexpected error", exc)
        console.print(f"[red]Unexpected error: {exc}[/red]")
        return 1
    finally:
        logger.info("=" * 60)
        logger.info("STRUCT CLI APPLICATION ENDED")
        logger.info("=" * 60)


if __name__ == "__main__":
    sys.exit(main())
