"""Company comparison command."""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from config import get_config
from core import get_standard_logger
from core.paths import get_runtime_dir
from domain.data_processors import process_and_enrich_company_data, sanitize_raw_company_data
from services.ai.providers import AIService
from services.factories import get_financial_service
from services.utils.calculations import make_json_serializable, validate_symbol

logger = get_standard_logger(__name__)


def _parse_symbols(raw: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"[\s,]+", raw) if p.strip()]
    seen = set()
    symbols: List[str] = []
    for part in parts:
        symbol = part.upper()
        if symbol in seen:
            continue
        symbols.append(symbol)
        seen.add(symbol)
    return symbols


def _prompt_symbols(console: Console) -> Optional[List[str]]:
    raw = Prompt.ask(
        "Stocks to compare (separate with commas, e.g., AAPL, MSFT)",
        default="",
    ).strip()
    if not raw:
        console.print("[red]❌ Stock list cannot be empty.[/red]")
        return None

    symbols = _parse_symbols(raw)
    if len(symbols) < 2:
        console.print("[red]❌ You must enter at least 2 stocks.[/red]")
        return None

    invalid = [symbol for symbol in symbols if not validate_symbol(symbol)]
    if invalid:
        console.print(f"[red]❌ Invalid symbols: {', '.join(invalid)}[/red]")
        return None

    return symbols


def _prompt_criteria() -> Optional[str]:
    if Confirm.ask("Would you like to add additional criteria?", default=False):
        criteria = Prompt.ask(
            "Enter criteria (e.g., dividend, growth, risk, sector fit)",
            default="",
        ).strip()
        return criteria or None
    return None


def _extract_metric_snapshot(metrics: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "current_price",
        "currency",
        "market_cap",
        "pe_ratio",
        "forward_pe",
        "price_to_book",
        "price_to_sales",
        "enterprise_value",
        "revenue_growth",
        "earnings_growth",
        "profit_margin",
        "roe",
        "roa",
        "beta",
        "debt_to_equity",
        "current_ratio",
        "dividend_yield",
        "payout_ratio",
        "free_cash_flow",
        "fifty_two_week_high",
        "fifty_two_week_low",
        "price_change_percent",
        "dcf",
        "dcf_diff",
        "rsi",
    ]
    snapshot = {}
    for key in keys:
        value = metrics.get(key)
        if value is None or value == "N/A":
            continue
        snapshot[key] = value
    return snapshot


def _build_company_snapshot(processed: Dict[str, Any]) -> Dict[str, Any]:
    metrics = processed.get("financial_metrics") or {}
    return {
        "symbol": processed.get("symbol"),
        "company_name": processed.get("company_name"),
        "sector": processed.get("sector"),
        "industry": processed.get("industry"),
        "overall_data_quality": processed.get("overall_data_quality"),
        "comparative_analysis": processed.get("comparative_analysis"),
        "macro_context": processed.get("macro_context"),
        "financial_metrics": _extract_metric_snapshot(metrics),
    }


def compare_companies(
    symbols: List[str],
    criteria: Optional[str] = None,
    depth_mode: str = "standard",
    console: Optional[Console] = None,
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare companies and return a reusable structured payload."""
    unique_symbols: List[str] = []
    seen = set()
    for value in symbols or []:
        symbol = str(value).strip().upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        unique_symbols.append(symbol)

    if len(unique_symbols) < 2:
        return {
            "error": "At least 2 valid stocks are required for comparison.",
            "symbols": unique_symbols,
            "missing_symbols": [],
        }

    financial_service = get_financial_service(get_config())
    raw_companies: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []

    bulk_fetch = getattr(financial_service, "get_company_data_bulk", None)
    fetched_map = bulk_fetch(unique_symbols) if callable(bulk_fetch) else {}

    for symbol in unique_symbols:
        data = fetched_map.get(symbol)
        if not data:
            missing.append(symbol)
            continue
        raw_companies[symbol] = sanitize_raw_company_data(data)

    if len(raw_companies) < 2:
        return {
            "error": "At least 2 stocks with available data are required for comparison.",
            "symbols": list(raw_companies.keys()),
            "missing_symbols": missing,
        }

    snapshots: List[Dict[str, Any]] = []
    for symbol, data in raw_companies.items():
        try:
            processed = process_and_enrich_company_data(data, raw_companies)
            processed_dict = asdict(processed) if is_dataclass(processed) else processed
            snapshots.append(_build_company_snapshot(processed_dict))
        except Exception as exc:
            logger.warning("Processing error (%s): %s", symbol, exc)

    snapshots = make_json_serializable(snapshots)

    ai_service = AIService(get_config())
    result = ai_service.compare_companies(
        tickers=list(raw_companies.keys()),
        criteria=criteria,
        initial_data=snapshots,
        console=console,
        depth_mode=depth_mode,
        session_id=session_id,
    )

    if not isinstance(result, dict):
        return {
            "error": "Comparison failed: invalid AI response.",
            "symbols": list(raw_companies.keys()),
            "missing_symbols": missing,
        }

    analysis_text = result.get("analysis")
    if analysis_text in ("</tool_call>", "<tool_call>"):
        analysis_text = ""

    payload: Dict[str, Any] = {
        "symbols": list(raw_companies.keys()),
        "missing_symbols": missing,
        "analysis": analysis_text or "",
        "raw_result": result,
        "session_id": result.get("session_id") if isinstance(result, dict) else None,
        "session_path": result.get("session_path") if isinstance(result, dict) else None,
        "resumed_from_snapshot": result.get("resumed_from_snapshot") if isinstance(result, dict) else None,
    }
    if result.get("error"):
        payload["error"] = str(result.get("error"))
    return payload


def handle_compare_companies(console: Console) -> None:
    symbols = _prompt_symbols(console)
    if not symbols:
        return

    criteria = _prompt_criteria()

    console.print("\n[dim]⚖️ Starting company comparison, please wait...[/dim]")

    result = compare_companies(
        symbols=symbols,
        criteria=criteria,
        depth_mode="standard",
        console=console,
    )

    if result.get("missing_symbols"):
        console.print(
            f"[yellow]⚠️ Symbols with no data: {', '.join(result['missing_symbols'])}[/yellow]"
        )

    if result.get("error"):
        error_msg = str(result.get("error"))
        logger.error("Comparison failed: %s", error_msg)
        console.print(f"[red]❌ Comparison failed: {error_msg}[/red]")
        return

    analysis_text = str(result.get("analysis") or "")

    if analysis_text:
        console.print(
            Panel(
                Markdown(analysis_text),
                title="⚖️ Company Comparison Report",
                border_style="cyan",
            )
        )

    session_id = str(result.get("session_id") or "").strip()
    session_path = str(result.get("session_path") or "").strip()
    if session_id:
        console.print(f"[dim]Session: {session_id}[/dim]")
    if session_path:
        console.print(f"[dim]Session file: {session_path}[/dim]")

        try:
            reports_dir = get_runtime_dir() / "instance" / "reports" / "comparisons"
            reports_dir.mkdir(parents=True, exist_ok=True)

            safe_symbols = re.sub(
                r"[^a-zA-Z0-9_\-\.]", "_", "-".join(result.get("symbols", []))
            )
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{safe_symbols}_comparison.md"
            filepath = reports_dir / filename

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(
                    f"# Company Comparison Report: {', '.join(result.get('symbols', []))}\n"
                )
                f.write(f"**Date:** {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")
                if criteria:
                    f.write(f"**Criteria:** {criteria}\n")
                f.write("\n---\n\n")
                f.write(analysis_text)

            console.print(
                f"\n[green]✅ Report saved:[/green] [link={filepath}]{filepath}[/link]"
            )
        except Exception as exc:
            logger.error("Report could not be saved: %s", exc)
            console.print(f"[red]❌ Report could not be saved: {exc}[/red]")
    else:
        console.print("[yellow]⚠️ Comparison result returned empty.[/yellow]")
