"""Output formatting utilities"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import re
from typing import Dict, Any, Optional

from config import get_config
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from core.console import console as core_console
from domain.utils import get_currency_symbol

# Module-level cached config slices (avoids deepcopy on every display call).
_output_cfg = get_config()
_output_reporting_cfg: dict = _output_cfg.get("ai", {}).get("reporting", {})
_output_ai_cfg: dict = _output_cfg.get("ai", {})
del _output_cfg

from domain.utils import safe_int_strict as _safe_int


def display_analysis_result(
    result: Dict[str, Any],
    console: Optional[Console] = None,
    readable_mode: Optional[bool] = None,
) -> None:
    """
    Display analysis result in table format with comprehensive results section

    Args:
        result: Analysis result dictionary
        console: Rich console for output
    """
    if console is None:
        console = core_console

    reporting_cfg = _output_reporting_cfg
    if readable_mode is None:
        readable_mode = bool(reporting_cfg.get("readable_mode_default", True))

    # Rich table output
    symbol = result.get("symbol", "N/A")
    company_data = result.get("company_data", {})
    ai_analysis = result.get("ai_analysis")
    if not isinstance(ai_analysis, dict):
        ai_analysis = (result.get("workflow") or {}).get("decision_summary", {})
    if not isinstance(ai_analysis, dict):
        ai_analysis = {}

    # Company info
    profile = company_data.get("company_profile", {})
    investment_horizon = result.get("investment_horizon")
    horizon_display = ""
    if investment_horizon:
        horizon_labels = {
            "short-term": "📅 Short-Term (days/weeks)",
            "medium-term": "📅 Medium-Term (months)",
            "long-term": "📅 Long-Term (years)",
        }
        horizon_display = (
            f" | {horizon_labels.get(investment_horizon, investment_horizon)}"
        )

    console.print(
        f"\n[bold cyan]📊 {profile.get('name', symbol)} ({symbol}){horizon_display}[/bold cyan]\n"
    )

    if profile.get("description"):
        console.print(
            Panel(
                (
                    profile["description"][:200] + "..."
                    if len(profile.get("description", "")) > 200
                    else profile.get("description", "")
                ),
                title="Company Summary",
                border_style="cyan",
            )
        )
        console.print()

    _display_why_this_result_card(result, console)

    # Display comprehensive AI results section
    _display_comprehensive_results(result, console, readable_mode=bool(readable_mode))


def _extract_plain_text(raw: str) -> str:
    if not raw:
        return ""
    cleaned = re.sub(r"`{1,3}.*?`{1,3}", " ", str(raw), flags=re.DOTALL)
    cleaned = re.sub(r"[#>*_\-]", " ", cleaned)
    cleaned = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _take_sentences(text: str, max_count: int = 3) -> list[str]:
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    picked = []
    for sentence in parts:
        s = sentence.strip()
        if len(s) < 20:
            continue
        picked.append(s)
        if len(picked) >= max_count:
            break
    return picked


def _display_why_this_result_card(
    result: Dict[str, Any], console: Optional[Console] = None
) -> None:
    if console is None:
        console = core_console

    ai_cfg = _output_ai_cfg
    reporting_cfg = ai_cfg.get("reporting", {}) if isinstance(ai_cfg, dict) else {}
    max_points = max(1, _safe_int(reporting_cfg.get("max_reason_points", 3), 3))

    workflow = result.get("workflow", {}) if isinstance(result, dict) else {}
    decision = result.get("ai_analysis", {}) if isinstance(result, dict) else {}
    if not isinstance(decision, dict):
        decision = {}

    recommendation = str(decision.get("decision") or "Not specified")
    raw_reasoning = (
        decision.get("reasoning")
        or decision.get("analysis")
        or decision.get("content")
        or ""
    )
    reasoning_text = _extract_plain_text(str(raw_reasoning))
    points = _take_sentences(reasoning_text, max_count=max_points)

    news_analysis = workflow.get("news_analysis") if isinstance(workflow, dict) else None
    if isinstance(news_analysis, dict):
        themes = news_analysis.get("key_themes")
        if isinstance(themes, list):
            for theme in themes:
                text = str(theme).strip()
                if text:
                    points.append(text)
                if len(points) >= max_points:
                    break

    if not points:
        points = [
            "Market data, financial indicators, and the latest news flow were evaluated together.",
            "The result is based on common signals from multiple sources rather than a single signal.",
        ]

    bullet_text = "\n".join([f"• {p}" for p in points[:max_points]])
    console.print(
        Panel(
            f"[bold]Recommendation:[/bold] {recommendation}\n\n[bold]Why this result?[/bold]\n{bullet_text}",
            title="🧭 Why This Result?",
            border_style="blue",
        )
    )
    console.print()


def _display_comprehensive_results(
    result: Dict[str, Any],
    console: Optional[Console] = None,
    readable_mode: bool = False,
) -> None:
    """
    Display comprehensive AI results section with all outputs in full.

    Shows:
    - News Research (News Analysis)
    - Google Research (Phase 2 Web Analysis)
    - Final Conclusion (Investment Decision)

    Args:
        result: Analysis result dictionary
        console: Rich console for output
    """
    if console is None:
        console = core_console

    # Extract workflow data
    workflow_data = result.get("workflow", {})
    if not isinstance(workflow_data, dict):
        workflow_data = {}

    # Title for results section
    console.print(
        "\n[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]"
    )
    console.print("[bold cyan]📋 DETAILED AI ANALYSIS[/bold cyan]")
    console.print(
        "[dim]Below is a detailed breakdown of the AI's findings and how it reached these results.[/dim]"
    )
    console.print(
        "[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]\n"
    )

    # Display AI search keywords if available
    ai_search_keywords = result.get("ai_search_keywords")
    if (
        ai_search_keywords
        and isinstance(ai_search_keywords, list)
        and len(ai_search_keywords) > 0
    ):
        keyword_display = []
        for kw in ai_search_keywords:
            if isinstance(kw, dict):
                term = kw.get("term", "")
                lang = kw.get("lang", "en")
                if term:
                    keyword_display.append(f"[cyan]{term}[/cyan] [dim]({lang})[/dim]")
            elif isinstance(kw, str) and kw.strip():
                keyword_display.append(f"[cyan]{kw}[/cyan]")

        if keyword_display:
            console.print("[bold blue]🔍 AI SEARCH KEYWORDS[/bold blue]")
            console.print("[dim]   Terms used for Google News search[/dim]\n")
            console.print(f"   {' • '.join(keyword_display)}\n")

    # 1. NEWS RESEARCH (News Analysis)
    console.print("[bold magenta]1️⃣  MARKET NEWS AND SENTIMENT[/bold magenta]")
    console.print("[dim]   What news sources say about this stock[/dim]\n")
    news_analysis = workflow_data.get("news_analysis")
    if isinstance(news_analysis, dict):
        # Check for structured detailed analysis first
        detailed_analysis = news_analysis.get("detailed_analysis")
        executive_summary = news_analysis.get("executive_summary")
        news_text = news_analysis.get("analysis", "")

        # Use detailed analysis if available, otherwise fall back to raw analysis text
        display_text = ""
        if detailed_analysis:
            parts = []
            if executive_summary:
                parts.append(f"**Executive Summary**\n\n{executive_summary}")
            parts.append(f"**Detailed Analysis**\n\n{detailed_analysis}")
            display_text = "\n\n".join(parts)

            # Add Key Themes
            if news_analysis.get("key_themes"):
                themes = "\n".join(
                    [f"- {t}" for t in news_analysis.get("key_themes", [])]
                )
                display_text += f"\n\n**Key Themes**\n\n{themes}"

            # Add Risks & Opportunities
            if news_analysis.get("risks"):
                risks = "\n".join([f"- {r}" for r in news_analysis.get("risks", [])])
                display_text += f"\n\n**Risks**\n\n{risks}"

            if news_analysis.get("opportunities"):
                opps = "\n".join(
                    [f"- {o}" for o in news_analysis.get("opportunities", [])]
                )
                display_text += f"\n\n**Opportunities**\n\n{opps}"

        elif news_text:
            display_text = news_text

        news_model = news_analysis.get("model_used", "N/A")

        if display_text:
            console.print(
                Panel(
                    Markdown(display_text),
                    title=f"📰 News Analysis",
                    border_style="cyan",
                    subtitle=f"Model: {news_model}",
                )
            )
        else:

            console.print(
                Panel(
                    "[dim]No news analysis content available[/dim]",
                    title="📰 News Analysis",
                    border_style="cyan",
                )
            )
    else:
        console.print(
            Panel(
                "[dim]No news analysis available[/dim]",
                title="📰 News Analysis",
                border_style="cyan",
            )
        )

    console.print()

    # 2. FINAL CONCLUSION (Investment Decision)
    console.print("[bold magenta]2️⃣  INVESTMENT RECOMMENDATION[/bold magenta]")
    console.print("[dim]   The AI's final decision based on all current data[/dim]\n")
    decision_summary = result.get("ai_analysis", {})
    if not isinstance(decision_summary, dict):
        decision_summary = workflow_data.get("decision_summary", {})
    if not isinstance(decision_summary, dict):
        decision_summary = {}

    # Extract raw content which is now Markdown
    # Use 'reasoning' (set in investment_reasoner.py) or 'analysis' or 'content' key
    raw_content = (
        decision_summary.get("reasoning")
        or decision_summary.get("analysis")
        or decision_summary.get("content")
        or ""
    )

    if raw_content:
        console.print(
            Panel(
                Markdown(raw_content),
                title="🎯 Investment Decision",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[dim]No final conclusion available[/dim]",
                title="🎯 Investment Decision",
                border_style="green",
            )
        )

    console.print()
    console.print(
        "[dim]Note: This analysis is not investment advice. Conduct your own research.[/dim]\n"
    )

    # Technical Metadata (Model & Usage)
    model_used = decision_summary.get("model_used") or (
        result.get("workflow") or {}
    ).get("investment_decision", {}).get("model", "unknown")
    usage = decision_summary.get("usage") or (result.get("workflow") or {}).get(
        "investment_decision", {}
    ).get("usage")

    if model_used or usage:
        tech_parts = [f"[dim]Model: {model_used}[/dim]"]
        if usage:
            tech_parts.append(
                f"[dim]Tokens: {usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out[/dim]"
            )
        console.print(f"   {' | '.join(tech_parts)}")

    # Summary footer
    console.print(
        "\n[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]"
    )
    console.print("[dim]✅ Analysis complete — All findings shown above.[/dim]")
    console.print(
        "[dim]💡 Tip: This analysis is for informational purposes only. Always do your own research before investing.[/dim]"
    )
    console.print(
        "[bold yellow]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold yellow]\n"
    )

    if readable_mode:
        plain = _extract_plain_text(raw_content)
        short_points = _take_sentences(plain, max_count=4)
        if not short_points:
            short_points = [
                "This report evaluates the company's current condition together with the impact of news.",
                "Risk level and time horizon should be considered when making a decision.",
            ]
        readable = "\n".join([f"- {point}" for point in short_points])
        console.print(
            Panel(
                f"This section simplifies the technical language:\n\n{readable}",
                title="🗣️ Plain-Language Summary",
                border_style="magenta",
            )
        )
        console.print()


def display_batch_results(
    results: Dict[str, Optional[Dict[str, Any]]], console: Optional[Console] = None
) -> None:
    """
    Display batch analysis results in table format

    Args:
        results: Dictionary mapping symbol to analysis results
        console: Rich console for output
    """
    if console is None:
        console = core_console

    # Rich table output

    def _ensure_mapping(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if is_dataclass(value):
            return asdict(value)
        return {}

    def _format_price(value: Any, curr_sym: str = "$") -> str:
        if value in (None, "N/A", ""):
            return "N/A"
        try:
            return f"{curr_sym}{float(value):,.2f}"
        except (TypeError, ValueError):
            return str(value)

    def _format_market_cap(value: Any, curr_sym: str = "$") -> str:
        if value in (None, "N/A", ""):
            return "N/A"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)

        abs_number = abs(number)
        if abs_number >= 1_000_000_000_000:
            return f"{curr_sym}{number / 1_000_000_000_000:.2f}T"
        if abs_number >= 1_000_000_000:
            return f"{curr_sym}{number / 1_000_000_000:.2f}B"
        if abs_number >= 1_000_000:
            return f"{curr_sym}{number / 1_000_000:.2f}M"
        if abs_number >= 1_000:
            return f"{curr_sym}{number / 1_000:.2f}K"
        return f"{curr_sym}{number:,.0f}"

    def _format_ratio(value: Any) -> str:
        if value in (None, "N/A", ""):
            return "N/A"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return str(value)

    # Calculate success/fail counts for subtitle
    success_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - success_count

    table = Table(
        title=f"📊 Batch Analysis Results",
        caption=f"[dim]✅ {success_count} successful | ❌ {fail_count} failed[/dim]",
    )
    table.add_column("Symbol", style="cyan", justify="center")
    table.add_column("Company Name", style="white")
    table.add_column("Current Price", style="green", justify="right")
    table.add_column("Market Cap", style="yellow", justify="right")
    table.add_column("P/E Ratio", style="blue", justify="right")
    table.add_column("AI Decision", style="magenta", justify="center")
    table.add_column("Status", style="white", justify="center")

    for symbol, data in results.items():
        if data:
            # New batch mode structure: result from analyze_single_stock
            # Raw data in company_data, AI decision in ai_analysis
            company_data = data.get("company_data", {}) or {}
            ai_analysis = data.get("ai_analysis", {}) or {}

            profile = _ensure_mapping(company_data.get("company_profile"))
            metrics = _ensure_mapping(company_data.get("financial_metrics"))

            name = (
                profile.get("name")
                or profile.get("company_name")
                or company_data.get("longName")
                or company_data.get("shortName")
                or company_data.get("symbol")
                or symbol
            )

            price_value = metrics.get("current_price")
            if price_value in (None, "N/A", ""):
                price_value = (
                    company_data.get("regularMarketPrice")
                    or company_data.get("currentPrice")
                    or company_data.get("price")
                )

            market_cap_value = metrics.get("market_cap")
            if market_cap_value in (None, "N/A", ""):
                market_cap_value = (
                    company_data.get("marketCap")
                    or company_data.get("market_cap")
                    or company_data.get("marketCapitalization")
                )

            pe_value = metrics.get("pe_ratio")
            if pe_value in (None, "N/A", ""):
                pe_value = company_data.get("trailingPE")

            # AI karar bilgisi
            ai_decision = ai_analysis.get("decision", "N/A")
            if ai_decision == "N/A":
                ai_decision_display = "[dim]N/A[/dim]"
            elif ai_decision.lower() in ("buy", "al", "strong buy"):
                ai_decision_display = f"[bold green]{ai_decision}[/bold green]"
            elif ai_decision.lower() in ("sell", "sat", "strong sell"):
                ai_decision_display = f"[bold red]{ai_decision}[/bold red]"
            elif ai_decision.lower() in ("hold", "tut", "neutral"):
                ai_decision_display = f"[yellow]{ai_decision}[/yellow]"
            else:
                ai_decision_display = f"[cyan]{ai_decision}[/cyan]"

            # Get currency symbol
            curr_sym = get_currency_symbol(metrics.get("currency"))

            table.add_row(
                symbol,
                str(name)[:30],
                _format_price(price_value, curr_sym),
                _format_market_cap(market_cap_value, curr_sym),
                _format_ratio(pe_value),
                ai_decision_display,
                "[green]✅ OK[/green]",
            )
        else:
            table.add_row(
                symbol,
                "[dim]Data could not be fetched[/dim]",
                "-",
                "-",
                "-",
                "-",
                "[red]❌ Failed[/red]",
            )

    console.print(table)
    console.print()


def display_history_entry(
    entry: Dict[str, Any], console: Optional[Console] = None
) -> None:
    """
    Display history entry details in a user-friendly format

    Args:
        entry: History entry dictionary
        console: Rich console for output
    """
    if console is None:
        console = core_console

    console.print(
        f"\n[bold cyan]📚 Saved Analysis #{entry.get('id', 'N/A')}[/bold cyan]\n"
    )
    console.print(f"  📌 Stock Symbol: [green]{entry.get('symbol', 'N/A')}[/green]")
    console.print(
        f"  📋 Analysis Type: [yellow]{entry.get('type', 'N/A').replace('_', ' ').title()}[/yellow]"
    )
    console.print(f"  🕒 Save Date: [dim]{entry.get('timestamp', 'N/A')}[/dim]\n")

    # Display data if available
    data = entry.get("data", {})
    if not data:
        console.print("[yellow]⚠️ No data available to display.[/yellow]")
        return

    # Check for simplified history format (from commands/history.py)
    is_simplified = isinstance(data, dict) and (
        "news_analysis_output" in data or "final_analysis_output" in data
    )

    if is_simplified:
        # --- Formatted Display (Similar to Analysis Result) ---

        # 1. News Analysis
        news_text = data.get("news_analysis_output", "")
        if news_text:
            console.print(
                Panel(
                    Markdown(news_text), title="📰 News Analysis", border_style="cyan"
                )
            )
        else:
            console.print(
                Panel(
                    "[dim]No news analysis content[/dim]",
                    title="📰 News Analysis",
                    border_style="cyan",
                )
            )

        console.print()

        # 2. Final Analysis
        final_text = data.get("final_analysis_output", "")
        model_used = data.get("model_used", "Unknown")

        if final_text:
            console.print(
                Panel(
                    Markdown(final_text),
                    title="🎯 Investment Decision",
                    border_style="green",
                    subtitle=f"Model: {model_used}",
                )
            )
        else:
            console.print(
                Panel(
                    "[dim]No investment decision content[/dim]",
                    title="🎯 Investment Decision",
                    border_style="green",
                )
            )

        console.print()

        # 3. Links
        links = data.get("site_links", [])
        if links and isinstance(links, list):
            console.print("[bold blue]🔍 Source Links[/bold blue]")
            for link in links:
                console.print(f"   • [link={link}]{link}[/link]")
            console.print()

        # Footer
        console.print("[dim]✅ History analysis displayed.[/dim]\n")

    else:
        # --- Fallback: Raw JSON Display ---
        console.print(
            Panel(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                title="📄 Full Analysis Data (Raw)",
                subtitle="[dim]Scroll to see all details[/dim]",
                border_style="cyan",
            )
        )
