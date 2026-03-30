"""Financial data query commands"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from core import (
    get_standard_logger,
    log_operation_start,
    log_operation_end,
    console as core_console,
)
from config import get_config
from commands.favorites import load_favorites
from services.finance import PortfolioOptimizer
from domain.utils import get_currency_symbol, safe_float
from services.factories import get_financial_service

logger = get_standard_logger(__name__)

def _format_trend_bar(
    value: float, min_value: float, max_value: float, width: int = 20
) -> str:
    if width <= 0:
        return ""
    if max_value <= min_value:
        return "█" * max(1, width // 2)

    normalized = (value - min_value) / (max_value - min_value)
    normalized = max(0.0, min(1.0, normalized))
    filled = max(1, int(round(normalized * width)))
    return "█" * filled + "░" * (width - filled)


def handle_company_financial_query(console: Console = core_console) -> None:
    """Handle interactive company-focused financial data query."""
    console.print("\n[bold cyan]🏢 Company Financial Data Query[/bold cyan]")
    console.print(
        "[dim]Select a company data category or perform portfolio optimization.[/dim]\n"
    )
    console.print("[bold]1[/bold] Overview & Key Metrics")
    console.print("[bold]2[/bold] Dividend History")
    console.print("[bold]3[/bold] Analyst Recommendations")
    console.print("[bold]4[/bold] Earnings Data")
    console.print("[bold]5[/bold] Ownership & Insiders")
    console.print("[bold]6[/bold] ESG / Sustainability")
    console.print("[bold]7[/bold] Portfolio Optimization\n")

    choice = Prompt.ask(
        "Select Operation",
        choices=["1", "2", "3", "4", "5", "6", "7"],
        default="1",
        show_choices=False,
    )
    if choice == "7":
        _handle_portfolio_optimization(console)
        return

    symbol = Prompt.ask("Stock Symbol").upper().strip()
    if not symbol:
        return

    service = get_financial_service()
    start_time = log_operation_start(logger, f"Financial Query [{choice}]: {symbol}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(description=f"Fetching data: {symbol}...", total=None)

            if choice == "1":
                data = service.get_company_data(symbol)
                if not data:
                    raise ValueError("Data not found or incomplete.")
                _display_financial_summary(symbol, data, console)

            elif choice == "2":
                data = service.get_dividends(symbol)
                if not data:
                    raise ValueError("No dividend data found.")
                _display_dividends(symbol, data, console)

            elif choice == "3":
                data = service.get_analyst(symbol)
                if not data:
                    raise ValueError("No analyst data found.")
                _display_analyst(symbol, data, console)

            elif choice == "4":
                data = service.get_earnings(symbol)
                if not data:
                    raise ValueError("No earnings data found.")
                _display_earnings(symbol, data, console)

            elif choice == "5":
                data = service.get_ownership(symbol)
                if not data:
                    raise ValueError("No ownership data found.")
                _display_ownership(symbol, data, console)

            elif choice == "6":
                data = service.get_sustainability(symbol)
                if not data:
                    raise ValueError("No sustainability/ESG data found.")
                _display_sustainability(symbol, data, console)

        log_operation_end(
            logger, f"Financial Query [{choice}]: {symbol}", start_time, success=True
        )

    except Exception as e:
        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
            console.print(
                "[red]❌ Error: Too many requests (Rate Limit). Please wait a while or check your proxy list.[/red]"
            )
        else:
            console.print(f"[red]❌ Error occurred: {e}[/red]")
        logger.error("Financial Query [%s] Failed: %s", choice, e)
        log_operation_end(
            logger, f"Financial Query [{choice}]: {symbol}", start_time, success=False
        )


def handle_index_etf_financial_query(console: Console = core_console) -> None:
    """Handle interactive index/ETF focused financial data query."""
    console.print("\n[bold cyan]📈 Index / ETF Data Query[/bold cyan]")
    console.print(
        "[dim]Enter an index or ETF symbol (e.g., XU100.IS, ^GSPC, SPY).[/dim]\n"
    )

    symbol = Prompt.ask("Index / ETF Symbol").upper().strip()
    if not symbol:
        return

    service = get_financial_service()
    start_time = log_operation_start(logger, f"Index/ETF Query: {symbol}")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(
                description=f"Fetching index/ETF data: {symbol}...", total=None
            )
            data = service.get_index_data(symbol)
            if not data:
                raise ValueError("No index/ETF data found.")
            _display_index_summary(symbol, data, console)

        log_operation_end(
            logger, f"Index/ETF Query: {symbol}", start_time, success=True
        )

    except Exception as e:
        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
            console.print(
                "[red]❌ Error: Too many requests (Rate Limit). Please wait a while or check your proxy list.[/red]"
            )
        else:
            console.print(f"[red]❌ Error occurred: {e}[/red]")
        logger.error("Index/ETF Query Failed: %s", e)
        log_operation_end(
            logger, f"Index/ETF Query: {symbol}", start_time, success=False
        )


def _display_financial_summary(symbol: str, info: dict, console: Console) -> None:
    """Display summarized financial data in a clean format."""

    from config import NA_VALUE

    currency = info.get("currency", "TRY")
    curr_sym = get_currency_symbol(currency)

    # Company Header
    name = info.get("longName", symbol)
    sector = info.get("sector", "N/A")
    industry = info.get("industry", "N/A")

    console.print(f"\n[bold yellow]🏢 {name} ({symbol})[/bold yellow]")
    console.print(f"[dim]{sector} | {industry}[/dim]\n")

    # Key Metrics Table
    table = Table(title="💎 Key Financial Data", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    # Helper for formatting values
    def fmt_num(val):
        if val is None or val == NA_VALUE:
            return "N/A"
        try:
            val = float(val)
            if val >= 1e12:
                return f"{curr_sym}{val/1e12:.2f}T"
            if val >= 1e9:
                return f"{curr_sym}{val/1e9:.2f}B"
            if val >= 1e6:
                return f"{curr_sym}{val/1e6:.2f}M"
            return f"{curr_sym}{val:,.0f}"
        except (TypeError, ValueError):
            return str(val)

    def fmt_ratio(val, is_percent=False, already_percent=False):
        if val is None or val == NA_VALUE:
            return "N/A"
        try:
            val = float(val)
            if already_percent:
                return f"%{val:.2f}"
            if is_percent:
                return f"%{val*100:.2f}"
            return f"{val:.2f}"
        except (TypeError, ValueError):
            return str(val)

    def fmt_price(val):
        if val is None or val == NA_VALUE:
            return "N/A"
        try:
            val = float(val)
            return f"{curr_sym}{val:,.2f}"
        except (TypeError, ValueError):
            return str(val)

    # Determine current price
    current_price = info.get("currentPrice")
    if current_price is None or current_price == NA_VALUE:
        current_price = info.get("regularMarketPrice")

    # Important Metrics Selection
    eps_value = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")

    metrics = [
        ("Current Price", fmt_price(current_price)),
        ("Market Cap", fmt_num(info.get("marketCap"))),
        ("Trailing P/E", fmt_ratio(info.get("trailingPE"))),
        ("Forward P/E", fmt_ratio(info.get("forwardPE"))),
        ("Earnings Per Share (EPS)", fmt_ratio(eps_value)),
        ("Dividend Yield", fmt_ratio(info.get("dividendYield"), already_percent=True)),
        ("52 Week High", fmt_price(info.get("fiftyTwoWeekHigh"))),
        ("52 Week Low", fmt_price(info.get("fiftyTwoWeekLow"))),
        ("Beta", fmt_ratio(info.get("beta"))),
        (
            "Return on Equity (ROE)",
            fmt_ratio(info.get("returnOnEquity"), is_percent=True),
        ),
    ]

    for label, value in metrics:
        table.add_row(label, str(value))

    console.print(table)

    # Brief Business Summary
    summary = info.get("longBusinessSummary") or ""
    if summary and summary != "Company summary not available.":
        # Show only first ~300 chars
        short_summary = summary[:300] + "..." if len(summary) > 300 else summary
        console.print(
            Panel(short_summary, title="Company Summary", border_style="blue")
        )

    console.print()


def _display_index_summary(symbol: str, data: dict, console: Console) -> None:
    """Display index/ETF centered summary with returns and trend table."""
    quote_type = str(data.get("quoteType") or "Unknown")
    if not data.get("isIndexAsset"):
        console.print(
            "[yellow]⚠️ This symbol was not recognized as an index/ETF. "
            "Use 'Overview & Key Metrics' (1) option for stock data.[/yellow]"
        )
        return

    currency = data.get("currency") or "TRY"
    curr_sym = get_currency_symbol(currency)
    name = data.get("longName") or symbol

    def fmt_price(val):
        parsed = safe_float(val)
        return "N/A" if parsed is None else f"{curr_sym}{parsed:,.2f}"

    def fmt_volume(val):
        parsed = safe_float(val)
        if parsed is None:
            return "N/A"
        return f"{parsed:,.0f}"

    def fmt_pct(val):
        parsed = safe_float(val)
        if parsed is None:
            return "N/A"
        if parsed > 0:
            return f"[green]+{parsed:.2f}%[/green]"
        if parsed < 0:
            return f"[red]{parsed:.2f}%[/red]"
        return f"{parsed:.2f}%"

    console.print(f"\n[bold yellow]📈 {name} ({symbol})[/bold yellow]")
    console.print(f"[dim]{quote_type}[/dim]\n")

    overview = Table(title="📊 Index Overview", show_header=True)
    overview.add_column("Metric", style="cyan")
    overview.add_column("Value", style="white", justify="right")
    overview.add_row("Current Price", fmt_price(data.get("regularMarketPrice")))
    overview.add_row("Previous Close", fmt_price(data.get("previousClose")))
    overview.add_row("Daily Change %", fmt_pct(data.get("dailyChangePercent")))
    overview.add_row("Volume", fmt_volume(data.get("volume")))
    overview.add_row("52 Week High", fmt_price(data.get("fiftyTwoWeekHigh")))
    overview.add_row("52 Week Low", fmt_price(data.get("fiftyTwoWeekLow")))
    console.print(overview)

    period_returns = data.get("returns") or {}
    returns_table = Table(title="📅 Periodic Returns", show_header=True)
    returns_table.add_column("Period", style="cyan")
    returns_table.add_column("Return", justify="right")
    for key in ("YTD", "1M", "3M", "6M", "1Y"):
        returns_table.add_row(key, fmt_pct(period_returns.get(key)))
    console.print(returns_table)

    candles = data.get("recentCandles") or []
    if isinstance(candles, list) and candles:
        close_values = [
            safe_float(entry.get("close"))
            for entry in candles
            if isinstance(entry, dict)
        ]
        close_values = [v for v in close_values if v is not None]

        if close_values:
            min_close = min(close_values)
            max_close = max(close_values)

            trend_table = Table(
                title="📈 Price Trend (last 20 sessions)", show_header=True
            )
            trend_table.add_column("Date", style="dim")
            trend_table.add_column("Close", justify="right")
            trend_table.add_column("Trend", style="cyan")

            for entry in candles:
                if not isinstance(entry, dict):
                    continue
                close_val = safe_float(entry.get("close"))
                if close_val is None:
                    continue
                bar = _format_trend_bar(close_val, min_close, max_close, width=20)
                trend_table.add_row(
                    str(entry.get("date", ""))[:10],
                    fmt_price(close_val),
                    bar,
                )

            if trend_table.row_count:
                console.print(trend_table)

    components = data.get("components") or []
    if isinstance(components, list) and components:
        components_table = Table(title="🏢 Top Components", show_header=True)
        components_table.add_column("#", style="dim", justify="right")
        components_table.add_column("Symbol", style="cyan")
        for idx, comp in enumerate(components[:10], start=1):
            components_table.add_row(str(idx), str(comp))
        if components_table.row_count:
            console.print(components_table)

    console.print()


def _display_dividends(symbol: str, data: dict, console: Console) -> None:
    """Dividend history and yield information."""
    from config import NA_VALUE

    console.print(
        f"\n[bold yellow]💰 Dividend & Split History — {symbol}[/bold yellow]\n"
    )

    table = Table(title="Dividend Overview", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    field_labels = {
        "dividendYield": "Dividend Yield (%)",
        "dividendFrequency": "Frequency",
        "lastDividendAmount": "Last Dividend Amount",
        "lastDividendDate": "Last Dividend Date",
        "ttmDividend": "Trailing 12m Dividend",
        "forwardAnnualDividendYield": "Forward Annual Yield (%)",
        "fiveYearAvgDividendYield": "5-Year Avg Yield (%)",
        "payoutRatio": "Payout Ratio",
    }
    for key, label in field_labels.items():
        val = data.get(key)
        if val not in (None, NA_VALUE, ""):
            table.add_row(label, str(val))

    console.print(table)

    # Dividend history list
    history = data.get("dividendHistory") or data.get("history") or []
    if history and isinstance(history, list):
        hist_table = Table(title="Recent Dividend Payments", show_header=True)
        hist_table.add_column("Date", style="dim")
        hist_table.add_column("Amount", justify="right")
        for entry in history[-10:]:
            if isinstance(entry, dict):
                hist_table.add_row(
                    str(entry.get("date", "")), str(entry.get("amount", ""))
                )
        if hist_table.row_count:
            console.print(hist_table)

    # Splits
    splits = data.get("splitsHistory") or data.get("splits") or []
    if splits and isinstance(splits, list):
        split_table = Table(title="Stock Splits", show_header=True)
        split_table.add_column("Date", style="dim")
        split_table.add_column("Ratio", justify="right")
        for entry in splits:
            if isinstance(entry, dict):
                split_table.add_row(
                    str(entry.get("date", "")), str(entry.get("ratio", ""))
                )
        if split_table.row_count:
            console.print(split_table)

    console.print()


def _display_analyst(symbol: str, data: dict, console: Console) -> None:
    """Analyst recommendations, price targets, and estimates."""
    from config import NA_VALUE

    console.print(
        f"\n[bold yellow]🎯 Analyst Recommendations — {symbol}[/bold yellow]\n"
    )

    # Core recommendation
    rec = data.get("analystRecommendations") or {}
    if rec:
        table = Table(title="Analyst Consensus", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white", justify="right")
        labels = {
            "ratingKey": "Rating",
            "ratingScore": "Mean Score (1=Strong Buy)",
            "numberOfAnalysts": "# Analysts",
        }
        for key, label in labels.items():
            val = rec.get(key)
            if val not in (None, NA_VALUE, ""):
                table.add_row(label, str(val))
        console.print(table)

    # Price targets
    targets = data.get("analystPriceTargets") or {}
    if targets:
        pt_table = Table(title="Price Targets", show_header=True)
        pt_table.add_column("Metric", style="cyan")
        pt_table.add_column("Value", style="white", justify="right")
        target_labels = {
            "current": "Current Price",
            "mean": "Mean Target",
            "low": "Low Target",
            "high": "High Target",
            "median": "Median Target",
            "upsidePotential": "Upside Potential (%)",
        }
        for key, label in target_labels.items():
            val = targets.get(key)
            if val not in (None, NA_VALUE, ""):
                pt_table.add_row(label, str(val))
        console.print(pt_table)

    # Upgrades/downgrades (last 5)
    upgrades = data.get("upgradesDowngrades") or []
    if upgrades and isinstance(upgrades, list):
        ud_table = Table(
            title="Recent Upgrades / Downgrades (last 5)", show_header=True
        )
        ud_table.add_column("Date", style="dim")
        ud_table.add_column("Firm", style="cyan")
        ud_table.add_column("Action")
        ud_table.add_column("From → To")
        for entry in upgrades[:5]:
            if isinstance(entry, dict):
                ud_table.add_row(
                    str(entry.get("date", entry.get("GradeDate", ""))),
                    str(entry.get("firm", entry.get("Firm", ""))),
                    str(entry.get("action", entry.get("Action", ""))),
                    f"{entry.get('fromGrade', entry.get('FromGrade', '?'))} → {entry.get('toGrade', entry.get('ToGrade', '?'))}",
                )
        if ud_table.row_count:
            console.print(ud_table)

    console.print()


def _display_earnings(symbol: str, data: dict, console: Console) -> None:
    """Earnings trend, EPS history, and upcoming dates."""

    console.print(f"\n[bold yellow]📈 Earnings Data — {symbol}[/bold yellow]\n")

    # Earnings trend
    trend = data.get("earningsTrend") or []
    if trend and isinstance(trend, list):
        t_table = Table(title="Earnings Trend", show_header=True)
        t_table.add_column("Period", style="cyan")
        t_table.add_column("EPS Estimate", justify="right")
        t_table.add_column("Revenue Estimate", justify="right")
        for entry in trend:
            if isinstance(entry, dict):
                t_table.add_row(
                    str(entry.get("period", "")),
                    str(entry.get("epsAverage", entry.get("earningsAverage", "N/A"))),
                    str(entry.get("revenueAverage", "N/A")),
                )
        if t_table.row_count:
            console.print(t_table)

    # EPS history
    eps_hist = data.get("epsHistory") or []
    if eps_hist and isinstance(eps_hist, list):
        eps_table = Table(title="EPS History (last 8 quarters)", show_header=True)
        eps_table.add_column("Date", style="dim")
        eps_table.add_column("EPS Actual", justify="right")
        eps_table.add_column("EPS Estimate", justify="right")
        eps_table.add_column("Surprise %", justify="right")
        for entry in eps_hist[-8:]:
            if isinstance(entry, dict):
                eps_table.add_row(
                    str(entry.get("quarter", entry.get("date", ""))),
                    str(entry.get("epsActual", "N/A")),
                    str(entry.get("epsEstimate", "N/A")),
                    str(entry.get("surprisePercent", "N/A")),
                )
        if eps_table.row_count:
            console.print(eps_table)

    # Upcoming earnings dates
    dates = data.get("earningsDates") or []
    if dates and isinstance(dates, list):
        date_table = Table(
            title="Upcoming / Recent Earnings Dates (next 3)", show_header=True
        )
        date_table.add_column("Date", style="cyan")
        date_table.add_column("EPS Estimate", justify="right")
        for entry in dates[:3]:
            if isinstance(entry, dict):
                date_table.add_row(
                    str(entry.get("date", "")),
                    str(entry.get("epsEstimate", "N/A")),
                )
        if date_table.row_count:
            console.print(date_table)

    console.print()


def _display_ownership(symbol: str, data: dict, console: Console) -> None:
    """Institutional holders, major holders, and insider transactions."""
    console.print(f"\n[bold yellow]🏦 Ownership & Insiders — {symbol}[/bold yellow]\n")

    # Institutional holders
    inst = data.get("institutionalHolders") or data.get("topInstitutionalHolders") or []
    if inst and isinstance(inst, list):
        i_table = Table(title="Top Institutional Holders", show_header=True)
        i_table.add_column("Holder", style="cyan")
        i_table.add_column("Shares", justify="right")
        i_table.add_column("% Held", justify="right")
        i_table.add_column("Date Reported", style="dim")
        for entry in inst[:10]:
            if isinstance(entry, dict):
                i_table.add_row(
                    str(entry.get("Holder", entry.get("holder", ""))),
                    str(entry.get("Shares", entry.get("shares", ""))),
                    str(entry.get("pctHeld", entry.get("% Out", ""))),
                    str(entry.get("Date Reported", entry.get("dateReported", ""))),
                )
        if i_table.row_count:
            console.print(i_table)

    # Insider transactions
    insider_txn = data.get("insiderTransactions") or []
    if insider_txn and isinstance(insider_txn, list):
        ins_table = Table(
            title="Recent Insider Transactions (last 5)", show_header=True
        )
        ins_table.add_column("Date", style="dim")
        ins_table.add_column("Insider", style="cyan")
        ins_table.add_column("Transaction")
        ins_table.add_column("Shares", justify="right")
        for entry in insider_txn[:5]:
            if isinstance(entry, dict):
                ins_table.add_row(
                    str(entry.get("startDate", entry.get("date", ""))),
                    str(entry.get("filerName", entry.get("name", ""))),
                    str(entry.get("transactionText", entry.get("transaction", ""))),
                    str(entry.get("shares", "")),
                )
        if ins_table.row_count:
            console.print(ins_table)

    # Major holders summary
    major = data.get("majorHolders") or {}
    if major and isinstance(major, dict):
        mh_table = Table(title="Major Holders Summary", show_header=True)
        mh_table.add_column("Metric", style="cyan")
        mh_table.add_column("Value", justify="right")
        for k, v in major.items():
            if v not in (None, ""):
                mh_table.add_row(str(k), str(v))
        if mh_table.row_count:
            console.print(mh_table)

    console.print()


def _display_sustainability(symbol: str, data: dict, console: Console) -> None:
    """ESG sustainability scores."""
    from config import NA_VALUE

    console.print(f"\n[bold yellow]🌱 ESG / Sustainability — {symbol}[/bold yellow]\n")

    table = Table(title="ESG Scores", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="white", justify="right")

    # Common top-level ESG keys
    top_keys = {
        "totalEsg": "Total ESG Score",
        "environmentScore": "Environmental (E)",
        "socialScore": "Social (S)",
        "governanceScore": "Governance (G)",
        "esgPerformance": "ESG Performance",
        "peerGroup": "Peer Group",
        "peerCount": "Peers Compared",
        "percentile": "Percentile in Peer Group",
        "ratingYear": "Rating Year",
        "ratingMonth": "Rating Month",
        "highestControversy": "Highest Controversy Level",
        "adult": "Adult Content",
        "alcoholic": "Alcohol",
        "animalTesting": "Animal Testing",
        "catholic": "Catholic Values",
        "controversialWeapons": "Controversial Weapons",
        "smallArms": "Small Arms",
        "furLeather": "Fur & Leather",
        "gambling": "Gambling",
        "gmo": "GMO",
        "militaryContract": "Military Contract",
        "nuclear": "Nuclear",
        "pesticides": "Pesticides",
        "palmOil": "Palm Oil",
        "coal": "Thermal Coal",
        "tobacco": "Tobacco",
    }
    for key, label in top_keys.items():
        val = data.get(key)
        if val not in (None, NA_VALUE, ""):
            table.add_row(label, str(val))

    # If nothing matched top keys, show raw keys
    if not table.row_count:
        for k, v in data.items():
            if v not in (None, NA_VALUE, ""):
                table.add_row(str(k), str(v))

    console.print(table)
    console.print()


def _handle_portfolio_optimization(console: Console = core_console) -> None:
    """Build an equal-weight portfolio baseline for favorite stocks."""
    console.print("\n[bold cyan]📈 Portfolio Allocation Baseline[/bold cyan]")
    console.print(
        "[dim]Creates a simple equal-weight allocation across favorite stocks.[/dim]\n"
    )

    favorites = load_favorites()
    if len(favorites) < 2:
        console.print(
            "[yellow]⚠️ At least 2 favorite stocks are required for portfolio optimization.[/yellow]"
        )
        return

    optimizer = PortfolioOptimizer(get_config())

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(
            description="Preparing price history and correlations...", total=None
        )
        result = optimizer.optimize_portfolio(favorites)

    if not result:
        console.print("[red]❌ Not enough data found for portfolio allocation baseline.[/red]")
        return

    table = Table(title="📊 Equal-Weight Portfolio Targets", show_header=True)
    table.add_column("Symbol", style="cyan")
    table.add_column("Target Weight", justify="right")

    for ticker in result.tickers_used:
        weight = result.weights.get(ticker, 0)
        table.add_row(
            ticker,
            f"{weight * 100:.2f}%",
        )

    console.print(table)

    # Note: OptimizationResult already returns annual values
    console.print(
        f"[bold]Expected annual return:[/bold] {result.expected_return * 100:.2f}% | "
        f"[bold]Annual volatility:[/bold] {result.volatility * 100:.2f}% | "
        f"[bold]Sharpe Ratio:[/bold] {result.sharpe_ratio:.2f}"
    )

    console.print(
        "\n[bold]Allocation method:[/bold] Equal weight baseline for favorites list"
    )

    if result.tickers_skipped:
        skipped = ", ".join(result.tickers_skipped)
        console.print(
            f"[yellow]⚠️ Skipped due to insufficient data: {skipped}[/yellow]"
        )

    _display_top_correlations(result.correlation, console)


def _display_top_correlations(corr_df, console: Console) -> None:
    """Shows highest correlation pairs."""
    if corr_df is None or corr_df.empty:
        return

    pairs = []
    tickers = list(corr_df.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a = tickers[i]
            b = tickers[j]
            val = corr_df.loc[a, b]
            pairs.append((a, b, float(val)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    top_pairs = pairs[:5]

    if not top_pairs:
        return

    table = Table(title="🔗 Top Correlations", show_header=True)
    table.add_column("Pair", style="cyan")
    table.add_column("Correlation", justify="right")

    for a, b, val in top_pairs:
        table.add_row(f"{a} - {b}", f"{val:.2f}")

    console.print("\n")
    console.print(table)
