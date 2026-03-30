"""
Core Data Processing and Merge Functions - Data Processors
ProcessedCompanyData creation and AI prompt formatting

Aligned with AGENTS.MD principles: KISS, DRY, YAGNI
"""

from typing import Dict, Any, List
from datetime import datetime
import copy

from .models import (
    ProcessedFinancialMetrics,
    ProcessedCompanyProfile,
    ProcessedComparativeAnalysis,
    ProcessedCompanyData,
    DataQuality,
)
from .utils import safe_float, safe_int
from .quality import assess_data_quality
from .utils import compute_comparative_score
from core import get_standard_logger

_log = get_standard_logger("data_processors")


def clean_keys(data: Any) -> Any:
    """
    Recursively strips whitespace from dictionary keys.
    """
    if isinstance(data, dict):
        return {k.strip(): clean_keys(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_keys(item) for item in data]
    return data


def _deduplicate_shares_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicates shares history by date, keeping the last entry for each date.
    """
    if not history:
        return []

    deduped = {}
    for entry in history:
        date = entry.get("date")
        if date:
            deduped[date] = entry

    # Return sorted by date
    return sorted(deduped.values(), key=lambda x: x.get("date", ""))


def sanitize_raw_company_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a cleaned/sanitized copy of raw company data for persistence and downstream use.
    - Strips whitespace from keys
    - Swaps last/next earnings & dividend dates if reversed
    - Normalizes percentage/ratio fields (dividendYield, debtToEquity)
    - Deduplicates shares history and aligns latestSharesOutstanding
    - Prefers impliedSharesOutstanding as canonical sharesOutstanding when present
    """
    if not isinstance(data, dict):
        return data

    cleaned = clean_keys(copy.deepcopy(data))

    # Swap earnings dates when reversed
    next_earnings = cleaned.get("nextEarningsDate")
    last_earnings = cleaned.get("lastEarningsDate")
    if (
        isinstance(next_earnings, str)
        and isinstance(last_earnings, str)
        and last_earnings > next_earnings
    ):
        cleaned["nextEarningsDate"], cleaned["lastEarningsDate"] = (
            last_earnings,
            next_earnings,
        )

    # Swap dividend dates when reversed
    next_dividend = cleaned.get("nextDividendDate")
    last_dividend = cleaned.get("lastDividendDate")
    if (
        isinstance(next_dividend, str)
        and isinstance(last_dividend, str)
        and last_dividend > next_dividend
    ):
        cleaned["nextDividendDate"], cleaned["lastDividendDate"] = (
            last_dividend,
            next_dividend,
        )

    # Prefer implied shares when available
    shares_outstanding = safe_float(cleaned.get("sharesOutstanding"))
    implied_shares = safe_float(cleaned.get("impliedSharesOutstanding"))
    if implied_shares and implied_shares > 0:
        cleaned["sharesOutstanding"] = implied_shares

    # Deduplicate shares history
    shares_full = cleaned.get("sharesFull")
    if isinstance(shares_full, dict):
        history = shares_full.get("history")
        if isinstance(history, list):
            shares_full["history"] = _deduplicate_shares_history(history)

        # Align latestSharesOutstanding with preferred figure
        if implied_shares and implied_shares > 0:
            shares_full["latestSharesOutstanding"] = implied_shares
        elif shares_outstanding and shares_outstanding > 0:
            shares_full["latestSharesOutstanding"] = shares_outstanding

        # Clarify dilution semantics: expose shareChangePercent and dilutionType
        dilution_pct = safe_float(shares_full.get("dilutionPercent"))
        if dilution_pct is not None:
            shares_full["shareChangePercent"] = dilution_pct
            if dilution_pct > 0:
                shares_full["dilutionType"] = "DILUTION"
            elif dilution_pct < 0:
                shares_full["dilutionType"] = "BUYBACK"
            else:
                shares_full["dilutionType"] = "FLAT"

    return cleaned


def _extract_earnings_estimate_from_dict(earnings_dict: Any) -> Any:
    """
    Extract the latest EPS estimate from an earningsEstimate dictionary.

    yfinance earningsEstimate format:
    {
        "0q": {"avg": 3.85, "low": 3.41, "high": 4.11, ...},
        "+1q": {"avg": 3.91, ...},
        "0y": {"avg": 15.72, ...},
        "+1y": {"avg": 18.62, ...}
    }

    Priority: +1q > 0q > +1y > 0y

    Args:
        earnings_dict: earningsEstimate dictionary or None

    Returns:
        Quarterly EPS estimate (float) or None
    """
    if not isinstance(earnings_dict, dict):
        return None

    # Try quarterly estimates first
    for period in ["+1q", "0q", "+1y", "0y"]:
        period_data = earnings_dict.get(period, {})
        if isinstance(period_data, dict):
            avg = period_data.get("avg")
            if avg is not None and avg != "N/A":
                return avg

    return None


def process_and_enrich_company_data(
    company_data: Dict[str, Any], all_companies: Dict[str, Any]
) -> ProcessedCompanyData:
    """
    Take raw company data, process it, and build a standard ProcessedCompanyData object.
    KISS principle: compute only the information that is used.

    Args:
        company_data: Raw company data
        all_companies: All companies for comparison

    Returns:
        Processed company data
    """
    try:
        # Sanitize raw data upfront so downstream consumers see consistent values
        company_data = sanitize_raw_company_data(company_data)

        # Data quality assessment
        data_quality = assess_data_quality(company_data)

        # Process sub-models
        financial_metrics = _process_financial_metrics(
            company_data, precomputed_quality=data_quality
        )
        company_profile = _process_company_profile(
            company_data, precomputed_quality=data_quality
        )
        comparative_analysis = _process_comparative_analysis(
            company_data, all_companies, precomputed_quality=data_quality
        )

        # YAGNI: processing_notes, missing_fields, and data_sources removed because they are unused.
        return ProcessedCompanyData(
            symbol=company_data.get("symbol", "UNKNOWN"),
            company_name=company_data.get(
                "longName", company_data.get("symbol", "UNKNOWN")
            ),
            sector=company_data.get("sector", "Unknown"),
            industry=company_data.get("industry", "Unknown"),
            financial_metrics=financial_metrics,
            company_profile=company_profile,
            comparative_analysis=comparative_analysis,
            overall_data_quality=data_quality,
            macro_context=company_data.get("macro_context"),
            last_updated=datetime.now().isoformat(),
        )

    except Exception as e:
        _log.error(f"Data processing error: {e}", exc_info=True)
        return _create_fallback_processed_data(company_data, str(e))


# ───────────────────────────────
# Helper Processing Functions
# ───────────────────────────────


def _process_financial_metrics(
    data: Dict[str, Any], precomputed_quality: DataQuality
) -> ProcessedFinancialMetrics:
    """
    Process financial metrics.

    Args:
        data: Raw company data
        precomputed_quality: Precomputed data quality

    Returns:
        Processed financial metrics
    """
    # 1. Date and shares logic - simplified (handled in sanitize_raw_company_data)
    # Just retrieve the values directly

    return ProcessedFinancialMetrics(
        current_price=safe_float(
            data.get("currentPrice")
            or data.get("regularMarketPrice")
            or data.get("price")
        ),
        currency=data.get("currency"),
        pe_ratio=safe_float(data.get("trailingPE")),
        forward_pe=safe_float(data.get("forwardPE")),
        price_to_book=safe_float(data.get("priceToBook")),
        price_to_sales=safe_float(data.get("priceToSales")),
        enterprise_value=safe_float(data.get("enterpriseValue")),
        revenue_growth=safe_float(data.get("revenueGrowth")),
        earnings_growth=safe_float(data.get("earningsGrowth")),
        profit_margin=safe_float(
            data.get("profitMargins")
            or data.get("profitMargin")
            or data.get("netProfitMargin")
        ),
        roe=safe_float(data.get("returnOnEquity") or data.get("roe")),
        roa=safe_float(data.get("returnOnAssets") or data.get("roa")),
        peg_ratio=safe_float(data.get("pegRatio") or data.get("peg")),
        beta=safe_float(data.get("beta")),
        debt_to_equity=safe_float(data.get("debtToEquity")),
        current_ratio=safe_float(data.get("currentRatio")),
        quick_ratio=safe_float(data.get("quickRatio")),
        dividend_yield=safe_float(data.get("dividendYield")),
        payout_ratio=safe_float(data.get("payoutRatio")),
        free_cash_flow=safe_float(data.get("freeCashFlow")),
        # Analyst estimates
        estimated_revenue_avg_next_quarter=safe_float(
            data.get("estimated_revenue_avg_next_quarter")
        ),
        estimated_eps_avg_next_quarter=safe_float(
            data.get("estimated_eps_avg_next_quarter")
        ),
        estimated_revenue_avg_next_year=safe_float(
            data.get("estimated_revenue_avg_next_year")
        ),
        estimated_eps_avg_next_year=safe_float(data.get("estimated_eps_avg_next_year")),
        analyst_estimates_count=safe_int(data.get("analyst_estimates_count")),
        market_cap=safe_float(data.get("marketCap")),
        fifty_two_week_high=safe_float(data.get("fiftyTwoWeekHigh")),
        fifty_two_week_low=safe_float(data.get("fiftyTwoWeekLow")),
        price_change_percent=safe_float(data.get("regularMarketChangePercent")),
        # Valuation metrics
        dcf=safe_float(data.get("dcf")),
        dcf_diff=safe_float(data.get("dcfDiff")),
        # Earnings and dividend analysis
        next_earnings_date=data.get("nextEarningsDate", "N/A"),
        last_earnings_date=data.get("lastEarningsDate", "N/A"),
        earnings_estimate=safe_float(
            data.get("estimated_eps_avg_next_quarter")
            or data.get("estimatedEPS")
            or _extract_earnings_estimate_from_dict(data.get("earningsEstimate"))
        ),
        earnings_actual=safe_float(
            data.get("earningsActual") or data.get("reportedEPS")
        ),
        earnings_surprise=safe_float(data.get("earningsSurprise")),
        last_dividend_date=data.get("lastDividendDate", "N/A"),
        last_dividend_amount=safe_float(data.get("lastDividendAmount")),
        dividend_frequency=data.get("dividendFrequency", "N/A"),
        dividend_growth_rate=safe_float(data.get("dividendGrowthRate")),
        # Dividend forward dates
        next_dividend_date=data.get("nextDividendDate", "N/A"),
        next_dividend_amount=safe_float(data.get("nextDividendAmount")),
        # Analyst ratings
        latest_grade=data.get("latestGrade", "N/A"),
        latest_grade_company=data.get("latestGradeCompany", "N/A"),
        latest_grade_date=data.get("latestGradeDate", "N/A"),
        latest_grade_action=data.get("latestGradeAction", "N/A"),
        previous_grade=data.get("previousGrade", "N/A"),
        # Price targets
        last_month_avg_price_target=safe_float(data.get("lastMonthAvgPriceTarget")),
        last_quarter_avg_price_target=safe_float(data.get("lastQuarterAvgPriceTarget")),
        last_year_avg_price_target=safe_float(data.get("lastYearAvgPriceTarget")),
        all_time_avg_price_target=safe_float(data.get("allTimeAvgPriceTarget")),
        analyst_count_last_year=safe_int(data.get("lastYearCount")),
        # Insider transactions
        insider_transaction_count=safe_int(
            (data.get("insider_trading_analysis") or {}).get("total_transactions")
        ),
        latest_insider_transaction=(
            (
                (data.get("insider_trading_analysis") or {}).get("latest_transaction")
                or {}
            ).get("date")
            or "N/A"
        ),
        latest_insider_name=(
            (
                (data.get("insider_trading_analysis") or {}).get("latest_transaction")
                or {}
            ).get("insider_name")
            or "N/A"
        ),
        latest_insider_type=(
            (
                (data.get("insider_trading_analysis") or {}).get("latest_transaction")
                or {}
            ).get("type")
            or "N/A"
        ),
        # Sustainability (ESG) metrics
        esg_total_score=safe_float((data.get("sustainability") or {}).get("totalEsg")),
        esg_environment_score=safe_float(
            (data.get("sustainability") or {}).get("environmentScore")
        ),
        esg_social_score=safe_float(
            (data.get("sustainability") or {}).get("socialScore")
        ),
        esg_governance_score=safe_float(
            (data.get("sustainability") or {}).get("governanceScore")
        ),
        esg_peer_group=(data.get("sustainability") or {}).get("peerGroup", "N/A"),
        # Shares and ownership information
        shares_outstanding=safe_float(data.get("sharesOutstanding")),
        float_shares=safe_float(data.get("floatShares")),
        implied_shares_outstanding=safe_float(data.get("impliedSharesOutstanding")),
        held_percent_insiders=safe_float(data.get("heldPercentInsiders")),
        held_percent_institutions=safe_float(data.get("heldPercentInstitutions")),
        # Short interest information
        short_ratio=safe_float(data.get("shortRatio")),
        short_percent_of_float=safe_float(data.get("shortPercentOfFloat")),
        shares_short=safe_float(data.get("sharesShort")),
        shares_short_prior_month=safe_float(data.get("sharesShortPriorMonth")),
        # Income statement metrics
        revenue=safe_float(data.get("revenue")),
        net_income=safe_float(data.get("netIncome")),
        gross_profit=safe_float(data.get("grossProfit")),
        operating_income=safe_float(data.get("operatingIncome")),
        ebitda=safe_float(data.get("ebitda")),
        eps=safe_float(data.get("eps")),
        # Margin metrics
        gross_margins=safe_float(data.get("grossMargins")),
        operating_margins=safe_float(data.get("operatingMargins")),
        # Technical indicators
        fifty_day_average=safe_float(data.get("fiftyDayAverage")),
        two_hundred_day_average=safe_float(data.get("twoHundredDayAverage")),
        rsi=safe_float(data.get("rsi")),
        data_quality=precomputed_quality,
        last_updated=datetime.now().isoformat(),
    )


def _process_company_profile(
    data: Dict[str, Any], precomputed_quality: DataQuality
) -> ProcessedCompanyProfile:
    """
    Process the company profile.

    Args:
        data: Raw company data
        precomputed_quality: Precomputed data quality

    Returns:
        Processed company profile
    """
    return ProcessedCompanyProfile(
        symbol=data.get("symbol", ""),
        company_name=data.get("longName", ""),
        sector=data.get("sector", ""),
        industry=data.get("industry", ""),
        description=data.get("longBusinessSummary", ""),
        business_model=data.get("sector", "General Business"),
        key_products=[],
        target_markets=[],
        employees=safe_int(data.get("fullTimeEmployees")),
        founded_year=safe_int(data.get("founded")),
        headquarters=", ".join(
            [part for part in [data.get("city"), data.get("country")] if part]
        ),
        data_quality=precomputed_quality,
        last_updated=datetime.now().isoformat(),
    )


def _process_comparative_analysis(
    data: Dict[str, Any],
    all_companies: Dict[str, Any],
    precomputed_quality: DataQuality,
) -> ProcessedComparativeAnalysis:
    """
    Process comparative analysis and produce a real sector comparison.

    Args:
        data: Raw company data
        all_companies: All companies for comparison
        precomputed_quality: Precomputed data quality

    Returns:
        Processed comparative analysis
    """
    sector = data.get("sector", "")

    peer_metrics = data.get("peerMetrics")
    peer_companies: Dict[str, Dict[str, Any]] = {}

    if peer_metrics and isinstance(peer_metrics, list):
        for peer in peer_metrics:
            if not isinstance(peer, dict):
                continue
            symbol = peer.get("symbol")
            if not symbol or symbol == data.get("symbol"):
                continue
            peer_companies[symbol] = {
                "symbol": symbol,
                "sector": peer.get("sector") or sector,
                "trailingPE": peer.get("trailingPE"),
                "forwardPE": peer.get("forwardPE"),
                "priceToBook": peer.get("priceToBook"),
                "beta": peer.get("beta"),
                "marketCap": peer.get("marketCap"),
                "debtToEquity": peer.get("debtToEquity"),
            }

    # Fallback: other companies from the cache
    if not peer_companies:
        peer_companies = all_companies or {}

    comp_result = compute_comparative_score(data, peer_companies)

    overall_score = 50.0
    sector_company_count = 0
    pe_vs_sector = None
    pb_vs_sector = None
    beta_vs_sector = None

    if isinstance(comp_result, dict) and "error" not in comp_result:
        overall_score = comp_result.get("overall_score", overall_score)
        sector_company_count = comp_result.get("sector_company_count", 0)

        for metric in comp_result.get("metric_results", []):
            try:
                if metric.get("sector_avg") in (None, 0):
                    continue
                ratio = metric.get("value") / metric.get("sector_avg")
            except (TypeError, ZeroDivisionError):
                ratio = None

            metric_name = metric.get("metric")
            if metric_name == "trailingPE":
                pe_vs_sector = ratio
            elif metric_name == "priceToBook":
                pb_vs_sector = ratio
            elif metric_name == "beta":
                beta_vs_sector = ratio

    return ProcessedComparativeAnalysis(
        sector_rank=None,
        sector_percentile=None,
        sector_company_count=sector_company_count,
        pe_vs_sector=pe_vs_sector,
        pb_vs_sector=pb_vs_sector,
        beta_vs_sector=beta_vs_sector,
        overall_score=overall_score,
        data_quality=precomputed_quality,
        last_updated=datetime.now().isoformat(),
    )


# ───────────────────────────────
# Fallback Data Processing (moved from domain/utils.py)
# ───────────────────────────────


def _create_fallback_processed_data(
    company_data: Dict[str, Any], error_message: str
) -> ProcessedCompanyData:
    """
    Advanced fallback data creation.
    Uses standard processors to preserve existing data (DRY).
    """
    quality = assess_data_quality(company_data)

    return ProcessedCompanyData(
        symbol=company_data.get("symbol", "UNKNOWN"),
        company_name=company_data.get(
            "longName", company_data.get("symbol", "UNKNOWN")
        ),
        sector=company_data.get("sector", "Unknown"),
        industry=company_data.get("industry", "Unknown"),
        overall_data_quality=quality,
        financial_metrics=_process_financial_metrics(company_data, quality),
        company_profile=_process_company_profile(company_data, quality),
        comparative_analysis=_process_comparative_analysis(company_data, {}, quality),
        last_updated=datetime.now().isoformat(),
        macro_context={"is_fallback": True, "error_source": error_message},
    )
