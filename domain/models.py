"""
Data Models - Struct
DataQuality enum and all data models
"""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

# ───────────────────────────────
# Data Quality and Core Models
# ───────────────────────────────


class DataQuality(Enum):
    """Data quality levels."""

    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    MISSING = "missing"


@dataclass
class ProcessedFinancialMetrics:
    """Processed financial metrics."""

    # Core valuation
    current_price: Optional[float] = None
    currency: Optional[str] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    enterprise_value: Optional[float] = None

    # Growth and profitability
    revenue_growth: Optional[float] = None
    earnings_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    peg_ratio: Optional[float] = None  # Price/Earnings to Growth ratio

    # Risk indicators
    beta: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None

    # Dividend and cash
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    free_cash_flow: Optional[float] = None

    # New - Analyst Estimates (Revenue/EPS)
    estimated_revenue_avg_next_quarter: Optional[float] = None
    estimated_eps_avg_next_quarter: Optional[float] = None
    estimated_revenue_avg_next_year: Optional[float] = None
    estimated_eps_avg_next_year: Optional[float] = None
    analyst_estimates_count: Optional[int] = None

    # Market performance
    market_cap: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    price_change_percent: Optional[float] = None

    # Valuation metrics
    dcf: Optional[float] = None
    dcf_diff: Optional[float] = None

    # New - Earnings and Dividend Analysis
    next_earnings_date: Optional[str] = None
    last_earnings_date: Optional[str] = None
    earnings_estimate: Optional[float] = None
    earnings_actual: Optional[float] = None
    earnings_surprise: Optional[float] = None
    last_dividend_date: Optional[str] = None
    last_dividend_amount: Optional[float] = None
    dividend_frequency: Optional[str] = None
    dividend_growth_rate: Optional[float] = None
    next_dividend_date: Optional[str] = None
    next_dividend_amount: Optional[float] = None

    # New - Analyst Ratings
    latest_grade: Optional[str] = None
    latest_grade_company: Optional[str] = None
    latest_grade_date: Optional[str] = None
    latest_grade_action: Optional[str] = None
    previous_grade: Optional[str] = None

    # New - Price Targets
    last_month_avg_price_target: Optional[float] = None
    last_quarter_avg_price_target: Optional[float] = None
    last_year_avg_price_target: Optional[float] = None
    all_time_avg_price_target: Optional[float] = None
    analyst_count_last_year: Optional[int] = None

    # New - Insider Transactions
    insider_transaction_count: Optional[int] = None
    latest_insider_transaction: Optional[str] = None
    latest_insider_name: Optional[str] = None
    latest_insider_type: Optional[str] = None

    # New - Sustainability (ESG) Metrics
    esg_total_score: Optional[float] = None
    esg_environment_score: Optional[float] = None
    esg_social_score: Optional[float] = None
    esg_governance_score: Optional[float] = None
    esg_peer_group: Optional[str] = None

    # New - Shares and Ownership Information
    shares_outstanding: Optional[float] = None
    float_shares: Optional[float] = None
    implied_shares_outstanding: Optional[float] = None
    held_percent_insiders: Optional[float] = None
    held_percent_institutions: Optional[float] = None

    # New - Short Interest Information
    short_ratio: Optional[float] = None
    short_percent_of_float: Optional[float] = None
    shares_short: Optional[float] = None
    shares_short_prior_month: Optional[float] = None

    # New - Income Statement Metrics
    revenue: Optional[float] = None
    net_income: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    ebitda: Optional[float] = None
    eps: Optional[float] = None

    # New - Margin Metrics
    gross_margins: Optional[float] = None
    operating_margins: Optional[float] = None

    # New - Technical Indicators
    fifty_day_average: Optional[float] = None
    two_hundred_day_average: Optional[float] = None
    rsi: Optional[float] = None  # Relative Strength Index (14-day)

    # Data quality
    data_quality: DataQuality = DataQuality.MISSING
    last_updated: str = ""


@dataclass
class ProcessedCompanyProfile:
    """Processed company profile."""

    # Core information
    symbol: str
    company_name: str
    sector: str
    industry: str
    description: str

    # Business model
    business_model: str
    key_products: List[str]
    target_markets: List[str]

    # Management and structure
    employees: Optional[int]
    founded_year: Optional[int]
    headquarters: str

    # Data quality
    data_quality: DataQuality
    last_updated: str


@dataclass
class ProcessedComparativeAnalysis:
    """Processed comparative analysis."""

    # Sector comparison
    sector_rank: Optional[int]
    sector_percentile: Optional[float]
    sector_company_count: int

    # Metric-based comparison
    pe_vs_sector: Optional[float]  # PE ratio vs sector average
    pb_vs_sector: Optional[float]  # Price/Book vs sector average
    beta_vs_sector: Optional[float]  # Beta vs sector average

    # Overall score
    overall_score: float  # 0-100 range
    data_quality: DataQuality
    last_updated: str


@dataclass
class ProcessedCompanyData:
    """Combined processed company data."""

    # Core data
    symbol: str
    company_name: str
    sector: str
    industry: str

    # Processed metrics
    financial_metrics: ProcessedFinancialMetrics
    company_profile: ProcessedCompanyProfile
    comparative_analysis: ProcessedComparativeAnalysis

    # Overall data quality
    overall_data_quality: DataQuality
    last_updated: str
    macro_context: Optional[dict] = None

    # YAGNI: processing_notes, missing_fields, and data_sources were removed.
    # These fields were computed but never used in AI analysis.
