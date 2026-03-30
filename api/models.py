"""API request and response models."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


def clean_symbol(value: str) -> str:
    """Normalize and validate ticker symbols."""
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError("Symbol cannot be empty")
    return text


def _clean_symbol_list(values: List[str], *, minimum: int = 1) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()

    for raw_value in values:
        symbol = clean_symbol(raw_value)
        if symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)

    if len(cleaned) < minimum:
        if minimum == 1:
            raise ValueError("At least one symbol is required")
        raise ValueError(f"At least {minimum} unique symbols are required")

    return cleaned


def _clean_text_list(values: Optional[List[str]]) -> Optional[List[str]]:
    if values is None:
        return None

    cleaned: List[str] = []
    seen: set[str] = set()
    for item in values:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
    return cleaned


class PeriodEnum(str, Enum):
    """Supported yfinance history periods."""

    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"
    SIX_MONTHS = "6mo"
    ONE_YEAR = "1y"
    TWO_YEARS = "2y"
    FIVE_YEARS = "5y"
    YTD = "ytd"
    MAX = "max"


class IntervalEnum(str, Enum):
    """Supported yfinance history intervals."""

    ONE_MINUTE = "1m"
    TWO_MINUTES = "2m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    SIXTY_MINUTES = "60m"
    NINETY_MINUTES = "90m"
    ONE_HOUR = "1h"
    ONE_DAY = "1d"
    FIVE_DAYS = "5d"
    ONE_WEEK = "1wk"
    ONE_MONTH = "1mo"
    THREE_MONTHS = "3mo"


class MarketRegionEnum(str, Enum):
    """Supported market summary regions."""

    US = "US"
    GB = "GB"
    ASIA = "ASIA"
    EUROPE = "EUROPE"
    RATES = "RATES"
    COMMODITIES = "COMMODITIES"
    CURRENCIES = "CURRENCIES"
    CRYPTOCURRENCIES = "CRYPTOCURRENCIES"


class ScreenerQueryTypeEnum(str, Enum):
    """Supported yfinance screener query families."""

    EQUITY = "EQUITY"
    FUND = "FUND"


class AnalysisStatus(str, Enum):
    PENDING = "Pending"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


class TypeFilterEnum(str, Enum):
    """Asset type filter for yfinance lookup."""

    STOCK = "stock"
    ETF = "etf"
    INDEX = "index"
    CRYPTO = "crypto"
    MUTUALFUND = "mutualfund"


class AlertDirectionEnum(str, Enum):
    ABOVE = "above"
    BELOW = "below"


class ProfileRiskToleranceEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProfileInvestmentHorizonEnum(str, Enum):
    SHORT_TERM = "short-term"
    MEDIUM_TERM = "medium-term"
    LONG_TERM = "long-term"


class ProfileMarketFocusEnum(str, Enum):
    BIST = "BIST"
    US = "US"
    CRYPTO = "Crypto"
    ALL = "All"


class ProfileAlertSensitivityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CategoryDataResponse(BaseModel):
    """Common wrapper used by finance category endpoints."""

    symbol: str
    category: str
    data: Any


class ScreenerCondition(BaseModel):
    operator: str = Field(..., description="Filter operator such as gt, lt, or eq")
    field: str = Field(..., description="yfinance screener field name")
    value: Any = Field(..., description="Comparison value")


class ScreenerRequest(BaseModel):
    query_type: ScreenerQueryTypeEnum = Field(
        ScreenerQueryTypeEnum.EQUITY,
        description="Screener target type",
    )
    conditions: List[ScreenerCondition] = Field(
        ..., description="List of screener conditions"
    )

    @field_validator("conditions")
    @classmethod
    def validate_conditions(
        cls, values: List[ScreenerCondition]
    ) -> List[ScreenerCondition]:
        if not values:
            raise ValueError("conditions cannot be empty")
        return values


class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol such as AAPL or THYAO.IS")
    investment_horizon: Optional[str] = Field(
        None,
        description="Optional investment horizon override",
    )
    user_context: Optional[str] = Field(
        None,
        description="Optional investor context or portfolio notes",
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return clean_symbol(value)


class AnalysisResponse(BaseModel):
    symbol: str
    status: AnalysisStatus
    task_id: Optional[str] = None
    news_summary: Optional[str] = Field(
        None,
        description="Extracted summary of the analyzed news flow",
    )
    final_decision: Optional[str] = Field(
        None,
        description="Final investment conclusion and reasoning",
    )
    full_result: Optional[Dict[str, Any]] = Field(
        None,
        description="Complete structured workflow output",
    )
    success: bool
    message: Optional[str] = None


class BatchAnalysisRequest(BaseModel):
    symbols: List[str] = Field(
        ..., description="Ticker symbols such as [AAPL, THYAO.IS, MSFT]"
    )
    investment_horizon: Optional[str] = Field(
        None,
        description="Optional investment horizon override",
    )
    user_context: Optional[str] = Field(
        None,
        description="Optional investor context or portfolio notes",
    )
    max_workers: Optional[int] = Field(
        None,
        ge=1,
        le=32,
        description="Parallel worker count",
    )

    @field_validator("symbols")
    @classmethod
    def validate_symbols(cls, values: List[str]) -> List[str]:
        return _clean_symbol_list(values)


class BatchAnalysisResponse(BaseModel):
    task_id: str
    status: AnalysisStatus
    success: bool
    message: str
    total_symbols: int


class BatchAnalysisStatusResponse(BaseModel):
    task_id: str
    status: AnalysisStatus
    success: bool
    message: str
    total_symbols: int
    symbols: List[str] = Field(
        default_factory=list,
        description="Symbols included in the batch request",
    )
    completed_count: Optional[int] = Field(
        None,
        description="Number of successfully completed analyses",
    )
    failed_count: Optional[int] = Field(
        None,
        description="Number of failed analyses",
    )
    total_count: Optional[int] = Field(
        None,
        description="Total number of processed analyses",
    )
    results: Optional[Dict[str, Any]] = Field(
        None,
        description="Per-symbol batch analysis results",
    )


class CompareRequest(BaseModel):
    symbols: List[str] = Field(
        ..., description="At least two ticker symbols for comparison"
    )
    criteria: Optional[str] = Field(
        None,
        description="Optional comparison prompt or criteria",
    )
    depth_mode: str = Field("standard", description="Comparison depth mode")

    @field_validator("symbols")
    @classmethod
    def validate_compare_symbols(cls, values: List[str]) -> List[str]:
        return _clean_symbol_list(values, minimum=2)


class CompareResponse(BaseModel):
    success: bool
    symbols: List[str]
    analysis: Optional[str] = None
    message: Optional[str] = None


class PreResearchRequest(BaseModel):
    exchange: str = Field(..., description="Exchange to research, such as BIST or NASDAQ")
    criteria: Optional[str] = Field(
        None,
        description="Optional filtering criteria or research angle",
    )
    depth_mode: str = Field(
        "standard",
        description="Research depth, usually standard or deep",
    )


class PreResearchResponse(BaseModel):
    task_id: str
    status: AnalysisStatus
    message: str = "Pre-research initiated"
    result: Optional[str] = Field(None, description="Generated research report in Markdown")


class TickerSearchResponse(BaseModel):
    query: str = Field(..., description="Search query")
    type_filter: Optional[str] = Field(
        None,
        description="Applied asset type filter",
    )
    results: Dict[str, Any] = Field(
        ...,
        description="Grouped lookup results such as quotes, ETFs, indices, or news",
    )
    total_results: int = Field(..., description="Total number of returned result items")


class KapListRequest(BaseModel):
    stock_codes: List[str] = Field(
        ...,
        description="Ticker symbols such as ['THYAO', 'AKBNK']",
    )
    category: Optional[str] = Field(
        default=None,
        description="KAP disclosure class such as FR, ODA, DG, or FON",
    )
    days: Optional[int] = Field(
        default=None,
        ge=1,
        description="Return disclosures from the last N days",
    )
    from_date: Optional[str] = Field(default=None, description="Lower bound in YYYY-MM-DD format")
    to_date: Optional[str] = Field(default=None, description="Upper bound in YYYY-MM-DD format")

    @field_validator("stock_codes")
    @classmethod
    def validate_stock_codes(cls, values: List[str]) -> List[str]:
        return _clean_symbol_list(values)


class KapBatchDetailRequest(BaseModel):
    disclosure_indexes: List[int] = Field(..., description="Disclosure index values")
    max_workers: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of parallel worker threads",
    )


class KapBatchDetailResponse(BaseModel):
    results: List[Optional[Any]]
    errors: Dict[int, str]


class FavoriteListResponse(BaseModel):
    favorites: List[str] = Field(default_factory=list, description="Normalized watchlist symbols")
    count: int = Field(..., description="Number of favorites")


class FavoriteMutationResponse(BaseModel):
    success: bool
    symbol: str
    message: str
    favorites: List[str] = Field(
        default_factory=list,
        description="Updated favorites list after the operation",
    )


class PortfolioPositionPayload(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    quantity: float = Field(..., gt=0, description="Position size")
    average_cost: float = Field(..., gt=0, description="Average entry price")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return clean_symbol(value)


class PortfolioPosition(BaseModel):
    symbol: str
    quantity: float
    average_cost: float


class PortfolioMutationResponse(BaseModel):
    success: bool
    message: str
    position: Optional[PortfolioPosition] = None


class PortfolioListResponse(BaseModel):
    positions: List[PortfolioPosition] = Field(default_factory=list)
    count: int


class PortfolioSnapshotPosition(BaseModel):
    symbol: str
    quantity: float
    average_cost: float
    current_price: Optional[float] = None
    cost_value: float
    market_value: float
    pnl: float
    pnl_pct: float
    price_unavailable: bool = False


class PortfolioSnapshotTotals(BaseModel):
    cost: float
    market_value: float
    pnl: float
    pnl_pct: float


class PortfolioSnapshotResponse(BaseModel):
    positions: List[PortfolioSnapshotPosition] = Field(default_factory=list)
    totals: PortfolioSnapshotTotals


class EventMessage(BaseModel):
    type: Optional[str] = None
    severity: Optional[str] = None
    symbol: Optional[str] = None
    message: str
    alert_id: Optional[str] = None


class PortfolioRiskPosition(PortfolioSnapshotPosition):
    weight_pct: float = 0.0
    sector: str = "Unknown"
    industry: str = "Unknown"
    confidence_pct: float = 0.0
    confidence_level: str = "very_low"
    data_quality: str = "unknown"


class PortfolioRiskSummary(BaseModel):
    position_count: int
    total_market_value: Optional[float] = None
    diversification_score: float
    average_confidence: float
    estimated_portfolio_volatility: Optional[float] = None
    risk_score: float


class PortfolioSectorExposure(BaseModel):
    sector: str
    weight_pct: float
    market_value: float


class PortfolioRiskSnapshotResponse(BaseModel):
    positions: List[PortfolioRiskPosition] = Field(default_factory=list)
    summary: PortfolioRiskSummary
    sector_exposure: List[PortfolioSectorExposure] = Field(default_factory=list)
    breaches: List[EventMessage] = Field(default_factory=list)
    correlation_alerts: List[EventMessage] = Field(default_factory=list)


class AlertCreateRequest(BaseModel):
    symbol: str = Field(..., description="Ticker symbol")
    target_price: float = Field(..., gt=0, description="Trigger price")
    direction: AlertDirectionEnum = Field(..., description="Alert trigger direction")
    note: str = Field(default="", description="Optional user note")

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        return clean_symbol(value)


class AlertRecord(BaseModel):
    id: Optional[str] = None
    type: str = "price"
    symbol: str
    target_price: float
    direction: str
    note: str = ""
    triggered: bool = False
    status: str = "active"
    created_at: Optional[str] = None
    triggered_at: Optional[str] = None
    current_price: Optional[float] = None


class AlertCreateResponse(BaseModel):
    message: str
    alert: AlertRecord


class AlertListResponse(BaseModel):
    count: int
    alerts: List[AlertRecord] = Field(default_factory=list)


class AlertCenterResponse(BaseModel):
    triggered_events: List[EventMessage] = Field(default_factory=list)
    price_events: List[EventMessage] = Field(default_factory=list)
    risk_events: List[EventMessage] = Field(default_factory=list)
    kap_events: List[EventMessage] = Field(default_factory=list)
    saved_alerts: AlertListResponse


class InvestorProfileUpdateRequest(BaseModel):
    profile_name: Optional[str] = Field(None, description="Display name for the investor profile")
    risk_tolerance: Optional[ProfileRiskToleranceEnum] = Field(
        None,
        description="Risk tolerance setting",
    )
    investment_horizon: Optional[ProfileInvestmentHorizonEnum] = Field(
        None,
        description="Preferred investment horizon",
    )
    market_focus: Optional[ProfileMarketFocusEnum] = Field(
        None,
        description="Primary market focus",
    )
    preferred_sectors: Optional[List[str]] = Field(
        None,
        description="Preferred sectors",
    )
    avoided_sectors: Optional[List[str]] = Field(
        None,
        description="Avoided sectors",
    )
    max_single_position_pct: Optional[float] = Field(
        None,
        ge=1,
        le=100,
        description="Maximum preferred single position size in percent",
    )
    alert_sensitivity: Optional[ProfileAlertSensitivityEnum] = Field(
        None,
        description="Alert sensitivity level",
    )
    active_playbook: Optional[str] = Field(
        None,
        description="Playbook key returned by /profile/playbooks",
    )

    @field_validator("profile_name")
    @classmethod
    def validate_profile_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("preferred_sectors", "avoided_sectors")
    @classmethod
    def validate_sector_lists(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        return _clean_text_list(value)

    @field_validator("active_playbook")
    @classmethod
    def validate_active_playbook(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class InvestorProfileResponse(BaseModel):
    profile_name: str
    risk_tolerance: str
    investment_horizon: str
    market_focus: str
    preferred_sectors: List[str] = Field(default_factory=list)
    avoided_sectors: List[str] = Field(default_factory=list)
    max_single_position_pct: float
    alert_sensitivity: str
    active_playbook: str
    playbook_summary: str


class PlaybookOptionResponse(BaseModel):
    key: str
    label: str
    summary: str


class PlaybookListResponse(BaseModel):
    playbooks: List[PlaybookOptionResponse] = Field(default_factory=list)


class HistoryListItem(BaseModel):
    id: int
    symbol: str
    type: str
    timestamp: str
    file_path: str


class HistoryListResponse(BaseModel):
    entries: List[HistoryListItem] = Field(default_factory=list)
    count: int


class HistoryEntryResponse(BaseModel):
    id: int
    symbol: str
    type: str
    data: Dict[str, Any]
    timestamp: str