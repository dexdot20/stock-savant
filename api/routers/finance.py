"""
Financial Data Query Router — Modular category endpoints
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Path, Query, Depends
from fastapi.concurrency import run_in_threadpool
from services.factories import get_financial_service
from api.models import (
    CategoryDataResponse,
    MarketRegionEnum,
    PeriodEnum,
    IntervalEnum,
    ScreenerRequest,
    TypeFilterEnum,
    TickerSearchResponse,
    clean_symbol,
)
from domain.data_processors import sanitize_raw_company_data
from core import get_standard_logger

router = APIRouter(prefix="/finance", tags=["Finance"])
logger = get_standard_logger(__name__)


def valid_symbol_dependency(
    symbol: str = Path(..., description="Financial symbol (stock/index/ETF)")
) -> str:
    """Dependency for symbol validation and cleanup."""
    try:
        return clean_symbol(symbol)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def financial_service_dependency():
    """Dependency for the financial service."""
    return get_financial_service()


def _require_data(data, symbol: str, category: str):
    """Raise 404 when no data is available."""
    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No '{category}' data found for symbol {symbol}.",
        )
    return data


def _require_index_asset(data, symbol: str):
    """Raise 422 if the data is not an index/ETF asset."""
    if not data.get("isIndexAsset"):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Symbol {symbol} was not recognized as an index/ETF. "
                "Use /finance/{symbol}/company for company data."
            ),
        )
    return data


async def _run_blocking(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Offload blocking finance service calls out of the event loop."""
    return await run_in_threadpool(func, *args, **kwargs)


# ---------------------------------------------------------------------------
# GET /search  — ticker / ETF / index / crypto search  (/{symbol}/... must come first)
# ---------------------------------------------------------------------------


@router.get(
    "/search",
    response_model=TickerSearchResponse,
    summary="Ticker Search",
    description=(
        "Search stocks, ETFs, indexes, or crypto. "
        "If `type_filter` is omitted, a general search (quotes) is performed; "
        "if provided, search only the specified asset type (lookup)."
    ),
    responses={
        400: {"description": "Empty query."},
        500: {"description": "Server error."},
    },
)
async def search_ticker(
    query: str = Query(
        ..., min_length=1, description="Search term (symbol or company name)"
    ),
    type_filter: TypeFilterEnum = Query(None, description="Asset type filter"),
    max_results: int = Query(10, ge=1, le=50, description="Maximum number of results"),
    news_results: int = Query(
        0, ge=0, le=10, description="Number of news results (general search only)"
    ),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: search — query=%s type_filter=%s", query, type_filter)
    if type_filter is not None:
        results = await _run_blocking(
            service.lookup_ticker,
            query,
            count=max_results,
            asset_type=type_filter.value,
        )
    else:
        results = await _run_blocking(
            service.search_ticker,
            query,
            max_results=max_results,
            news_count=news_results,
        )
    if not results:
        results = {}
    # total_results: sum all lists
    total = sum(len(v) for v in results.values() if isinstance(v, list))
    return TickerSearchResponse(
        query=query,
        type_filter=type_filter.value if type_filter else None,
        results=results,
        total_results=total,
    )


@router.get(
    "/market/summary",
    summary="Market Summary",
    description="Return yfinance market summary data for a specific region.",
)
async def get_market_summary(
    region: MarketRegionEnum = Query(MarketRegionEnum.US, description="Market region"),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: market summary — %s", region.value)
    data = await _run_blocking(service.get_market_summary, region.value)
    return _require_data(data, region.value, "market_summary")


@router.post(
    "/screener",
    summary="Screener Query",
    description="Run a yfinance equity/fund screener query.",
)
async def screen_tickers(
    payload: ScreenerRequest,
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: screener — %s", payload.query_type.value)
    data = await _run_blocking(
        service.screen_tickers,
        query_type=payload.query_type.value,
        filters={
            "conditions": [
                (item.operator, item.field, item.value) for item in payload.conditions
            ]
        },
    )
    return {
        "query_type": payload.query_type.value,
        "conditions": [item.model_dump() for item in payload.conditions],
        "results": data,
        "count": len(data or []),
    }


# ---------------------------------------------------------------------------
# GET /{symbol}/company  — company-focused wide data package
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/company",
    response_model=CategoryDataResponse,
    summary="Company Data (Wide Package)",
    description=(
        "Return a broad company-focused data set for the symbol. "
        "It may include identity, price, core metrics, dividends/splits, trend, analyst, and ownership fields."
    ),
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        500: {"description": "Server error."},
    },
)
async def get_company_data(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: company — %s", symbol)
    data = await _run_blocking(service.get_company_data, symbol)
    return CategoryDataResponse(
        symbol=symbol, category="company", data=_require_data(data, symbol, "company")
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/index  — index/ETF-focused data package
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/index",
    response_model=CategoryDataResponse,
    summary="Index / ETF Data",
    description=(
        "Return an index/ETF-focused data set for the symbol. "
        "Includes current price, daily change, period returns, trend summary, and components if available."
    ),
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        422: {"description": "Symbol is not an index/ETF."},
        500: {"description": "Server error."},
    },
)
async def get_index_data(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: index — %s", symbol)
    data = await _run_blocking(service.get_index_data, symbol)
    data = _require_data(data, symbol, "index")
    return CategoryDataResponse(
        symbol=symbol, category="index", data=_require_index_asset(data, symbol)
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/overview  — identity, current price, and core metrics
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/overview",
    response_model=CategoryDataResponse,
    summary="Overview",
    description=(
        "Return the symbol's core identity, current price, market cap, "
        "P/E, margins, and other core metrics. "
        "Heavy categories (historical price, analyst, ownership) are excluded."
    ),
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        500: {"description": "Server error."},
    },
)
async def get_overview(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: overview — %s", symbol)
    data = await _run_blocking(service.get_overview, symbol)
    return CategoryDataResponse(
        symbol=symbol, category="overview", data=_require_data(data, symbol, "overview")
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/price  — price history + technical indicators
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/price",
    response_model=CategoryDataResponse,
    summary="Price History",
    description=(
        "Return OHLCV candle data, MA50/MA200, and RSI values for the specified period and interval. "
        "If no parameters are provided, period=1y and interval=1d are used by default."
    ),
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        422: {"description": "Invalid period/interval value."},
        500: {"description": "Server error."},
    },
)
async def get_price_history(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
    period: PeriodEnum = Query(
        PeriodEnum.ONE_YEAR,
        description="Data period — yfinance supports: 1d 5d 1mo 3mo 6mo 1y 2y 5y ytd max",
    ),
    interval: IntervalEnum = Query(
        IntervalEnum.ONE_DAY,
        description="Data interval — yfinance supports: 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo 3mo",
    ),
):
    logger.info(
        "API Request: price — %s | period=%s interval=%s",
        symbol,
        period.value,
        interval.value,
    )
    data = await _run_blocking(
        service.get_price_history,
        symbol,
        period=period.value,
        interval=interval.value,
    )
    return CategoryDataResponse(
        symbol=symbol, category="price", data=_require_data(data, symbol, "price")
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/dividends  — dividends, splits, and corporate actions
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/dividends",
    response_model=CategoryDataResponse,
    summary="Dividend and Split History",
    description="Return dividend payments, stock splits, and corporate action history.",
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        500: {"description": "Server error."},
    },
)
async def get_dividends(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: dividends — %s", symbol)
    data = await _run_blocking(service.get_dividends, symbol)
    return CategoryDataResponse(
        symbol=symbol,
        category="dividends",
        data=_require_data(data, symbol, "dividends"),
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/analyst  — analyst recommendations and forecasts
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/analyst",
    response_model=CategoryDataResponse,
    summary="Analyst Recommendations and Forecasts",
    description=(
        "Return analyst buy/sell recommendations, price targets, EPS revisions, "
        "growth forecasts, and earnings estimates."
    ),
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        500: {"description": "Server error."},
    },
)
async def get_analyst(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: analyst — %s", symbol)
    data = await _run_blocking(service.get_analyst, symbol)
    return CategoryDataResponse(
        symbol=symbol, category="analyst", data=_require_data(data, symbol, "analyst")
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/earnings  — earnings history and dates
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/earnings",
    response_model=CategoryDataResponse,
    summary="Earnings History and Dates",
    description="Return quarterly earnings trends, EPS history, and earnings announcement dates.",
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        500: {"description": "Server error."},
    },
)
async def get_earnings(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: earnings — %s", symbol)
    data = await _run_blocking(service.get_earnings, symbol)
    return CategoryDataResponse(
        symbol=symbol, category="earnings", data=_require_data(data, symbol, "earnings")
    )


@router.get(
    "/{symbol}/financials",
    response_model=CategoryDataResponse,
    summary="Financial Statements",
    description="Return a summary of the income statement, balance sheet, cash flow, and SEC filings.",
)
async def get_financials(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: financials — %s", symbol)
    data = await _run_blocking(service.get_financial_statements, symbol)
    return CategoryDataResponse(
        symbol=symbol,
        category="financials",
        data=_require_data(data, symbol, "financials"),
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/ownership  — institutional ownership and insider transactions
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/ownership",
    response_model=CategoryDataResponse,
    summary="Institutional Ownership and Insider Transactions",
    description="Return institutional ownership structure, major holders, and insider trading analysis.",
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        500: {"description": "Server error."},
    },
)
async def get_ownership(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: ownership — %s", symbol)
    data = await _run_blocking(service.get_ownership, symbol)
    return CategoryDataResponse(
        symbol=symbol,
        category="ownership",
        data=_require_data(data, symbol, "ownership"),
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/sustainability  — ESG / sustainability
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/sustainability",
    response_model=CategoryDataResponse,
    summary="ESG Sustainability Scores",
    description="Return environmental (E), social (S), and governance (G) scores and sub-metrics.",
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        500: {"description": "Server error."},
    },
)
async def get_sustainability(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: sustainability — %s", symbol)
    data = await _run_blocking(service.get_sustainability, symbol)
    return CategoryDataResponse(
        symbol=symbol,
        category="sustainability",
        data=_require_data(data, symbol, "sustainability"),
    )


@router.get(
    "/{symbol}/news",
    response_model=CategoryDataResponse,
    summary="Ticker News",
    description="Return the latest yfinance ticker-based news summary.",
)
async def get_news(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: news — %s", symbol)
    data = await _run_blocking(service.get_news, symbol)
    return CategoryDataResponse(
        symbol=symbol,
        category="news",
        data=_require_data(data, symbol, "news"),
    )


# ---------------------------------------------------------------------------
# GET /{symbol}/raw  — raw company data
# ---------------------------------------------------------------------------


@router.get(
    "/{symbol}/raw",
    response_model=CategoryDataResponse,
    summary="Raw Company Data",
    description=(
        "Return all available raw yfinance data in sanitized JSON format. "
        "Use it for full data access during analysis or debugging."
    ),
    responses={
        400: {"description": "Invalid symbol."},
        404: {"description": "No data found."},
        500: {"description": "Server error."},
    },
)
async def get_raw(
    symbol: str = Depends(valid_symbol_dependency),
    service=Depends(financial_service_dependency),
):
    logger.info("API Request: raw — %s", symbol)
    raw_fn = getattr(service, "get_raw_company_data", None) or service.get_company_data
    data = await _run_blocking(raw_fn, symbol)
    if data:
        data = sanitize_raw_company_data(data)
    return CategoryDataResponse(
        symbol=symbol, category="raw", data=_require_data(data, symbol, "raw")
    )
