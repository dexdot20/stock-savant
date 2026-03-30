"""
Domain Layer - Shared Utilities
================================

This module contains common utility functions used in the domain layer.
It is independent from the services layer and follows Hexagonal Architecture principles.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math

from config import LOGISTIC_SCALING_FACTOR, NA_VALUE
from core import get_standard_logger

_log = get_standard_logger("domain.utils")


def safe_numeric(value: Any, type_func: Callable, default: Any = None) -> Any:
    """Generic safe conversion helper."""
    if value is None:
        return default

    if isinstance(value, str):
        value = value.strip()
        if not value or value.upper() in ("N/A", "NA", "NULL", "NONE"):
            return default

    try:
        result = type_func(value)
        if type_func is float and not math.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Safely convert a value to float."""
    return safe_numeric(value, float, default)


def safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    """Safely convert a value to int."""
    res = safe_numeric(value, int)
    if res is not None:
        return res

    try:
        f_val = safe_numeric(value, float)
        return int(f_val) if f_val is not None else default
    except (TypeError, ValueError):
        return default


def safe_int_strict(value: Any, default: int = 0) -> int:
    """Safely convert a value to int and always return an int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_symbol(
    value: Any,
    *,
    drop_suffix_after_dot: bool = False,
) -> str:
    """Normalize symbols for cross-service comparisons and storage."""
    normalized = str(value or "").strip().upper()
    if drop_suffix_after_dot and "." in normalized:
        normalized = normalized.split(".", 1)[0]
    return normalized


def utc_now_iso() -> str:
    """Return a timezone-aware ISO-8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


def make_json_serializable(obj: Any) -> Any:
    """Convert an object into a JSON-serializable structure."""
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (datetime, date)):
        return obj.isoformat()

    if isinstance(obj, Decimal):
        return float(obj)

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]

    if isinstance(obj, set):
        return list(obj)

    if hasattr(obj, "__dict__"):
        return make_json_serializable(obj.__dict__)

    try:
        return str(obj)
    except Exception:
        return "<Unserializable Object>"


COMPARISON_METRICS: List[Tuple[str, str]] = [
    ("trailingPE", "P/E (Trailing)"),
    ("forwardPE", "P/E (Forward)"),
    ("priceToBook", "P/B"),
    ("beta", "Beta"),
    ("debtToEquity", "Debt/Equity"),
    ("marketCap", "Market Cap"),
]


def compute_comparative_score(
    company_info: Dict[str, Any], all_companies: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute a sector-comparison score for a company."""
    try:
        sector = company_info.get("sector")
        if not sector:
            return {"error": "Sector information not found"}

        sector_companies = []
        for symbol, company in all_companies.items():
            if company.get("sector") == sector and symbol != company_info.get("symbol"):
                sector_companies.append(company)

        if len(sector_companies) < 3:
            return {
                "error": f"At least 3 companies are required for a meaningful sector comparison (current: {len(sector_companies)})"
            }

        metric_results = []
        total_score = 0

        for metric_name, metric_label in COMPARISON_METRICS:
            company_value = company_info.get(metric_name)
            if company_value is None or company_value == NA_VALUE:
                continue

            try:
                company_value_float = float(company_value)

                sector_values = []
                for company in sector_companies:
                    value = company.get(metric_name)
                    if value is not None and value != NA_VALUE:
                        try:
                            sector_values.append(float(value))
                        except (ValueError, TypeError):
                            continue

                if len(sector_values) < 2:
                    continue

                sector_avg = sum(sector_values) / len(sector_values)
                sector_std = math.sqrt(
                    sum((x - sector_avg) ** 2 for x in sector_values)
                    / len(sector_values)
                )

                if sector_std == 0:
                    continue

                z_score = (company_value_float - sector_avg) / sector_std
                scaled_score = 50 + (50 * math.tanh(z_score / LOGISTIC_SCALING_FACTOR))

                better = None
                if metric_name in [
                    "trailingPE",
                    "forwardPE",
                    "priceToBook",
                    "beta",
                    "debtToEquity",
                ]:
                    better = company_value_float < sector_avg
                elif metric_name == "marketCap":
                    better = company_value_float > sector_avg

                metric_results.append(
                    {
                        "metric": metric_name,
                        "label": metric_label,
                        "value": company_value_float,
                        "sector_avg": sector_avg,
                        "zscore": z_score,
                        "better": better,
                        "score": scaled_score,
                    }
                )

                total_score += scaled_score

            except (ValueError, TypeError) as exc:
                _log.debug("Metric calculation failed for %s: %s", metric_name, exc)
                continue

        if not metric_results:
            return {"error": "No metrics could be calculated"}

        overall_score = total_score / len(metric_results)

        return {
            "overall_score": round(overall_score, 1),
            "sector_company_count": len(sector_companies),
            "metric_results": metric_results,
            "sector": sector,
        }

    except Exception as exc:
        _log.error("Comparative score calculation failed: %s", exc)
        return {"error": f"Calculation error: {exc}"}


def apply_fallback_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """Apply fallback values for missing fields."""
    basic_fallback_values = {
        "marketCap": NA_VALUE,
        "regularMarketPrice": NA_VALUE,
        "trailingPE": NA_VALUE,
        "priceToBook": NA_VALUE,
        "priceToSales": NA_VALUE,
        "beta": NA_VALUE,
        "dividendYield": NA_VALUE,
        "fiftyTwoWeekHigh": NA_VALUE,
        "fiftyTwoWeekLow": NA_VALUE,
    }

    for key, fallback_value in basic_fallback_values.items():
        if key not in data or data[key] is None:
            data[key] = fallback_value

    if data.get("marketCap") in (NA_VALUE, None):
        for candidate in (
            data.get("mktCap"),
            data.get("marketCapitalization"),
            data.get("market_cap"),
        ):
            if candidate not in (None, NA_VALUE, ""):
                data["marketCap"] = candidate
                break

    return data


def format_currency(value: Union[int, float, str], currency: str = "USD") -> str:
    """Format a currency value."""
    if value is None or value == NA_VALUE:
        return NA_VALUE

    try:
        num_value = float(value)

        if num_value >= 1e12:
            return f"{num_value/1e12:.2f} T{currency}"
        if num_value >= 1e9:
            return f"{num_value/1e9:.2f} B{currency}"
        if num_value >= 1e6:
            return f"{num_value/1e6:.2f} M{currency}"
        if num_value >= 1e3:
            return f"{num_value/1e3:.2f} K{currency}"
        return f"{num_value:,.0f} {currency}"
    except (ValueError, TypeError):
        return str(value)


def get_currency_symbol(currency: Optional[str]) -> str:
    """Get the currency symbol from a code."""
    if not currency:
        return "$"

    symbols = {
        "TRY": "₺",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥",
    }
    return symbols.get(currency.upper(), f"{currency} ")


def quality_from_score(score: Any) -> str:
    """Normalize quality labels from a numerical score."""
    try:
        value = float(score)
    except (TypeError, ValueError):
        value = 0.0

    if value >= 80:
        return "excellent"
    if value >= 60:
        return "good"
    if value >= 40:
        return "fair"
    if value >= 20:
        return "poor"
    return "missing"


def get_ai_company_context(company_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a lightweight company context for AI providers."""
    if not isinstance(company_data, dict):
        return {}

    return {
        "symbol": company_data.get("symbol", "N/A"),
        "longName": company_data.get("longName", "N/A"),
        "shortName": company_data.get("shortName", ""),
        "sector": company_data.get("sector", "N/A"),
        "industry": company_data.get("industry", "N/A"),
        "country": company_data.get("country", "N/A"),
        "currency": company_data.get("currency", "USD"),
        "exchange": company_data.get("exchange", ""),
        "quoteType": company_data.get("quoteType", ""),
    }


__all__ = [
    "safe_float",
    "safe_int",
    "safe_numeric",
    "compute_comparative_score",
    "get_ai_company_context",
    "apply_fallback_values",
    "format_currency",
    "get_currency_symbol",
    "quality_from_score",
    "make_json_serializable",
    "COMPARISON_METRICS",
]