"""
Domain Quality
==============

This module contains data quality assessment operations.
"""

from typing import Any, Dict

from core import get_standard_logger
from config import NA_VALUE
from .models import DataQuality

_log = get_standard_logger("domain.quality")

def assess_data_quality(data: Dict[str, Any]) -> DataQuality:
    """Assess data quality using dynamic completeness (market-aware, non-weighted)."""

    def _is_missing(value: Any) -> bool:
        if value is None or value == NA_VALUE:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        return False

    def _has_any(keys: list[str]) -> bool:
        return any(not _is_missing(data.get(key)) for key in keys)

    def _is_bist_asset() -> bool:
        symbol = str(data.get("symbol") or "").upper()
        exchange = str(data.get("exchange") or "").upper()
        return symbol.endswith(".IS") or exchange in {
            "BIST",
            "IST",
            "ISTANBUL",
            "XIST",
        }

    if not isinstance(data, dict) or not data:
        return DataQuality.MISSING

    if not _has_any(["currentPrice", "regularMarketPrice", "price"]):
        return DataQuality.MISSING

    # Use a narrower metric set for BIST assets because Yahoo data is sparse.
    if _is_bist_asset():
        candidate_fields = [
            "marketCap",
            "sector",
            "industry",
            "trailingPE",
            "priceToBook",
            "beta",
            "totalRevenue",
            "averageVolume",
            "regularMarketVolume",
            "lastEarningsDate",
        ]
    else:
        candidate_fields = [
            "marketCap",
            "sector",
            "industry",
            "trailingPE",
            "forwardPE",
            "priceToBook",
            "beta",
            "enterpriseValue",
            "totalRevenue",
            "averageVolume",
            "regularMarketVolume",
            "lastEarningsDate",
            "earningsGrowth",
            "revenueGrowth",
        ]

    available = sum(1 for field in candidate_fields if not _is_missing(data.get(field)))
    completeness = available / max(1, len(candidate_fields))

    if completeness >= 0.80:
        return DataQuality.EXCELLENT
    if completeness >= 0.55:
        return DataQuality.GOOD
    if completeness >= 0.30:
        return DataQuality.FAIR
    if completeness > 0:
        return DataQuality.POOR
    return DataQuality.MISSING
