"""
Service Utilities - Configuration and Validation Helpers

This module contains infrastructure-specific helper functions for the service layer.
Domain layer functions are re-exported from domain.utils.

KISS principle: Only functions that are actually used here.
YAGNI principle: safe_str was removed - it wasn't used.
"""

import re
from typing import Dict, Any

from core import get_standard_logger
from config import get_config

# BACKWARD COMPATIBILITY: Re-export from Domain (only used ones)
from domain.utils import (
    safe_float,
    safe_int,
    safe_numeric,
    compute_comparative_score,
    apply_fallback_values,
    format_currency,
    quality_from_score,
    make_json_serializable,
    COMPARISON_METRICS,
)

_log = get_standard_logger("services.utils.calculations")


_SYMBOL_PATTERN = re.compile(r"^[A-Za-z0-9.-]+$")


def get_file_constants() -> Dict[str, Any]:
    """Gets required file management constants from config."""
    from config import CACHE_FILE

    try:
        config = get_config()
        files_config = config.get("files", {})

        market_data_config = config.get("market_data", {})
        cache_ttl_hours = market_data_config.get("default_cache_ttl_hours", 24)
        cache_duration_seconds = cache_ttl_hours * 3600

        return {
            "cache_file": files_config.get("cache_file", CACHE_FILE),
            "json_indent": int(files_config.get("json_indent", 2)),
            "cache_duration": cache_duration_seconds,
        }
    except Exception as e:
        _log.warning("File constants config could not be read, loading defaults: %s", e)
        return {"cache_file": CACHE_FILE, "json_indent": 2, "cache_duration": 86400}


def validate_symbol(symbol: str) -> bool:
    """Return True if the given ticker symbol matches allowed characters."""
    if not symbol or not isinstance(symbol, str):
        return False

    cleaned = symbol.strip().upper()
    if not cleaned or len(cleaned) > 15:
        return False

    return bool(_SYMBOL_PATTERN.match(cleaned))


__all__ = [
    # Re-exported from domain.utils
    "safe_float",
    "safe_int",
    "safe_numeric",
    "compute_comparative_score",
    "apply_fallback_values",
    "format_currency",
    "quality_from_score",
    "make_json_serializable",
    "COMPARISON_METRICS",
    # Local functions
    "get_file_constants",
    "validate_symbol",
]
