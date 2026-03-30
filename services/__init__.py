"""Public interface for the services package.

Services are organized in sub-packages:
- services.ai: AI provider integrations
- services.finance: Financial services
- services.network: Network utilities
- services.utils: Helper functions

Usage (recommended):
    from services.finance import MarketDataService
    from services.ai import AIService
    from services.network import HTTPClient
"""

# Import from new structure
from .utils import (
    COMPARISON_METRICS,
    apply_fallback_values,
    compute_comparative_score,
    format_currency,
    get_file_constants,
    quality_from_score,
    safe_float,
    safe_int,
    safe_numeric,
)
from .finance import DataValidationService, MarketDataProvider, MarketDataService

from .factories import (
    get_ai_service,
    get_data_validation_service,
    get_financial_service,
    get_investor_profile_context,
    get_kap_intelligence_service,
)

__all__ = [
    # Utils
    "COMPARISON_METRICS",
    "apply_fallback_values",
    "compute_comparative_score",
    "format_currency",
    "get_file_constants",
    "quality_from_score",
    "safe_float",
    "safe_int",
    "safe_numeric",
    # Finance
    "DataValidationService",
    "MarketDataService",
    "MarketDataProvider",
    # Factories
    "get_financial_service",
    "get_data_validation_service",
    "get_ai_service",
    "get_investor_profile_context",
    "get_kap_intelligence_service",
]
