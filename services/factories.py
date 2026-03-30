"""Service factory helpers with explicit dependency wiring.

This module provides factory functions for creating service instances with
proper configuration and dependency injection.

## Market Data Providers

The market data service now supports pluggable providers through the
MarketDataProvider interface. By default, YahooFinanceMarketDataProvider
is used, but additional providers can be implemented.

Example usage:
    # Get default yfinance provider
    service = get_market_data_service()

    # Get by explicit provider name
    service = get_market_data_service(provider="yfinance")
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, TYPE_CHECKING
import threading

if TYPE_CHECKING:
    from .finance import (
        MarketDataService,
        MarketDataProvider,
        DataValidationService,
        PortfolioRiskService,
    )
    from .ai import AIService
    from .kap_intelligence import KapIntelligenceService

from config import get_config

_rag_service_instance = None
_rag_service_lock = threading.Lock()
_shared_memory_pool_instance = None
_shared_memory_pool_lock = threading.Lock()


def _resolve_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Resolve configuration with fallback to global config."""
    return config or get_config()


@lru_cache(maxsize=1)
def _get_default_financial_service() -> "MarketDataService":
    from .finance import MarketDataService

    return MarketDataService(_resolve_config())


@lru_cache(maxsize=1)
def _get_default_portfolio_risk_service() -> "PortfolioRiskService":
    from .finance import PortfolioRiskService

    return PortfolioRiskService(_resolve_config())


@lru_cache(maxsize=1)
def _get_default_kap_intelligence_service() -> "KapIntelligenceService":
    from .kap_intelligence import KapIntelligenceService

    return KapIntelligenceService(_resolve_config())


@lru_cache(maxsize=1)
def _get_default_ai_service() -> "AIService":
    from .ai import AIService

    return AIService(_resolve_config())


def get_financial_service(
    config: Optional[Dict[str, Any]] = None,
) -> "MarketDataService":
    """
    Get the default financial/market data service (yfinance provider).

    Args:
        config: Optional configuration dictionary

    Returns:
        MarketDataService instance (yfinance provider)
    """
    if config is None:
        return _get_default_financial_service()

    from .finance import MarketDataService

    return MarketDataService(_resolve_config(config))


def get_market_data_provider(
    provider: str = "yfinance", config: Optional[Dict[str, Any]] = None
) -> "MarketDataProvider":
    """
    Factory function for market data providers.

    Args:
        provider: Provider name ("yfinance" is currently the only supported provider)
        config: Optional configuration dictionary

    Returns:
        MarketDataProvider instance

    Raises:
        ValueError: If provider name is not recognized

    Example:
        service = get_market_data_provider("yfinance")
        data = service.get_company_data("AAPL")
    """
    resolved = _resolve_config(config)

    if provider.lower() == "yfinance":
        from .finance import MarketDataService

        return MarketDataService(resolved)
    else:
        raise ValueError(
            f"Unknown market data provider: {provider!r}. "
            f"Supported providers: 'yfinance'"
        )


def get_data_validation_service() -> DataValidationService:
    """
    Get the data validation service instance.

    Returns:
        DataValidationService instance
    """
    from .finance import DataValidationService

    return DataValidationService()


def get_portfolio_risk_service(
    config: Optional[Dict[str, Any]] = None,
) -> "PortfolioRiskService":
    if config is None:
        return _get_default_portfolio_risk_service()

    from .finance import PortfolioRiskService

    return PortfolioRiskService(_resolve_config(config))


def get_kap_intelligence_service(config: Optional[Dict[str, Any]] = None) -> "KapIntelligenceService":
    if config is None:
        return _get_default_kap_intelligence_service()

    from .kap_intelligence import KapIntelligenceService

    return KapIntelligenceService(_resolve_config(config))


def get_investor_profile_context(extra_context: Optional[str] = None) -> str:
    from .investor_profile import build_investor_context

    return build_investor_context(extra_context)


def get_ai_service(config: Optional[Dict[str, Any]] = None) -> "AIService":
    """
    Get the AI service instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        AIService instance
    """
    if config is None:
        return _get_default_ai_service()

    from .ai import AIService

    return AIService(_resolve_config(config))


def get_rag_service(config: Optional[Dict[str, Any]] = None):
    """Get singleton RAG memory service instance (thread-safe)."""
    global _rag_service_instance
    if _rag_service_instance is None:
        with _rag_service_lock:
            if _rag_service_instance is None:
                from .ai.memory import RAGMemory

                resolved = _resolve_config(config)
                _rag_service_instance = RAGMemory(resolved)
    return _rag_service_instance


def get_shared_memory_pool(config: Optional[Dict[str, Any]] = None):
    """Get singleton shared memory pool instance (thread-safe)."""
    global _shared_memory_pool_instance
    if _shared_memory_pool_instance is None:
        with _shared_memory_pool_lock:
            if _shared_memory_pool_instance is None:
                from .ai.shared_memory_pool import SharedMemoryPool

                resolved = _resolve_config(config)
                wm_cfg = resolved.get("ai", {}).get("working_memory", {})
                similarity_threshold = float(
                    wm_cfg.get("shared_pool_similarity_threshold", 0.6) or 0.6
                )
                _shared_memory_pool_instance = SharedMemoryPool(
                    similarity_threshold=similarity_threshold
                )
    return _shared_memory_pool_instance


__all__ = [
    "get_financial_service",
    "get_market_data_provider",
    "get_data_validation_service",
    "get_portfolio_risk_service",
    "get_kap_intelligence_service",
    "get_investor_profile_context",
    "get_ai_service",
    "get_rag_service",
    "get_shared_memory_pool",
]
