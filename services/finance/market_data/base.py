"""Base MarketDataService components."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from core import get_standard_logger

from ...network import ProxyManager, PROXY_STEP_MARKET_DATA
from ..validation import DataValidationService

# Note: External library loggers respect LOGGING_FILE_LEVEL and LOGGING_CONSOLE_LEVEL
# from .env configuration. Respecting user logging preferences is critical.


class MarketDataProvider(ABC):
    """Abstract base contract for market data providers."""

    @abstractmethod
    def get_company_data(
        self,
        ticker: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return company data for the given ticker or ``None`` on failure."""

    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """Return ``True`` when the provider recognises the ticker."""


class MarketDataBase(MarketDataProvider):
    """Basic infrastructure and configuration for MarketDataService."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.logger = get_standard_logger(__name__)
        # Log level is configured from LOGGING_CONSOLE_LEVEL and LOGGING_FILE_LEVEL in .env

        self.validation_service = DataValidationService()
        self.market_config = (self.config.get("market_data") or {}).copy()
        self.market_config.setdefault("api_max_retries", 3)
        self.market_config.setdefault("api_retry_delay", 0.75)
        self.market_config.setdefault("api_max_backoff_delay", 10.0)
        self.market_config.setdefault("api_total_timeout", 30.0)
        self.market_config.setdefault("api_circuit_breaker_threshold", 5)
        self.market_config.setdefault("api_circuit_breaker_timeout", 60.0)
        self.market_config.setdefault("api_request_timeout", 10.0)
        self.market_config.setdefault("history_repair_enabled", True)
        self.market_config.setdefault("history_repair_non_us_only", True)
        self.market_config.setdefault("yfinance_news_limit", 5)
        self.market_config.setdefault("prefer_proxy_for_yfinance", False)

        proxy_config: Dict[str, Any] = {}
        for candidate in (self.market_config.get("proxy"), self.config.get("proxy")):
            if isinstance(candidate, dict):
                proxy_config = candidate.copy()
                break
        self.proxy_manager = ProxyManager(proxy_config, step=PROXY_STEP_MARKET_DATA)

        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure: Optional[float] = None

    # ------------------------------------------------------------------
    # General utilities
    # ------------------------------------------------------------------

    def _call_with_retry(self, func, *args, name=None, **kwargs):
        """Wrap API calls with retry and circuit breaker policies."""
        circuit_threshold = int(
            self.market_config.get("api_circuit_breaker_threshold", 5)
        )
        circuit_timeout = float(
            self.market_config.get("api_circuit_breaker_timeout", 60.0)
        )

        half_open = False
        if self._circuit_breaker_failures >= circuit_threshold:
            if self._circuit_breaker_last_failure:
                elapsed = time.time() - self._circuit_breaker_last_failure
                if elapsed < circuit_timeout:
                    raise Exception(
                        "Circuit breaker active. Too many failures (%d). Reset in %.1f seconds."
                        % (
                            self._circuit_breaker_failures,
                            circuit_timeout - elapsed,
                        )
                    )

                half_open = True
                self.logger.info(
                    "Circuit breaker half-open: allowing limited retry after %.1f seconds.",
                    elapsed,
                )
            else:
                half_open = True

        retries = max(1, int(self.market_config.get("api_max_retries", 3)))
        delay = float(self.market_config.get("api_retry_delay", 0.75))
        max_backoff_delay = float(self.market_config.get("api_max_backoff_delay", 10.0))
        total_timeout = float(self.market_config.get("api_total_timeout", 30.0))

        start_time = time.time()
        last_exc: Optional[Exception] = None

        max_attempts = 1 if half_open else retries

        # Debug logging - only when needed
        func_name = name or getattr(func, "__name__", str(func))
        self.logger.debug(f"[RETRY] Calling {func_name} | attempts: {max_attempts}")

        for attempt in range(max_attempts):
            elapsed = time.time() - start_time
            if elapsed >= total_timeout:
                raise TimeoutError(
                    f"Total timeout exceeded ({total_timeout}s) after {attempt + 1} attempts."
                )

            # Sadece retry durumunda log
            if attempt > 0:
                self.logger.info(
                    f"🔄 [RETRY] {func_name} attempt {attempt + 1}/{max_attempts}"
                )

            try:
                result = func(*args, **kwargs)
                # Success log removed - creates too much spam
                # self.logger.debug(f"[RETRY] {func_name} successful")

                if self._circuit_breaker_failures > 0:
                    self._circuit_breaker_failures = 0
                    self._circuit_breaker_last_failure = None
                    self.logger.debug("Circuit breaker failures reset on success.")
                    if half_open:
                        self.logger.info(
                            "Circuit breaker closed after successful half-open attempt."
                        )
                return result
            except Exception as exc:
                last_exc = exc
                now = time.time()
                self._circuit_breaker_last_failure = now

                # Hata tespiti
                exc_type = type(exc).__name__
                exc_msg = str(exc)[:150]

                # ⚠️ YFinance-specific errors - these are normal situations, don't retry
                exc_msg_lower = exc_msg.lower()
                is_yfinance_warning = (
                    "possibly delisted" in exc_msg_lower
                    or "no price data found" in exc_msg_lower
                    or "no data found" in exc_msg_lower
                    or "no fundamentals data found" in exc_msg_lower
                    or "yahoo web request" in exc_msg_lower
                    or "http error 404" in exc_msg_lower
                    or "quotesummary" in exc_msg_lower
                    or "not found" in exc_msg_lower
                )

                if is_yfinance_warning:
                    # This is normal - indicates Yahoo Finance API cannot provide data
                    # Reset circuit breaker and return None silently
                    if self._circuit_breaker_failures > 0:
                        self._circuit_breaker_failures = 0
                        self._circuit_breaker_last_failure = None
                    return None

                # ⚠️ Proxy/Network errors - special handling, don't trigger circuit breaker
                is_proxy_error = (
                    "ProxyError" in exc_type
                    or "CONNECT tunnel failed" in exc_msg
                    or "Failed to connect" in exc_msg
                    or "Connection refused" in exc_msg
                    or "response 500" in exc_msg
                    or "response 502" in exc_msg
                    or "response 503" in exc_msg
                    or "Could not connect to server" in exc_msg
                    or "curl" in exc_msg.lower()
                    or "tls" in exc_msg.lower()
                    or "ssl" in exc_msg.lower()
                )

                if is_proxy_error:
                    # Log only on first error
                    if attempt == 0:
                        self.logger.warning(f"⚠️ [PROXY] {exc_type}: {exc_msg}")
                    # Don't increment circuit breaker on proxy error
                    # Just raise exception and retry
                    if attempt < max_attempts - 1:
                        sleep_time = min(delay * (2**attempt), max_backoff_delay)
                        remaining_time = total_timeout - elapsed
                        if sleep_time > remaining_time:
                            raise TimeoutError(
                                f"Next retry would exceed total timeout. Remaining: {remaining_time:.1f}s"
                            )
                        self.logger.debug(
                            "Proxy error - will retry in %.2f seconds (attempt %d/%d)",
                            sleep_time,
                            attempt + 1,
                            max_attempts,
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        # Final attempt also failed - but don't trigger circuit breaker
                        self.logger.debug(
                            "[PROXY] All attempts exhausted, circuit breaker skipped"
                        )
                        raise

                if half_open:
                    self._circuit_breaker_failures = circuit_threshold
                    self.logger.error(
                        "Circuit breaker half-open attempt failed: %s. Circuit remains open.",
                        exc,
                    )
                    raise

                self._circuit_breaker_failures += 1

                if attempt == max_attempts - 1:
                    self.logger.error(
                        "All retry attempts failed for %s. Circuit breaker failures: %d | Error: %s",
                        func_name,
                        self._circuit_breaker_failures,
                        exc_msg,
                    )
                    raise

                sleep_time = min(delay * (2**attempt), max_backoff_delay)
                remaining_time = total_timeout - elapsed
                if sleep_time > remaining_time:
                    raise TimeoutError(
                        f"Next retry would exceed total timeout. Remaining: {remaining_time:.1f}s"
                    )

                self.logger.debug(
                    "API call failed (%s). Will retry in %.2f seconds (attempt %d/%d)",
                    exc,
                    sleep_time,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(sleep_time)

        if last_exc:
            raise last_exc
