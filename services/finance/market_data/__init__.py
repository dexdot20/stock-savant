"""Market data servis paketi."""

from __future__ import annotations

import copy
import threading
from typing import Any, Dict, Iterable, List, Optional

from diskcache import Cache
import yfinance as yf

from config import CACHE_FILE, NA_VALUE
from domain.confidence import auto_complete_company_data, calculate_company_confidence
from domain.utils import quality_from_score

from ...utils import apply_fallback_values
from .base import MarketDataBase, MarketDataProvider
from .yfinance_adapter import YFinanceAdapterMixin


class MarketDataService(YFinanceAdapterMixin, MarketDataBase):
    """Service providing live data from Yahoo Finance data provider."""

    _disk_cache: Optional[Cache] = None
    _disk_cache_lock = threading.Lock()

    @classmethod
    def _get_disk_cache(cls) -> Optional[Cache]:
        if cls._disk_cache is not None:
            return cls._disk_cache
        with cls._disk_cache_lock:
            if cls._disk_cache is None:
                try:
                    cls._disk_cache = Cache(CACHE_FILE)
                except Exception:
                    cls._disk_cache = None
        return cls._disk_cache

    def _is_market_cache_enabled(self) -> bool:
        return bool(self.config.get("cache_enabled", True))

    def _market_cache_ttl_seconds(self) -> int:
        hours = (self.market_config or {}).get("default_cache_ttl_hours", 24)
        try:
            ttl = int(float(hours) * 3600)
        except (TypeError, ValueError):
            ttl = 24 * 3600
        return max(60, ttl)

    def _market_cache_key(self, ticker: str) -> str:
        return f"company_data:v3:{ticker}"

    def _normalize_tickers(self, tickers: Iterable[str]) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        for ticker in tickers or []:
            symbol = self._normalize_ticker(str(ticker))
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            cleaned.append(symbol)
        return cleaned

    def get_company_data(
        self,
        ticker: str,
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            self.logger.error("Invalid symbol parameter: %s", ticker)
            return None

        cache_enabled = self._is_market_cache_enabled()
        cache = self._get_disk_cache() if cache_enabled else None
        cache_key = self._market_cache_key(cleaned)
        if cache_enabled and cache is not None:
            cached = cache.get(cache_key)
            if isinstance(cached, dict):
                self.logger.debug(
                    "🔍 get_company_data(%s): user_context=%s | cache_hit",
                    cleaned,
                    user_context,
                )
                return copy.deepcopy(cached)

        self.logger.debug(
            "🔍 get_company_data(%s): user_context=%s | cache_%s",
            cleaned,
            user_context,
            "enabled" if cache_enabled else "disabled",
        )

        try:
            raw = self._fetch_from_yfinance(cleaned)
            if not raw:
                self.logger.debug("🔍 %s: yfinance data not found", cleaned)
                return None

            prepared = apply_fallback_values(raw)
            completion_report = auto_complete_company_data(prepared)
            validated = self.validation_service.validate_financial_data(prepared)

            if not self.validation_service.has_minimum_required_data(validated):
                self.logger.warning("Insufficient data - ticker skipped: %s", cleaned)
                return None

            # Risk score has been removed - now calculated by AI
            quality_report = self.validation_service.get_data_quality_score(validated)
            confidence_report = calculate_company_confidence(
                validated,
                quality_report=quality_report,
                completion_report=completion_report,
            )
            validated["data_quality"] = quality_from_score(
                quality_report.get("overall_score", 0)
            )
            validated["data_quality_report"] = quality_report
            validated["data_completion"] = completion_report
            validated["confidence"] = confidence_report.get("confidence", 0.0)
            validated["confidence_level"] = confidence_report.get(
                "confidence_level", "very_low"
            )
            validated["confidence_report"] = confidence_report
            validated.setdefault("data_sources", ["yfinance"])
            validated["data_source"] = "live"

            # Add Macro-Economic Context
            try:
                macro = self.get_macro_context(cleaned)
                if macro:
                    validated["macro_context"] = macro
            except Exception as macro_exc:
                self.logger.debug("Macro context fetching failed: %s", macro_exc)

            sanitized = self._sanitize_company_data(validated)
            if cache_enabled and cache is not None:
                try:
                    cache.set(
                        cache_key,
                        copy.deepcopy(sanitized),
                        expire=self._market_cache_ttl_seconds(),
                    )
                except Exception as cache_exc:
                    self.logger.debug(
                        "Market cache write skipped (%s): %s", cleaned, cache_exc
                    )
            return sanitized
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("%s data could not be retrieved: %s", cleaned, exc)
            return None

    def get_raw_company_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch raw company data from yfinance without cache."""

        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            self.logger.error("Invalid ticker parameter: %s", ticker)
            return None

        try:
            return self._fetch_from_yfinance(cleaned)
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.error("%s raw data could not be retrieved: %s", cleaned, exc)
            return None

    def get_latest_prices_bulk(self, tickers: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch latest prices for multiple symbols using yfinance batch download."""
        cleaned = self._normalize_tickers(tickers)
        if not cleaned:
            return {}

        previous_proxy = getattr(getattr(yf, "config", None), "network", None)
        previous_proxy = getattr(previous_proxy, "proxy", None)
        proxy = (
            self.proxy_manager.get_proxy()
            if self.market_config.get("prefer_proxy_for_yfinance", False)
            else None
        )
        session = self._create_http_session(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        )
        if proxy:
            session.proxies = {"http": proxy, "https": proxy}
        self._set_yfinance_proxy(proxy)

        interval = "1d"
        repair_history = any(
            self._should_enable_history_repair(symbol, interval) for symbol in cleaned
        )

        try:
            frame = yf.download(
                cleaned,
                period="5d",
                interval=interval,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=True,
                repair=repair_history,
                timeout=self.market_config.get("api_request_timeout", 10.0),
                session=session,
                group_by="ticker",
                multi_level_index=True,
            )
        except Exception as exc:
            self.logger.debug("Bulk price download failed: %s", exc)
            return {}
        finally:
            try:
                session.close()
            except Exception as exc:
                self.logger.debug("Bulk price session close failed: %s", exc)
            self._set_yfinance_proxy(previous_proxy)

        try:
            import pandas as pd  # type: ignore
        except ImportError:  # pragma: no cover
            return {}

        result: Dict[str, Dict[str, Any]] = {}
        if frame is None or frame.empty:
            return result

        for symbol in cleaned:
            try:
                if isinstance(frame.columns, pd.MultiIndex):
                    if symbol not in frame.columns.get_level_values(0):
                        continue
                    ticker_frame = frame[symbol].dropna(how="all")
                else:
                    ticker_frame = frame.dropna(how="all") if len(cleaned) == 1 else None

                if ticker_frame is None or ticker_frame.empty:
                    continue

                closes = ticker_frame.get("Close")
                opens = ticker_frame.get("Open")
                if closes is None:
                    continue

                closes = closes.dropna()
                if closes.empty:
                    continue

                current_price = float(closes.iloc[-1])
                previous_close = None
                if len(closes) >= 2:
                    previous_close = float(closes.iloc[-2])
                elif opens is not None:
                    opens = opens.dropna()
                    if not opens.empty:
                        previous_close = float(opens.iloc[-1])

                result[symbol] = {
                    "regularMarketPrice": current_price,
                    "currentPrice": current_price,
                    "previousClose": previous_close,
                }
            except Exception as exc:
                self.logger.debug("Bulk price parse skipped for %s: %s", symbol, exc)

        return result

    def get_company_data_bulk(
        self,
        tickers: Iterable[str],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Fetch multiple company payloads and enrich missing prices from batch data."""
        cleaned = self._normalize_tickers(tickers)
        if not cleaned:
            return {}

        price_map = self.get_latest_prices_bulk(cleaned)
        result: Dict[str, Dict[str, Any]] = {}
        for symbol in cleaned:
            data = self.get_company_data(symbol, user_context=user_context)
            if not data:
                continue

            price_data = price_map.get(symbol) or {}
            if data.get("regularMarketPrice") in (None, NA_VALUE):
                data["regularMarketPrice"] = price_data.get("regularMarketPrice")
            if data.get("currentPrice") in (None, NA_VALUE):
                data["currentPrice"] = price_data.get("currentPrice")
            if data.get("previousClose") in (None, NA_VALUE):
                data["previousClose"] = price_data.get("previousClose")
            result[symbol] = data

        return result

    # ------------------------------------------------------------------
    # Modular category methods
    # ------------------------------------------------------------------

    def get_overview(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return basic identity, price and metric information for symbol."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            return self._fetch_overview(cleaned)
        except Exception as exc:
            self.logger.error("%s overview could not be retrieved: %s", cleaned, exc)
            return None

    def get_index_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return index/ETF-focused summary data set."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            raw = self._fetch_index_data(cleaned)
            if not raw:
                return None
            return self._sanitize_company_data(raw)
        except Exception as exc:
            self.logger.error("%s index data could not be retrieved: %s", cleaned, exc)
            return None

    def get_price_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> Optional[Dict[str, Any]]:
        """Return price history and technical indicators."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            return self._fetch_price_history(cleaned, period=period, interval=interval)
        except Exception as exc:
            self.logger.error(
                "%s price history could not be retrieved: %s", cleaned, exc
            )
            return None

    def get_dividends(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return dividend, stock split and corporate action data."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            ticker_obj, _, _, _ = self._create_ticker(cleaned)
            if ticker_obj is None:
                return None
            info = self._safe_get_info(ticker_obj)
            price = info.get("regularMarketPrice")
            result: Dict[str, Any] = self._build_dividend_snapshot(
                ticker_obj, price, info
            )
            splits = self._build_splits_snapshot(ticker_obj)
            if splits:
                result["splits"] = splits
            actions = self._build_actions_snapshot(ticker_obj)
            if actions:
                result["actions"] = actions
            capital_gains = self._build_capital_gains_snapshot(ticker_obj)
            if capital_gains:
                result["capitalGains"] = capital_gains
            events = self._build_events_snapshot(ticker_obj)
            if events:
                next_dividend = self._extract_next_dividend_date(
                    ticker_obj, info, events
                )
                if next_dividend and next_dividend != NA_VALUE:
                    result["nextDividendDate"] = next_dividend
            return result if result else None
        except Exception as exc:
            self.logger.error(
                "%s dividend data could not be retrieved: %s", cleaned, exc
            )
            return None

    def get_financial_statements(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return detailed financial statements and filing snapshots."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            ticker_obj, _, _, _ = self._create_ticker(cleaned)
            if ticker_obj is None:
                return None
            result = self._build_financial_statement_snapshot(ticker_obj)
            filings = self._build_sec_filings_snapshot(ticker_obj)
            if filings:
                result["secFilings"] = filings
            return result if result else None
        except Exception as exc:
            self.logger.error(
                "%s financial statements could not be retrieved: %s", cleaned, exc
            )
            return None

    def get_analyst(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return analyst recommendations, price targets and forecasts."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            ticker_obj, _, _, _ = self._create_ticker(cleaned)
            if ticker_obj is None:
                return None
            info = self._safe_get_info(ticker_obj)
            result: Dict[str, Any] = {}
            analyst_snapshot = self._build_analyst_snapshot(info, ticker_obj)
            if analyst_snapshot:
                result["analystRecommendations"] = analyst_snapshot
            recommendations = self._build_recommendations_snapshot(ticker_obj)
            if recommendations:
                result["recommendations"] = recommendations
            recommendations_summary = self._build_recommendations_summary(ticker_obj)
            if recommendations_summary:
                result["recommendationsSummary"] = recommendations_summary
            analyst_price_targets = self._build_analyst_price_targets_snapshot(
                ticker_obj
            )
            if analyst_price_targets:
                result["analystPriceTargets"] = analyst_price_targets
            eps_revisions = self._build_eps_revisions_snapshot(ticker_obj)
            if eps_revisions:
                result["epsRevisions"] = eps_revisions
            eps_trend = self._build_eps_trend_snapshot(ticker_obj)
            if eps_trend:
                result["epsTrend"] = eps_trend
            growth_estimates = self._build_growth_estimates_snapshot(ticker_obj)
            if growth_estimates:
                result["growthEstimates"] = growth_estimates
            earnings_estimate = self._build_earnings_estimate_snapshot(ticker_obj)
            if earnings_estimate:
                result["earningsEstimate"] = earnings_estimate
            upgrades_downgrades = self._build_upgrades_downgrades_snapshot(ticker_obj)
            if upgrades_downgrades:
                result["upgradesDowngrades"] = upgrades_downgrades
            return result if result else None
        except Exception as exc:
            self.logger.error(
                "%s analyst data could not be retrieved: %s", cleaned, exc
            )
            return None

    def get_earnings(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return earnings history, dates and quarterly income statement."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            ticker_obj, _, _, _ = self._create_ticker(cleaned)
            if ticker_obj is None:
                return None
            result: Dict[str, Any] = {}
            earnings_trend = self._build_earnings_trend(ticker_obj)
            if earnings_trend:
                result["earningsTrend"] = earnings_trend
            eps_history = self._build_eps_history_lookup(ticker_obj)
            if eps_history:
                result["epsHistory"] = eps_history
            earnings_dates = self._build_earnings_dates_snapshot(ticker_obj)
            if earnings_dates:
                result["earningsDates"] = earnings_dates
            return result if result else None
        except Exception as exc:
            self.logger.error(
                "%s earnings data could not be retrieved: %s", cleaned, exc
            )
            return None

    def get_ownership(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return institutional ownership, insider trading and share history."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            ticker_obj, _, _, _ = self._create_ticker(cleaned)
            if ticker_obj is None:
                return None
            info = self._safe_get_info(ticker_obj)
            result: Dict[str, Any] = (
                self._build_holders_snapshot(ticker_obj, info) or {}
            )
            insider = self._build_insider_snapshot(ticker_obj)
            if insider:
                result.update(insider)
            shares_full = self._build_shares_full_snapshot(ticker_obj)
            if shares_full:
                result["sharesFullHistory"] = shares_full
            return result if result else None
        except Exception as exc:
            self.logger.error("%s ownership data could not be retrieved: %s", cleaned, exc)
            return None

    def get_sustainability(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return ESG sustainability scores."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            ticker_obj, _, _, _ = self._create_ticker(cleaned)
            if ticker_obj is None:
                return None
            return self._build_sustainability_snapshot(ticker_obj)
        except Exception as exc:
            self.logger.error("%s sustainability data could not be retrieved: %s", cleaned, exc)
            return None

    def get_news(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return trimmed ticker news from yfinance."""
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return None
        try:
            ticker_obj, _, _, _ = self._create_ticker(cleaned)
            if ticker_obj is None:
                return None
            news = self._build_news_snapshot(ticker_obj)
            return {"items": news, "count": len(news)} if news else None
        except Exception as exc:
            self.logger.error("%s news data could not be retrieved: %s", cleaned, exc)
            return None

    def validate_ticker(self, ticker: str) -> bool:
        cleaned = self._normalize_ticker(ticker)
        if not cleaned:
            return False

        try:
            data = self._fetch_from_yfinance(cleaned)
            return bool(data)
        except Exception:
            return False

    def _normalize_ticker(self, ticker: str) -> Optional[str]:
        if not ticker or not isinstance(ticker, str):
            return None

        cleaned = ticker.strip().upper()
        normalized = cleaned[1:] if cleaned.startswith("^") else cleaned
        if len(cleaned) > 15 or not normalized.replace(".", "").replace("-", "").isalnum():
            return None

        return cleaned

    def _sanitize_company_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Clear Sentinel N/A values and filter out empty collections."""

        def sanitize(value: Any):
            if isinstance(value, dict):
                result: Dict[str, Any] = {}
                for key, inner_value in value.items():
                    sanitized = sanitize(inner_value)
                    if sanitized is None:
                        result[key] = None
                        continue
                    if isinstance(sanitized, dict) and not sanitized:
                        continue
                    if isinstance(sanitized, list) and not sanitized:
                        continue
                    result[key] = sanitized
                return result if result else None

            if isinstance(value, list):
                result_list = []
                for item in value:
                    sanitized_item = sanitize(item)
                    if sanitized_item is None:
                        result_list.append(None)
                        continue
                    if isinstance(sanitized_item, dict) and not sanitized_item:
                        continue
                    if isinstance(sanitized_item, list) and not sanitized_item:
                        continue
                    result_list.append(sanitized_item)
                return result_list if result_list else None

            if isinstance(value, str) and value == NA_VALUE:
                return None

            return value

        sanitized_data = sanitize(data)
        if not isinstance(sanitized_data, dict) or sanitized_data is None:
            return {}

        return sanitized_data


__all__ = ["MarketDataService", "MarketDataProvider"]
