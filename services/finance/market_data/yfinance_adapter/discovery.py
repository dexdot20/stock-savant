"""Discovery and market context mixins for YFinance adapter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
import yfinance as yf

try:
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover
    curl_requests = None


class YFinanceSearchMixin:
    """Provides search and screener capabilities."""

    _YF_UA = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(
            s in msg for s in ("too many requests", "rate limit", "rate limited", "429")
        )

    def _make_session(self, proxy: Optional[str]) -> requests.Session:
        session_factory = curl_requests.Session if curl_requests else requests.Session
        session = (
            session_factory(impersonate="chrome")
            if curl_requests
            else session_factory()
        )
        session.headers.update({"User-Agent": self._YF_UA})
        if proxy and self.market_config.get("prefer_proxy_for_yfinance", False):
            session.proxies = {"http": proxy, "https": proxy}
        return session

    def search_ticker(
        self, query: str, max_results: int = 10, news_count: int = 0
    ) -> Dict[str, Any]:
        """
        Search for tickers, news, and research.
        """
        max_retries = int(self.market_config.get("api_max_retries", 3))
        for attempt in range(max_retries):
            proxy = self.proxy_manager.get_proxy()
            try:
                session = self._make_session(proxy)
                search = yf.Search(
                    query,
                    max_results=max_results,
                    news_count=news_count,
                    session=session,
                )
                return {
                    "quotes": search.quotes,
                    "news": search.news if news_count > 0 else [],
                    "research": search.research,
                }
            except Exception as exc:
                if self._is_rate_limited(exc):
                    if proxy:
                        self.proxy_manager.mark_proxy_failed(proxy)
                    self.logger.warning(
                        "⚠️ [SEARCH] Rate limited (attempt %d/%d), rotating proxy...",
                        attempt + 1,
                        max_retries,
                    )
                    continue
                self.logger.warning(
                    "❌ [SEARCH] Failed to search for '%s': %s", query, exc
                )
                return {}
        self.logger.warning(
            "❌ [SEARCH] All %d proxy attempts exhausted for '%s'", max_retries, query
        )
        return {}

    def lookup_ticker(
        self, query: str, count: int = 10, asset_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Lookup tickers by type (Stock, ETF, Index, etc.).

        Args:
            query: Search term.
            count: Maximum number of results.
            asset_type: One of 'stock', 'equity', 'etf', 'index', 'crypto',
                        'cryptocurrency', 'mutualfund'. If None, returns all.
        """
        max_retries = int(self.market_config.get("api_max_retries", 3))
        for attempt in range(max_retries):
            proxy = self.proxy_manager.get_proxy()
            try:
                session = self._make_session(proxy)
                lookup = yf.Lookup(query, session=session)

                _TYPE_MAP = {
                    "stock": ("stocks", lookup.get_stock),
                    "equity": ("stocks", lookup.get_stock),
                    "etf": ("etfs", lookup.get_etf),
                    "index": ("indices", lookup.get_index),
                    "crypto": ("cryptocurrencies", lookup.get_cryptocurrency),
                    "cryptocurrency": ("cryptocurrencies", lookup.get_cryptocurrency),
                    "mutualfund": ("mutualfunds", lookup.get_mutualfund),
                }

                if asset_type and asset_type.lower() in _TYPE_MAP:
                    key, getter = _TYPE_MAP[asset_type.lower()]
                    return {key: self._dataframe_to_records(getter(count=count))}

                return {"all": self._dataframe_to_records(lookup.get_all(count=count))}
            except Exception as exc:
                if self._is_rate_limited(exc):
                    if proxy:
                        self.proxy_manager.mark_proxy_failed(proxy)
                    self.logger.warning(
                        "⚠️ [LOOKUP] Rate limited (attempt %d/%d), rotating proxy...",
                        attempt + 1,
                        max_retries,
                    )
                    continue
                self.logger.warning("❌ [LOOKUP] Failed to lookup '%s': %s", query, exc)
                return {}
        self.logger.warning(
            "❌ [LOOKUP] All %d proxy attempts exhausted for '%s'", max_retries, query
        )
        return {}

    def screen_tickers(
        self,
        query_type: str = "EQUITY",
        filters: Dict[str, Any] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Screen tickers using EquityQuery or FundQuery.

        Args:
            query_type: "EQUITY" or "FUND"
            filters: Dictionary of filters.
                     Supports constructing simple queries if filters are provided as list of conditions.
                     Example: filters={'conditions': [('gt', 'per', 15)]}
        """
        try:
            if not filters:
                return []

            conditions = filters.get("conditions")
            if conditions:
                queries = []
                QueryClass = yf.EquityQuery if query_type == "EQUITY" else yf.FundQuery

                for op, field, value in conditions:
                    operand = [field, *value] if isinstance(value, (list, tuple)) else [field, value]
                    queries.append(QueryClass(op, operand))

                if not queries:
                    return []

                # Combine with 'and' if multiple
                if len(queries) > 1:
                    q = QueryClass("and", queries)
                else:
                    q = queries[0]

                results = yf.screen(q)
                return self._normalize_screen_results(results, limit=limit)

            self.logger.info("Screener: No valid filters provided.")
            return []

        except Exception as exc:
            self.logger.warning("❌ [SCREENER] Failed to screen tickers: %s", exc)
            return []

    def _normalize_screen_results(
        self, raw: Any, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        max_rows = max(1, int(limit or 25))

        if isinstance(raw, dict):
            rows = raw.get("quotes")
            if not isinstance(rows, list):
                rows = raw.get("results") if isinstance(raw.get("results"), list) else []

            normalized_rows: List[Dict[str, Any]] = []
            for row in rows[:max_rows]:
                if not isinstance(row, dict):
                    continue
                normalized_rows.append(
                    {
                        key: self._coerce_numeric(value)
                        for key, value in row.items()
                    }
                )
            return normalized_rows

        return self._dataframe_to_records(raw, limit=max_rows)

    def screen_equities_by_market(
        self,
        *,
        region: Optional[str],
        exchange: Optional[str],
        sector: Optional[str] = None,
        limit: int = 25,
    ) -> Dict[str, Any]:
        conditions = []

        normalized_region = str(region or "").strip().lower() or None
        normalized_exchange = str(exchange or "").strip().upper() or None
        normalized_sector = str(sector or "").strip() or None

        if normalized_region:
            conditions.append(("eq", "region", normalized_region))
        if normalized_exchange:
            conditions.append(("is-in", "exchange", normalized_exchange))
        if normalized_sector and normalized_sector.lower() != "all":
            conditions.append(("eq", "sector", normalized_sector))

        quotes = self.screen_tickers(
            query_type="EQUITY",
            filters={"conditions": conditions},
            limit=limit,
        )

        return {
            "region": normalized_region,
            "exchange": normalized_exchange,
            "sector": normalized_sector,
            "count": len(quotes),
            "quotes": quotes,
        }


class YFinanceMarketMixin:
    """Provides market summary and status information."""

    def get_market_summary(self, region: str = "US") -> Dict[str, Any]:
        """
        Get market summary for a specific region.
        Regions: US, GB, ASIA, EUROPE, RATES, COMMODITIES, CURRENCIES, CRYPTOCURRENCIES
        """
        try:
            proxy = self.proxy_manager.get_proxy()
            session = self._make_session(proxy)

            market = yf.Market(region, session=session)

            # Fetch status and summary
            # Note: yfinance Market object properties might trigger network calls
            status = self._call_with_retry(
                lambda: market.status, name=f"market.status({region})"
            )
            summary = self._call_with_retry(
                lambda: market.summary, name=f"market.summary({region})"
            )

            return {
                "region": region,
                "status": status,
                "summary": (
                    self._dataframe_to_records(summary) if summary is not None else []
                ),
            }
        except Exception as exc:
            self.logger.warning(
                "❌ [MARKET] Failed to get market summary for %s: %s", region, exc
            )
            return {}

    def get_macro_context(self, ticker: str) -> Dict[str, Any]:
        """
        Fetches relevant macro-economic context data.
        """
        macro_tickers = {
            "S&P 500": "^GSPC",
            "Nasdaq": "^IXIC",
            "Dow Jones": "^DJI",
            "US 10Y Yield": "^TNX",
            "Gold": "GC=F",
            "Crude Oil": "CL=F",
        }

        # Add BIST 100 if it's a Turkish stock
        if ticker.upper().endswith(".IS"):
            macro_tickers["BIST 100"] = "XU100.IS"

        results = {}
        proxy = self.proxy_manager.get_proxy()

        for name, sym in macro_tickers.items():
            try:
                session = self._make_session(proxy)

                t = yf.Ticker(sym, session=session)
                # Use fast_info for quick price access
                fast = t.fast_info
                results[name] = {
                    "price": float(fast.last_price),
                    "change_pct": float(fast.regular_market_change_percent or 0.0)
                    * 100,
                    "symbol": sym,
                }
            except Exception as exc:
                self.logger.debug(f"[MACRO] Failed to fetch {name} ({sym}): {exc}")

        return results


class YFinanceSectorIndustryMixin:
    """Provides sector and industry information."""

    def get_sector_info(self, sector_key: str) -> Dict[str, Any]:
        """
        Get detailed information about a sector.
        """
        try:
            proxy = self.proxy_manager.get_proxy()
            session = self._make_session(proxy)

            sector = yf.Sector(sector_key, session=session)

            return {
                "key": sector.key,
                "name": sector.name,
                "overview": sector.overview,
                "top_companies": self._dataframe_to_records(sector.top_companies),
                "top_etfs": self._dataframe_to_records(sector.top_etfs),
                "top_mutual_funds": self._dataframe_to_records(sector.top_mutual_funds),
                "industries": self._dataframe_to_records(sector.industries),
            }
        except Exception as exc:
            self.logger.warning(
                "❌ [SECTOR] Failed to get info for sector '%s': %s", sector_key, exc
            )
            return {}

    def get_industry_info(self, industry_key: str) -> Dict[str, Any]:
        """
        Get detailed information about an industry.
        """
        try:
            proxy = self.proxy_manager.get_proxy()
            session = self._make_session(proxy)

            industry = yf.Industry(industry_key, session=session)

            return {
                "key": industry.key,
                "name": industry.name,
                "sector_key": industry.sector_key,
                "sector_name": industry.sector_name,
                "top_performing_companies": self._dataframe_to_records(
                    industry.top_performing_companies
                ),
                "top_growth_companies": self._dataframe_to_records(
                    industry.top_growth_companies
                ),
            }
        except Exception as exc:
            self.logger.warning(
                "❌ [INDUSTRY] Failed to get info for industry '%s': %s",
                industry_key,
                exc,
            )
            return {}


class YFinanceOptionsMixin:
    """Provides options data capabilities."""

    def get_options_data(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get options data including expiration dates and option chains.
        """
        try:
            # Get expiration dates
            expirations = self._call_with_retry(
                lambda: ticker.options, name="ticker.options"
            )

            return {"expirations": expirations, "has_options": bool(expirations)}
        except Exception as exc:
            self.logger.debug("Option data could not be retrieved: %s", exc)
            return {"expirations": [], "has_options": False}

    def get_option_chain(self, ticker: yf.Ticker, date: str = None) -> Dict[str, Any]:
        """
        Get option chain for a specific expiration date.
        If date is None, gets the nearest expiration.
        """
        try:
            chain = self._call_with_retry(
                lambda: ticker.option_chain(date), name=f"ticker.option_chain({date})"
            )

            return {
                "calls": self._dataframe_to_records(chain.calls),
                "puts": self._dataframe_to_records(chain.puts),
                "underlying": chain.underlying,
            }
        except Exception as exc:
            self.logger.warning(
                "❌ [OPTIONS] Failed to get option chain for date %s: %s", date, exc
            )
            return {}
