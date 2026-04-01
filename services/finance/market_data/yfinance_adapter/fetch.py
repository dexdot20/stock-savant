"""Fetching layer for YFinance adapter."""

from __future__ import annotations

from typing import Any, Dict, Optional
from datetime import date, datetime
import time
import requests
import yfinance as yf
from config import NA_VALUE

try:
    from curl_cffi import requests as curl_requests
except ImportError:  # pragma: no cover
    curl_requests = None


class YFinanceFetchMixin:
    """Mixin for fetching data from YFinance API."""

    _TICKER_CACHE_TTL_SECONDS = 300
    _TICKER_CACHE_MAX_SIZE = 20

    def _init_ticker_cache(self) -> None:
        if not hasattr(self, "_ticker_cache"):
            self._ticker_cache: Dict[str, Any] = {}
            self._ticker_cache_expiry: Dict[str, float] = {}

    def _prune_ticker_cache(self) -> None:
        self._init_ticker_cache()
        now = time.time()
        expired_keys = [
            k for k, expiry in self._ticker_cache_expiry.items() if expiry <= now
        ]
        for key in expired_keys:
            self._ticker_cache.pop(key, None)
            self._ticker_cache_expiry.pop(key, None)

        while len(self._ticker_cache) > self._TICKER_CACHE_MAX_SIZE:
            oldest_key = min(
                self._ticker_cache_expiry,
                key=self._ticker_cache_expiry.get,
                default=None,
            )
            if oldest_key is None:
                break
            self._ticker_cache.pop(oldest_key, None)
            self._ticker_cache_expiry.pop(oldest_key, None)

    def _create_http_session(self, user_agent: str) -> requests.Session:
        session_factory = curl_requests.Session if curl_requests else requests.Session
        session = (
            session_factory(impersonate="chrome")
            if curl_requests
            else session_factory()
        )
        session.headers.update({"User-Agent": user_agent})
        return session

    def _set_yfinance_proxy(self, proxy: Optional[str]) -> None:
        try:
            yf.config.network.proxy = (
                {"http": proxy, "https": proxy} if proxy else None
            )
        except AttributeError as exc:
            self.logger.debug("yfinance proxy config unavailable: %s", exc)

    def _should_enable_history_repair(self, ticker: str, interval: str) -> bool:
        if not bool(self.market_config.get("history_repair_enabled", True)):
            return False

        if interval not in {"1d", "5d", "1wk", "1mo", "3mo"}:
            return False

        if not bool(self.market_config.get("history_repair_non_us_only", True)):
            return True

        upper_ticker = ticker.upper()
        return "." in upper_ticker or upper_ticker.startswith("^")

    def _filter_future_date(self, date_str: Any) -> Any:
        """Filter out past dates from future event predictions."""
        if date_str in (None, NA_VALUE, ""):
            return NA_VALUE
        try:
            # Check if it's already a string in ISO format
            if isinstance(date_str, str):
                dt = datetime.fromisoformat(date_str).date()
                if dt >= date.today():
                    return date_str
            # If it's a datetime/date object (unlikely here as we sanitized before)
            elif isinstance(date_str, (date, datetime)):
                d = date_str.date() if isinstance(date_str, datetime) else date_str
                if d >= date.today():
                    return d.isoformat()
        except (TypeError, ValueError):
            return NA_VALUE
        return NA_VALUE

    def _safe_get_info(self, ticker_obj):
        """
        Güvenli şekilde ticker.info property'sini çağırır.
        """
        try:
            # info property'si bazen 404 veya JSONDecodeError verebiliyor.
            # Bu yüzden try-except bloğu içinde çağırıyoruz.
            info = ticker_obj.info
            if info is None:
                return {}
            return info
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "404" in exc_msg or "not found" in exc_msg:
                self.logger.debug(
                    f"[FETCH] Info not found for {ticker_obj.ticker}: {exc_msg}"
                )
            else:
                self.logger.warning(
                    f"❌ [FETCH] Error fetching info for {ticker_obj.ticker}: {str(exc)[:100]}"
                )
            return {}

    def _safe_get_fast_info(self, ticker_obj):
        try:
            fast_info_dict = {}
            for attr in dir(ticker_obj.fast_info):
                if not attr.startswith("_"):
                    try:
                        value = getattr(ticker_obj.fast_info, attr)
                        fast_info_dict[attr] = value
                    except (AttributeError, TypeError, ValueError, KeyError):
                        continue
                    except Exception:
                        continue

            self.logger.debug(
                f"[FETCH] ticker.fast_info: {len(fast_info_dict)} attributes"
            )
            return fast_info_dict
        except Exception as exc:
            # Suppress "possibly delisted" errors - these are expected for some tickers
            exc_msg = str(exc).lower()
            if "possibly delisted" in exc_msg or "no price data found" in exc_msg:
                self.logger.debug(f"[FETCH] {ticker_obj.ticker}: {str(exc)[:100]}")
            else:
                self.logger.error(
                    f"❌ [FETCH] fast_info property failed: {str(exc)[:100]}"
                )
            return {}

    def _safe_get_isin(self, ticker_obj):
        """Safely fetch ISIN to avoid external lookup failures (e.g. businessinsider)."""
        try:
            val = ticker_obj.isin
            return val if val else NA_VALUE
        except Exception as exc:
            self.logger.debug(
                f"[FETCH] ISIN fetch failed for {ticker_obj.ticker}: {exc}"
            )
            return NA_VALUE

    def _get_history_metadata(self, ticker_obj) -> Dict[str, Any]:
        """
        Get history metadata (trading hours, timezone, etc.).
        """
        try:
            metadata = ticker_obj.history_metadata
            if not metadata:
                return {}
            return metadata
        except Exception as exc:
            self.logger.debug(
                f"[FETCH] History metadata not found for {ticker_obj.ticker}: {exc}"
            )
            return {}

    # ------------------------------------------------------------------
    # Yardımcı: proxy/session hazırlığı ve yf.Ticker nesnesi oluşturma
    # ------------------------------------------------------------------

    def _create_ticker(self, ticker: str):
        """
        Proxy ve requests.Session konfigürasyonunu yaparak yf.Ticker döndürür.
        Başarısız olursa None döner.
        Dönüş: (ticker_obj, session, proxy, ua)
        """
        ua = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
        proxy = (
            self.proxy_manager.get_proxy()
            if self.market_config.get("prefer_proxy_for_yfinance", False)
            else None
        )
        cache_key = f"{ticker}|{proxy or 'direct'}"
        self._init_ticker_cache()
        self._prune_ticker_cache()

        now = time.time()
        cached_expiry = self._ticker_cache_expiry.get(cache_key, 0.0)
        cached_entry = self._ticker_cache.get(cache_key)
        if cached_entry and cached_expiry > now:
            ticker_obj, session = cached_entry
            return ticker_obj, session, proxy, ua

        session = self._create_http_session(ua)

        try:
            if proxy:
                session.proxies = {"http": proxy, "https": proxy}
                self._set_yfinance_proxy(proxy)
            else:
                self._set_yfinance_proxy(None)

            ticker_obj = yf.Ticker(ticker, session=session)
            self._ticker_cache[cache_key] = (ticker_obj, session)
            self._ticker_cache_expiry[cache_key] = now + self._TICKER_CACHE_TTL_SECONDS
            self._prune_ticker_cache()
            return ticker_obj, session, proxy, ua
        except Exception as exc:
            self.logger.warning(
                "❌ [TICKER] Cannot create Ticker (%s): %s", ticker, str(exc)[:100]
            )
            if proxy:
                self.proxy_manager.mark_proxy_failed(proxy)
            return None, session, proxy, ua

    # ------------------------------------------------------------------
    # Kategori metodu: Fiyat geçmişi
    # ------------------------------------------------------------------

    def _fetch_price_history(
        self,
        ticker: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> Optional[Dict[str, Any]]:
        """
        Belirtilen sembol için yalnızca fiyat geçmişi ve teknik göstergeleri getirir.
        period / interval yfinance'in desteklediği değerleri alır.
        """
        ticker_obj, session, proxy, ua = self._create_ticker(ticker)
        if ticker_obj is None:
            return None

        timeout = self.market_config.get("api_request_timeout", 10.0)
        repair_history = self._should_enable_history_repair(ticker, interval)

        # yfinance: repair=True requires actions=True to access 'Stock Splits' column
        # internally; passing actions=False with repair=True raises KeyError('Stock Splits').
        # We fetch with actions=True when repair is enabled, then strip the action columns.
        fetch_actions = True if repair_history else False

        try:
            price_history = ticker_obj.history(
                period=period,
                interval=interval,
                repair=repair_history,
                actions=fetch_actions,
                timeout=timeout,
            )

            # Proxy başarısız olduysa direkt dene
            if (price_history is None or price_history.empty) and proxy:
                self.logger.warning(
                    "⚠️ [PRICE] Proxy %s no data for %s, retrying directly...",
                    proxy,
                    ticker,
                )
                old_proxies = dict(getattr(session, "proxies", {}) or {})
                try:
                    session.proxies = {}
                    self._set_yfinance_proxy(None)
                    price_history = ticker_obj.history(
                        period=period,
                        interval=interval,
                        repair=repair_history,
                        actions=fetch_actions,
                        timeout=timeout,
                    )
                finally:
                    session.proxies = old_proxies
                    self._set_yfinance_proxy(proxy)

        except Exception as exc:
            self.logger.debug("Price history could not be retrieved (%s): %s", ticker, exc)
            return None

        if price_history is None or price_history.empty:
            return None

        # Drop corporate-action columns that were fetched only to satisfy repair internals.
        if fetch_actions:
            action_cols = [c for c in ("Dividends", "Stock Splits", "Capital Gains") if c in price_history.columns]
            if action_cols:
                price_history = price_history.drop(columns=action_cols)

        result: Dict[str, Any] = {
            "period": period,
            "interval": interval,
            "candles": self._summarize_history(price_history),
        }
        history_metadata = self._get_history_metadata(ticker_obj)
        if history_metadata:
            result["historyMetadata"] = history_metadata

        # Teknik göstergeler (sadece günlük/haftalık aralıklarda anlamlı)
        if "Close" in price_history.columns:
            close_prices = price_history["Close"]

            if interval in ("1d", "5d", "1wk", "1mo", "3mo"):
                if len(close_prices) >= 50:
                    ma50_values = close_prices.iloc[-50:].dropna()
                    if len(ma50_values) > 0:
                        ma50 = float(ma50_values.mean())
                        if not (ma50 != ma50 or ma50 == float("inf")):
                            result["ma50"] = ma50

                if len(close_prices) >= 200:
                    ma200_values = close_prices.iloc[-200:].dropna()
                    if len(ma200_values) > 0:
                        ma200 = float(ma200_values.mean())
                        if not (ma200 != ma200 or ma200 == float("inf")):
                            result["ma200"] = ma200

            if len(close_prices) >= 15:
                try:
                    delta = close_prices.diff()
                    up = delta.where(delta > 0, 0.0)
                    down = -delta.where(delta < 0, 0.0)
                    ma_up = up.ewm(com=13, adjust=False, min_periods=14).mean()
                    ma_down = down.ewm(com=13, adjust=False, min_periods=14).mean()
                    rs = ma_up / ma_down
                    rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])
                    if not (rsi_val != rsi_val or rsi_val == float("inf")):
                        result["rsi"] = rsi_val
                except Exception as rsi_exc:
                    self.logger.debug(
                        "RSI calculation failed (%s): %s", ticker, rsi_exc
                    )

        return result

    def _fetch_index_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Fetch index/ETF oriented snapshot with returns, trend, and optional components."""
        ticker_obj, session, proxy, ua = self._create_ticker(ticker)
        if ticker_obj is None:
            return None

        info = self._safe_get_info(ticker_obj)
        fast_info = self._safe_get_fast_info(ticker_obj)

        if not info and not fast_info and proxy:
            self.logger.warning(
                "⚠️ [INDEX] No data with proxy, retrying without proxies: %s", ticker
            )
            try:
                session2 = self._create_http_session(ua)
                self._set_yfinance_proxy(None)
                ticker_obj2 = yf.Ticker(ticker, session=session2)
                info = self._safe_get_info(ticker_obj2)
                fast_info = self._safe_get_fast_info(ticker_obj2)
                ticker_obj = ticker_obj2
            except Exception as exc:
                self.logger.warning(
                    "❌ [INDEX] Proxyless attempt failed: %s", str(exc)[:100]
                )

        if not info and not fast_info:
            return None

        quote_type = (
            str(info.get("quoteType") or fast_info.get("quoteType") or "").upper()
            or None
        )
        supported_types = {"INDEX", "ETF", "MUTUALFUND", "CLOSEDFUND"}

        current_price = self._pick_first(
            info.get("regularMarketPrice"), fast_info.get("lastPrice")
        )
        previous_close = self._pick_first(
            info.get("previousClose"), fast_info.get("previousClose")
        )

        daily_change_percent = None
        try:
            if (
                isinstance(current_price, (int, float))
                and isinstance(previous_close, (int, float))
                and previous_close
            ):
                daily_change_percent = (
                    (float(current_price) - float(previous_close))
                    / float(previous_close)
                ) * 100
        except Exception:
            daily_change_percent = None

        if daily_change_percent is None:
            raw_change = info.get("regularMarketChangePercent")
            if isinstance(raw_change, (int, float)):
                daily_change_percent = float(raw_change)

        returns: Dict[str, Optional[float]] = {}
        recent_candles = []
        timeout = self.market_config.get("api_request_timeout", 10.0)

        period_map = {
            "YTD": "ytd",
            "1M": "1mo",
            "3M": "3mo",
            "6M": "6mo",
            "1Y": "1y",
        }

        for label, period in period_map.items():
            try:
                repair_history = self._should_enable_history_repair(ticker, "1d")
                history = ticker_obj.history(
                    period=period,
                    interval="1d",
                    repair=repair_history,
                    actions=False,
                    timeout=timeout,
                )
            except Exception:
                history = None

            if history is None or history.empty or "Close" not in history.columns:
                returns[label] = None
                continue

            closes = history["Close"].dropna()
            if len(closes) < 2:
                returns[label] = None
                continue

            try:
                start = float(closes.iloc[0])
                end = float(closes.iloc[-1])
                returns[label] = ((end - start) / start) * 100 if start else None
            except Exception:
                returns[label] = None

            if label == "1M":
                try:
                    tail = history.tail(20)
                    for idx, row in tail.iterrows():
                        close_val = row.get("Close")
                        if close_val is None:
                            continue
                        try:
                            close_float = float(close_val)
                        except Exception:
                            continue

                        date_str = None
                        if hasattr(idx, "date"):
                            try:
                                date_str = idx.date().isoformat()
                            except Exception:
                                date_str = None
                        if date_str is None:
                            date_str = str(idx)

                        recent_candles.append({"date": date_str, "close": close_float})
                except Exception:
                    recent_candles = []

        components = []
        try:
            raw_components = ticker_obj.components
            if raw_components is not None:
                if hasattr(raw_components, "columns") and "symbol" in list(
                    raw_components.columns
                ):
                    values = raw_components["symbol"].dropna().tolist()
                    components = [str(v).strip() for v in values if str(v).strip()][:10]
                elif hasattr(raw_components, "index"):
                    values = list(raw_components.index)
                    components = [str(v).strip() for v in values if str(v).strip()][:10]
                elif isinstance(raw_components, (list, tuple, set)):
                    components = [
                        str(v).strip() for v in raw_components if str(v).strip()
                    ][:10]
        except Exception:
            components = []

        if not components:
            screen_scope = None
            upper_ticker = str(ticker or "").strip().upper()
            if upper_ticker in {"XU100.IS", "^XU100", "XU100"}:
                screen_scope = {"region": "tr", "exchange": "IST"}

            if screen_scope:
                try:
                    screened = self.screen_equities_by_market(
                        region=screen_scope["region"],
                        exchange=screen_scope["exchange"],
                        limit=10,
                    )
                    quotes = screened.get("quotes") if isinstance(screened, dict) else []
                    if isinstance(quotes, list):
                        components = [
                            str(item.get("symbol") or "").strip()
                            for item in quotes
                            if isinstance(item, dict)
                            and str(item.get("symbol") or "").strip()
                        ][:10]
                except Exception as exc:
                    self.logger.debug(
                        "Index components screener fallback failed for %s: %s",
                        ticker,
                        exc,
                    )

        data: Dict[str, Any] = {
            "symbol": ticker,
            "longName": info.get("longName") or info.get("shortName") or ticker,
            "quoteType": quote_type,
            "isIndexAsset": bool(quote_type in supported_types),
            "currency": self._pick_first(
                info.get("currency"),
                fast_info.get("currency"),
                info.get("financialCurrency"),
            ),
            "regularMarketPrice": current_price,
            "previousClose": previous_close,
            "dailyChangePercent": daily_change_percent,
            "volume": self._pick_first(info.get("volume"), fast_info.get("lastVolume")),
            "fiftyTwoWeekHigh": self._pick_first(
                info.get("fiftyTwoWeekHigh"), fast_info.get("yearHigh")
            ),
            "fiftyTwoWeekLow": self._pick_first(
                info.get("fiftyTwoWeekLow"), fast_info.get("yearLow")
            ),
            "returns": returns,
            "recentCandles": recent_candles,
            "components": components,
        }

        return data

    # ------------------------------------------------------------------
    # Kategori metodu: Genel bakış (overview)
    # ------------------------------------------------------------------

    def _fetch_overview(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Sembol için temel kimlik, fiyat ve temel metrik bilgilerini getirir.
        Ağır veri kategorilerini (history, analyst, ownership vb.) çekmez.
        """
        from config import NA_VALUE  # noqa: PLC0415

        ticker_obj, session, proxy, ua = self._create_ticker(ticker)
        if ticker_obj is None:
            return None

        info = self._safe_get_info(ticker_obj)
        fast_info = self._safe_get_fast_info(ticker_obj)

        if not info and not fast_info and proxy:
            self.logger.warning(
                "⚠️ [OVERVIEW] Proxy ile veri yok, proxiesiz deneniyor: %s", ticker
            )
            try:
                session2 = self._create_http_session(ua)
                self._set_yfinance_proxy(None)
                ticker_obj2 = yf.Ticker(ticker, session=session2)
                info = self._safe_get_info(ticker_obj2)
                fast_info = self._safe_get_fast_info(ticker_obj2)
                ticker_obj = ticker_obj2
            except Exception as exc:
                self.logger.warning(
                    "❌ [OVERVIEW] Proxyless attempt failed: %s", str(exc)[:100]
                )

        if not info and not fast_info:
            return None

        summary_profile = (
            info.get("summaryProfile")
            if isinstance(info.get("summaryProfile"), dict)
            else {}
        )
        financial_data = (
            info.get("financialData")
            if isinstance(info.get("financialData"), dict)
            else {}
        )
        default_key_statistics = (
            info.get("defaultKeyStatistics")
            if isinstance(info.get("defaultKeyStatistics"), dict)
            else {}
        )
        quote_type = (
            str(info.get("quoteType") or fast_info.get("quoteType") or "").upper()
            or None
        )

        data: Dict[str, Any] = {
            "symbol": ticker,
            "isin": self._safe_get_isin(ticker_obj),
            "longName": info.get("longName") or info.get("shortName") or ticker,
            "quoteType": quote_type,
            "sector": info.get("sector") or summary_profile.get("sector") or NA_VALUE,
            "industry": info.get("industry")
            or summary_profile.get("industry")
            or NA_VALUE,
            "longBusinessSummary": (
                info.get("longBusinessSummary")
                or summary_profile.get("longBusinessSummary")
                or NA_VALUE
            ),
            "country": info.get("country")
            or summary_profile.get("country")
            or NA_VALUE,
            "fullTimeEmployees": info.get("fullTimeEmployees") or NA_VALUE,
            "currency": self._pick_first(
                info.get("currency"),
                fast_info.get("currency"),
                info.get("financialCurrency"),
            ),
            "regularMarketPrice": self._pick_first(
                info.get("regularMarketPrice"), fast_info.get("lastPrice")
            ),
            "previousClose": info.get("previousClose")
            or fast_info.get("previousClose"),
            "regularMarketChangePercent": info.get("regularMarketChangePercent"),
            "fiftyTwoWeekHigh": self._pick_first(
                info.get("fiftyTwoWeekHigh"), fast_info.get("yearHigh")
            ),
            "fiftyTwoWeekLow": self._pick_first(
                info.get("fiftyTwoWeekLow"), fast_info.get("yearLow")
            ),
            "volume": info.get("volume") or fast_info.get("lastVolume"),
            "marketCap": self._pick_first(
                info.get("marketCap"), fast_info.get("marketCap")
            ),
            "trailingPE": self._pick_first(
                info.get("trailingPE"), fast_info.get("peRatio")
            ),
            "forwardPE": info.get("forwardPE"),
            "trailingEps": self._pick_first(
                info.get("trailingEps"),
                financial_data.get("trailingEps"),
                default_key_statistics.get("trailingEps"),
            ),
            "priceToBook": info.get("priceToBook"),
            "priceToSales": info.get("priceToSalesTrailing12Months"),
            "beta": self._pick_first(info.get("beta"), fast_info.get("beta")),
            "dividendYield": self._convert_dividend_yield(info.get("dividendYield")),
            "enterpriseValue": self._calculate_enterprise_value(info, ticker_obj),
            "grossMargins": self._pick_first(
                info.get("grossMargins"), financial_data.get("grossMargins")
            ),
            "operatingMargins": self._pick_first(
                info.get("operatingMargins"), financial_data.get("operatingMargins")
            ),
            "profitMargins": self._pick_first(
                info.get("profitMargins"), financial_data.get("profitMargins")
            ),
            "debtToEquity": info.get("debtToEquity"),
            "currentRatio": self._pick_first(
                info.get("currentRatio"), financial_data.get("currentRatio")
            ),
            "returnOnEquity": self._calculate_roe(info, ticker_obj),
            "returnOnAssets": info.get("returnOnAssets"),
            "freeCashFlow": info.get("freeCashflow") or info.get("operatingCashflow"),
            "ebitda": info.get("ebitda"),
            "sharesOutstanding": self._pick_first(
                info.get("sharesOutstanding"),
                default_key_statistics.get("sharesOutstanding"),
                fast_info.get("sharesOutstanding"),
            ),
        }

        return data

    def _fetch_from_yfinance(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Fetch data from yfinance.

        Using requests.Session to avoid curl_cffi TLS errors and improve reliability.
        """
        ticker_obj, session, proxy, ua = self._create_ticker(ticker)
        if ticker_obj is None:
            return None

        try:
            _ = ticker_obj.fast_info
            if proxy:
                self.proxy_manager.mark_proxy_success(proxy)
        except Exception as exc:
            exc_msg = str(exc).lower()
            if "possibly delisted" in exc_msg or "no price data found" in exc_msg:
                self.logger.debug(f"[TEST] {ticker}: {str(exc)[:100]}")
            else:
                self.logger.warning(
                    "❌ [TEST] YFinance test failed (%s): %s", ticker, str(exc)[:100]
                )
                if proxy:
                    self.proxy_manager.mark_proxy_failed(proxy)
            # Continue anyway, individual methods might still work

        info = self._safe_get_info(ticker_obj)
        fast_info = self._safe_get_fast_info(ticker_obj)
        if not info and not fast_info and proxy:
            # Proxy kaynaklı bağlantı sorunlarında bir kez proxiesiz dene
            self.logger.warning(
                "⚠️ [FETCH] Proxy ile veri yok, proxiesiz tekrar denenecek: %s", ticker
            )
            try:
                session = self._create_http_session(ua)
                self._set_yfinance_proxy(None)
                ticker_obj = yf.Ticker(ticker, session=session)
                info = self._safe_get_info(ticker_obj)
                fast_info = self._safe_get_fast_info(ticker_obj)
            except Exception as exc:
                self.logger.warning(
                    "❌ [FETCH] Proxyless attempt failed: %s", str(exc)[:100]
                )

        if not info and not fast_info:
            self.logger.warning("yfinance returned no data: %s", ticker)
            return None

        summary_profile = (
            info.get("summaryProfile")
            if isinstance(info.get("summaryProfile"), dict)
            else {}
        )
        financial_data = (
            info.get("financialData")
            if isinstance(info.get("financialData"), dict)
            else {}
        )
        default_key_statistics = (
            info.get("defaultKeyStatistics")
            if isinstance(info.get("defaultKeyStatistics"), dict)
            else {}
        )

        quote_type = (
            str(info.get("quoteType") or fast_info.get("quoteType") or "").upper()
            or None
        )

        data: Dict[str, Any] = {
            "symbol": ticker,
            "isin": self._safe_get_isin(ticker_obj),
            "longName": info.get("longName")
            or info.get("shortName")
            or summary_profile.get("longBusinessSummary")
            or ticker,
            "quoteType": quote_type,
            "sector": info.get("sector") or summary_profile.get("sector") or "Unknown",
            "industry": info.get("industry")
            or summary_profile.get("industry")
            or "Unknown",
            "longBusinessSummary": info.get("longBusinessSummary")
            or summary_profile.get("longBusinessSummary")
            or "Company summary not available.",
            "country": info.get("country")
            or summary_profile.get("country")
            or NA_VALUE,
            "fullTimeEmployees": info.get("fullTimeEmployees") or NA_VALUE,
            "currency": self._pick_first(
                info.get("currency"),
                fast_info.get("currency"),
                info.get("financialCurrency"),
            ),
            "regularMarketPrice": self._pick_first(
                info.get("regularMarketPrice"),
                fast_info.get("lastPrice"),
            ),
            "previousClose": info.get("previousClose")
            or fast_info.get("previousClose"),
            "fiftyTwoWeekHigh": self._pick_first(
                info.get("fiftyTwoWeekHigh"),
                fast_info.get("yearHigh"),
            ),
            "fiftyTwoWeekLow": self._pick_first(
                info.get("fiftyTwoWeekLow"),
                fast_info.get("yearLow"),
            ),
            "regularMarketChangePercent": info.get("regularMarketChangePercent"),
            "volume": info.get("volume") or fast_info.get("lastVolume"),
            "marketCap": self._pick_first(
                info.get("marketCap"), fast_info.get("marketCap")
            ),
            "trailingPE": self._pick_first(
                info.get("trailingPE"), fast_info.get("peRatio")
            ),
            "forwardPE": info.get("forwardPE"),
            "trailingEps": self._pick_first(
                info.get("trailingEps"),
                financial_data.get("trailingEps"),
                default_key_statistics.get("trailingEps"),
            ),
            "epsTrailingTwelveMonths": self._pick_first(
                info.get("epsTrailingTwelveMonths"),
                default_key_statistics.get("epsTrailingTwelveMonths"),
            ),
            "priceToBook": info.get("priceToBook"),
            "priceToSales": info.get("priceToSalesTrailing12Months"),
            "enterpriseValue": self._calculate_enterprise_value(info, ticker_obj),
            "beta": self._pick_first(info.get("beta"), fast_info.get("beta")),
            "dividendYield": self._convert_dividend_yield(info.get("dividendYield")),
            "lastDividendAmount": self._pick_first(
                info.get("lastDividendValue"),
                info.get("dividendRate"),
            ),
            "currentRatio": self._pick_first(
                info.get("currentRatio"),
                financial_data.get("currentRatio"),
            ),
            "quickRatio": self._pick_first(
                info.get("quickRatio"),
                financial_data.get("quickRatio"),
            ),
            "grossMargins": self._pick_first(
                info.get("grossMargins"),
                financial_data.get("grossMargins"),
            ),
            "operatingMargins": self._pick_first(
                info.get("operatingMargins"),
                financial_data.get("operatingMargins"),
            ),
            "profitMargins": self._pick_first(
                info.get("profitMargins"),
                financial_data.get("profitMargins"),
            ),
            "pegRatio": self._pick_first(
                info.get("pegRatio"),
                financial_data.get("pegRatio"),
                default_key_statistics.get("pegRatio"),
            ),
            "freeCashFlow": info.get("freeCashflow") or info.get("operatingCashflow"),
            "ebitda": info.get("ebitda"),
            "debtToEquity": info.get("debtToEquity"),
            "returnOnEquity": self._calculate_roe(info, ticker_obj),
            "returnOnAssets": info.get("returnOnAssets"),
            "nextEarningsDate": self._pick_first(
                self._timestamp_to_iso(info.get("earningsTimestamp")),
                self._timestamp_to_iso(info.get("earningsDate")),
            ),
            "lastEarningsDate": self._pick_first(
                self._timestamp_to_iso(info.get("earningsTimestampStart")),
                self._timestamp_to_iso(info.get("mostRecentQuarter")),
            ),
            "lastDividendDate": self._timestamp_to_iso(info.get("lastDividendDate")),
            "nextDividendDate": self._filter_future_date(
                self._pick_first(
                    self._timestamp_to_iso(info.get("dividendDate")),
                    self._timestamp_to_iso(info.get("exDividendDate")),
                )
            ),
            "sharesOutstanding": self._pick_first(
                info.get("sharesOutstanding"),
                default_key_statistics.get("sharesOutstanding"),
                fast_info.get("sharesOutstanding"),
            ),
            "impliedSharesOutstanding": self._pick_first(
                info.get("impliedSharesOutstanding"),
                fast_info.get("sharesOutstanding"),
            ),
            "floatShares": self._pick_first(
                info.get("floatShares"),
                default_key_statistics.get("floatShares"),
            ),
            "heldPercentInsiders": info.get("heldPercentInsiders"),
            "heldPercentInstitutions": info.get("heldPercentInstitutions"),
            "shortRatio": self._pick_first(
                info.get("shortRatio"),
                default_key_statistics.get("shortRatio"),
            ),
            "shortPercentOfFloat": self._pick_first(
                info.get("shortPercentOfFloat"),
                default_key_statistics.get("shortPercentOfFloat"),
            ),
            "sharesShort": self._pick_first(
                info.get("sharesShort"),
                default_key_statistics.get("sharesShort"),
            ),
            "sharesShortPriorMonth": self._pick_first(
                info.get("sharesShortPriorMonth"),
                default_key_statistics.get("sharesShortPriorMonth"),
            ),
        }

        options_summary = self._build_options_summary(ticker_obj)
        if options_summary:
            data["options"] = options_summary

        if quote_type and quote_type in {"ETF", "MUTUALFUND", "CLOSEDFUND", "INDEX"}:
            funds_snapshot = self._build_funds_snapshot_lite(ticker_obj)
            if funds_snapshot:
                data["fundProfile"] = funds_snapshot

        dividend_snapshot = self._build_dividend_snapshot(
            ticker_obj,
            data.get("regularMarketPrice"),
            info,
        )
        if dividend_snapshot:
            data.update(dividend_snapshot)

        # Add splits and actions
        splits_snapshot = self._build_splits_snapshot(ticker_obj)
        if splits_snapshot:
            data["splits"] = splits_snapshot
            # Add last split info to top level
            if len(splits_snapshot) > 0:
                last_split = splits_snapshot[-1]
                data["lastSplitDate"] = last_split.get("date")
                data["lastSplitFactor"] = last_split.get("ratio")

        # Add sector and industry details if keys are available
        sector_key = info.get("sectorKey")
        if sector_key:
            data["sectorDetails"] = self.get_sector_info(sector_key)

        industry_key = info.get("industryKey")
        if industry_key:
            data["industryDetails"] = self.get_industry_info(industry_key)

        history_cfg = self.market_config.copy()
        try:
            period = history_cfg.get("history_period", "1y")
            interval = history_cfg.get("history_interval", "1d")
            timeout = self.market_config.get("api_request_timeout", 10.0)
            repair_history = self._should_enable_history_repair(ticker, interval)

            try:
                price_history = ticker_obj.history(
                    period=period,
                    interval=interval,
                    repair=repair_history,
                    actions=False,
                    timeout=timeout,
                )

                # Fallback: If proxy failed (empty data), try direct connection
                if (price_history is None or price_history.empty) and proxy:
                    self.logger.warning(
                        f"⚠️ [YFinance] Proxy {proxy} returned no data for history, retrying directly..."
                    )
                    old_proxies = dict(getattr(session, "proxies", {}) or {})
                    try:
                        session.proxies = {}  # Clear proxies in session
                        self._set_yfinance_proxy(None)
                        price_history = ticker_obj.history(
                            period=period,
                            interval=interval,
                            repair=repair_history,
                            actions=False,
                            timeout=timeout,
                        )
                    finally:
                        # Restore proxy
                        session.proxies = old_proxies
                        self._set_yfinance_proxy(proxy)
            except Exception as history_exc:
                if proxy:
                    self.logger.warning(
                        f"⚠️ [YFinance] Proxy {proxy} error for history: {history_exc}, retrying directly..."
                    )
                    old_proxies = dict(getattr(session, "proxies", {}) or {})
                    try:
                        session.proxies = {}  # Clear proxies in session
                        self._set_yfinance_proxy(None)
                        price_history = ticker_obj.history(
                            period=period,
                            interval=interval,
                            repair=repair_history,
                            actions=False,
                            timeout=timeout,
                        )
                    finally:
                        session.proxies = old_proxies
                        self._set_yfinance_proxy(proxy)
                else:
                    raise history_exc

            data["priceHistory"] = self._summarize_history(price_history)
            history_metadata = self._get_history_metadata(ticker_obj)
            if history_metadata:
                data["priceHistoryMetadata"] = history_metadata

            # Calculate 50-day and 200-day moving averages from price history
            try:
                if (
                    price_history is not None
                    and not price_history.empty
                    and "Close" in price_history.columns
                ):
                    close_prices = price_history["Close"]

                    # 50-day MA - son 50 gün ortalaması
                    if len(close_prices) >= 50:
                        ma50_values = close_prices.iloc[-50:].dropna()
                        if len(ma50_values) > 0:
                            ma50 = float(ma50_values.mean())
                            if not (
                                ma50 != ma50 or ma50 == float("inf")
                            ):  # Check for NaN and inf
                                data["fiftyDayAverage"] = ma50

                    # 200-day MA - son 200 gün ortalaması
                    if len(close_prices) >= 200:
                        ma200_values = close_prices.iloc[-200:].dropna()
                        if len(ma200_values) > 0:
                            ma200 = float(ma200_values.mean())
                            if not (
                                ma200 != ma200 or ma200 == float("inf")
                            ):  # Check for NaN and inf
                                data["twoHundredDayAverage"] = ma200

                    # RSI (14-period) - Wilder's Smoothing
                    if len(close_prices) >= 15:
                        delta = close_prices.diff()

                        # Calculate gains and losses
                        up = delta.where(delta > 0, 0.0)
                        down = -delta.where(delta < 0, 0.0)

                        # Use Exponential Moving Average with com=13 (alpha=1/14)
                        # This matches the standard Wilder's RSI calculation used by TradingView/Finviz
                        ma_up = up.ewm(com=13, adjust=False, min_periods=14).mean()
                        ma_down = down.ewm(com=13, adjust=False, min_periods=14).mean()

                        rs = ma_up / ma_down
                        rsi_series = 100 - (100 / (1 + rs))

                        rsi = rsi_series.iloc[-1]

                        if not (
                            rsi != rsi or rsi == float("inf")
                        ):  # Check for NaN and inf
                            data["rsi"] = float(rsi)
            except Exception as indicator_exc:
                self.logger.debug(
                    "Technical indicator calculation failed (%s): %s",
                    ticker,
                    indicator_exc,
                )
        except Exception as exc:
            self.logger.debug("Price history could not be retrieved (%s): %s", ticker, exc)

        statements = self._build_financial_statement_snapshot(ticker_obj)
        if statements:
            data["financialStatements"] = statements

        sec_filings = self._build_sec_filings_snapshot(ticker_obj)
        if sec_filings:
            data["secFilings"] = sec_filings

        earnings_trend = self._build_earnings_trend(ticker_obj)
        if earnings_trend:
            data["earningsTrend"] = earnings_trend

        analyst_snapshot = self._build_analyst_snapshot(info, ticker_obj)
        if analyst_snapshot:
            data["analystRecommendations"] = analyst_snapshot

        recommendations = self._build_recommendations_snapshot(ticker_obj)
        if recommendations:
            data["recommendations"] = recommendations

        recommendations_summary = self._build_recommendations_summary(ticker_obj)
        if recommendations_summary:
            data["recommendationsSummary"] = recommendations_summary

        eps_revisions = self._build_eps_revisions_snapshot(ticker_obj)
        if eps_revisions:
            data["epsRevisions"] = eps_revisions

        eps_trend = self._build_eps_trend_snapshot(ticker_obj)
        if eps_trend:
            data["epsTrend"] = eps_trend

        growth_estimates = self._build_growth_estimates_snapshot(ticker_obj)
        if growth_estimates:
            data["growthEstimates"] = growth_estimates

            # Flatten growth estimates to top level for backward compatibility
            # Extract earningsGrowth from Long-Term Growth (LTG) or +1y
            ltg_data = growth_estimates.get("LTG", {})
            earnings_growth = ltg_data.get("stock")

            if earnings_growth is None:
                py_data = growth_estimates.get("+1y", {})
                earnings_growth = py_data.get("stock")

            if earnings_growth is not None:
                data["earningsGrowth"] = earnings_growth

            # Calculate PEG Ratio if not available from yfinance
            # PEG = P/E Ratio / Growth Rate (%)
            # Note: growth_estimates stores rates as decimals (0.0883 = 8.83%), convert to percentage
            if data.get("pegRatio") in (None, NA_VALUE):
                try:
                    pe_ratio = data.get("trailingPE")
                    if not isinstance(pe_ratio, (int, float)) or pe_ratio <= 0:
                        raise ValueError("Invalid P/E ratio")

                    # Get growth rate from LTG (Long-Term Growth), fallback to +1y
                    ltg_data = growth_estimates.get("LTG", {})
                    growth_rate = ltg_data.get("stock")

                    if growth_rate is None:
                        py_data = growth_estimates.get("+1y", {})
                        growth_rate = py_data.get("stock")

                    if isinstance(growth_rate, (int, float)) and growth_rate > 0:
                        # Convert decimal to percentage (0.0883 → 8.83)
                        growth_rate_percent = growth_rate * 100
                        calculated_peg = pe_ratio / growth_rate_percent

                        # Sanity check: PEG typically ranges from 0.5 to 50
                        if 0 < calculated_peg < 1000:
                            data["pegRatio"] = calculated_peg
                except Exception as peg_exc:
                    self.logger.debug(
                        "PEG calculation failed (%s): %s", ticker, peg_exc
                    )

        earnings_estimate = self._build_earnings_estimate_snapshot(ticker_obj)
        if earnings_estimate:
            data["earningsEstimate"] = earnings_estimate

            # Flatten quarterly EPS estimates to top level
            # Extract from current quarter (0q) or next quarter (+1q)
            cq_eps = earnings_estimate.get("0q", {})
            nq_eps = earnings_estimate.get("+1q", {})

            # Next quarter EPS: prefer +1q, fallback to 0q
            eps_next_quarter = (
                nq_eps.get("avg") if nq_eps.get("avg") else cq_eps.get("avg")
            )
            if eps_next_quarter is not None:
                data["estimated_eps_avg_next_quarter"] = eps_next_quarter

            # Next year EPS: prefer +1y, fallback to 0y
            cy_eps = earnings_estimate.get("0y", {})
            ny_eps = earnings_estimate.get("+1y", {})
            eps_next_year = (
                ny_eps.get("avg") if ny_eps.get("avg") else cy_eps.get("avg")
            )
            if eps_next_year is not None:
                data["estimated_eps_avg_next_year"] = eps_next_year

            # Analyst count from most populated period
            analyst_counts = [
                nq_eps.get("numberOfAnalysts"),
                cq_eps.get("numberOfAnalysts"),
                ny_eps.get("numberOfAnalysts"),
                cy_eps.get("numberOfAnalysts"),
            ]
            valid_counts = [c for c in analyst_counts if c is not None and c != "N/A"]
            if valid_counts:
                data["analyst_estimates_count"] = max(valid_counts)

        revenue_estimate = self._build_revenue_estimate_snapshot(ticker_obj)
        if revenue_estimate:
            data["revenueEstimate"] = revenue_estimate

            # Flatten quarterly revenue estimates to top level
            cq_rev = revenue_estimate.get("0q", {})
            nq_rev = revenue_estimate.get("+1q", {})

            # Next quarter revenue: prefer +1q, fallback to 0q
            rev_next_quarter = (
                nq_rev.get("avg") if nq_rev.get("avg") else cq_rev.get("avg")
            )
            if rev_next_quarter is not None:
                data["estimated_revenue_avg_next_quarter"] = rev_next_quarter

            # Next year revenue: prefer +1y, fallback to 0y
            cy_rev = revenue_estimate.get("0y", {})
            ny_rev = revenue_estimate.get("+1y", {})
            rev_next_year = (
                ny_rev.get("avg") if ny_rev.get("avg") else cy_rev.get("avg")
            )
            if rev_next_year is not None:
                data["estimated_revenue_avg_next_year"] = rev_next_year

            # Flatten revenue growth to top level for backward compatibility
            # Extract revenueGrowth from next quarter (+1q) or current quarter (0q)
            pq_data = revenue_estimate.get("+1q", {})
            revenue_growth = pq_data.get("growth")

            if revenue_growth is None:
                cq_data = revenue_estimate.get("0q", {})
                revenue_growth = cq_data.get("growth")

            if revenue_growth is not None:
                data["revenueGrowth"] = revenue_growth

        upgrades_downgrades = self._build_upgrades_downgrades_snapshot(ticker_obj)
        if upgrades_downgrades:
            data["upgradesDowngrades"] = upgrades_downgrades

        sustainability = self._build_sustainability_snapshot(ticker_obj)
        if sustainability:
            data["sustainability"] = sustainability

        insider_snapshot = self._build_insider_snapshot(ticker_obj)
        if insider_snapshot:
            data.update(insider_snapshot)

        events_snapshot = self._build_events_snapshot(ticker_obj)
        if events_snapshot:
            data["corporateEvents"] = events_snapshot

            # Extract last reported earnings date from events
            latest_reported = self._extract_last_reported_earnings_date(events_snapshot)

            # Fallback: If earningsDates is empty/old, try earningsTrend for most recent period
            if not latest_reported and data.get("earningsTrend"):
                # earningsTrend is sorted newest first, get the first valid entry
                for trend_entry in data["earningsTrend"]:
                    if isinstance(trend_entry, dict):
                        period = trend_entry.get("period")
                        if period and period not in (None, "", NA_VALUE):
                            try:
                                # Extract date from period string (e.g., "2025-06-30 00:00:00")
                                period_str = str(period).split()[
                                    0
                                ]  # Get "2025-06-30" part
                                parsed = datetime.fromisoformat(period_str).date()
                                if parsed <= date.today():
                                    latest_reported = parsed.isoformat()
                                    break
                            except (ValueError, IndexError):
                                continue

            if latest_reported:
                data["lastEarningsDate"] = latest_reported

            # Update next earnings date with better source
            next_earnings = self._extract_next_earnings_date(events_snapshot)
            if next_earnings:
                data["nextEarningsDate"] = next_earnings

            # Update next dividend date with better logic
            next_dividend = self._extract_next_dividend_date(
                ticker_obj, info, events_snapshot
            )
            if next_dividend and next_dividend != NA_VALUE:
                data["nextDividendDate"] = next_dividend

        holders_snapshot = self._build_holders_snapshot(ticker_obj, info)
        if holders_snapshot:
            data["ownership"] = holders_snapshot
            self._deduplicate_ownership_summary(data)

        # New metrics - Analyst Price Targets
        analyst_price_targets = self._build_analyst_price_targets_snapshot(ticker_obj)
        if analyst_price_targets:
            data["analystPriceTargets"] = analyst_price_targets

        # New metrics - Capital Gains
        capital_gains = self._build_capital_gains_snapshot(ticker_obj)
        if capital_gains:
            data["capitalGains"] = capital_gains

        if events_snapshot and events_snapshot.get("upcomingEvents"):
            data["upcomingEvents"] = events_snapshot.get("upcomingEvents")

        news_snapshot = self._build_news_snapshot(ticker_obj)
        if news_snapshot:
            data["news"] = news_snapshot

        # Prune low-signal metadata before returning to AI
        self._prune_low_signal_fields(data)

        return data

    def _build_options_summary(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """Return a compact options snapshot with minimal token footprint."""
        try:
            meta = self.get_options_data(ticker)
            if not meta or not meta.get("has_options"):
                return {}

            expirations = meta.get("expirations") or []
            expirations = expirations[:5]

            summary: Dict[str, Any] = {"expirations": expirations}

            if expirations:
                nearest = expirations[0]
                chain = self.get_option_chain(ticker, nearest)
                calls = chain.get("calls") or []
                puts = chain.get("puts") or []

                put_volume = sum(
                    self._coerce_numeric(p.get("volume")) or 0 for p in puts
                )
                call_volume = sum(
                    self._coerce_numeric(c.get("volume")) or 0 for c in calls
                )

                ratio = None
                if call_volume:
                    ratio = round(put_volume / call_volume, 4)

                summary["nearestExpiration"] = {
                    "date": nearest,
                    "putVolume": put_volume,
                    "callVolume": call_volume,
                    "putCallRatio": ratio,
                }

            return summary
        except Exception as exc:
            self.logger.debug("Options summary could not be created: %s", exc)
            return {}

    def _build_funds_snapshot_lite(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """Return a trimmed fund snapshot to avoid overloading the AI payload."""
        try:
            snapshot = self._build_funds_snapshot(ticker)
            if not snapshot:
                return {}

            top_holdings = snapshot.get("top_holdings") or []
            trimmed_holdings = (
                top_holdings[:10] if isinstance(top_holdings, list) else []
            )

            cleaned = {
                "description": snapshot.get("description"),
                "sector_weightings": snapshot.get("sector_weightings"),
                "asset_classes": snapshot.get("asset_classes"),
                "top_holdings": trimmed_holdings,
            }

            return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
        except Exception as exc:
            self.logger.debug("Fund summary could not be created: %s", exc)
            return {}

    def _prune_low_signal_fields(self, data: Dict[str, Any]) -> None:
        """Drop noisy info fields that add token cost but little analytical value."""
        low_signal_keys = {
            "logo_url",
            "logoUrl",
            "address1",
            "address2",
            "city",
            "state",
            "zip",
            "phone",
            "fax",
            "website",
            "uuid",
            "messageBoardId",
            "gmtOffSetMilliseconds",
            "quoteSourceName",
        }

        for key in low_signal_keys:
            data.pop(key, None)

    # Removed: _format_proxy method is no longer needed
    # yfinance handles proxy parameter directly
