"""Data accessor mixins for YFinance adapter - consolidated utility mixins."""

from __future__ import annotations

from typing import Any, Dict, List

import yfinance as yf


class YFinanceNewsMixin:
    """Builds news snapshots from yfinance output."""

    def _build_news_snapshot(self, ticker: yf.Ticker) -> List[Dict[str, Any]]:
        """
        Get trimmed ticker news using property access.
        """
        limit = int(self.market_config.get("yfinance_news_limit", 5))
        if limit <= 0:
            return []

        try:
            news_items = self._call_with_retry(lambda: ticker.news, name="ticker.news")
        except Exception as exc:
            self.logger.debug("Ticker news could not be retrieved: %s", exc)
            return []

        if not isinstance(news_items, list):
            return []

        snapshots: List[Dict[str, Any]] = []
        for item in news_items[:limit]:
            if not isinstance(item, dict):
                continue

            payload = item.get("content") if isinstance(item.get("content"), dict) else item

            canonical = payload.get("canonicalUrl")
            click_through = payload.get("clickThroughUrl")
            provider = payload.get("provider")

            title = str(payload.get("title") or payload.get("headline") or "").strip()
            link = str(
                payload.get("link")
                or (canonical.get("url") if isinstance(canonical, dict) else canonical)
                or (click_through.get("url") if isinstance(click_through, dict) else click_through)
                or ""
            ).strip()
            publisher = str(
                payload.get("publisher")
                or (provider.get("displayName") if isinstance(provider, dict) else provider)
                or ""
            ).strip()
            summary = str(
                payload.get("summary") or payload.get("description") or ""
            ).strip()

            snapshot = {
                "title": title,
                "publisher": publisher,
                "link": link,
                "publishedAt": self._timestamp_to_iso(
                    payload.get("providerPublishTime")
                    or payload.get("pubDate")
                    or payload.get("displayTime")
                ),
                "summary": summary,
                "type": payload.get("type") or payload.get("contentType") or "news",
                "relatedTickers": [
                    str(value).strip()
                    for value in (payload.get("relatedTickers") or [])
                    if str(value).strip()
                ],
            }

            cleaned = {
                key: value
                for key, value in snapshot.items()
                if value not in (None, "", [], {})
            }
            if cleaned.get("title") and cleaned.get("link"):
                snapshots.append(cleaned)

        return snapshots


class YFinanceSustainabilityMixin:
    """Builds sustainability snapshots from ticker data."""

    def _build_sustainability_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get sustainability/ESG scores using property access.
        According to yfinance documentation, use ticker.sustainability instead of ticker.get_sustainability().
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                sustainability = self._call_with_retry(
                    lambda: ticker.sustainability, name="ticker.sustainability"
                )
        except Exception as exc:
            self.logger.debug("Sustainability data could not be retrieved: %s", exc)
            sustainability = None

        df = self._ensure_dataframe(sustainability)
        if df is None or df.empty:
            return {}

        try:
            df_t = df.transpose()
            if df_t.empty:
                return {}
            series = df_t.iloc[0]
        except Exception:
            return {}

        metrics: Dict[str, Any] = {}
        for key in [
            "environmentScore",
            "socialScore",
            "governanceScore",
            "totalEsg",
            "peerGroup",
            "percentile",
            "esgPerformance",
            "highestControversy",
        ]:
            value = series.get(key)
            if value is not None and value == value:
                metrics[key] = (
                    self._coerce_numeric(value)
                    if key != "peerGroup" and key != "esgPerformance"
                    else value
                )

        return metrics


class YFinanceFundsMixin:
    """Provides funds (ETF/Mutual Fund) data capabilities."""

    def _build_funds_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get funds data including holdings and sector weightings.
        """
        try:
            # funds_data property returns a FundsData object
            funds_data = self._call_with_retry(
                lambda: ticker.funds_data, name="ticker.funds_data"
            )

            if not funds_data:
                return {}

            return {
                "description": funds_data.description,
                "top_holdings": self._dataframe_to_records(funds_data.top_holdings),
                "sector_weightings": funds_data.sector_weightings,
                "asset_classes": funds_data.asset_classes,
                "fund_overview": funds_data.fund_overview,
                "fund_operations": funds_data.fund_operations,
            }
        except Exception as exc:
            # Not all tickers are funds, so this is expected for stocks
            self.logger.debug("Fund data could not be retrieved (may be a regular stock): %s", exc)
            return {}
