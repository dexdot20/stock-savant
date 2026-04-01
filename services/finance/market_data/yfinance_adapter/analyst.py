"""Analyst and recommendation helpers for YFinance adapter."""

from __future__ import annotations

from typing import Any, Dict, List

import yfinance as yf

from config import NA_VALUE


class YFinanceAnalystMixin:
    """Provides analyst related snapshot builders."""

    def _build_analyst_snapshot(
        self, info: Dict[str, Any], ticker: yf.Ticker = None
    ) -> Dict[str, Any]:
        """Build analyst snapshot using both info dict and dedicated API methods."""
        if not info:
            return {}

        snapshot = {
            "ratingKey": info.get("recommendationKey", NA_VALUE),
            "ratingScore": info.get("recommendationMean", NA_VALUE),
            "targetMeanPrice": info.get("targetMeanPrice", NA_VALUE),
            "targetHighPrice": info.get("targetHighPrice", NA_VALUE),
            "targetLowPrice": info.get("targetLowPrice", NA_VALUE),
            "targetMedianPrice": info.get("targetMedianPrice", NA_VALUE),
            "numberOfAnalysts": info.get("numberOfAnalystOpinions", NA_VALUE),
        }

        # Fallback: Use info dict values if available, otherwise empty
        # Dedicated snapshots (get_analyst_price_targets) are called separately
        return {
            key: value
            for key, value in snapshot.items()
            if value not in (None, "", NA_VALUE)
        }

    def _build_analyst_price_targets_snapshot(
        self, ticker: yf.Ticker
    ) -> Dict[str, Any]:
        """
        Get analyst price targets using property access.
        According to yfinance documentation, use ticker.analyst_price_targets instead of
        ticker.get_analyst_price_targets().

        Returns detailed analyst price targets including current, mean, low, high, median prices.
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                price_targets = self._call_with_retry(
                    lambda: ticker.analyst_price_targets,
                    name="ticker.analyst_price_targets",
                )
        except Exception as exc:
            self.logger.debug("Analyst price targets could not be retrieved: %s", exc)
            return {}

        if not isinstance(price_targets, dict):
            return {}

        try:
            snapshot = {
                "current": self._coerce_numeric(price_targets.get("current")),
                "mean": self._coerce_numeric(price_targets.get("mean")),
                "low": self._coerce_numeric(price_targets.get("low")),
                "high": self._coerce_numeric(price_targets.get("high")),
                "median": self._coerce_numeric(price_targets.get("median")),
            }

            # Calculate upside/downside potential if current and mean are available
            current = snapshot.get("current")
            mean = snapshot.get("mean")
            if (
                isinstance(current, (int, float))
                and isinstance(mean, (int, float))
                and current > 0
            ):
                upside = ((mean - current) / current) * 100
                snapshot["upsidePotential"] = round(upside, 2)

            return {k: v for k, v in snapshot.items() if v not in (None, "", NA_VALUE)}
        except Exception as exc:
            self.logger.debug("Analyst price targets could not be processed: %s", exc)
            return {}

    def _build_recommendations_snapshot(
        self, ticker: yf.Ticker
    ) -> List[Dict[str, Any]]:
        """
        Get recommendations using property access.
        According to yfinance documentation, use ticker.recommendations instead of ticker.get_recommendations().
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = self._call_with_retry(
                    lambda: ticker.recommendations, name="ticker.recommendations"
                )
        except Exception as exc:
            self.logger.debug("Recommendations could not be retrieved: %s", exc)
            df = None

        frame = self._ensure_dataframe(df)
        if frame is None or frame.empty:
            return []

        try:
            records = []
            for index, row in frame.iterrows():
                # Convert period to ISO format, skip if invalid (e.g., 1970-01-01 placeholder)
                period_iso = self._timestamp_to_iso(index)

                # Filter out invalid placeholder dates (1970-01-01)
                if period_iso == "1970-01-01" or period_iso == NA_VALUE:
                    continue

                record: Dict[str, Any] = {
                    "period": period_iso,
                    "strongBuy": self._coerce_numeric(row.get("strongBuy")),
                    "buy": self._coerce_numeric(row.get("buy")),
                    "hold": self._coerce_numeric(row.get("hold")),
                    "sell": self._coerce_numeric(row.get("sell")),
                    "strongSell": self._coerce_numeric(row.get("strongSell")),
                }
                records.append(record)
            return records
        except Exception as exc:
            self.logger.debug("Recommendations could not be processed: %s", exc)
            return []

    def _build_recommendations_summary(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get summary of analyst recommendations using property access.
        According to yfinance documentation, use ticker.recommendations_summary instead of ticker.get_recommendations_summary().
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                summary = self._call_with_retry(
                    lambda: ticker.recommendations_summary,
                    name="ticker.recommendations_summary",
                )
        except Exception as exc:
            self.logger.debug("Recommendations summary could not be retrieved: %s", exc)
            return {}

        if isinstance(summary, dict):
            return {
                "strongBuy": self._coerce_numeric(summary.get("strongBuy", 0)),
                "buy": self._coerce_numeric(summary.get("buy", 0)),
                "hold": self._coerce_numeric(summary.get("hold", 0)),
                "sell": self._coerce_numeric(summary.get("sell", 0)),
                "strongSell": self._coerce_numeric(summary.get("strongSell", 0)),
            }

        return {}

    def _build_growth_estimates_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get growth estimates using property access.
        According to yfinance documentation, use ticker.growth_estimates instead of ticker.get_growth_estimates().

        Note: yfinance returns columns 'stockTrend' and 'indexTrend', not 'stock' and 'index'
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = self._call_with_retry(
                    lambda: ticker.growth_estimates, name="ticker.growth_estimates"
                )
        except Exception as exc:
            self.logger.debug("Growth estimates could not be retrieved: %s", exc)
            df = None

        frame = self._ensure_dataframe(df)
        if frame is None or frame.empty:
            return {}

        # Helper to extract numeric value while preserving None (not converting to 'N/A')
        # This is critical for _sanitize_company_data to work correctly
        def extract_numeric_or_none(value):
            """Convert value to float or None, preserving None instead of 'N/A'."""
            if value is None:
                return None
            try:
                numeric = float(value)
                return None if numeric != numeric else numeric  # NaN -> None
            except (TypeError, ValueError):
                return None

        try:
            result: Dict[str, Any] = {}
            for period in frame.index:
                period_str = str(period)
                row = frame.loc[period]

                # Use correct column names from yfinance: stockTrend and indexTrend
                result[period_str] = {
                    "stock": extract_numeric_or_none(row.get("stockTrend")),
                    "industry": extract_numeric_or_none(row.get("industry")),
                    "sector": extract_numeric_or_none(row.get("sector")),
                    "index": extract_numeric_or_none(row.get("indexTrend")),
                }
            return result
        except Exception as exc:
            self.logger.debug("Growth estimates could not be processed: %s", exc)
            return {}

    def _build_revenue_estimate_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get revenue estimates using property access.
        According to yfinance documentation, use ticker.revenue_estimate instead of ticker.get_revenue_estimate().
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = self._call_with_retry(
                    lambda: ticker.revenue_estimate, name="ticker.revenue_estimate"
                )
        except Exception as exc:
            self.logger.debug("Revenue estimates could not be retrieved: %s", exc)
            df = None

        frame = self._ensure_dataframe(df)
        if frame is None or frame.empty:
            return {}

        try:
            result: Dict[str, Any] = {}
            for period in frame.index:
                period_str = str(period)
                row = frame.loc[period]
                result[period_str] = {
                    "avg": self._coerce_numeric(row.get("avg")),
                    "low": self._coerce_numeric(row.get("low")),
                    "high": self._coerce_numeric(row.get("high")),
                    "numberOfAnalysts": self._coerce_numeric(
                        row.get("numberOfAnalysts")
                    ),
                    "yearAgoRevenue": self._coerce_numeric(row.get("yearAgoRevenue")),
                    "growth": self._coerce_numeric(row.get("growth")),
                }
            return result
        except Exception as exc:
            self.logger.debug("Revenue estimates could not be processed: %s", exc)
            return {}

    def _build_earnings_estimate_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get earnings estimates for future periods using property access.
        According to yfinance documentation, use ticker.earnings_estimate instead of ticker.get_earnings_estimate().
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = self._call_with_retry(
                    lambda: ticker.earnings_estimate, name="ticker.earnings_estimate"
                )
        except Exception as exc:
            self.logger.debug("Earnings estimates could not be retrieved: %s", exc)
            df = None

        frame = self._ensure_dataframe(df)
        if frame is None or frame.empty:
            return {}

        try:
            result: Dict[str, Any] = {}
            for period in frame.index:
                period_str = str(period)
                row = frame.loc[period]
                result[period_str] = {
                    "avg": self._coerce_numeric(row.get("avg")),
                    "low": self._coerce_numeric(row.get("low")),
                    "high": self._coerce_numeric(row.get("high")),
                    "numberOfAnalysts": self._coerce_numeric(
                        row.get("numberOfAnalysts")
                    ),
                    "yearAgoEarnings": self._coerce_numeric(row.get("yearAgoEarnings")),
                    "growth": self._coerce_numeric(row.get("growth")),
                }
            return result
        except Exception as exc:
            self.logger.debug("Earnings estimates could not be processed: %s", exc)
            return {}

    def _build_upgrades_downgrades_snapshot(
        self, ticker: yf.Ticker
    ) -> List[Dict[str, Any]]:
        """
        Get upgrades and downgrades using property access.
        According to yfinance documentation, use ticker.upgrades_downgrades instead of ticker.get_upgrades_downgrades().

        Note: yfinance returns columns with capital letters: 'Firm', 'ToGrade', 'FromGrade', 'Action'
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = self._call_with_retry(
                    lambda: ticker.upgrades_downgrades,
                    name="ticker.upgrades_downgrades",
                )
        except Exception as exc:
            self.logger.debug("Upgrades/Downgrades could not be retrieved: %s", exc)
            df = None

        frame = self._ensure_dataframe(df)
        if frame is None or frame.empty:
            return []

        try:
            records = []
            for index, row in frame.head(20).iterrows():
                # Use capitalized column names as returned by yfinance
                record: Dict[str, Any] = {
                    "date": self._timestamp_to_iso(index),
                    "firm": row.get("Firm"),  # Capital F
                    "toGrade": row.get("ToGrade"),  # Capital T and G
                    "fromGrade": row.get("FromGrade"),  # Capital F and G
                    "action": row.get("Action"),  # Capital A
                }
                records.append(record)
            return records
        except Exception as exc:
            self.logger.debug("Upgrades/Downgrades could not be processed: %s", exc)
            return []
