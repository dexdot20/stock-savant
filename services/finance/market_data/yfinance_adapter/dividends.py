"""Dividend snapshot utilities for YFinance adapter."""

from __future__ import annotations

from typing import Any, Dict, List

import yfinance as yf

from config import NA_VALUE


class YFinanceDividendMixin:
    """Builds dividend related snapshots."""

    def _build_dividend_snapshot(
        self,
        ticker: yf.Ticker,
        current_price: Any,
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build dividend snapshot using property access.
        According to yfinance documentation, use ticker.dividends instead of ticker.get_dividends().
        """
        snapshot: Dict[str, Any] = {}

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # Use property access instead of get_dividends()
                dividends_raw = self._call_with_retry(
                    lambda: ticker.dividends, name="ticker.dividends"
                )
        except Exception as exc:
            self.logger.debug("Dividend data could not be retrieved: %s", exc)
            dividends_raw = None

        series = self._ensure_series(dividends_raw)
        if series is None or series.empty:
            return {
                "dividendYield": self._convert_dividend_yield(
                    info.get("dividendYield")
                ),
                "dividendFrequency": info.get("dividendDate") and "annual" or NA_VALUE,
                "lastDividendDate": self._timestamp_to_iso(
                    info.get("lastDividendDate")
                ),
            }

        try:
            series = series.dropna()
            if series.empty:
                return {}

            series = series.sort_index()
            last_amount = self._coerce_numeric(series.iloc[-1])
            last_date = self._normalize_calendar_value(series.index[-1])
            if last_amount not in (None, NA_VALUE):
                snapshot["lastDividendAmount"] = last_amount
            if last_date not in (None, NA_VALUE):
                snapshot["lastDividendDate"] = last_date

            frequency = self._determine_dividend_frequency(series.index)
            if frequency:
                snapshot["dividendFrequency"] = frequency

            price_value = self._coerce_numeric(current_price)
            if isinstance(price_value, (int, float)) and price_value > 0:
                try:
                    import pandas as pd  # type: ignore

                    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=365)
                    index = series.index
                    if hasattr(index, "tz") and index.tz is not None:
                        cutoff = cutoff.tz_convert(index.tz)
                    else:
                        cutoff = cutoff.tz_localize(None)
                    recent = series[index >= cutoff]
                except Exception:
                    recent = series.tail(4)
                if recent.empty:
                    recent = series.tail(4)
                ttm_dividend = float(recent.sum()) if not recent.empty else 0.0
                if ttm_dividend > 0:
                    yield_value = (ttm_dividend / float(price_value)) * 100
                    if 0 <= yield_value <= 20:
                        snapshot["dividendYield"] = round(yield_value, 4)
        except Exception as exc:
            self.logger.debug("Dividend data could not be processed: %s", exc)

        snapshot.setdefault(
            "dividendYield", self._convert_dividend_yield(info.get("dividendYield"))
        )
        snapshot.setdefault(
            "lastDividendDate", self._timestamp_to_iso(info.get("lastDividendDate"))
        )

        # Don't set nextDividendDate here - it will be calculated separately with better logic
        # snapshot.setdefault("nextDividendDate", self._timestamp_to_iso(info.get("dividendDate")))

        return snapshot

    def _build_splits_snapshot(self, ticker: yf.Ticker) -> List[Dict[str, Any]]:
        """
        Get stock splits using property access.
        According to yfinance documentation, use ticker.splits instead of ticker.get_splits().

        Returns list of historical stock splits with dates and ratios.
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                splits_raw = self._call_with_retry(
                    lambda: ticker.splits, name="ticker.splits"
                )
        except Exception as exc:
            self.logger.debug("Splits data could not be retrieved: %s", exc)
            return []

        series = self._ensure_series(splits_raw)
        if series is None or series.empty:
            return []

        try:
            series = series.dropna()
            if series.empty:
                return []

            splits_list = []
            for index, value in series.items():
                split_date = self._normalize_calendar_value(index)
                split_ratio = self._coerce_numeric(value)

                if split_date not in (None, NA_VALUE) and split_ratio not in (
                    None,
                    NA_VALUE,
                ):
                    splits_list.append(
                        {
                            "date": split_date,
                            "ratio": split_ratio,
                        }
                    )

            return splits_list
        except Exception as exc:
            self.logger.debug("Splits data could not be processed: %s", exc)
            return []

    def _build_actions_snapshot(self, ticker: yf.Ticker) -> List[Dict[str, Any]]:
        """
        Get corporate actions (dividends + splits combined) using property access.
        According to yfinance documentation, use ticker.actions instead of ticker.get_actions().

        Returns combined list of dividends and splits in chronological order.
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                actions_raw = self._call_with_retry(
                    lambda: ticker.actions, name="ticker.actions"
                )
        except Exception as exc:
            self.logger.debug("Actions data could not be retrieved: %s", exc)
            return []

        df = self._ensure_dataframe(actions_raw)
        if df is None or df.empty:
            return []

        try:
            df = df.dropna(how="all")
            if df.empty:
                return []

            actions_list = []
            for index, row in df.tail(50).iterrows():  # Son 50 aksiyon
                action_date = self._normalize_calendar_value(index)

                dividend = self._coerce_numeric(row.get("Dividends"))
                split = self._coerce_numeric(row.get("Stock Splits"))

                if dividend not in (None, NA_VALUE, 0):
                    actions_list.append(
                        {
                            "date": action_date,
                            "type": "dividend",
                            "value": dividend,
                        }
                    )

                if split not in (None, NA_VALUE, 0, 1):
                    actions_list.append(
                        {
                            "date": action_date,
                            "type": "split",
                            "ratio": split,
                        }
                    )

            return actions_list
        except Exception as exc:
            self.logger.debug("Actions data could not be processed: %s", exc)
            return []

    def _build_capital_gains_snapshot(self, ticker: yf.Ticker) -> List[Dict[str, Any]]:
        """
        Get capital gains using property access.
        According to yfinance documentation, use ticker.capital_gains instead of ticker.get_capital_gains().

        Returns list of capital gains distributions (primarily for funds/ETFs).
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                capital_gains_raw = self._call_with_retry(
                    lambda: ticker.capital_gains, name="ticker.capital_gains"
                )
        except Exception as exc:
            self.logger.debug("Capital gains data could not be retrieved: %s", exc)
            return []

        series = self._ensure_series(capital_gains_raw)
        if series is None or series.empty:
            return []

        try:
            series = series.dropna()
            if series.empty:
                return []

            gains_list = []
            for index, value in series.items():
                gain_date = self._normalize_calendar_value(index)
                gain_amount = self._coerce_numeric(value)

                if gain_date not in (None, NA_VALUE) and gain_amount not in (
                    None,
                    NA_VALUE,
                ):
                    gains_list.append(
                        {
                            "date": gain_date,
                            "amount": gain_amount,
                        }
                    )

            return gains_list
        except Exception as exc:
            self.logger.debug("Capital gains data could not be processed: %s", exc)
            return []

    def _extract_next_dividend_date(
        self, ticker: yf.Ticker, info: Dict[str, Any], events_snapshot: Dict[str, Any]
    ) -> Any:
        """
        Extract or estimate next dividend date from multiple sources.

        Priority order:
        1. Calendar upcoming events (most reliable)
        2. API info fields (if future date)
        3. Historical pattern estimation (last resort)
        """
        from datetime import date, datetime, timedelta

        today = date.today()

        # 1. Try calendar upcoming events
        upcoming = events_snapshot.get("upcomingEvents", {}).get("raw", {})
        dividend_date = upcoming.get("Dividend Date") or upcoming.get(
            "Ex-Dividend Date"
        )
        if dividend_date:
            iso_date = self._timestamp_to_iso(dividend_date)
            if iso_date != NA_VALUE:
                try:
                    parsed = datetime.fromisoformat(iso_date).date()
                    if parsed >= today:
                        self.logger.debug(f"Next dividend from calendar: {iso_date}")
                        return iso_date
                except (ValueError, AttributeError):
                    pass

        # 2. Try API info fields - check if they're future dates
        api_dates = [
            info.get("dividendDate"),
            info.get("exDividendDate"),
        ]
        for api_date in api_dates:
            if api_date:
                iso_date = self._timestamp_to_iso(api_date)
                if iso_date != NA_VALUE:
                    try:
                        parsed = datetime.fromisoformat(iso_date).date()
                        if parsed >= today:
                            self.logger.debug(f"Next dividend from API: {iso_date}")
                            return iso_date
                    except (ValueError, AttributeError):
                        pass

        # 3. Estimate from historical pattern
        last_div_date = info.get("lastDividendDate") or info.get("exDividendDate")
        dividend_freq = info.get("dividendFrequency", "annual")

        if last_div_date:
            try:
                last_date = datetime.fromisoformat(str(last_div_date)).date()

                # Map frequency to days
                freq_days = {
                    "annual": 365,
                    "quarterly": 90,
                    "monthly": 30,
                    "semi-annual": 182,
                    "biannual": 182,
                }
                days_delta = freq_days.get(
                    str(dividend_freq).lower(), 90
                )  # Default to quarterly

                # Estimate next date
                next_estimated = last_date + timedelta(days=days_delta)

                # Only return if it's a future date
                if next_estimated >= today:
                    self.logger.debug(
                        f"Next dividend estimated: {next_estimated.isoformat()} "
                        f"(from {last_date.isoformat()} + {days_delta} days)"
                    )
                    return next_estimated.isoformat()
            except Exception as exc:
                self.logger.debug(f"Dividend date estimation failed: {exc}")

        return NA_VALUE
