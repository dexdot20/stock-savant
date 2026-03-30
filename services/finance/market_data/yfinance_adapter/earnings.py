"""Earnings related helpers for YFinance adapter."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf

from config import NA_VALUE


class YFinanceEarningsMixin:
    """Processes earnings history and corporate events."""

    def _build_eps_history_lookup(self, ticker: yf.Ticker) -> Dict[str, Dict[str, Any]]:
        """
        Build EPS history lookup using property access.
        According to yfinance documentation, use ticker.earnings_history instead of ticker.get_earnings_history().
        """
        lookup: Dict[str, Dict[str, Any]] = {}

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # Use property access instead of get_earnings_history()
                raw_history = self._call_with_retry(
                    lambda: ticker.earnings_history, name="ticker.earnings_history"
                )
        except Exception as exc:
            self.logger.debug("Earnings geçmişi alınamadı: %s", exc)
            raw_history = None

        if raw_history is None:
            return lookup

        frame = self._ensure_dataframe(raw_history)

        if frame is not None and not frame.empty:
            try:
                import pandas as pd  # type: ignore

                if not isinstance(frame, pd.DataFrame):
                    frame = pd.DataFrame(frame)
            except ImportError:  # pragma: no cover
                frame = None

        if frame is not None and not frame.empty:
            try:
                frame = frame.dropna(how="all")
                for idx in frame.index:
                    row = frame.loc[idx]
                    record = row.to_dict() if hasattr(row, "to_dict") else dict(row)

                    period = self._normalize_calendar_value(idx)
                    if period in (None, "", NA_VALUE):
                        period = record.get("period")
                    if (
                        isinstance(period, str)
                        and len(period) == 6
                        and period.isdigit()
                    ):
                        try:
                            year = int(period[:4])
                            month = int(period[4:])
                            period = (
                                datetime(year, month, 1, tzinfo=timezone.utc)
                                .date()
                                .isoformat()
                            )
                        except Exception:
                            period = str(period)

                    if period in (None, "", NA_VALUE):
                        continue

                    lookup[str(period)] = {
                        "epsActual": self._coerce_numeric(record.get("epsActual")),
                        "epsEstimate": self._coerce_numeric(record.get("epsEstimate")),
                        "surprisePercent": self._coerce_numeric(
                            record.get("surprisePercent")
                        ),
                    }
            except Exception as exc:
                self.logger.debug("Earnings history frame işlenemedi: %s", exc)

        if not lookup and isinstance(raw_history, list):
            for item in raw_history:
                if not isinstance(item, dict):
                    continue
                period_value = (
                    item.get("startdatetime")
                    or item.get("startDatetime")
                    or item.get("date")
                    or item.get("period")
                )
                period = self._normalize_calendar_value(period_value)
                if period in (None, "", NA_VALUE):
                    continue

                lookup[str(period)] = {
                    "epsActual": self._coerce_numeric(item.get("epsActual")),
                    "epsEstimate": self._coerce_numeric(item.get("epsEstimate")),
                    "surprisePercent": self._coerce_numeric(
                        item.get("surprisePercent")
                    ),
                }

        return lookup

    def _build_earnings_dates_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get future earnings dates and past surprises.
        """
        try:
            # earnings_dates property returns a DataFrame
            dates_df = self._call_with_retry(
                lambda: ticker.earnings_dates, name="ticker.earnings_dates"
            )

            if dates_df is None or dates_df.empty:
                return {}

            # Convert index (Timestamp) to string for JSON serialization
            # Note: The index is usually the earnings date
            dates_df.index = dates_df.index.astype(str)

            return self._dataframe_to_records(dates_df)
        except Exception as exc:
            self.logger.debug("Gelecek bilanço tarihleri alınamadı: %s", exc)
            return {}

    def _build_earnings_trend(self, ticker: yf.Ticker) -> List[Dict[str, Any]]:
        """
        Build earnings trend with robust column name matching.
        Handles multiple column name variations from yfinance API.
        """
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            # Alternative column names for revenue and earnings
            revenue_columns = ["Total Revenue", "Revenue", "Total Revenues", "Revenues"]
            earnings_columns = [
                "Net Income",
                "Net Income Common Stockholders",
                "Earnings",
                "Net Earnings",
            ]

            # Try quarterly_income_stmt first
            income_df = self._ensure_dataframe(
                getattr(ticker, "quarterly_income_stmt", None)
            )
            if income_df is not None and not income_df.empty:
                try:
                    income_df = income_df.transpose().dropna(how="all")
                    if not income_df.empty:
                        # Find which columns exist
                        revenue_col = None
                        earnings_col = None

                        for col in revenue_columns:
                            if col in income_df.columns:
                                revenue_col = col
                                break

                        for col in earnings_columns:
                            if col in income_df.columns:
                                earnings_col = col
                                break

                        trend: List[Dict[str, Any]] = []
                        for index, row in income_df.head(8).iterrows():
                            revenue = (
                                self._coerce_numeric(row.get(revenue_col))
                                if revenue_col
                                else NA_VALUE
                            )
                            earnings = (
                                self._coerce_numeric(row.get(earnings_col))
                                if earnings_col
                                else NA_VALUE
                            )

                            # Include row if at least one value is valid
                            if revenue not in (None, NA_VALUE) or earnings not in (
                                None,
                                NA_VALUE,
                            ):
                                trend.append(
                                    {
                                        "period": str(index),
                                        "revenue": revenue,
                                        "earnings": earnings,
                                    }
                                )

                        if trend:
                            self.logger.debug(
                                f"Earnings trend built from income_stmt: {len(trend)} periods (revenue_col={revenue_col}, earnings_col={earnings_col})"
                            )
                            return trend
                except Exception as exc:
                    self.logger.debug("quarterly_income_stmt işlenemedi: %s", exc)

            # Fallback to quarterly_earnings
            try:
                df = self._ensure_dataframe(getattr(ticker, "quarterly_earnings", None))
                if df is not None and not df.empty:
                    df = df.dropna(how="all")
                    if not df.empty:
                        trend = []
                        for index, row in df.head(8).iterrows():
                            revenue = self._coerce_numeric(row.get("Revenue"))
                            earnings = self._coerce_numeric(row.get("Earnings"))

                            # Include row if at least one value is valid
                            if revenue not in (None, NA_VALUE) or earnings not in (
                                None,
                                NA_VALUE,
                            ):
                                trend.append(
                                    {
                                        "period": str(index),
                                        "revenue": revenue,
                                        "earnings": earnings,
                                    }
                                )

                        if trend:
                            self.logger.debug(
                                f"Earnings trend built from quarterly_earnings: {len(trend)} periods"
                            )
                            return trend
            except Exception as exc:
                self.logger.debug("quarterly_earnings işlenemedi: %s", exc)

        return []

    def _build_eps_revisions_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get EPS revisions using property access.
        According to yfinance documentation, use ticker.eps_revisions instead of ticker.get_eps_revisions().
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = self._call_with_retry(
                    lambda: ticker.eps_revisions, name="ticker.eps_revisions"
                )
        except Exception as exc:
            self.logger.debug("EPS revizyonları alınamadı: %s", exc)
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
                    "upLast7days": self._coerce_numeric(row.get("upLast7days")),
                    "upLast30days": self._coerce_numeric(row.get("upLast30days")),
                    "downLast7days": self._coerce_numeric(
                        row.get("downLast7Days")
                    ),  # Note: API uses 'downLast7Days' with capital D
                    "downLast30days": self._coerce_numeric(row.get("downLast30days")),
                }
            return result
        except Exception as exc:
            self.logger.debug("EPS revizyonları işlenemedi: %s", exc)
            return {}

    def _build_eps_trend_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get EPS trend using property access.
        According to yfinance documentation, use ticker.eps_trend instead of ticker.get_eps_trend().
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = self._call_with_retry(
                    lambda: ticker.eps_trend, name="ticker.eps_trend"
                )
        except Exception as exc:
            self.logger.debug("EPS trend alınamadı: %s", exc)
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
                    "current": self._coerce_numeric(row.get("current")),
                    "7daysAgo": self._coerce_numeric(row.get("7daysAgo")),
                    "30daysAgo": self._coerce_numeric(row.get("30daysAgo")),
                    "60daysAgo": self._coerce_numeric(row.get("60daysAgo")),
                    "90daysAgo": self._coerce_numeric(row.get("90daysAgo")),
                }
            return result
        except Exception as exc:
            self.logger.debug("EPS trend işlenemedi: %s", exc)
            return {}

    def _build_events_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Build events snapshot using property access.
        According to yfinance documentation, use ticker.earnings_dates instead of ticker.get_earnings_dates().
        """
        snapshot: Dict[str, Any] = {}

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                # earnings_dates sık sık TLS hatası veriyor, retry yapmadan doğrudan çağır
                # Başarısız olursa sessizce None döndür, circuit breaker tetiklenmesin
                earnings_raw = ticker.earnings_dates
        except Exception as exc:
            self.logger.debug(
                "Earnings tarihleri alınamadı (retry yok): %s", str(exc)[:100]
            )
            earnings_raw = None

        eps_lookup = self._build_eps_history_lookup(ticker)

        earnings_df = self._ensure_dataframe(earnings_raw)
        if earnings_df is not None and not earnings_df.empty:
            try:
                earnings_df = earnings_df.dropna(how="all")
                if not earnings_df.empty:
                    earnings_df = earnings_df.sort_index(ascending=False)
                earnings_list: List[Dict[str, Any]] = []
                for index, row in earnings_df.head(12).iterrows():
                    period = self._timestamp_to_iso(index)
                    if period == NA_VALUE:
                        period = str(index)

                    # Note: yfinance returns 'EPS Estimate', 'Reported EPS', 'Surprise(%)'
                    eps_estimate = self._coerce_numeric(
                        row.get("EPS Estimate")  # Current yfinance format
                        or row.get("epsEstimate")
                        or row.get("EPS_Estimate")
                    )
                    eps_actual = self._coerce_numeric(
                        row.get("Reported EPS")  # Current yfinance format
                        or row.get("EPS Actual")
                        or row.get("epsActual")
                        or row.get("EPS_Actual")
                    )
                    surprise = self._coerce_numeric(
                        row.get("Surprise(%)")  # Current yfinance format
                        or row.get("Surprise %")
                        or row.get("Surprise")
                        or row.get("surprisePercent")
                    )

                    history_entry = eps_lookup.get(period)
                    if history_entry:
                        if eps_actual in (None, NA_VALUE):
                            eps_actual = history_entry.get("epsActual", eps_actual)
                        if eps_estimate in (None, NA_VALUE):
                            eps_estimate = history_entry.get(
                                "epsEstimate", eps_estimate
                            )
                        if surprise in (None, NA_VALUE):
                            surprise = history_entry.get("surprisePercent", surprise)

                    if (
                        surprise in (None, NA_VALUE)
                        and isinstance(eps_actual, (int, float))
                        and isinstance(eps_estimate, (int, float))
                        and eps_estimate not in (0, -0.0)
                    ):
                        surprise = (
                            (eps_actual - eps_estimate) / abs(eps_estimate)
                        ) * 100

                    earnings_list.append(
                        {
                            "date": period,
                            "epsEstimate": eps_estimate,
                            "epsActual": eps_actual,
                            "surprisePercent": surprise,
                        }
                    )
                if earnings_list:
                    snapshot["earningsDates"] = earnings_list
            except Exception as exc:
                self.logger.debug("Earnings tarihleri işlenemedi: %s", exc)

        calendar_df = self._ensure_dataframe(getattr(ticker, "calendar", None))
        if calendar_df is not None and not calendar_df.empty:
            try:
                calendar_df = calendar_df.transpose().dropna(how="all")
                if not calendar_df.empty:
                    row = calendar_df.iloc[0]
                    translation_map = {
                        "Earnings Date": "Kazanç Açıklama Tarihi",
                        "Earnings Call": "Kazanç Konferans Çağrısı",
                        "Ex-Dividend Date": "Temettü Kesim Tarihi",
                        "Dividend Date": "Temettü Ödeme Tarihi",
                        "Conference Call": "Konferans Çağrısı",
                        "Annual Shareholders Meeting Date": "Genel Kurul Tarihi",
                    }

                    filtered_events: Dict[str, Any] = {}
                    filtered_localized: Dict[str, Any] = {}

                    for key, value in row.items():
                        normalized_key = str(key)
                        translated_key = translation_map.get(
                            normalized_key, normalized_key
                        )
                        iso_candidate = self._timestamp_to_iso(value)
                        if iso_candidate != NA_VALUE:
                            if self._is_future_date(iso_candidate):
                                filtered_events[normalized_key] = iso_candidate
                                filtered_localized[translated_key] = iso_candidate
                            continue

                        normalized_value = self._normalize_calendar_value(value)
                        iso_candidate = self._timestamp_to_iso(normalized_value)
                        if iso_candidate != NA_VALUE:
                            if self._is_future_date(iso_candidate):
                                filtered_events[normalized_key] = iso_candidate
                                filtered_localized[translated_key] = iso_candidate
                            continue

                        filtered_events[normalized_key] = value
                        filtered_localized[translated_key] = value

                    if filtered_events:
                        snapshot["upcomingEvents"] = {
                            "raw": filtered_events,
                            "localized": filtered_localized,
                        }
            except Exception as exc:
                self.logger.debug("Takvim verisi işlenemedi: %s", exc)

        return snapshot

    def _extract_last_reported_earnings_date(
        self, events_snapshot: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract the most recent reported earnings date.
        Checks earningsDates for historical data with epsActual values.
        Only returns dates that have already occurred (not future dates).
        """
        earnings = events_snapshot.get("earningsDates")
        if not isinstance(earnings, list):
            return None

        latest_date: Optional[str] = None

        for entry in earnings:
            if not isinstance(entry, dict):
                continue
            raw_date = entry.get("date")
            actual = entry.get("epsActual")
            if raw_date in (None, "", NA_VALUE) or actual in (
                None,
                "",
                NA_VALUE,
                "N/A",
            ):
                continue

            try:
                parsed_date = datetime.fromisoformat(str(raw_date)).date()
            except ValueError:
                continue

            # Only include past dates
            if parsed_date > date.today():
                continue

            if (
                latest_date is None
                or parsed_date > datetime.fromisoformat(latest_date).date()
            ):
                latest_date = parsed_date.isoformat()

        return latest_date

    def _extract_next_earnings_date(
        self, events_snapshot: Dict[str, Any]
    ) -> Optional[str]:
        """
        Extract next upcoming earnings date from events snapshot.
        Tries multiple sources: earnings_dates list and upcomingEvents calendar.
        """
        today = date.today()

        # 1. Try earnings_dates list first
        earnings = events_snapshot.get("earningsDates")
        if isinstance(earnings, list):
            for entry in earnings:
                if not isinstance(entry, dict):
                    continue
                raw_date = entry.get("date")
                if raw_date in (None, "", NA_VALUE):
                    continue

                try:
                    parsed_date = datetime.fromisoformat(str(raw_date)).date()
                    if parsed_date >= today:
                        return parsed_date.isoformat()
                except ValueError:
                    continue

        # 2. Try upcomingEvents calendar
        upcoming = events_snapshot.get("upcomingEvents", {}).get("raw", {})
        earnings_date = upcoming.get("Earnings Date") or upcoming.get("Earnings Call")
        if earnings_date:
            iso_date = self._timestamp_to_iso(earnings_date)
            if iso_date != NA_VALUE:
                try:
                    parsed = datetime.fromisoformat(iso_date).date()
                    if parsed >= today:
                        return iso_date
                except (ValueError, AttributeError):
                    pass

        return None
