"""Helper utilities for YFinance adapter mixins."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any, Dict, List

from config import NA_VALUE


class YFinanceHelperMixin:
    """Shared helper methods used across YFinance adapter mixins."""

    @staticmethod
    def _pick_first(*values: Any) -> Any:
        for value in values:
            if value not in (None, NA_VALUE, ""):
                return value
        return NA_VALUE

    @staticmethod
    def _convert_dividend_yield(raw_value: Any) -> Any:
        if raw_value is None:
            return NA_VALUE
        try:
            yield_value = float(raw_value) * 100
            if yield_value < 0 or yield_value > 20:
                return NA_VALUE
            return yield_value
        except (TypeError, ValueError):
            return NA_VALUE

    @staticmethod
    def _timestamp_to_iso(raw_value: Any) -> Any:
        if raw_value in (None, NA_VALUE, ""):
            return NA_VALUE
        try:
            if isinstance(raw_value, (int, float)):
                dt_obj = datetime.fromtimestamp(raw_value, tz=timezone.utc)
                return dt_obj.date().isoformat()
            if isinstance(raw_value, datetime):
                return raw_value.date().isoformat()
            if isinstance(raw_value, str):
                try:
                    parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
                    return parsed.date().isoformat()
                except ValueError:
                    pass
            return str(raw_value)
        except Exception:
            return NA_VALUE

    @staticmethod
    def _summarize_history(history) -> Dict[str, Any]:  # type: ignore[override]
        try:
            if history is None or history.empty:
                return {"dataPoints": 0}

            close = history["Close"] if "Close" in history else history.iloc[:, 0]
            high = float(close.max()) if not close.empty else None
            low = float(close.min()) if not close.empty else None
            average = float(close.mean()) if not close.empty else None
            last_close = float(close.iloc[-1]) if not close.empty else None
            first_close = float(close.iloc[0]) if not close.empty else None
            range_width = (
                float(high - low) if high is not None and low is not None else None
            )

            period_return_pct = NA_VALUE
            range_position_pct = NA_VALUE
            distance_from_low_pct = NA_VALUE
            distance_from_high_pct = NA_VALUE
            is_near_period_low = False

            try:
                if first_close not in (None, 0) and last_close is not None:
                    period_return_pct = ((last_close - first_close) / first_close) * 100
            except Exception:
                period_return_pct = NA_VALUE

            try:
                if (
                    range_width is not None
                    and range_width > 0
                    and last_close is not None
                    and low is not None
                ):
                    range_position_pct = ((last_close - low) / range_width) * 100
                    is_near_period_low = bool(range_position_pct <= 25.0)
            except Exception:
                range_position_pct = NA_VALUE
                is_near_period_low = False

            try:
                if low not in (None, 0) and last_close is not None:
                    distance_from_low_pct = ((last_close - low) / low) * 100
            except Exception:
                distance_from_low_pct = NA_VALUE

            try:
                if high not in (None, 0) and last_close is not None:
                    distance_from_high_pct = ((high - last_close) / high) * 100
            except Exception:
                distance_from_high_pct = NA_VALUE

            # Build a complete price series for full analysis (no window limit)
            series = []
            try:
                # Use full history instead of last 180 points to fix data gaps
                tail = close
                for idx, (ts, val) in enumerate(tail.items()):
                    try:
                        y_val = float(val)
                        # Skip NaN and inf values
                        if not (
                            y_val != y_val
                            or y_val == float("inf")
                            or y_val == float("-inf")
                        ):
                            series.append(
                                {
                                    "x": idx,
                                    "label": (
                                        str(ts.date())
                                        if hasattr(ts, "date")
                                        else str(ts)
                                    ),
                                    "y": y_val,
                                }
                            )
                    except Exception:
                        continue
            except Exception:
                series = []

            return {
                "dataPoints": int(close.shape[0]),
                "periodHigh": high if high is not None else NA_VALUE,
                "periodLow": low if low is not None else NA_VALUE,
                "priceRange": range_width if range_width is not None else NA_VALUE,
                "averagePrice": average or NA_VALUE,
                "lastDate": (
                    str(close.index[-1].date()) if not close.empty else NA_VALUE
                ),
                "firstClose": first_close if first_close is not None else NA_VALUE,
                "lastClose": last_close if last_close is not None else NA_VALUE,
                "periodReturnPct": period_return_pct,
                "rangePositionPct": range_position_pct,
                "distanceFromLowPct": distance_from_low_pct,
                "distanceFromHighPct": distance_from_high_pct,
                "isNearPeriodLow": is_near_period_low,
                "series": series,
            }
        except Exception:
            return {"dataPoints": 0}

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        try:
            numeric = float(value)
            return not (math.isnan(numeric) or math.isinf(numeric))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _ensure_series(raw: Any):
        try:
            import pandas as pd  # type: ignore
        except ImportError:  # pragma: no cover - defensive
            return None

        if raw is None:
            return None

        if isinstance(raw, pd.Series):
            return raw

        if isinstance(raw, pd.DataFrame):
            if raw.shape[1] == 1:
                return raw.iloc[:, 0]
            return None

        try:
            series = pd.Series(raw)
            return series
        except Exception:
            return None

    @staticmethod
    def _determine_dividend_frequency(index) -> Any:
        try:
            import pandas as pd  # type: ignore
        except ImportError:  # pragma: no cover - defensive
            return NA_VALUE

        if index is None or len(index) < 2:
            return NA_VALUE

        try:
            if not isinstance(index, pd.Index):
                index = pd.Index(index)
            if index.empty:
                return NA_VALUE
            diffs = index.to_series().sort_values().diff().dropna()
            if diffs.empty:
                return NA_VALUE
            median_days = float(diffs.dt.days.median())
        except Exception:
            return NA_VALUE

        if median_days <= 32:
            return "monthly"
        if median_days <= 95:
            return "quarterly"
        if median_days <= 190:
            return "semiannual"
        if median_days <= 370:
            return "annual"
        return "irregular"

    def _normalize_calendar_value(self, value: Any) -> Any:
        if value in (None, "", NA_VALUE):
            return NA_VALUE

        iso_value = self._timestamp_to_iso(value)
        if iso_value != NA_VALUE:
            return iso_value

        try:
            if hasattr(value, "date"):
                return value.date().isoformat()
            if hasattr(value, "isoformat"):
                return value.isoformat()
        except (AttributeError, TypeError, ValueError):
            return str(value)

        return str(value)

    def _is_future_date(self, value: Any) -> bool:
        if value in (None, "", NA_VALUE):
            return False

        iso_candidate = self._timestamp_to_iso(value)
        if iso_candidate == NA_VALUE:
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
                today_val = datetime.now(parsed.tzinfo or timezone.utc).date()
                return parsed.date() >= today_val
            except ValueError:
                return True

        try:
            parsed = datetime.fromisoformat(iso_candidate)
        except ValueError:
            return True

        today_val = date.today()
        return parsed.date() >= today_val

    @staticmethod
    def _ensure_dataframe(raw: Any):
        try:
            import pandas as pd  # type: ignore
        except ImportError:  # pragma: no cover - defensive
            return None

        if raw is None:
            return None

        if isinstance(raw, pd.DataFrame):
            return raw

        try:
            return pd.DataFrame(raw)
        except Exception:
            return None

    @staticmethod
    def _coerce_numeric(value: Any) -> Any:
        if value is None:
            return NA_VALUE

        try:
            if hasattr(value, "item"):
                value = value.item()
        except (AttributeError, TypeError, ValueError):
            pass

        try:
            numeric = float(value)
            if numeric != numeric:
                return NA_VALUE
            return numeric
        except (TypeError, ValueError):
            return value

    def _dataframe_to_records(self, raw: Any, limit: int = 10) -> List[Dict[str, Any]]:
        frame = self._ensure_dataframe(raw)
        if frame is None or frame.empty:
            return []

        try:
            frame = frame.dropna(how="all").reset_index(drop=True)
            if frame.empty:
                return []
            rows = frame.head(limit).to_dict(orient="records")
            return [
                {key: self._coerce_numeric(value) for key, value in row.items()}
                for row in rows
            ]
        except Exception:
            return []
