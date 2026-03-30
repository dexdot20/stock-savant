"""Ownership and insider trading helpers for YFinance adapter."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf

from config import NA_VALUE
from domain.utils import safe_float


def _summarize_insider_transactions(
    insider_data: List[Dict[str, Any]],
    logger: Any,
) -> Dict[str, Any]:
    """Aggregate insider transactions into a single weighted snapshot."""
    try:
        if not insider_data:
            return {
                "insider_trading_analysis": {
                    "total_transactions": 0,
                    "recent_acquisitions": 0,
                    "recent_dispositions": 0,
                    "net_insider_sentiment": "NEUTRAL",
                    "latest_transaction": None,
                    "transaction_types": {},
                    "owner_types": {},
                    "analysis_notes": ["Insider trading data not found"],
                    "data_quality": "MISSING",
                    "weighted_signal_score": 50,
                    "normalized_signal": 0.0,
                    "volume_weighted_net_shares": 0.0,
                    "weighted_breakdown": [],
                }
            }

        now = datetime.now(timezone.utc)

        def _determine_role_weight(transaction: Dict[str, Any]) -> float:
            role_text = " ".join(
                filter(
                    None,
                    [
                        transaction.get("reportingTitle"),
                        transaction.get("reportingName"),
                        transaction.get("typeOfOwner"),
                    ],
                )
            ).lower()
            role_map = [
                ("chief executive", 1.45),
                ("ceo", 1.45),
                ("chief financial", 1.35),
                ("cfo", 1.35),
                ("president", 1.3),
                ("director", 1.15),
                ("10%", 1.25),
                ("owner", 1.2),
                ("officer", 1.1),
            ]
            for keyword, weight in role_map:
                if keyword in role_text:
                    return weight
            return 1.0

        def _parse_transaction_datetime(raw_value: Any) -> Optional[datetime]:
            if not raw_value:
                return None
            if isinstance(raw_value, datetime):
                return (
                    raw_value
                    if raw_value.tzinfo
                    else raw_value.replace(tzinfo=timezone.utc)
                )
            value_str = str(raw_value).strip().replace("Z", "")
            value_str = value_str.split(".")[0]
            for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    dt = datetime.strptime(value_str, fmt)
                    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            return None

        def _determine_recency_weight(raw_date: Any) -> float:
            parsed = _parse_transaction_datetime(raw_date)
            if not parsed:
                return 0.6
            age_days = max(0.0, (now - parsed).days + (now - parsed).seconds / 86400)
            return max(0.2, 1.0 / (1.0 + age_days / 30.0))

        transaction_count = len(insider_data)
        acquisitions = 0
        dispositions = 0
        transaction_types: Dict[str, int] = {}
        owner_types: Dict[str, int] = {}
        weight_sum = 0.0
        weight_magnitude = 0.0
        volume_weighted_net_shares = 0.0
        weighted_breakdown: List[Dict[str, Any]] = []
        latest_transaction: Optional[Dict[str, Any]] = None
        latest_transaction_dt: Optional[datetime] = None

        for transaction in insider_data:
            t_type = transaction.get("transactionType", "Unknown") or "Unknown"
            transaction_types[t_type] = transaction_types.get(t_type, 0) + 1

            o_type = transaction.get("typeOfOwner", "Unknown") or "Unknown"
            owner_types[o_type] = owner_types.get(o_type, 0) + 1

            shares = safe_float(transaction.get("securitiesTransacted")) or 0.0

            acq_flag = transaction.get("acquisitionOrDisposition")
            if not acq_flag and t_type:
                lower_t = t_type.lower()
                if any(
                    x in lower_t
                    for x in [
                        "purchase",
                        "buy",
                        "grant",
                        "award",
                        "exercise",
                        "acquisition",
                    ]
                ):
                    acq_flag = "A"
                elif any(x in lower_t for x in ["sale", "sell", "disposition"]):
                    acq_flag = "D"
                elif "gift" in lower_t:
                    acq_flag = "D"

            if acq_flag == "A":
                acquisitions += 1
            elif acq_flag == "D":
                dispositions += 1

            parsed_dt = _parse_transaction_datetime(transaction.get("transactionDate"))
            if parsed_dt and (
                latest_transaction_dt is None or parsed_dt > latest_transaction_dt
            ):
                latest_transaction_dt = parsed_dt
                latest_transaction = {
                    "date": transaction.get("transactionDate"),
                    "type": transaction.get("transactionType"),
                    "insider_name": transaction.get("reportingName"),
                    "shares": safe_float(transaction.get("securitiesTransacted")),
                    "acquisition_or_disposition": acq_flag,
                    "owner_type": transaction.get("typeOfOwner"),
                }

            direction = 1 if acq_flag == "A" else -1 if acq_flag == "D" else 0
            if direction == 0 or shares in (None, 0):
                continue

            role_weight = _determine_role_weight(transaction)
            recency_weight = _determine_recency_weight(
                transaction.get("transactionDate")
            )
            size_weight = math.log1p(abs(shares))

            contribution = direction * size_weight * role_weight * recency_weight
            weight_sum += contribution
            weight_magnitude += abs(size_weight * role_weight * recency_weight)
            volume_weighted_net_shares += (
                direction * shares * role_weight * recency_weight
            )

            weighted_breakdown.append(
                {
                    "date": transaction.get("transactionDate"),
                    "direction": "BUY" if direction > 0 else "SELL",
                    "shares": shares,
                    "role_weight": round(role_weight, 2),
                    "recency_weight": round(recency_weight, 2),
                    "contribution": round(contribution, 2),
                }
            )

        analysis_notes = []
        if transaction_count > 10:
            analysis_notes.append("High insider activity")
        elif transaction_count > 5:
            analysis_notes.append("Orta seviye insider aktivitesi")
        else:
            analysis_notes.append("Low insider activity")

        normalized_signal = 0.0
        if weight_magnitude > 0:
            normalized_signal = max(-1.0, min(1.0, weight_sum / weight_magnitude))

        weighted_signal_score = int(round(50 + 50 * normalized_signal))

        if normalized_signal > 0.15:
            net_sentiment = "POSITIVE"
            analysis_notes.append("Weighted insider signal positive")
        elif normalized_signal < -0.15:
            net_sentiment = "NEGATIVE"
            analysis_notes.append("Weighted insider signal negative")
        else:
            net_sentiment = "NEUTRAL"
            analysis_notes.append("Ağırlıklı insider sinyali nötr")

        if acquisitions and dispositions:
            analysis_notes.append(f"Alış/Satış oranı: {acquisitions}/{dispositions}")
        elif acquisitions:
            analysis_notes.append("Tüm işlemler alış yönlü")
        elif dispositions:
            analysis_notes.append("Tüm işlemler satış yönlü")

        return {
            "insider_trading_analysis": {
                "total_transactions": transaction_count,
                "recent_acquisitions": acquisitions,
                "recent_dispositions": dispositions,
                "net_insider_sentiment": net_sentiment,
                "latest_transaction": latest_transaction,
                "transaction_types": transaction_types,
                "owner_types": owner_types,
                "analysis_notes": analysis_notes,
                "data_quality": "GOOD" if transaction_count > 0 else "MISSING",
                "weighted_signal_score": weighted_signal_score,
                "normalized_signal": round(normalized_signal, 3),
                "volume_weighted_net_shares": volume_weighted_net_shares,
                "weighted_breakdown": weighted_breakdown,
            }
        }

    except Exception as exc:
        if logger:
            logger.error("Insider trading veri işleme hatası: %s", exc, exc_info=True)
        return {
            "insider_trading_analysis": {
                "total_transactions": 0,
                "recent_acquisitions": 0,
                "recent_dispositions": 0,
                "net_insider_sentiment": "NEUTRAL",
                "latest_transaction": None,
                "transaction_types": {},
                "owner_types": {},
                "analysis_notes": [f"Veri işleme hatası: {exc}"],
                "data_quality": "ERROR",
                "weighted_signal_score": 50,
                "normalized_signal": 0.0,
                "volume_weighted_net_shares": 0.0,
                "weighted_breakdown": [],
            }
        }


class YFinanceInsiderMixin:
    """Processes insider trading snapshots."""

    def _build_insider_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Build insider snapshot using property access.
        According to yfinance documentation, use ticker.insider_transactions and ticker.insider_purchases
        instead of ticker.get_insider_transactions() and ticker.get_insider_purchases().
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                df = self._call_with_retry(
                    lambda: ticker.insider_transactions,
                    name="ticker.insider_transactions",
                )
        except Exception as exc:
            self.logger.debug("İçeriden öğrenen işlemleri alınamadı: %s", exc)
            df = None

        frame = self._ensure_dataframe(df)

        # Initialize lists to hold transactions from different sources
        all_transactions = []

        # Helper to process a dataframe
        def process_frame(df_in, source_context="main"):
            if df_in is None or df_in.empty:
                return []

            # Normalize columns
            df_in.columns = [str(c).strip() for c in df_in.columns]

            # Find date column
            d_col = None
            for col in [
                "Date",
                "Start Date",
                "date",
                "Date Reported",
                "Transaction Date",
            ]:
                if col in df_in.columns:
                    d_col = col
                    break

            # If no date column found, check index
            if not d_col:
                # Check if index has date-like properties
                import pandas as pd

                if isinstance(df_in.index, pd.DatetimeIndex):
                    # Reset index to make it a column
                    df_in = df_in.reset_index()
                    d_col = df_in.columns[0]  # Usually 'Date' or 'index'
                elif "Date" in df_in.index.names:
                    df_in = df_in.reset_index()
                    d_col = "Date"

            if not d_col:
                self.logger.debug(
                    f"Date column not found in {source_context} frame. Columns: {df_in.columns}"
                )

            processed = []
            for _, row_data in df_in.iterrows():
                t_date = row_data.get(d_col) if d_col else NA_VALUE

                # Try to clean up date if NA
                if t_date in (None, NA_VALUE) and d_col:
                    # fallback logic if needed
                    pass

                processed.append(
                    {
                        "reportingName": row_data.get("Insider")
                        or row_data.get("Buyer")
                        or row_data.get("Owner")
                        or NA_VALUE,
                        "transactionDate": t_date,
                        "transactionType": self._categorize_transaction(row_data),
                        "securitiesTransacted": self._coerce_numeric(
                            row_data.get("Shares")
                            or row_data.get("Qty")
                            or row_data.get("Value")
                        ),
                        "typeOfOwner": row_data.get("Relationship")
                        or row_data.get("Title")
                        or "Unknown",
                        "acquisitionOrDisposition": self._infer_acq_flag(row_data),
                    }
                )
            return processed

        # Process main frame (insider_transactions)
        if frame is not None and not frame.empty:
            all_transactions.extend(process_frame(frame, "transactions"))

        # Check if data is stale (older than 60 days)
        # We check the dates we just extracted
        is_stale = False
        if all_transactions:
            try:
                # Find max date
                dates = []
                for t in all_transactions:
                    parsed = self._parse_date_helper(t["transactionDate"])
                    if parsed:
                        dates.append(parsed)

                if dates:
                    last_date = max(dates)
                    if (datetime.now() - last_date).days > 60:
                        is_stale = True
                        self.logger.debug(
                            "İçeriden öğrenen verisi güncel değil (stale), ek kaynaklar taranıyor."
                        )
            except Exception as e:
                self.logger.debug(f"Stale check failed: {e}")
        else:
            # If empty, treat as stale/missing to trigger fallback
            is_stale = True

        # If data is stale or empty, try insider_purchases
        if is_stale or not all_transactions:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    df_purchases = self._call_with_retry(
                        lambda: ticker.insider_purchases,
                        name="ticker.insider_purchases",
                    )
                    frame_purchases = self._ensure_dataframe(df_purchases)

                    if frame_purchases is not None and not frame_purchases.empty:
                        new_items = process_frame(frame_purchases, "purchases")

                        # Merge strategy: Add new items.
                        # Ideally we deduplicate, but for now let's just add them.
                        # Simple deduplication by date+name+shares
                        existing_keys = set()
                        for t in all_transactions:
                            # rudimentary key
                            key = (
                                str(t["transactionDate"]),
                                str(t["reportingName"]),
                                str(t["securitiesTransacted"]),
                            )
                            existing_keys.add(key)

                        for item in new_items:
                            item_key = (
                                str(item["transactionDate"]),
                                str(item["reportingName"]),
                                str(item["securitiesTransacted"]),
                            )
                            if item_key not in existing_keys:
                                all_transactions.append(item)

            except Exception as exc:
                self.logger.debug("İçeriden öğrenen alımları alınamadı: %s", exc)

        if not all_transactions:
            return {}

        # Convert back to list format expected by _summarize_insider_transactions
        # (It actually expects the exact dict structure we created above)

        # Sort by date descending
        def get_sort_date(x):
            d = self._parse_date_helper(x["transactionDate"])
            return d if d else datetime.min

        all_transactions.sort(key=get_sort_date, reverse=True)

        return _summarize_insider_transactions(all_transactions, self.logger)

    def _parse_date_helper(self, raw_date):
        if not raw_date or raw_date == NA_VALUE:
            return None
        if isinstance(raw_date, datetime):
            return raw_date
        try:
            import pandas as pd

            return pd.to_datetime(raw_date).to_pydatetime()
        except Exception:
            return None

    def _categorize_transaction(self, row: Any) -> str:
        raw_transaction = str(row.get("Transaction") or "").strip()
        if not raw_transaction:
            raw_transaction = str(row.get("Text") or "").strip()

        transaction = raw_transaction.lower()

        buy_keywords = {"buy", "purchase", "acquire", "acquired", "bought", "exercise"}
        sell_keywords = {"sell", "sold", "dispose", "disposition", "sale"}
        neutral_keywords = {"gift", "award", "grant"}

        if any(keyword in transaction for keyword in buy_keywords):
            return "Purchase"
        if any(keyword in transaction for keyword in sell_keywords):
            return "Sale"
        if any(keyword in transaction for keyword in neutral_keywords):
            return "Gift"
        return "Unknown"

    def _infer_acq_flag(self, row: Any) -> Optional[str]:
        acq_flag = row.get("Type")
        if acq_flag:
            return acq_flag

        transaction_type = self._categorize_transaction(row)
        if "purchase" in transaction_type.lower() or "buy" in transaction_type.lower():
            return "A"
        if "sale" in transaction_type.lower():
            return "D"
        return None


class YFinanceHoldersMixin:
    """Builds ownership related snapshots."""

    def _build_holders_snapshot(
        self, ticker: yf.Ticker, info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Build holders snapshot using property access.
        According to yfinance documentation, use properties like ticker.institutional_holders
        instead of get_institutional_holders().
        """
        snapshot: Dict[str, Any] = {}

        def fetch(property_name: str, limit: int = 10) -> List[Dict[str, Any]]:
            """Fetch data using property access instead of method calls."""
            try:
                result = self._call_with_retry(
                    lambda: getattr(ticker, property_name),
                    name=f"ticker.{property_name}",
                )
            except Exception as exc:
                self.logger.debug("%s verisi alınamadı: %s", property_name, exc)
                result = None

            if property_name == "institutional_holders" and result is not None:
                df = self._ensure_dataframe(result)
                if df is not None and not df.empty and "pctHeld" in df.columns:
                    df = df.rename(columns={"pctHeld": "% Out"})
                    return self._dataframe_to_records(df, limit)

            return self._dataframe_to_records(result, limit)

        major_raw = None
        try:
            major_raw = self._call_with_retry(
                lambda: ticker.major_holders, name="ticker.major_holders"
            )
        except Exception as exc:
            self.logger.debug("major_holders alınamadı: %s", exc)

        if major_raw is not None:
            major_df = self._ensure_dataframe(major_raw)
            if major_df is not None and not major_df.empty:
                try:
                    major_list = []
                    for idx, row in major_df.iterrows():
                        label = str(idx) if idx else "Unknown"
                        value = self._coerce_numeric(
                            row.get(0) if 0 in row.index else row.iloc[0]
                        )
                        major_list.append({"label": label, "value": value})
                    if major_list:
                        snapshot["majorHolders"] = major_list
                except Exception as exc:
                    self.logger.debug("majorHolders parsing hatası: %s", exc)

        institutional = fetch("institutional_holders", limit=10)
        if institutional:
            snapshot["institutionalHolders"] = institutional

        mutual = fetch("mutualfund_holders", limit=10)
        if mutual:
            snapshot["mutualFundHolders"] = mutual

        insider = fetch("insider_roster_holders", limit=10)
        if insider:
            snapshot["insiderRosterHolders"] = insider

        ownership_summary = {
            "heldPercentInsiders": info.get("heldPercentInsiders"),
            "heldPercentInstitutions": info.get("heldPercentInstitutions"),
            "sharesOutstanding": self._pick_first(
                info.get("sharesOutstanding"),
                info.get("impliedSharesOutstanding"),
            ),
            "floatShares": info.get("floatShares"),
            "shortRatio": info.get("shortRatio"),
            "shortPercentOfFloat": info.get("shortPercentOfFloat"),
            "sharesShort": info.get("sharesShort"),
            "sharesShortPriorMonth": info.get("sharesShortPriorMonth"),
        }

        summary = {
            k: v for k, v in ownership_summary.items() if v not in (None, "", NA_VALUE)
        }
        if summary:
            snapshot["summary"] = summary

        return snapshot

    def _deduplicate_ownership_summary(self, data: Dict[str, Any]) -> None:
        ownership = data.get("ownership")
        if not isinstance(ownership, dict):
            return

        major_holders = ownership.get("majorHolders")
        summary = ownership.get("summary")
        if not isinstance(summary, dict):
            return

        major_values = set()
        if isinstance(major_holders, list):
            for holder in major_holders:
                if isinstance(holder, dict) and "value" in holder:
                    major_values.add(holder["value"])

        duplicate_keys: List[str] = []
        for key, value in summary.items():
            if key in data and data[key] == value:
                duplicate_keys.append(key)
            elif value in major_values:
                duplicate_keys.append(key)

        for key in duplicate_keys:
            summary.pop(key, None)

        if not summary:
            ownership.pop("summary", None)

    def _build_shares_full_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Get detailed shares outstanding history using property access.
        According to yfinance documentation, use ticker.get_shares_full() method.

        Returns historical shares outstanding data over time to track dilution.
        Note: This is one of the few methods (not a property) in yfinance.
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                shares_raw = self._call_with_retry(
                    lambda: ticker.get_shares_full(), name="ticker.get_shares_full()"
                )
        except Exception as exc:
            self.logger.debug("Shares full verisi alınamadı: %s", exc)
            return {}

        df = self._ensure_dataframe(shares_raw)
        if df is None or df.empty:
            return {}

        try:
            df = df.dropna(how="all")
            if df.empty:
                return {}

            latest_shares = None
            if not df.empty:
                try:
                    latest_shares = self._coerce_numeric(df.iloc[-1].values[0])
                except (IndexError, TypeError, ValueError):
                    latest_shares = None

            dilution_percent = None
            if len(df) > 1:
                try:
                    first_shares = self._coerce_numeric(df.iloc[0].values[0])
                    last_shares = self._coerce_numeric(df.iloc[-1].values[0])

                    if (
                        isinstance(first_shares, (int, float))
                        and isinstance(last_shares, (int, float))
                        and first_shares > 0
                    ):
                        dilution_percent = (
                            (last_shares - first_shares) / first_shares
                        ) * 100
                except (IndexError, TypeError, ValueError, ZeroDivisionError):
                    dilution_percent = None

            history = []
            for index, row in df.tail(12).iterrows():
                date = self._normalize_calendar_value(index)
                shares = self._coerce_numeric(
                    row.values[0] if len(row.values) > 0 else None
                )

                if date not in (None, NA_VALUE) and shares not in (None, NA_VALUE):
                    history.append(
                        {
                            "date": date,
                            "shares": shares,
                        }
                    )

            snapshot = {}
            if latest_shares not in (None, NA_VALUE):
                snapshot["latestSharesOutstanding"] = latest_shares
            if dilution_percent is not None:
                snapshot["dilutionPercent"] = round(dilution_percent, 2)
            if history:
                snapshot["history"] = history

            return snapshot
        except Exception as exc:
            self.logger.debug("Shares full verisi işlenemedi: %s", exc)
            return {}
