"""Financial statement related helpers for YFinance adapter."""

from __future__ import annotations

import math
from typing import Any, Dict, List

import yfinance as yf

from config import NA_VALUE


class YFinanceStatementsMixin:
    """Provides helpers for financial statement data."""

    def _statement_frame_to_records(self, frame: Any, limit: int = 4) -> List[Dict[str, Any]]:
        df = self._ensure_dataframe(frame)
        if df is None or df.empty:
            return []

        try:
            df_t = df.transpose().dropna(how="all")
            if df_t.empty:
                return []

            records: List[Dict[str, Any]] = []
            for index, row in df_t.head(limit).iterrows():
                record = {"period": self._normalize_calendar_value(index)}
                for key, value in row.items():
                    record[str(key)] = self._coerce_numeric(value)
                records.append(record)
            return records
        except Exception:
            return []

    def _calculate_enterprise_value(
        self, info: Dict[str, Any], ticker: yf.Ticker
    ) -> Any:
        """
        Calculate Enterprise Value with fallback to manual calculation.
        Enterprise Value = Market Cap + Total Debt - Cash and Cash Equivalents
        """
        enterprise_value = info.get("enterpriseValue")
        if enterprise_value not in (None, NA_VALUE) and isinstance(
            enterprise_value, (int, float)
        ):
            if math.isfinite(enterprise_value) and enterprise_value > 0:
                return enterprise_value

        # Manual calculation fallback
        try:
            market_cap = info.get("marketCap")
            if not market_cap:
                fast_info = getattr(ticker, "fast_info", None)
                if fast_info:
                    market_cap = getattr(fast_info, "market_cap", None)

            if not market_cap:
                return NA_VALUE

            balance_sheet = ticker.balance_sheet
            if balance_sheet is None or balance_sheet.empty:
                return NA_VALUE

            # Get latest values
            latest = balance_sheet.iloc[:, 0]

            total_debt = latest.get("Total Debt")
            if total_debt is None or math.isnan(total_debt):
                # Try alternative names or sum short+long term
                short_term = latest.get("Short Long Term Debt", 0)
                long_term = latest.get("Long Term Debt", 0)
                total_debt = short_term + long_term

            cash = latest.get("Cash And Cash Equivalents")

            if total_debt is not None and cash is not None:
                ev = market_cap + total_debt - cash
                return ev

        except Exception as exc:
            self.logger.debug("Enterprise Value manual calculation error: %s", exc)

        return NA_VALUE

    def _calculate_roe(self, info: Dict[str, Any], ticker: yf.Ticker) -> Any:
        """
        Calculate Return on Equity (ROE).
        According to yfinance documentation, use properties like ticker.income_stmt, ticker.balance_sheet
        instead of get_income_stmt(), get_balance_sheet() methods.
        """
        roe = info.get("returnOnEquity")
        if roe not in (None, NA_VALUE) and isinstance(roe, (int, float)):
            if math.isfinite(roe):
                return roe
            self.logger.debug(
                "ROE API value is infinite (%.4f). Manual calculation will be attempted.",
                roe,
            )
        elif roe not in (None, NA_VALUE):
            return roe

        # Manual calculation fallback
        try:
            income_stmt = ticker.income_stmt
            balance_sheet = ticker.balance_sheet

            if (
                income_stmt is None
                or income_stmt.empty
                or balance_sheet is None
                or balance_sheet.empty
            ):
                return NA_VALUE

            net_income = income_stmt.iloc[:, 0].get("Net Income")
            total_equity = balance_sheet.iloc[:, 0].get("Stockholders Equity")

            if net_income and total_equity and total_equity != 0:
                return net_income / total_equity

        except Exception as exc:
            self.logger.debug("ROE manual calculation error: %s", exc)

        return NA_VALUE

    def _extract_statement_fields(
        self, frame: Any, mapping: Dict[str, str]
    ) -> Dict[str, Any]:
        df = self._ensure_dataframe(frame)
        if df is None or df.empty:
            return {}

        try:
            df_t = df.transpose().dropna(how="all")
            if df_t.empty:
                return {}
            latest_row = df_t.iloc[0]
        except Exception:
            return {}

        snapshot: Dict[str, Any] = {}
        for key, label in mapping.items():
            try:
                value = latest_row.get(label)
            except Exception:
                value = None
            snapshot[key] = self._coerce_numeric(value)

        return snapshot

    def _build_financial_statement_snapshot(self, ticker: yf.Ticker) -> Dict[str, Any]:
        """
        Build financial statement snapshot using property access.
        """
        statement_map = {
            "incomeStatement": {
                "annual": getattr(ticker, "income_stmt", None),
                "quarterly": getattr(ticker, "quarterly_income_stmt", None),
                "ttm": getattr(ticker, "ttm_income_stmt", None),
            },
            "balanceSheet": {
                "annual": getattr(ticker, "balance_sheet", None),
            },
            "cashflow": {
                "annual": getattr(ticker, "cashflow", None),
                "quarterly": getattr(ticker, "quarterly_cashflow", None),
                "ttm": getattr(ticker, "ttm_cashflow", None),
            },
        }

        result: Dict[str, Any] = {}
        for section, variants in statement_map.items():
            section_payload: Dict[str, Any] = {}
            for variant, frame in variants.items():
                records = self._statement_frame_to_records(frame)
                if records:
                    section_payload[variant] = records
            if section_payload:
                result[section] = section_payload

        summary = {
            "incomeStatement": self._extract_statement_fields(
                getattr(ticker, "income_stmt", None),
                {
                    "revenue": "Total Revenue",
                    "grossProfit": "Gross Profit",
                    "operatingIncome": "Operating Income",
                    "netIncome": "Net Income",
                    "ebitda": "EBITDA",
                    "dilutedEPS": "Diluted EPS",
                },
            ),
            "balanceSheet": self._extract_statement_fields(
                getattr(ticker, "balance_sheet", None),
                {
                    "cash": "Cash And Cash Equivalents",
                    "totalDebt": "Total Debt",
                    "totalAssets": "Total Assets",
                    "totalLiabilities": "Total Liabilities Net Minority Interest",
                    "stockholdersEquity": "Stockholders Equity",
                },
            ),
            "cashflow": self._extract_statement_fields(
                getattr(ticker, "cashflow", None),
                {
                    "operatingCashFlow": "Operating Cash Flow",
                    "investingCashFlow": "Investing Cash Flow",
                    "financingCashFlow": "Financing Cash Flow",
                    "capitalExpenditure": "Capital Expenditure",
                    "freeCashFlow": "Free Cash Flow",
                },
            ),
        }

        cleaned_summary = {
            section: {k: v for k, v in values.items() if v not in (None, NA_VALUE)}
            for section, values in summary.items()
            if isinstance(values, dict)
        }
        cleaned_summary = {
            section: values for section, values in cleaned_summary.items() if values
        }
        if cleaned_summary:
            result["summary"] = cleaned_summary

        return result

    def _build_sec_filings_snapshot(self, ticker: yf.Ticker) -> List[Dict[str, Any]]:
        """
        Get SEC filings using property access.
        According to yfinance documentation, use ticker.sec_filings instead of ticker.get_sec_filings().

        Returns list of recent SEC filings (10-K, 10-Q, 8-K, etc.) with dates and links.
        """
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                filings_raw = self._call_with_retry(
                    lambda: ticker.sec_filings, name="ticker.sec_filings"
                )
        except Exception as exc:
            self.logger.debug("SEC filings could not be retrieved: %s", exc)
            return []

        df = self._ensure_dataframe(filings_raw)
        if df is None or df.empty:
            return []

        try:
            filings_list = []
            for index, row in df.head(20).iterrows():  # Last 20 filings
                filing = {
                    "date": self._timestamp_to_iso(row.get("date") or index),
                    "type": str(row.get("type", "Unknown")),
                    "title": str(row.get("title", "")),
                    "edgarUrl": str(row.get("edgarUrl", "")),
                }

                # Only add valid filings
                if filing["type"] != "Unknown" or filing["title"]:
                    filings_list.append(filing)

            return filings_list
        except Exception as exc:
            self.logger.debug("SEC filings could not be processed: %s", exc)
            return []
