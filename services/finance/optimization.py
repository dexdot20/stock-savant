"""Portfolio optimization service (Modern Portfolio Theory)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd

from core import get_standard_logger

logger = get_standard_logger(__name__)


@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    correlation: pd.DataFrame
    covariance: pd.DataFrame
    data_points: int
    tickers_used: List[str]
    tickers_skipped: List[str]
    method: str


class PortfolioOptimizer:
    """Build a low-complexity equal-weight allocation baseline for favorite stocks."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

        # Lazy import to avoid circular dependency with services.factories
        from services.factories import get_financial_service

        self.market_service = get_financial_service(self.config)

        self.risk_free_rate = 0.0  # Default risk-free rate (assumed 0 on a daily basis)

    def optimize_portfolio(
        self,
        tickers: List[str],
        objective: Literal["min_volatility", "max_sharpe"] = "min_volatility",
        min_points: int = 60,
        **kwargs,
    ) -> Optional[OptimizationResult]:
        """
        Build a portfolio allocation using a stable equal-weight baseline.

        Args:
            tickers: List of stock symbols
            objective: Preserved for compatibility; equal-weight is always used.
            min_points: Required minimum number of data points
        """
        if not tickers or len(tickers) < 2:
            logger.warning("Portfolio optimization requires at least 2 stocks.")
            return None

        prices, used, skipped = self._build_price_frame(tickers)
        if prices is None or prices.empty or len(used) < 2:
            logger.warning(
                "Insufficient price history: used=%s skipped=%s", used, skipped
            )
            return None

        returns = prices.pct_change().dropna(how="any")
        if returns.shape[0] < min_points:
            logger.warning("Insufficient data points: %s", returns.shape[0])
            return None

        if objective != "min_volatility":
            logger.info(
                "Portfolio objective '%s' requested; using equal-weight baseline instead.",
                objective,
            )

        weights = self._build_equal_weights(len(returns.columns))

        cov = returns.cov()
        corr = returns.corr()

        # Annualization factor (trading days)
        trading_days = 252

        # Calculate statistics
        mean_returns = returns.mean()
        pf_return = float(np.sum(mean_returns * weights) * trading_days)
        pf_volatility = float(
            np.sqrt(np.dot(weights.T, np.dot(cov * trading_days, weights)))
        )
        pf_sharpe = (
            (pf_return - (self.risk_free_rate * trading_days)) / pf_volatility
            if pf_volatility > 0
            else 0.0
        )

        weight_map = {
            ticker: float(weights[idx]) for idx, ticker in enumerate(prices.columns)
        }

        return OptimizationResult(
            weights=weight_map,
            expected_return=pf_return,
            volatility=pf_volatility,
            sharpe_ratio=pf_sharpe,
            correlation=corr,
            covariance=cov,
            data_points=int(returns.shape[0]),
            tickers_used=list(prices.columns),
            tickers_skipped=skipped,
            method="equal_weight",
        )

    def _build_price_frame(
        self,
        tickers: List[str],
    ) -> Tuple[Optional[pd.DataFrame], List[str], List[str]]:
        series_map: Dict[str, pd.Series] = {}
        used: List[str] = []
        skipped: List[str] = []

        for ticker in tickers:
            info = self.market_service.get_company_data(ticker)
            if not info:
                skipped.append(ticker)
                continue

            price_series = self._extract_price_series(info)
            if price_series is None or price_series.empty:
                skipped.append(ticker)
                continue

            series_map[ticker] = price_series
            used.append(ticker)

        if len(series_map) < 2:
            return None, used, skipped

        prices = pd.concat(series_map, axis=1).dropna(how="any")
        return prices, used, skipped

    def _extract_price_series(self, info: dict) -> Optional[pd.Series]:
        history = info.get("priceHistory", {})
        series = history.get("series", [])
        if not series:
            return None

        rows = []
        for item in series:
            label = item.get("label")
            value = item.get("y")
            if label is None or value is None:
                continue
            rows.append((label, value))

        if not rows:
            return None

        df = pd.DataFrame(rows, columns=["date", "price"]).dropna()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).sort_values("date")
        df = df.drop_duplicates(subset=["date"], keep="last")
        df = df.set_index("date")

        try:
            return df["price"].astype(float)
        except Exception:
            return None

    @staticmethod
    def _build_equal_weights(asset_count: int) -> np.ndarray:
        if asset_count <= 0:
            return np.array([])
        return np.array([1.0 / asset_count] * asset_count)


__all__ = ["PortfolioOptimizer", "OptimizationResult"]
