from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from config import get_config
from core import get_standard_logger
from services.factories import get_financial_service

logger = get_standard_logger(__name__)


class PortfolioRiskService:
    """Build a lightweight CLI-oriented portfolio risk cockpit."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or get_config()
        self.market_service = get_financial_service(self.config)

    def build_snapshot(
        self,
        entries: Iterable[Dict[str, Any]],
    ) -> Dict[str, Any]:
        positions = [dict(item) for item in entries if isinstance(item, dict)]
        if not positions:
            return {
                "positions": [],
                "summary": {
                    "position_count": 0,
                    "diversification_score": 0.0,
                    "average_confidence": 0.0,
                    "estimated_portfolio_volatility": None,
                    "risk_score": 0.0,
                },
                "sector_exposure": [],
                "correlation_alerts": [],
                "breaches": [],
            }

        enriched_positions = self._enrich_positions(positions)
        total_market_value = sum(
            float(item.get("market_value", 0.0)) for item in enriched_positions
        )
        if total_market_value <= 0:
            total_market_value = sum(
                float(item.get("cost_value", 0.0)) for item in enriched_positions
            )

        for item in enriched_positions:
            market_value = float(item.get("market_value", 0.0))
            weight_pct = ((market_value / total_market_value) * 100.0) if total_market_value > 0 else 0.0
            item["weight_pct"] = round(weight_pct, 2)

        sector_exposure = self._build_sector_exposure(enriched_positions, total_market_value)
        breaches = self._build_breaches(enriched_positions, sector_exposure)
        correlation_alerts, estimated_volatility = self._build_correlation_alerts(
            enriched_positions
        )

        average_confidence = mean(
            [float(item.get("confidence_pct", 0.0)) for item in enriched_positions]
        )
        diversification_score = max(
            0.0,
            100.0 - max((item.get("weight_pct", 0.0) for item in enriched_positions), default=0.0),
        )
        risk_score = self._compute_risk_score(
            enriched_positions,
            breaches,
            correlation_alerts,
            estimated_volatility,
            average_confidence,
        )

        return {
            "positions": enriched_positions,
            "summary": {
                "position_count": len(enriched_positions),
                "total_market_value": round(total_market_value, 2),
                "diversification_score": round(diversification_score, 2),
                "average_confidence": round(average_confidence, 2),
                "estimated_portfolio_volatility": estimated_volatility,
                "risk_score": risk_score,
            },
            "sector_exposure": sector_exposure,
            "correlation_alerts": correlation_alerts,
            "breaches": breaches,
        }

    def _enrich_positions(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for entry in entries:
            symbol = str(entry.get("symbol") or "").upper().strip()
            if not symbol:
                continue

            meta = self.market_service.get_company_data(symbol) or {}
            quantity = float(entry.get("quantity", 0.0) or 0.0)
            average_cost = float(entry.get("average_cost", 0.0) or 0.0)
            current_price = entry.get("current_price")
            if current_price is None:
                current_price = meta.get("currentPrice") or meta.get("regularMarketPrice")
            try:
                current_price_f = float(current_price) if current_price is not None else average_cost
            except (TypeError, ValueError):
                current_price_f = average_cost

            cost_value = float(entry.get("cost_value", quantity * average_cost))
            market_value = float(entry.get("market_value", quantity * current_price_f))
            confidence_report = meta.get("confidence_report") or {}
            enriched.append(
                {
                    **entry,
                    "symbol": symbol,
                    "quantity": quantity,
                    "average_cost": average_cost,
                    "current_price": current_price_f,
                    "cost_value": cost_value,
                    "market_value": market_value,
                    "pnl": float(entry.get("pnl", market_value - cost_value)),
                    "pnl_pct": float(
                        entry.get(
                            "pnl_pct",
                            ((market_value - cost_value) / cost_value * 100.0)
                            if cost_value > 0
                            else 0.0,
                        )
                    ),
                    "sector": str(meta.get("sector") or "Unknown"),
                    "industry": str(meta.get("industry") or "Unknown"),
                    "confidence_pct": float(confidence_report.get("confidence_pct") or 0.0),
                    "confidence_level": str(
                        meta.get("confidence_level")
                        or confidence_report.get("confidence_level")
                        or "very_low"
                    ),
                    "data_quality": str(meta.get("data_quality") or "unknown"),
                }
            )
        return enriched

    def _build_sector_exposure(
        self,
        positions: List[Dict[str, Any]],
        total_market_value: float,
    ) -> List[Dict[str, Any]]:
        buckets: Dict[str, float] = defaultdict(float)
        for item in positions:
            buckets[str(item.get("sector") or "Unknown")] += float(
                item.get("market_value", 0.0)
            )

        exposure = []
        for sector, value in sorted(buckets.items(), key=lambda x: x[1], reverse=True):
            weight_pct = ((value / total_market_value) * 100.0) if total_market_value > 0 else 0.0
            exposure.append(
                {
                    "sector": sector,
                    "market_value": round(value, 2),
                    "weight_pct": round(weight_pct, 2),
                }
            )
        return exposure

    def _build_breaches(
        self,
        positions: List[Dict[str, Any]],
        sector_exposure: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        protection_cfg = self.config.get("portfolio", {}).get("protection", {})
        max_single_pct = float(protection_cfg.get("max_single_position_pct", 40.0) or 40.0)
        sector_limit_pct = float(
            self.config.get("portfolio", {}).get("risk", {}).get("max_sector_weight_pct", 55.0) or 55.0
        )
        low_confidence_threshold = float(
            self.config.get("portfolio", {}).get("risk", {}).get("low_confidence_threshold_pct", 55.0) or 55.0
        )

        breaches: List[Dict[str, Any]] = []
        for item in positions:
            if float(item.get("weight_pct", 0.0)) > max_single_pct:
                breaches.append(
                    {
                        "type": "position_concentration",
                        "severity": "high",
                        "symbol": item.get("symbol"),
                        "message": (
                            f"{item.get('symbol')} weight %{float(item.get('weight_pct', 0.0)):.2f}; "
                            f"limit %{max_single_pct:.2f}."
                        ),
                    }
                )
            if float(item.get("confidence_pct", 0.0)) < low_confidence_threshold:
                breaches.append(
                    {
                        "type": "low_confidence_position",
                        "severity": "medium",
                        "symbol": item.get("symbol"),
                        "message": (
                            f"{item.get('symbol')} confidence %{float(item.get('confidence_pct', 0.0)):.2f}; "
                            f"threshold %{low_confidence_threshold:.2f}."
                        ),
                    }
                )

        for sector in sector_exposure:
            if float(sector.get("weight_pct", 0.0)) > sector_limit_pct:
                breaches.append(
                    {
                        "type": "sector_concentration",
                        "severity": "medium",
                        "sector": sector.get("sector"),
                        "message": (
                            f"{sector.get('sector')} sector weight %{float(sector.get('weight_pct', 0.0)):.2f}; "
                            f"sector limit %{sector_limit_pct:.2f}."
                        ),
                    }
                )
        return breaches

    def _build_correlation_alerts(
        self,
        positions: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[float]]:
        series_map: Dict[str, pd.Series] = {}
        for item in positions:
            symbol = str(item.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            meta = self.market_service.get_company_data(symbol) or {}
            price_history = ((meta.get("priceHistory") or {}).get("series") or [])
            rows = []
            for row in price_history:
                if not isinstance(row, dict):
                    continue
                label = row.get("label")
                value = row.get("y")
                if label is None or value is None:
                    continue
                rows.append((label, value))
            if len(rows) < 20:
                continue
            frame = pd.DataFrame(rows, columns=["date", "price"]).dropna()
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            frame = frame.dropna(subset=["date"]).sort_values("date")
            frame = frame.drop_duplicates(subset=["date"], keep="last").set_index("date")
            try:
                series_map[symbol] = frame["price"].astype(float)
            except Exception:
                continue

        if len(series_map) < 2:
            return [], None

        prices = pd.concat(series_map, axis=1).dropna(how="any")
        if prices.shape[0] < 20:
            return [], None

        returns = prices.pct_change().dropna(how="any")
        if returns.empty:
            return [], None

        corr = returns.corr()
        alerts: List[Dict[str, Any]] = []
        symbols = list(corr.columns)
        for idx, left in enumerate(symbols):
            for right in symbols[idx + 1 :]:
                value = float(corr.loc[left, right])
                if value >= 0.80:
                    alerts.append(
                        {
                            "type": "high_correlation",
                            "severity": "medium",
                            "symbols": [left, right],
                            "message": f"{left} ve {right} korelasyonu yüksek ({value:.2f}).",
                            "correlation": round(value, 4),
                        }
                    )

        estimated_volatility = float(returns.mean(axis=1).std() * (252 ** 0.5) * 100.0)
        return alerts, round(estimated_volatility, 2)

    @staticmethod
    def _compute_risk_score(
        positions: List[Dict[str, Any]],
        breaches: List[Dict[str, Any]],
        correlation_alerts: List[Dict[str, Any]],
        estimated_volatility: Optional[float],
        average_confidence: float,
    ) -> float:
        max_weight = max((float(item.get("weight_pct", 0.0)) for item in positions), default=0.0)
        base = max_weight * 0.45
        base += min(len(breaches) * 8.0, 25.0)
        base += min(len(correlation_alerts) * 5.0, 15.0)
        base += min(max(0.0, (estimated_volatility or 0.0) - 18.0), 20.0)
        base += max(0.0, (60.0 - average_confidence) * 0.3)
        return round(min(100.0, max(0.0, base)), 2)
