from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from config import get_config
from core import get_standard_logger
from core.paths import get_alerts_path
from domain.utils import normalize_symbol, utc_now_iso
from services.finance.risk import PortfolioRiskService

logger = get_standard_logger(__name__)


def load_alerts() -> List[Dict[str, Any]]:
    alerts_path = get_alerts_path()
    if not alerts_path.exists():
        return []
    try:
        with open(alerts_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except Exception as exc:
        logger.debug("Alerts read failed: %s", exc)
    return []


def save_alerts(alerts: List[Dict[str, Any]]) -> None:
    alerts_path = get_alerts_path()
    with open(alerts_path, "w", encoding="utf-8") as handle:
        json.dump(alerts, handle, ensure_ascii=False, indent=2)


def create_price_alert(
    symbol: str,
    target_price: float,
    direction: str,
    note: str = "",
) -> Dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    normalized_direction = str(direction or "").strip().lower()
    if not normalized_symbol:
        return {"error": "symbol is required"}
    if normalized_direction not in {"above", "below"}:
        return {"error": "direction must be 'above' or 'below'"}

    try:
        target = float(target_price)
    except Exception:
        return {"error": "target_price must be numeric"}

    if target <= 0:
        return {"error": "target_price must be greater than 0"}

    alerts = load_alerts()
    alert_entry = {
        "id": str(uuid4()),
        "type": "price",
        "symbol": normalized_symbol,
        "target_price": target,
        "direction": normalized_direction,
        "note": str(note or "").strip(),
        "triggered": False,
        "status": "active",
        "created_at": utc_now_iso(),
    }
    alerts.append(alert_entry)
    save_alerts(alerts)
    return {"status": "set", "alert": alert_entry}


def list_alerts(
    *,
    symbol: Optional[str] = None,
    include_triggered: bool = False,
) -> Dict[str, Any]:
    normalized_symbol = normalize_symbol(symbol)
    alerts = []
    for item in load_alerts():
        if normalized_symbol and str(item.get("symbol", "")).upper() != normalized_symbol:
            continue
        if not include_triggered and bool(item.get("triggered", False)):
            continue
        alerts.append(item)
    return {"count": len(alerts), "alerts": alerts}


def evaluate_price_alerts(
    price_lookup: Optional[callable] = None,
) -> List[Dict[str, Any]]:
    from services.factories import get_financial_service

    alerts = load_alerts()
    if not alerts:
        return []

    changed = False
    triggered_events: List[Dict[str, Any]] = []
    service = None

    def _resolve_price(symbol: str) -> Optional[float]:
        nonlocal service
        if callable(price_lookup):
            return price_lookup(symbol)
        if service is None:
            service = get_financial_service()
        data = service.get_company_data(symbol)
        if not isinstance(data, dict):
            return None
        price = data.get("currentPrice") or data.get("regularMarketPrice")
        try:
            return float(price) if price is not None else None
        except (TypeError, ValueError):
            return None

    for alert in alerts:
        if str(alert.get("type") or "price") != "price":
            continue
        if bool(alert.get("triggered", False)):
            continue

        symbol = normalize_symbol(alert.get("symbol"))
        direction = str(alert.get("direction") or "").strip().lower()
        try:
            target_price = float(alert.get("target_price") or 0.0)
        except (TypeError, ValueError):
            target_price = 0.0
        if not symbol or target_price <= 0 or direction not in {"above", "below"}:
            continue

        current_price = _resolve_price(symbol)
        if current_price is None or current_price <= 0:
            continue

        is_triggered = (
            direction == "above" and current_price >= target_price
        ) or (
            direction == "below" and current_price <= target_price
        )
        if not is_triggered:
            continue

        alert["triggered"] = True
        alert["status"] = "triggered"
        alert["triggered_at"] = utc_now_iso()
        alert["current_price"] = current_price
        changed = True
        triggered_events.append(
            {
                "type": "price",
                "severity": "medium",
                "symbol": symbol,
                "message": (
                    f"Price alert triggered: {symbol} target={target_price:.4f}, "
                    f"direction={direction}, current={current_price:.4f}"
                ),
                "alert_id": alert.get("id"),
            }
        )

    if changed:
        save_alerts(alerts)
    return triggered_events


def evaluate_portfolio_risk_alerts(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    from commands.portfolio import load_portfolio

    entries = load_portfolio()
    if not entries:
        return []
    risk_service = PortfolioRiskService(config)
    snapshot = risk_service.build_snapshot(entries)
    return list(snapshot.get("breaches") or []) + list(snapshot.get("correlation_alerts") or [])


def evaluate_kap_alerts(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    from commands.favorites import load_favorites
    from commands.portfolio import load_portfolio
    from services.kap_intelligence import KapIntelligenceService

    symbols = set(load_favorites())
    symbols.update(
        normalize_symbol(item.get("symbol"))
        for item in load_portfolio()
        if isinstance(item, dict)
    )
    symbols = {symbol for symbol in symbols if symbol}
    if not symbols:
        return []

    service = KapIntelligenceService(config)
    return service.evaluate_alertable_events(sorted(symbols))


def evaluate_alert_center(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = config or get_config()
    price_events = evaluate_price_alerts()
    risk_events = evaluate_portfolio_risk_alerts(cfg)
    kap_events = evaluate_kap_alerts(cfg)

    return {
        "triggered_events": price_events + risk_events + kap_events,
        "price_events": price_events,
        "risk_events": risk_events,
        "kap_events": kap_events,
        "saved_alerts": list_alerts(include_triggered=True),
    }
