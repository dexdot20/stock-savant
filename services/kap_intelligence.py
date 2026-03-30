from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import get_config
from core import get_standard_logger
from core.paths import get_kap_state_path, get_runtime_dir
from news_scraper.async_utils import run_async
from services.tools import execute_tool
from domain.utils import normalize_symbol, safe_int_strict as _safe_int, utc_now_iso

logger = get_standard_logger(__name__)


class KapIntelligenceService:
    """Builds a lightweight KAP event intelligence layer on top of existing tools."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or get_config()
        self._kap_cfg = self.config.get("kap_intelligence", {})

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return normalize_symbol(symbol, drop_suffix_after_dot=True)

    def _state_path(self) -> Path:
        return get_kap_state_path()

    def _load_state(self) -> Dict[str, Any]:
        path = self._state_path()
        if not path.exists():
            return {"seen_indexes": [], "last_scan_at": None}
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                data.setdefault("seen_indexes", [])
                return data
        except Exception as exc:
            logger.debug("KAP state read skipped: %s", exc)
        return {"seen_indexes": [], "last_scan_at": None}

    def _save_state(self, state: Dict[str, Any]) -> None:
        path = self._state_path()
        payload = dict(state)
        payload["last_scan_at"] = utc_now_iso()
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _extract_body_preview(detail: Dict[str, Any], max_chars: int = 320) -> str:
        summary = str(detail.get("summary") or "").strip()
        if summary:
            return summary[:max_chars]

        body = str(detail.get("disclosureBodyText") or "").strip()
        if not body:
            return ""
        body = re.sub(r"\s+", " ", body)
        return body[:max_chars] + ("..." if len(body) > max_chars else "")

    @staticmethod
    def _classify_event(disclosure_class: str, text: str) -> str:
        content = f"{disclosure_class} {text}".lower()
        if any(token in content for token in ("temett", "dividend", "kar payı")):
            return "dividend"
        if any(token in content for token in ("geri al", "buyback", "pay geri")):
            return "buyback"
        if any(token in content for token in ("finansal", "financial", "bilan", "earnings")):
            return "financials"
        if any(token in content for token in ("yatırım", "investment", "ihale", "tender", "capacity", "kapasite")):
            return "investment"
        if any(token in content for token in ("dava", "lawsuit", "investigation", "soruştur")):
            return "legal"
        if any(token in content for token in ("yönetim", "board", "director", "governance")):
            return "governance"
        return "general"

    @staticmethod
    def _severity_score(event_type: str, disclosure_class: str, text: str, attachment_count: Any) -> int:
        base_map = {
            "legal": 82,
            "financials": 76,
            "investment": 74,
            "buyback": 72,
            "dividend": 68,
            "governance": 60,
            "general": 50,
        }
        score = base_map.get(event_type, 50)
        content = f"{disclosure_class} {text}".lower()

        boosts = {
            8: ("birleşme", "merger", "acquisition", "devralma"),
            7: ("bedelsiz", "capital increase", "sermaye artır"),
            6: ("default", "temerr", "material", "önemli", "critical"),
            5: ("guidance", "forecast", "projeksiyon", "revenue"),
            4: ("credit", "rating", "not artır", "not indir"),
        }
        for boost, tokens in boosts.items():
            if any(token in content for token in tokens):
                score += boost

        try:
            if int(attachment_count or 0) > 0:
                score += 3
        except (TypeError, ValueError):
            pass

        return max(0, min(100, score))

    @staticmethod
    def _severity_label(score: int) -> str:
        if score >= 85:
            return "critical"
        if score >= 70:
            return "high"
        if score >= 55:
            return "medium"
        return "low"

    def _search_disclosures(
        self,
        stock_codes: List[str],
        *,
        days: Optional[int] = None,
        limit: Optional[int] = None,
        category: Optional[str] = None,
    ) -> Any:
        return run_async(
            execute_tool(
                "kap_search_disclosures",
                args={
                    "stock_codes": stock_codes,
                    "days": days,
                    "limit": limit,
                    "category": category,
                },
            )
        )

    def _fetch_batch_details(self, disclosure_indexes: List[int], max_workers: int = 5) -> Any:
        return run_async(
            execute_tool(
                "kap_batch_disclosure_details",
                args={
                    "disclosure_indexes": disclosure_indexes,
                    "max_workers": max_workers,
                },
            )
        )

    @staticmethod
    def _coerce_search_items(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get("root"), list):
                return [item for item in payload.get("root") if isinstance(item, dict)]
            if isinstance(payload.get("results"), list):
                return [item for item in payload.get("results") if isinstance(item, dict)]
        return []

    @staticmethod
    def _coerce_detail_items(payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            if isinstance(payload.get("results"), list):
                return [item for item in payload.get("results") if isinstance(item, dict)]
            if payload.get("disclosureIndex") is not None:
                return [payload]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _normalize_events(
        self,
        search_items: List[Dict[str, Any]],
        detail_items: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        detail_map = {
            int(item.get("disclosureIndex")): item
            for item in detail_items
            if item.get("disclosureIndex") is not None
        }

        events: List[Dict[str, Any]] = []
        for item in search_items:
            try:
                disclosure_index = int(item.get("disclosureIndex"))
            except (TypeError, ValueError):
                continue
            detail = detail_map.get(disclosure_index, {})
            stock_code = self._normalize_symbol(item.get("stockCode") or detail.get("stockCode") or "")
            company_title = str(item.get("companyTitle") or detail.get("companyTitle") or stock_code)
            disclosure_class = str(item.get("disclosureClass") or detail.get("disclosureClass") or "General")
            preview = self._extract_body_preview(detail)
            event_type = self._classify_event(disclosure_class, preview)
            severity_score = self._severity_score(
                event_type,
                disclosure_class,
                preview,
                detail.get("attachmentCount"),
            )
            events.append(
                {
                    "disclosure_index": disclosure_index,
                    "symbol": stock_code,
                    "company_title": company_title,
                    "disclosure_class": disclosure_class,
                    "publish_date": str(item.get("publishDate") or detail.get("publishDate") or ""),
                    "event_type": event_type,
                    "severity_score": severity_score,
                    "severity_label": self._severity_label(severity_score),
                    "summary": preview,
                    "attachment_count": int(detail.get("attachmentCount") or 0),
                }
            )

        return sorted(
            events,
            key=lambda item: (
                -int(item.get("severity_score") or 0),
                str(item.get("publish_date") or ""),
            ),
        )

    def get_company_intelligence(
        self,
        symbol: str,
        *,
        days: Optional[int] = None,
        limit: Optional[int] = None,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_symbol = self._normalize_symbol(symbol)
        if not normalized_symbol:
            return {"error": "symbol is required"}

        days = _safe_int(days or self._kap_cfg.get("default_days", 30), 30)
        limit = _safe_int(limit or self._kap_cfg.get("default_limit", 5), 5)
        search_payload = self._search_disclosures(
            [normalized_symbol], days=days, limit=limit, category=category
        )
        if isinstance(search_payload, dict) and search_payload.get("error"):
            return search_payload

        search_items = self._coerce_search_items(search_payload)
        disclosure_indexes: List[int] = []
        for item in search_items[:limit]:
            try:
                disclosure_indexes.append(int(item.get("disclosureIndex")))
            except (TypeError, ValueError):
                continue

        detail_items: List[Dict[str, Any]] = []
        if disclosure_indexes:
            detail_payload = self._fetch_batch_details(disclosure_indexes)
            if isinstance(detail_payload, dict) and detail_payload.get("error"):
                logger.warning("KAP batch detail fetch warning: %s", detail_payload.get("error"))
            else:
                detail_items = self._coerce_detail_items(detail_payload)

        events = self._normalize_events(search_items[:limit], detail_items)
        return {
            "symbol": normalized_symbol,
            "days": days,
            "limit": limit,
            "events": events,
            "count": len(events),
        }

    def get_watchlist_intelligence(
        self,
        symbols: Iterable[str],
        *,
        days: Optional[int] = None,
        per_symbol_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        results = []
        for symbol in symbols:
            payload = self.get_company_intelligence(
                symbol,
                days=days,
                limit=per_symbol_limit,
            )
            if payload.get("error"):
                continue
            results.append(payload)
        return {
            "symbols": [item.get("symbol") for item in results],
            "results": results,
            "count": sum(int(item.get("count", 0)) for item in results),
        }

    def evaluate_alertable_events(
        self,
        symbols: Iterable[str],
        *,
        days: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        days = _safe_int(days or self._kap_cfg.get("alert_days", 7), 7)
        min_score = _safe_int(self._kap_cfg.get("min_alert_severity_score", 70), 70)
        state = self._load_state()
        seen_indexes = {int(item) for item in state.get("seen_indexes", []) if isinstance(item, int) or str(item).isdigit()}

        watchlist_payload = self.get_watchlist_intelligence(
            symbols,
            days=days,
            per_symbol_limit=_safe_int(self._kap_cfg.get("default_limit", 5), 5),
        )
        alerts: List[Dict[str, Any]] = []
        newly_seen = set(seen_indexes)

        for company_payload in watchlist_payload.get("results", []):
            for event in company_payload.get("events", []):
                index = int(event.get("disclosure_index") or 0)
                if index <= 0:
                    continue
                if index in seen_indexes:
                    continue
                newly_seen.add(index)
                if int(event.get("severity_score") or 0) < min_score:
                    continue
                alerts.append(
                    {
                        "type": "kap",
                        "severity": event.get("severity_label", "medium"),
                        "symbol": event.get("symbol"),
                        "message": (
                            f"KAP olayı: {event.get('symbol')} | {event.get('disclosure_class')} | "
                            f"şiddet={event.get('severity_score')} | {event.get('summary') or 'Özet yok'}"
                        ),
                        "event": event,
                    }
                )

        state["seen_indexes"] = sorted(newly_seen)
        self._save_state(state)
        return alerts

    def save_markdown_report(self, payload: Dict[str, Any]) -> Optional[Path]:
        symbol = str(payload.get("symbol") or "").strip().upper()
        if not symbol:
            return None
        reports_dir = get_runtime_dir() / "instance" / "reports" / "kap"
        reports_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}_kap.md"
        path = reports_dir / filename

        lines = [f"# KAP Intelligence Report: {symbol}", ""]
        for event in payload.get("events", []):
            lines.extend(
                [
                    f"## {event.get('disclosure_class')} [{event.get('severity_label')}]",
                    f"- Publish Date: {event.get('publish_date')}",
                    f"- Event Type: {event.get('event_type')}",
                    f"- Severity Score: {event.get('severity_score')}",
                    f"- Summary: {event.get('summary') or '-'}",
                    "",
                ]
            )
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
        return path