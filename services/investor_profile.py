from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from core import get_standard_logger
from core.paths import get_investor_profile_path

logger = get_standard_logger(__name__)


DEFAULT_PLAYBOOKS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "label": "Balanced",
        "summary": "Dengeli risk/getiri; kalite, değerleme ve haber akışını birlikte değerlendir.",
    },
    "dividend": {
        "label": "Dividend",
        "summary": "Temettü sürdürülebilirliği, nakit akışı ve savunmacı metriklere öncelik ver.",
    },
    "growth": {
        "label": "Growth",
        "summary": "Gelir büyümesi, kârlılık trendi ve ileriye dönük katalistleri öne çıkar.",
    },
    "defensive": {
        "label": "Defensive",
        "summary": "Volatilite kontrolü, bilanço kalitesi ve risk azaltımını merkeze al.",
    },
}

_RISK_TOLERANCE_VALUES = {"low", "medium", "high"}
_INVESTMENT_HORIZON_VALUES = {"short-term", "medium-term", "long-term"}
_MARKET_FOCUS_VALUES = {"BIST", "US", "Crypto", "All"}
_ALERT_SENSITIVITY_VALUES = {"low", "medium", "high"}


def _default_profile() -> Dict[str, Any]:
    return {
        "profile_name": "Default",
        "risk_tolerance": "medium",
        "investment_horizon": "long-term",
        "market_focus": "BIST",
        "preferred_sectors": [],
        "avoided_sectors": [],
        "max_single_position_pct": 25.0,
        "alert_sensitivity": "medium",
        "active_playbook": "balanced",
    }


def _normalize_sector_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []

    seen: set[str] = set()
    normalized: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        lowered = text.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        normalized.append(text)
    return normalized


def _normalize_choice(value: Any, allowed: set[str], fallback: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else fallback


def _normalize_profile(raw_profile: Any) -> Dict[str, Any]:
    payload = _default_profile()
    if isinstance(raw_profile, dict):
        payload.update(raw_profile)

    profile_name = str(payload.get("profile_name") or "Default").strip() or "Default"
    playbook = str(payload.get("active_playbook") or "balanced").strip()
    if playbook not in DEFAULT_PLAYBOOKS:
        playbook = "balanced"

    try:
        max_single_position_pct = max(1.0, min(100.0, float(payload.get("max_single_position_pct", 25.0))))
    except (TypeError, ValueError):
        max_single_position_pct = 25.0

    return {
        "profile_name": profile_name,
        "risk_tolerance": _normalize_choice(
            payload.get("risk_tolerance"),
            _RISK_TOLERANCE_VALUES,
            "medium",
        ),
        "investment_horizon": _normalize_choice(
            payload.get("investment_horizon"),
            _INVESTMENT_HORIZON_VALUES,
            "long-term",
        ),
        "market_focus": _normalize_choice(
            payload.get("market_focus"),
            _MARKET_FOCUS_VALUES,
            "BIST",
        ),
        "preferred_sectors": _normalize_sector_list(payload.get("preferred_sectors")),
        "avoided_sectors": _normalize_sector_list(payload.get("avoided_sectors")),
        "max_single_position_pct": max_single_position_pct,
        "alert_sensitivity": _normalize_choice(
            payload.get("alert_sensitivity"),
            _ALERT_SENSITIVITY_VALUES,
            "medium",
        ),
        "active_playbook": playbook,
    }


def load_investor_profile() -> Dict[str, Any]:
    path = get_investor_profile_path()
    if not path.exists():
        return _default_profile()

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return _normalize_profile(data)
    except Exception as exc:
        logger.debug("Investor profile read failed: %s", exc)

    return _default_profile()


def save_investor_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    payload = _normalize_profile(profile)

    path = get_investor_profile_path()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return payload


def get_playbook_choices() -> List[str]:
    return list(DEFAULT_PLAYBOOKS.keys())


def get_playbook_summary(playbook_key: str) -> str:
    return str(
        DEFAULT_PLAYBOOKS.get(playbook_key, DEFAULT_PLAYBOOKS["balanced"]).get("summary")
        or ""
    )


def get_analysis_horizon_default() -> str:
    profile = load_investor_profile()
    return str(profile.get("investment_horizon") or "long-term")


def build_investor_context(extra_context: Optional[str] = None) -> str:
    profile = load_investor_profile()
    playbook_key = str(profile.get("active_playbook") or "balanced")
    preferred = ", ".join(profile.get("preferred_sectors") or []) or "None"
    avoided = ", ".join(profile.get("avoided_sectors") or []) or "None"

    parts = [
        f"Investor Profile: {profile.get('profile_name')}",
        f"Risk tolerance: {profile.get('risk_tolerance')}",
        f"Investment horizon: {profile.get('investment_horizon')}",
        f"Market focus: {profile.get('market_focus')}",
        f"Preferred sectors: {preferred}",
        f"Avoided sectors: {avoided}",
        f"Max single position preference: %{float(profile.get('max_single_position_pct', 25.0)):.1f}",
        f"Alert sensitivity: {profile.get('alert_sensitivity')}",
        f"Active playbook: {playbook_key} — {get_playbook_summary(playbook_key)}",
    ]
    if extra_context:
        parts.append(f"Custom user context: {extra_context}")
    return "\n".join(parts)