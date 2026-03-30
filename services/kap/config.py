from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from config import DEFAULT_USER_AGENTS, get_config
from core.paths import get_app_root, get_instance_dir, resolve_path

BASE_URL = "https://kap.org.tr"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_kap_config() -> Dict[str, Any]:
    config = get_config()
    legacy_cfg = config.get("kap_api", {}) or {}
    current_cfg = config.get("kap", {}) or {}
    merged = dict(legacy_cfg)
    merged.update(current_cfg)
    return merged


_KAP_CFG = _load_kap_config()
_DEFAULT_USER_AGENT = (
    DEFAULT_USER_AGENTS[0]
    if isinstance(DEFAULT_USER_AGENTS, list) and DEFAULT_USER_AGENTS
    else (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
    )
)
_KAP_DIR = get_instance_dir() / "kap"
_KAP_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": str(_KAP_CFG.get("user_agent") or _DEFAULT_USER_AGENT),
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TIMEOUT = max(1, _safe_int(_KAP_CFG.get("timeout_seconds"), 30))
DEFAULT_BATCH_WORKERS = max(
    1,
    min(20, _safe_int(_KAP_CFG.get("default_batch_workers"), 5)),
)
MEMBER_MAP_CACHE_FILE = _KAP_DIR / "member_map_cache.json"
MEMBER_MAP_CACHE_TTL = max(
    60,
    _safe_int(_KAP_CFG.get("member_map_cache_ttl_seconds"), 86400),
)
PROXY_FILE_PATH = resolve_path(
    Path(str(_KAP_CFG.get("proxy_file") or "proxies.txt")),
    get_app_root(),
)
PROXY_ERROR_THRESHOLD = max(
    1,
    _safe_int(_KAP_CFG.get("proxy_error_threshold"), 3),
)
PROXY_RESET_INTERVAL = max(
    1.0,
    _safe_float(_KAP_CFG.get("proxy_reset_interval_seconds"), 300.0),
)