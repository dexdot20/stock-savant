from __future__ import annotations

import difflib
import json
import re
import threading
import time
from datetime import date, timedelta
from typing import Optional

from bs4 import BeautifulSoup

from core import get_standard_logger

from ..config import BASE_URL, HEADERS, MEMBER_MAP_CACHE_FILE, MEMBER_MAP_CACHE_TTL, TIMEOUT
from ..models import DisclosureItem
from ..proxy_manager import ProxyManager, fetch_with_retry

logger = get_standard_logger(__name__)

_CRITERIA_URL = f"{BASE_URL}/tr/api/disclosure/members/byCriteria"
_FORM_URL = f"{BASE_URL}/tr/bildirim-sorgu"

_cache_lock = threading.Lock()
_member_map_cache: Optional[dict] = None
_member_map_loaded_at: float = 0.0


def _extract_rsc_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    chunks: list[str] = []
    for script in soup.find_all("script"):
        raw = script.string or ""
        if "__next_f" not in raw:
            continue
        for match in re.finditer(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', raw):
            try:
                chunks.append(json.loads(f'"{match.group(1)}"'))
            except json.JSONDecodeError:
                chunks.append(match.group(1))
    return "\n".join(chunks)


def _load_member_map(
    proxy_manager: Optional[ProxyManager] = None,
    force_refresh: bool = False,
) -> dict:
    global _member_map_cache, _member_map_loaded_at

    if not force_refresh and _member_map_cache is not None:
        if (time.time() - _member_map_loaded_at) < MEMBER_MAP_CACHE_TTL:
            return _member_map_cache

    with _cache_lock:
        if not force_refresh and _member_map_cache is not None:
            if (time.time() - _member_map_loaded_at) < MEMBER_MAP_CACHE_TTL:
                return _member_map_cache

        if not force_refresh and MEMBER_MAP_CACHE_FILE.exists():
            try:
                stored = json.loads(MEMBER_MAP_CACHE_FILE.read_text(encoding="utf-8"))
                if "updated_at" in stored and "data" in stored:
                    age = time.time() - stored["updated_at"]
                    if age < MEMBER_MAP_CACHE_TTL:
                        _member_map_cache = stored["data"]
                        _member_map_loaded_at = stored["updated_at"]
                        logger.info(
                            "KAP member map loaded from disk cache: %d entries age=%.0fs",
                            len(_member_map_cache),
                            age,
                        )
                        return _member_map_cache
            except (KeyError, json.JSONDecodeError, OSError):
                pass

        logger.info("Refreshing KAP member map from source")
        response = fetch_with_retry(
            "get",
            _FORM_URL,
            headers=HEADERS,
            timeout=TIMEOUT,
            proxy_manager=proxy_manager,
        )
        rsc = _extract_rsc_text(response.text)

        mapping = {}
        for match in re.finditer(r'"mkkMemberOid"\s*:\s*"([^"]+)"', rsc):
            oid = match.group(1)
            segment = rsc[match.start() : match.start() + 3000]
            stock_code_match = re.search(r'"stockCode"\s*:\s*"([^"]+)"', segment)
            if stock_code_match:
                mapping[stock_code_match.group(1).upper()] = oid

        now = time.time()
        cache_payload = {"updated_at": now, "data": mapping}
        tmp_path = MEMBER_MAP_CACHE_FILE.with_suffix(".tmp")
        try:
            tmp_path.write_text(json.dumps(cache_payload, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(MEMBER_MAP_CACHE_FILE)
        except OSError as exc:
            logger.warning("Failed to persist KAP member map cache: %s", exc)

        _member_map_cache = mapping
        _member_map_loaded_at = now
        logger.info("KAP member map refreshed: %d entries", len(mapping))
        return mapping


def lookup_member_oid(stock_code: str) -> Optional[str]:
    mapping = _load_member_map()
    oid = mapping.get(stock_code.upper())
    if oid:
        return oid

    suggestions = difflib.get_close_matches(stock_code.upper(), mapping.keys(), n=5, cutoff=0.4)
    if not suggestions:
        return None

    while True:
        try:
            raw = input(
                "Select a suggested stock code (1-%d) or 0 to cancel: " % len(suggestions)
            ).strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if raw == "0":
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(suggestions):
            return mapping[suggestions[int(raw) - 1]]


def lookup_member_oid_noninteractive(
    stock_code: str,
    proxy_manager: Optional[ProxyManager] = None,
    force_refresh: bool = False,
) -> dict:
    mapping = _load_member_map(proxy_manager=proxy_manager, force_refresh=force_refresh)
    code = stock_code.upper()
    oid = mapping.get(code)
    if oid:
        return {"stock_code": code, "oid": oid, "suggestions": [], "found": True}
    suggestions = difflib.get_close_matches(code, mapping.keys(), n=5, cutoff=0.4)
    return {"stock_code": code, "oid": None, "suggestions": suggestions, "found": False}


def list_disclosures(
    member_oids: list[str],
    category_filter: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    proxy_manager: Optional[ProxyManager] = None,
) -> list[DisclosureItem]:
    if not member_oids:
        return []

    today = date.today()
    payload = {
        "fromDate": from_date or (today - timedelta(days=365)).strftime("%Y-%m-%d"),
        "toDate": to_date or today.strftime("%Y-%m-%d"),
        "memberType": "IGS",
        "mkkMemberOidList": member_oids,
        "inactiveMkkMemberOidList": [],
        "disclosureClass": (
            category_filter.upper()
            if category_filter and category_filter.upper() != "ALL"
            else ""
        ),
        "subjectList": [],
        "isLate": "",
        "mainSector": "",
        "sector": "",
        "subSector": "",
        "marketOid": "",
        "index": "",
        "bdkReview": "",
        "bdkMemberOidList": [],
        "year": "",
        "term": "",
        "ruleType": "",
        "period": "",
        "fromSrc": False,
        "srcCategory": "",
        "disclosureIndexList": [],
    }

    api_headers = {
        **HEADERS,
        "Content-Type": "application/json",
        "Origin": BASE_URL,
        "Referer": _FORM_URL,
    }

    response = fetch_with_retry(
        "post",
        _CRITERIA_URL,
        headers=api_headers,
        timeout=TIMEOUT,
        proxy_manager=proxy_manager,
        json=payload,
    )
    raw_list = response.json()

    items: list[DisclosureItem] = []
    for entry in raw_list:
        raw_related = entry.get("relatedStocks") or ""
        related_list = [stock.strip() for stock in raw_related.split(",") if stock.strip()]
        mapped = {
            "disclosureIndex": entry.get("disclosureIndex"),
            "stockCode": related_list[0] if len(related_list) == 1 else entry.get("stockCode"),
            "relatedStocks": related_list or None,
            "companyTitle": entry.get("kapTitle") or entry.get("companyTitle"),
            "disclosureClass": entry.get("disclosureClass"),
            "disclosureType": entry.get("disclosureType"),
            "disclosureCategory": entry.get("disclosureCategory"),
            "publishDate": entry.get("publishDate"),
        }
        try:
            item = DisclosureItem.model_validate(mapped)
            item.detail_url = f"{BASE_URL}/tr/Bildirim/{item.disclosure_index}"
            items.append(item)
        except Exception as exc:
            logger.warning("Skipping invalid KAP disclosure entry: %s", exc)

    logger.info("Fetched %d KAP disclosures", len(items))
    return items