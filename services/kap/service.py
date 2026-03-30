from __future__ import annotations

import difflib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core import get_standard_logger

from .config import DEFAULT_BATCH_WORKERS
from .models import DisclosureDetail, DisclosureItem
from .proxy_manager import ProxyManager
from .scraper.detail_parser import fetch_and_parse
from .scraper.member_scraper import _load_member_map, list_disclosures

logger = get_standard_logger(__name__)

_proxy_manager: Optional[ProxyManager] = None
_proxy_manager_lock = threading.Lock()


class KapLookupError(ValueError):
    def __init__(self, details: Dict[str, object]) -> None:
        super().__init__("Some stock codes were not found")
        self.details = details


def get_proxy_manager(reset: bool = False) -> ProxyManager:
    global _proxy_manager
    if _proxy_manager is not None and not reset:
        return _proxy_manager

    with _proxy_manager_lock:
        if _proxy_manager is None or reset:
            _proxy_manager = ProxyManager()
        return _proxy_manager


def clear_proxy_manager_cache() -> None:
    global _proxy_manager
    with _proxy_manager_lock:
        _proxy_manager = None


def normalize_stock_codes(stock_codes: Sequence[str]) -> List[str]:
    normalized_codes: List[str] = []
    seen = set()
    for code in stock_codes:
        if not isinstance(code, str):
            continue
        normalized = code.strip().upper()
        if "." in normalized:
            normalized = normalized.split(".", 1)[0]
        if normalized and normalized not in seen:
            normalized_codes.append(normalized)
            seen.add(normalized)
    return normalized_codes


def _resolve_member_oids(stock_codes: Sequence[str]) -> Tuple[List[str], List[Dict[str, object]]]:
    manager = get_proxy_manager()
    mapping = _load_member_map(proxy_manager=manager)
    member_oids: List[str] = []
    not_found: List[Dict[str, object]] = []

    for code in normalize_stock_codes(stock_codes):
        oid = mapping.get(code)
        if oid:
            member_oids.append(oid)
            continue
        suggestions = difflib.get_close_matches(code, mapping.keys(), n=5, cutoff=0.4)
        not_found.append({"stock_code": code, "suggestions": suggestions})

    return member_oids, not_found


def search_disclosures(
    stock_codes: Sequence[str],
    *,
    category: Optional[str] = None,
    days: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[DisclosureItem]:
    normalized_codes = normalize_stock_codes(stock_codes)
    if not normalized_codes:
        raise ValueError("stock_codes must contain at least one symbol")

    member_oids, not_found = _resolve_member_oids(normalized_codes)
    if not_found:
        raise KapLookupError(
            {
                "message": "Some stock codes were not found.",
                "errors": not_found,
            }
        )

    effective_from_date = from_date
    if days is not None:
        try:
            effective_days = max(1, int(days))
        except (TypeError, ValueError):
            effective_days = None
        if effective_days is not None:
            effective_from_date = (date.today() - timedelta(days=effective_days)).strftime("%Y-%m-%d")

    items = list_disclosures(
        member_oids=member_oids,
        category_filter=category,
        from_date=effective_from_date,
        to_date=to_date,
        proxy_manager=get_proxy_manager(),
    )

    if limit is not None:
        try:
            safe_limit = max(1, int(limit))
        except (TypeError, ValueError):
            safe_limit = None
        if safe_limit is not None:
            return items[:safe_limit]
    return items


def get_disclosure_detail(disclosure_index: int) -> Optional[DisclosureDetail]:
    return fetch_and_parse(int(disclosure_index), proxy_manager=get_proxy_manager())


def batch_get_disclosure_details(
    disclosure_indexes: Sequence[int],
    max_workers: Optional[int] = None,
) -> Tuple[List[Optional[DisclosureDetail]], Dict[int, str]]:
    safe_indexes = [int(index) for index in disclosure_indexes]
    if not safe_indexes:
        return [], {}

    try:
        safe_workers = int(max_workers) if max_workers is not None else DEFAULT_BATCH_WORKERS
    except (TypeError, ValueError):
        safe_workers = DEFAULT_BATCH_WORKERS
    safe_workers = max(1, min(20, safe_workers))

    results: List[Optional[DisclosureDetail]] = [None] * len(safe_indexes)
    errors: Dict[int, str] = {}
    manager = get_proxy_manager()

    with ThreadPoolExecutor(max_workers=safe_workers) as pool:
        future_to_position = {
            pool.submit(fetch_and_parse, disclosure_index, manager): position
            for position, disclosure_index in enumerate(safe_indexes)
        }
        for future in as_completed(future_to_position):
            position = future_to_position[future]
            disclosure_index = safe_indexes[position]
            try:
                detail = future.result()
                if detail is None:
                    errors[disclosure_index] = "Disclosure detail could not be parsed"
                    continue
                results[position] = detail
            except Exception as exc:
                logger.error("KAP disclosure #%d detail fetch failed: %s", disclosure_index, exc)
                errors[disclosure_index] = str(exc)

    return results, errors

    return results, errors