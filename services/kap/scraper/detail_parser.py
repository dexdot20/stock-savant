from __future__ import annotations

import json
import re
import threading
import time
from typing import Optional

from bs4 import BeautifulSoup

from core import get_standard_logger

from ..config import BASE_URL, HEADERS, TIMEOUT
from ..models import DisclosureDetail, SignatureInfo
from ..proxy_manager import ProxyManager, fetch_with_retry

logger = get_standard_logger(__name__)

_rate_lock = threading.Lock()
_last_call_at: float = 0.0
_MIN_INTERVAL: float = 0.2


def _rate_limit() -> None:
    global _last_call_at
    with _rate_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
        _last_call_at = time.monotonic()


def _extract_rsc_strings(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    chunks: list[str] = []

    for script in soup.find_all("script"):
        raw = script.string or ""
        if "__next_f" not in raw:
            continue
        for match in re.finditer(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)', raw):
            escaped = match.group(1)
            try:
                chunks.append(json.loads(f'"{escaped}"'))
            except json.JSONDecodeError:
                chunks.append(escaped)

    return "\n".join(chunks)


def _extract_json_object(text: str, key: str) -> Optional[dict]:
    marker = f'"{key}":{{'
    pos = text.find(marker)
    if pos == -1:
        return None

    brace_start = pos + len(marker) - 1
    depth = 0
    index = brace_start
    in_string = False
    escaped = False

    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start : index + 1])
                    except json.JSONDecodeError:
                        return None
        index += 1
    return None


def _extract_json_array(text: str, key: str) -> Optional[list]:
    marker = f'"{key}":['
    pos = text.find(marker)
    if pos == -1:
        return None

    arr_start = pos + len(marker) - 1
    depth = 0
    index = arr_start
    in_string = False
    escaped = False

    while index < len(text):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\" and in_string:
            escaped = True
        elif char == '"':
            in_string = not in_string
        elif not in_string:
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[arr_start : index + 1])
                    except json.JSONDecodeError:
                        return None
        index += 1
    return None


def _find_chunk(rsc: str, hex_key: str) -> Optional[str]:
    text_match = re.search(rf'(?:^|\n){re.escape(hex_key)}:T(\d+),', rsc)
    if text_match:
        length = int(text_match.group(1))
        return rsc[text_match.end() : text_match.end() + length]

    json_match = re.search(rf'(?:^|\n){re.escape(hex_key)}:([^\n]+)', rsc)
    if json_match:
        return json_match.group(1).strip()
    return None


def _resolve_refs(value, rsc: str, depth: int = 0):
    if depth > 4:
        return value
    if isinstance(value, str) and value.startswith("$"):
        content = _find_chunk(rsc, value[1:])
        if content is None:
            return value
        try:
            return _resolve_refs(json.loads(content), rsc, depth + 1)
        except json.JSONDecodeError:
            return content
    if isinstance(value, dict):
        return {key: _resolve_refs(item, rsc, depth + 1) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_refs(item, rsc, depth + 1) for item in value]
    return value


def _get_ref_key(rsc: str, key: str) -> Optional[str]:
    match = re.search(rf'"{re.escape(key)}":"\$([0-9a-f]+)"', rsc)
    return match.group(1) if match else None


def _get_array_ref_key(rsc: str, key: str) -> Optional[str]:
    match = re.search(rf'"{re.escape(key)}":\["\$([0-9a-f]+)"\]', rsc)
    return match.group(1) if match else None


def fetch_html(disclosure_index: int, proxy_manager: Optional[ProxyManager] = None) -> str:
    _rate_limit()
    url = f"{BASE_URL}/tr/Bildirim/{disclosure_index}"
    response = fetch_with_retry(
        "get",
        url,
        headers=HEADERS,
        timeout=TIMEOUT,
        proxy_manager=proxy_manager,
    )
    return response.text


def parse_disclosure(html: str) -> Optional[DisclosureDetail]:
    rsc = _extract_rsc_strings(html)

    basic = _extract_json_object(rsc, "disclosureBasic")
    if not basic:
        return None

    detail = {}
    detail_ref = _get_ref_key(rsc, "disclosureDetail")
    if detail_ref:
        chunk = _find_chunk(rsc, detail_ref)
        if chunk:
            try:
                detail = _resolve_refs(json.loads(chunk), rsc)
            except json.JSONDecodeError:
                detail = {}
    else:
        extracted = _extract_json_object(rsc, "disclosureDetail")
        if extracted:
            detail = _resolve_refs(extracted, rsc)

    body = None
    body_ref = _get_array_ref_key(rsc, "disclosureBody")
    if body_ref:
        body = _find_chunk(rsc, body_ref)
    else:
        body = detail.pop("disclosureBody", None)

    signatures_raw = _extract_json_array(rsc, "signatures")
    if not signatures_raw:
        signatures_ref = _get_ref_key(rsc, "signatures")
        if signatures_ref:
            chunk = _find_chunk(rsc, signatures_ref)
            if chunk:
                try:
                    signatures_raw = json.loads(chunk)
                except json.JSONDecodeError:
                    signatures_raw = None

    signatures: Optional[list[SignatureInfo]] = None
    if signatures_raw:
        signatures = [
            SignatureInfo(**{key: value for key, value in item.items() if value is not None})
            for item in signatures_raw
            if isinstance(item, dict)
        ]

    merged = {**basic, **detail}
    body_text = None
    if body:
        body_text = BeautifulSoup(body, "html.parser").get_text(separator="\n", strip=True) or None

    return DisclosureDetail(
        disclosureIndex=merged.get("disclosureIndex"),
        mkkMemberOid=merged.get("mkkMemberOid"),
        companyTitle=merged.get("companyTitle"),
        stockCode=merged.get("stockCode"),
        disclosureClass=merged.get("disclosureClass"),
        disclosureType=merged.get("disclosureType"),
        disclosureCategory=merged.get("disclosureCategory"),
        publishDate=merged.get("publishDate"),
        summary=merged.get("summary"),
        attachmentCount=merged.get("attachmentCount"),
        year=merged.get("year"),
        period=merged.get("period"),
        isLate=merged.get("isLate"),
        isChanged=merged.get("isChanged"),
        isBlocked=merged.get("isBlocked"),
        senderType=merged.get("senderType"),
        relatedDisclosureOid=merged.get("relatedDisclosureOid"),
        ftNiteligi=merged.get("ftNiteligi"),
        opinion=merged.get("opinion"),
        opinionType=merged.get("opinionType"),
        auditType=merged.get("auditType"),
        mainDisclosureDocumentId=merged.get("mainDisclosureDocumentId"),
        memberType=merged.get("memberType"),
        signatures=signatures,
        disclosureBody=body,
        disclosureBodyText=body_text,
    )


def fetch_and_parse(
    disclosure_index: int,
    proxy_manager: Optional[ProxyManager] = None,
) -> Optional[DisclosureDetail]:
    html = fetch_html(disclosure_index, proxy_manager=proxy_manager)
    result = parse_disclosure(html)
    if result is None:
        logger.warning("KAP disclosure #%d could not be parsed", disclosure_index)
    else:
        logger.debug("KAP disclosure #%d parsed successfully", disclosure_index)
    return result