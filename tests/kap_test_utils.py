from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from services.kap.scraper import member_scraper as member_scraper_module

SAMPLE_MEMBER_MAP = {
    "AKBNK": "oid-001",
    "THYAO": "oid-002",
    "GARAN": "oid-003",
    "ISCTR": "oid-004",
}


def build_member_rsc_html(mapping: dict[str, str]) -> str:
    fragments = []
    for stock_code, oid in mapping.items():
        fragments.append(f'"mkkMemberOid":"{oid}","stockCode":"{stock_code.lower()}"')
    rsc_content = ",".join(fragments)
    escaped = rsc_content.replace("\\", "\\\\").replace('"', '\\"')
    return f'<script>self.__next_f.push([1,"{escaped}"])</script>'


def build_disclosure_rsc_html(**fields: Any) -> str:
    inner = json.dumps(fields, ensure_ascii=False, separators=(",", ":"))[1:-1]
    escaped = inner.replace("\\", "\\\\").replace('"', '\\"')
    return f'<script>self.__next_f.push([1,"{escaped}"])</script>'


def write_cache_file(path: Path, data: dict, age_seconds: float = 0.0) -> None:
    payload = {"updated_at": time.time() - age_seconds, "data": data}
    path.write_text(json.dumps(payload), encoding="utf-8")


def reset_member_cache() -> None:
    member_scraper_module._member_map_cache = None
    member_scraper_module._member_map_loaded_at = 0.0
    member_scraper_module._member_map_loaded_at = 0.0