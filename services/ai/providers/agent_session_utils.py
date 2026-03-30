from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from core.paths import get_sessions_path


def _slugify(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return "session"
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "session"


def generate_agent_session_id(agent_name: str, *parts: Any) -> str:
    pieces = [_slugify(agent_name)]
    for part in parts:
        slug = _slugify(part)
        if slug and slug != "session":
            pieces.append(slug)
            break
    pieces.append(uuid.uuid4().hex[:8])
    return "-".join(pieces)


def get_agent_session_path(agent_name: str, session_id: str):
    safe_session_id = _slugify(session_id)
    safe_agent_name = _slugify(agent_name)
    return get_sessions_path() / f"{safe_agent_name}_{safe_session_id}.json"


def load_agent_session(agent_name: str, session_id: str) -> Optional[Dict[str, Any]]:
    path = get_agent_session_path(agent_name, session_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def save_agent_session(
    agent_name: str,
    session_id: str,
    payload: Dict[str, Any],
) -> str:
    path = get_agent_session_path(agent_name, session_id)
    now_iso = datetime.now().isoformat()

    record: Dict[str, Any] = dict(payload or {})
    record["agent_name"] = str(agent_name)
    record["session_id"] = str(session_id)
    record["last_updated"] = now_iso

    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict):
                record["created_at"] = str(existing.get("created_at") or now_iso)
        except Exception:
            record["created_at"] = now_iso
    else:
        record["created_at"] = now_iso

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, default=str)
    return str(path)
