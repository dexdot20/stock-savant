from __future__ import annotations

from datetime import datetime
import json
import re
from typing import Any, Dict, List, Optional

from services.ai.providers.tool_call_parser import normalize_tool_calls


def normalize_history_archives(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        segment = item.get("segment")
        if not isinstance(segment, list):
            continue
        normalized.append(
            {
                "reason": str(item.get("reason") or "history_summarized"),
                "archived_at": str(item.get("archived_at") or datetime.now().isoformat()),
                "segment": segment,
            }
        )
    return normalized


def archive_history_segment(
    archives: List[Dict[str, Any]],
    *,
    segment: List[Dict[str, Any]],
    reason: str = "history_summarized",
) -> None:
    if not segment:
        return
    archives.append(
        {
            "reason": str(reason or "history_summarized"),
            "archived_at": datetime.now().isoformat(),
            "segment": segment,
        }
    )


def extract_tool_name_from_tool_result(
    content: Any,
    *,
    history: Optional[List[Dict[str, Any]]] = None,
    message_index: Optional[int] = None,
) -> str:
    if isinstance(content, dict):
        role = str(content.get("role") or "").strip().lower()
        if role == "tool":
            resolved = _resolve_tool_name_from_history(
                history,
                message_index,
                str(content.get("tool_call_id") or "").strip(),
            )
            if resolved:
                return resolved
        content = content.get("content", "")

    if not isinstance(content, str):
        return "tool"
    match = re.search(r'<tool_result\s+name="([^"]+)"', content)
    if not match:
        return "tool"
    return match.group(1)


def extract_tool_payload_from_result(content: Any) -> Any:
    if isinstance(content, dict):
        content = content.get("content", content)
    if not isinstance(content, str):
        return content

    match = re.search(r"<tool_result\s+name=\"[^\"]+\">\s*(.*?)\s*</tool_result>", content, re.DOTALL)
    payload_text = match.group(1).strip() if match else content.strip()
    if not payload_text:
        return ""

    try:
        return json.loads(payload_text)
    except Exception:
        return payload_text


def _resolve_tool_name_from_history(
    history: Optional[List[Dict[str, Any]]],
    message_index: Optional[int],
    tool_call_id: str,
) -> str:
    if not history or message_index is None or not tool_call_id:
        return ""

    for idx in range(message_index - 1, -1, -1):
        message = history[idx]
        if str(message.get("role") or "").strip().lower() != "assistant":
            continue
        for tool_call in normalize_tool_calls(message.get("tool_calls")):
            if str(tool_call.get("id") or "").strip() == tool_call_id:
                return str(tool_call.get("name") or "").strip()
    return ""


def build_ephemeral_evidence_record(
    tool_name: str,
    tool_content: Any,
    *,
    max_preview_chars: int = 700,
) -> Optional[Dict[str, Any]]:
    payload = extract_tool_payload_from_result(tool_content)
    source_id = ""
    title = ""
    url = ""
    payload_kind = type(payload).__name__

    if isinstance(payload, dict):
        if payload.get("error"):
            return None
        base_payload = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        if isinstance(base_payload, dict):
            url = str(base_payload.get("url") or base_payload.get("link") or "").strip()
            title = str(base_payload.get("title") or base_payload.get("name") or "").strip()
            source_id = str(
                base_payload.get("source_id")
                or url
                or title
                or base_payload.get("source")
                or base_payload.get("source_info")
                or ""
            ).strip()
            compact_payload = {
                key: base_payload.get(key)
                for key in (
                    "url",
                    "title",
                    "source",
                    "source_info",
                    "published_at",
                    "author",
                    "summary",
                    "content",
                    "mode",
                    "count",
                )
                if base_payload.get(key)
            }
            if not compact_payload:
                compact_payload = base_payload
            preview = json.dumps(
                compact_payload,
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            )
        else:
            source_id = str(payload.get("source_id") or payload.get("url") or tool_name).strip()
            preview = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    elif isinstance(payload, list):
        source_id = tool_name
        preview = json.dumps(payload[:3], ensure_ascii=False, separators=(",", ":"), default=str)
    else:
        source_id = tool_name
        preview = str(payload)

    preview = preview.strip()[: max(200, int(max_preview_chars))]
    if not preview:
        return None

    return {
        "tool_name": str(tool_name or "tool").strip() or "tool",
        "source_id": source_id or str(tool_name or "tool"),
        "title": title,
        "url": url,
        "captured_at": datetime.now().isoformat(),
        "preview": preview,
        "payload_kind": payload_kind,
        "tags": ["ephemeral", str(tool_name or "tool").strip() or "tool"],
    }


def build_ephemeral_fallback_fact(
    tool_name: str,
    tool_content: Any,
    *,
    max_preview_chars: int = 1200,
) -> Optional[str]:
    preview = ""

    if isinstance(tool_content, str):
        try:
            parsed = json.loads(tool_content)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            if parsed.get("error"):
                return None
            preview = json.dumps(
                parsed,
                ensure_ascii=False,
                separators=(",", ":"),
            )[:max(200, int(max_preview_chars))]
        else:
            preview = tool_content[:max(200, int(max_preview_chars))]
    else:
        preview = str(tool_content)[:max(200, int(max_preview_chars))]

    preview = preview.strip()
    if not preview:
        return None

    return (
        f"[Fallback:{tool_name}] Ephemeral output automatically backed up before stub: "
        f"{preview}"
    )
