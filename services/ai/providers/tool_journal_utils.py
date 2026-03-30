from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence


def _compact_tool_value(value: Any, *, max_length: int = 96) -> str:
    if isinstance(value, (dict, list, tuple, set)):
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = str(value)
    else:
        text = str(value)

    text = " ".join(text.split())
    if len(text) <= max_length:
        return text
    return text[: max(0, max_length - 3)].rstrip() + "..."


def _format_tool_args(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""

    preferred_keys = (
        "query",
        "url",
        "symbol",
        "ticker",
        "company_name",
        "stock_codes",
        "disclosure_indexes",
        "hit_ids",
        "collection",
        "limit",
        "max_results",
        "max_workers",
        "category",
        "from_date",
        "to_date",
        "depth_mode",
        "criteria",
    )
    parts: List[str] = []
    seen_keys: set[str] = set()

    for key in preferred_keys:
        if key in args and args.get(key) not in (None, "", [], {}):
            parts.append(f"{key}={_compact_tool_value(args.get(key))}")
            seen_keys.add(key)

    for key in sorted(args.keys()):
        if key in seen_keys:
            continue
        value = args.get(key)
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}={_compact_tool_value(value)}")

    if not parts:
        return ""

    return " (" + ", ".join(parts) + ")"


def _infer_result_count(value: Any) -> Optional[int]:
    if isinstance(value, list):
        return len(value)
    if not isinstance(value, dict):
        return None

    for key in ("results", "quotes", "news", "items", "signatures"):
        nested = value.get(key)
        if isinstance(nested, list):
            return len(nested)

    data = value.get("data")
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        return _infer_result_count(data)
    return None


def _extract_metadata_snippets(value: Any) -> List[str]:
    if not isinstance(value, dict):
        return []

    snippets: List[str] = []
    key_map = (
        ("url", "url"),
        ("title", "title"),
        ("symbol", "symbol"),
        ("stockCode", "stock"),
        ("companyTitle", "company"),
        ("disclosureIndex", "disclosure"),
        ("query", "query"),
        ("task_id", "task"),
    )
    for key, label in key_map:
        if key in value and value.get(key) not in (None, "", [], {}):
            snippets.append(f"{label}={_compact_tool_value(value.get(key), max_length=48)}")
        if len(snippets) >= 2:
            break
    return snippets


def _compact_text_summary(value: Any, *, max_length: int = 160) -> str:
    text = _compact_tool_value(value, max_length=max_length)
    return text.strip()


def summarize_memory_update_args(args: Any) -> Dict[str, Any]:
    payload = args if isinstance(args, dict) else {}
    facts = len(payload.get("new_facts") or []) if isinstance(payload.get("new_facts"), list) else 0
    questions = len(payload.get("new_questions") or []) if isinstance(payload.get("new_questions"), list) else 0
    contradictions = len(payload.get("contradictions") or []) if isinstance(payload.get("contradictions"), list) else 0
    milestones = len(payload.get("research_milestones") or []) if isinstance(payload.get("research_milestones"), list) else 0
    resolved = len(payload.get("resolve_questions") or []) if isinstance(payload.get("resolve_questions"), list) else 0
    parts: List[str] = []
    if facts:
        parts.append(f"facts={facts}")
    if questions:
        parts.append(f"questions={questions}")
    if contradictions:
        parts.append(f"contradictions={contradictions}")
    if milestones:
        parts.append(f"milestones={milestones}")
    if resolved:
        parts.append(f"resolved={resolved}")
    if not parts and payload.get("source_summary"):
        parts.append(f"source={_compact_text_summary(payload.get('source_summary'), max_length=64)}")
    return {
        "summary": ", ".join(parts) if parts else "empty",
        "facts": facts,
        "questions": questions,
        "contradictions": contradictions,
        "milestones": milestones,
        "resolved": resolved,
    }


def summarize_tool_result(tool_name: str, result: Any) -> Dict[str, Any]:
    payload = result
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {"raw": payload}

    success = True
    error_text = ""
    data = payload
    if isinstance(payload, dict) and "success" in payload:
        success = bool(payload.get("success"))
        error_text = str(payload.get("error") or payload.get("error_code") or "").strip()
        data = payload.get("data")
    elif isinstance(payload, dict):
        error_text = str(payload.get("error") or payload.get("error_code") or "").strip()
        success = not bool(error_text)
        if "data" in payload:
            data = payload.get("data")

    status = "ok" if success else "error"
    summary_parts: List[str] = []
    if error_text:
        summary_parts.append(_compact_text_summary(error_text, max_length=88))
    else:
        count = _infer_result_count(data)
        if count is not None:
            summary_parts.append(f"items={count}")
        metadata = _extract_metadata_snippets(data if isinstance(data, dict) else payload if isinstance(payload, dict) else {})
        summary_parts.extend(metadata[:2])
        if not summary_parts and data not in (None, "", [], {}):
            if isinstance(data, (str, int, float, bool)):
                summary_parts.append(_compact_text_summary(data, max_length=72))
            elif isinstance(data, dict):
                keys = ",".join(list(data.keys())[:3])
                if keys:
                    summary_parts.append(f"keys={keys}")

    return {
        "name": str(tool_name or "tool").strip() or "tool",
        "status": status,
        "summary": ", ".join(part for part in summary_parts if part),
    }


def normalize_tool_journal_step(
    step: int,
    tool_calls: Sequence[Dict[str, Any]],
    deferred_tools: Optional[Sequence[str]] = None,
    assistant_summary: Optional[str] = None,
    memory_updates: Optional[Sequence[Dict[str, Any]]] = None,
    tool_results: Optional[Sequence[Dict[str, Any]]] = None,
    notes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    tools: List[Dict[str, Any]] = []
    for tool_call in tool_calls or []:
        if not isinstance(tool_call, dict):
            continue
        tool_name = str(tool_call.get("name") or "tool").strip() or "tool"
        args = tool_call.get("args")
        tools.append(
            {
                "name": tool_name,
                "args": args if isinstance(args, dict) else {},
            }
        )

    record: Dict[str, Any] = {
        "step": int(step),
        "tools": tools,
    }
    if assistant_summary:
        cleaned = _compact_text_summary(assistant_summary, max_length=180)
        if cleaned:
            record["assistant_summary"] = cleaned
    if isinstance(memory_updates, Sequence):
        normalized_updates = []
        for item in memory_updates:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary") or "").strip()
            if not summary:
                continue
            normalized_updates.append({"summary": summary})
        if normalized_updates:
            record["memory_updates"] = normalized_updates
    if isinstance(tool_results, Sequence):
        normalized_results = []
        for item in tool_results:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "tool").strip() or "tool"
            status = str(item.get("status") or "ok").strip() or "ok"
            summary = str(item.get("summary") or "").strip()
            normalized_results.append(
                {
                    "name": name,
                    "status": status,
                    "summary": summary,
                }
            )
        if normalized_results:
            record["tool_results"] = normalized_results
    if deferred_tools:
        cleaned = [str(tool).strip() for tool in deferred_tools if str(tool).strip()]
        if cleaned:
            record["deferred_tools"] = cleaned
    if notes:
        cleaned_notes = [
            _compact_text_summary(note, max_length=120)
            for note in notes
            if _compact_text_summary(note, max_length=120)
        ]
        if cleaned_notes:
            record["notes"] = cleaned_notes
    return record


def normalize_tool_journal(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []

    journal: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        raw_tools = item.get("tools")
        if not isinstance(raw_tools, list):
            continue

        step = item.get("step")
        try:
            step_value = int(step)
        except (TypeError, ValueError):
            step_value = len(journal) + 1

        tools: List[Dict[str, Any]] = []
        for tool in raw_tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name") or "tool").strip() or "tool"
            args = tool.get("args")
            tools.append(
                {
                    "name": tool_name,
                    "args": args if isinstance(args, dict) else {},
                }
            )

        has_context = bool(
            item.get("assistant_summary")
            or item.get("memory_updates")
            or item.get("tool_results")
            or item.get("notes")
        )
        if not tools and not has_context:
            continue

        record: Dict[str, Any] = {
            "step": step_value,
            "tools": tools,
        }
        assistant_summary = str(item.get("assistant_summary") or "").strip()
        if assistant_summary:
            record["assistant_summary"] = assistant_summary

        raw_memory_updates = item.get("memory_updates")
        if isinstance(raw_memory_updates, list):
            memory_updates = []
            for update in raw_memory_updates:
                if not isinstance(update, dict):
                    continue
                summary = str(update.get("summary") or "").strip()
                if summary:
                    memory_updates.append({"summary": summary})
            if memory_updates:
                record["memory_updates"] = memory_updates

        raw_tool_results = item.get("tool_results")
        if isinstance(raw_tool_results, list):
            tool_results = []
            for tool_result in raw_tool_results:
                if not isinstance(tool_result, dict):
                    continue
                name = str(tool_result.get("name") or "tool").strip() or "tool"
                status = str(tool_result.get("status") or "ok").strip() or "ok"
                summary = str(tool_result.get("summary") or "").strip()
                tool_results.append(
                    {
                        "name": name,
                        "status": status,
                        "summary": summary,
                    }
                )
            if tool_results:
                record["tool_results"] = tool_results

        deferred_tools = item.get("deferred_tools")
        if isinstance(deferred_tools, list):
            cleaned = [str(tool).strip() for tool in deferred_tools if str(tool).strip()]
            if cleaned:
                record["deferred_tools"] = cleaned
        raw_notes = item.get("notes")
        if isinstance(raw_notes, list):
            notes = [str(note).strip() for note in raw_notes if str(note).strip()]
            if notes:
                record["notes"] = notes
        journal.append(record)

    return journal


def format_tool_journal_for_prompt(
    tool_journal: Sequence[Dict[str, Any]],
    *,
    title: str = "Chronological step trace in this run",
) -> str:
    if not tool_journal:
        return ""

    normalized = normalize_tool_journal(list(tool_journal))
    if not normalized:
        return ""

    lines: List[str] = [title + ":"]
    for record in normalized:
        step = record.get("step")
        step_label = str(step) if step is not None else "?"
        tools = record.get("tools", []) if isinstance(record.get("tools"), list) else []
        parts: List[str] = []
        assistant_summary = str(record.get("assistant_summary") or "").strip()
        if assistant_summary:
            parts.append(f"assistant: {assistant_summary}")

        memory_updates = record.get("memory_updates", []) if isinstance(record.get("memory_updates"), list) else []
        for update in memory_updates:
            if not isinstance(update, dict):
                continue
            summary = str(update.get("summary") or "").strip()
            if summary:
                parts.append(f"memory_update({summary})")

        tool_results = record.get("tool_results", []) if isinstance(record.get("tool_results"), list) else []
        result_index = 0
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_name = str(tool.get("name") or "tool").strip() or "tool"
            if tool_name == "update_working_memory":
                continue
            tool_text = f"{tool_name}{_format_tool_args(tool.get('args'))}"
            result_item = None
            if tool_name != "finish":
                while result_index < len(tool_results):
                    candidate = tool_results[result_index]
                    result_index += 1
                    if not isinstance(candidate, dict):
                        continue
                    candidate_name = str(candidate.get("name") or "tool").strip() or "tool"
                    if candidate_name == tool_name:
                        result_item = candidate
                        break
            if isinstance(result_item, dict):
                status = str(result_item.get("status") or "ok").strip() or "ok"
                summary = str(result_item.get("summary") or "").strip()
                if summary:
                    tool_text += f" [{status}: {summary}]"
                else:
                    tool_text += f" [{status}]"
            parts.append(tool_text)

        if not parts:
            continue

        line = f"- Step {step_label}: " + " -> ".join(parts)
        deferred_tools = record.get("deferred_tools")
        if isinstance(deferred_tools, list) and deferred_tools:
            line += " -> deferred: " + ", ".join(
                str(tool).strip() for tool in deferred_tools if str(tool).strip()
            )
        notes = record.get("notes")
        if isinstance(notes, list) and notes:
            line += " -> notes: " + " | ".join(
                str(note).strip() for note in notes if str(note).strip()
            )
        lines.append(line)

    return "\n".join(lines)