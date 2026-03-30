from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _safe_importance(value: Any, default: int = 5) -> int:
    try:
        return max(1, min(10, int(float(value))))
    except (TypeError, ValueError):
        return max(1, min(10, int(default)))


def _clean_provenance(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    cleaned: Dict[str, Any] = {}
    for key, raw in value.items():
        key_text = str(key).strip()
        if not key_text or raw is None:
            continue
        if isinstance(raw, (str, int, float, bool)):
            if isinstance(raw, str) and not raw.strip():
                continue
            cleaned[key_text] = raw
    return cleaned


def _extract_fact_parts(fact: Any) -> Tuple[str, int, List[str], Dict[str, Any]]:
    if hasattr(fact, "text"):
        text = str(getattr(fact, "text", "") or "").strip()
        importance = _safe_importance(getattr(fact, "importance", 5), 5)
        tags = [str(tag).strip() for tag in getattr(fact, "tags", []) if str(tag).strip()]
        provenance = _clean_provenance(getattr(fact, "provenance", {}))
        return text, importance, tags, provenance

    if isinstance(fact, dict):
        text = str(fact.get("text", "") or "").strip()
        importance = _safe_importance(fact.get("importance", 5), 5)
        tags_raw = fact.get("tags", [])
        tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()] if isinstance(tags_raw, list) else []
        provenance = _clean_provenance(fact.get("provenance", {}))
        return text, importance, tags, provenance

    return str(fact or "").strip(), 5, [], {}


def format_fact_for_rag(fact: Any) -> str:
    text, _, tags, provenance = _extract_fact_parts(fact)
    if not text:
        return ""

    parts: List[str] = []
    if tags:
        parts.append(f"[tags={', '.join(tags)}]")
    parts.append(text)

    if provenance:
        ordered_keys = [
            "evidence_type",
            "source_id",
            "doc_group_id",
            "chunk_index",
            "retrieval_score",
            "timestamp",
        ]
        provenance_parts: List[str] = []
        for key in ordered_keys:
            if key in provenance:
                provenance_parts.append(f"{key}={provenance[key]}")
        for key, value in provenance.items():
            if key not in ordered_keys:
                provenance_parts.append(f"{key}={value}")
        if provenance_parts:
            parts.append(f"[provenance: {'; '.join(provenance_parts)}]")

    return " ".join(part for part in parts if part).strip()


def derive_spill_confidence(facts: Iterable[Any], default: float = 0.55) -> float:
    scores: List[float] = []
    for fact in facts:
        _, importance, _, _ = _extract_fact_parts(fact)
        scores.append(max(1.0, min(10.0, float(importance))) / 10.0)

    if not scores:
        return max(0.0, min(1.0, float(default)))

    average = sum(scores) / len(scores)
    return max(0.35, min(1.0, average))


def derive_spill_data_gap_count(facts: Iterable[Any]) -> int:
    normalized = list(facts)
    if not normalized:
        return 0

    missing = 0
    for fact in normalized:
        _, _, _, provenance = _extract_fact_parts(fact)
        if not provenance:
            missing += 1
    return 0 if missing <= (len(normalized) // 2) else 1


def build_warm_memory_entries(
    hits: Iterable[Dict[str, Any]],
    *,
    max_chars: int = 800,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for hit in hits:
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        content = str(hit.get("content") or "").strip()
        if not content:
            continue

        doc_type = str(metadata.get("doc_type") or hit.get("collection") or "memory")
        timestamp = str(metadata.get("timestamp") or "")
        preview = content[: max(120, int(max_chars))].strip()
        if not preview:
            continue

        quality_score = hit.get("final_score")
        if not isinstance(quality_score, (int, float)):
            quality_score = hit.get("quality_adjusted_score")
        if not isinstance(quality_score, (int, float)):
            quality_score = hit.get("normalized_score")
        retrieval_score = round(float(quality_score or 0.0), 4)
        hit_id = ""
        collection_name = str(hit.get("collection") or "").strip()
        doc_id = str(hit.get("id") or "").strip()
        if collection_name and doc_id:
            hit_id = f"{collection_name}:{doc_id}"

        entries.append(
            {
                "text": f"[RAG:{doc_type}@{timestamp}] {preview}",
                "provenance": {
                    "evidence_type": "rag",
                    "source_id": hit_id or str(metadata.get("source_id") or "rag"),
                    "doc_group_id": str(metadata.get("doc_group_id") or "").strip(),
                    "chunk_index": metadata.get("chunk_index"),
                    "retrieval_score": retrieval_score,
                    "timestamp": timestamp,
                },
            }
        )

    return entries


async def generate_hypothetical_rag_document(
    query: str,
    *,
    request_executor: Any,
    language: str = "English",
    timeout_seconds: float = 15.0,
    max_chars: int = 900,
) -> str:
    normalized_query = str(query or "").strip()
    if not normalized_query or request_executor is None:
        return ""

    system_prompt = (
        "You write short factual hypothetical documents for vector retrieval. "
        "Return plain text only. Do not hedge, do not mention missing information, "
        "and do not use bullet points unless the query clearly asks for a list."
    )
    user_prompt = (
        f"Language: {language}.\n"
        f"Search query: {normalized_query}\n\n"
        "Write a compact hypothetical note that is likely to contain the answer to this query. "
        "Favor entities, dates, metrics, catalysts, risks, and historical references over generic prose."
    )

    try:
        response = await request_executor.send_async(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            request_type="summarizer",
            timeout_override=max(5.0, float(timeout_seconds or 15.0)),
        )
    except Exception:
        return ""

    content = str((response or {}).get("content") or "").strip()
    if not content:
        return ""

    content = re.sub(r"\s+", " ", content).strip()
    return content[: max(200, int(max_chars or 900))]
